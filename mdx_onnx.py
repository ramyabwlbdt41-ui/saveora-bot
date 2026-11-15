"""
MDX-Net vocal separation — numpy + onnxruntime only (no torch, no librosa).

إعادة تنفيذ لمسار MDX في UVR / audio-separator (رخصة MIT) لكن بالاعتماد على
numpy و onnxruntime فقط، حتى يعمل على جهاز بلا GPU وبتحميل صغير.

النموذج المدعوم: Kim_Vocal_2.onnx (MDX-Net B, n_fft=6144, dim_f=3072, dim_t=256)

الاستخدام:
    python mdx_onnx.py <in.wav> <out_vocals.wav> [--inst out_inst.wav]
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
HOP = 1024


def _hann(n: int) -> np.ndarray:
    """hann دوري (مطابق لـ torch.hann_window(periodic=True))."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


# ---------------------------------------------------------------------------
# WAV إدخال/إخراج
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"عرض عيّنة غير مدعوم: {sw*8} bit")

    data = data.reshape(-1, ch).T
    if ch == 1:
        data = np.vstack([data, data])
    elif ch > 2:
        data = data[:2]
    return np.ascontiguousarray(data), sr


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    data = np.clip(data, -1.0, 1.0)
    interleaved = data.T.reshape(-1)
    pcm = (interleaved * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(x.shape[-1] * sr_out / sr_in))
    idx = np.linspace(0, x.shape[-1] - 1, n_out, dtype=np.float64)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, x.shape[-1] - 1)
    frac = (idx - lo).astype(np.float32)
    return (x[:, lo] * (1 - frac) + x[:, hi] * frac).astype(np.float32)


# ---------------------------------------------------------------------------
# المحرّك
# ---------------------------------------------------------------------------

class MDXSeparator:
    def __init__(
        self,
        model_path: str | Path,
        n_fft: int = 6144,
        dim_f: int = 3072,
        dim_t: int = 256,
        hop: int = HOP,
        overlap: float = 0.25,
        progress=None,
    ) -> None:
        import onnxruntime as ort  # استيراد مؤجّل

        self.model_path = str(model_path)
        self.n_fft = n_fft
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.hop = hop
        self.overlap = overlap
        self.n_bins = n_fft // 2 + 1
        self.trim = n_fft // 2
        self.progress = progress

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 0
        self.session = ort.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    # -- إعدادات من داخل ملف ONNX إن وُجدت --------------------------------
    def read_model_metadata(self) -> dict:
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            for key in ("model_data", "model_data_json"):
                if key in meta:
                    return json.loads(meta[key])
        except Exception:
            pass
        return {}

    # -- STFT / iSTFT ----------------------------------------------------
    def stft(self, x: np.ndarray) -> np.ndarray:
        """x: (2, T) → (4, dim_f, frames) بترتيب [ch0.real, ch0.imag, ch1.real, ch1.imag]."""
        pad = self.n_fft // 2
        xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(xp, self.n_fft, axis=-1)
        frames = frames[:, :: self.hop, :] * _hann(self.n_fft).astype(np.float32)
        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)  # (2, T, bins)
        spec = np.transpose(spec, (0, 2, 1))  # (2, bins, T)
        packed = np.stack([spec.real, spec.imag], axis=1).astype(np.float32)  # (2,2,bins,T)
        packed = packed.reshape(x.shape[0] * 2, self.n_bins, spec.shape[-1])
        return np.ascontiguousarray(packed[:, : self.dim_f, :])

    def istft(self, spec: np.ndarray, length: int) -> np.ndarray:
        """spec: (4, dim_f, frames) → (2, length)."""
        n_ch = spec.shape[0] // 2
        bins = np.zeros((n_ch, 2, self.n_bins, spec.shape[-1]), dtype=np.float32)
        bins[:, :, : self.dim_f, :] = spec.reshape(n_ch, 2, self.dim_f, spec.shape[-1])
        comp = bins[:, 0, :, :] + 1j * bins[:, 1, :, :]  # (ch, bins, T)
        frames = np.fft.irfft(comp, n=self.n_fft, axis=1)  # (ch, n_fft, T)
        frames *= _hann(self.n_fft).astype(np.float32)[None, :, None]

        hop = self.hop
        n_frames = frames.shape[-1]
        total = (n_frames - 1) * hop + self.n_fft
        out = np.zeros((n_ch, total), dtype=np.float32)
        wsum = np.zeros((1, total), dtype=np.float32)
        win = _hann(self.n_fft).astype(np.float32)
        win_sq = (win * win)[None, :]

        for i in range(n_frames):
            a = i * hop
            out[:, a : a + self.n_fft] += frames[:, :, i]
            wsum[:, a : a + self.n_fft] += win_sq

        out /= np.maximum(wsum, 1e-8)
        pad = self.n_fft // 2
        return out[:, pad : pad + length]

    # -- قطعة واحدة ------------------------------------------------------
    def run_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (2, chunk_size) → (2, chunk_size)."""
        spec = self.stft(chunk)
        spec[:, :3, :] = 0.0  # مطابق لـ UVR: تصفير أول 3 ترددات
        pred = self.session.run(None, {self.input_name: spec[None, ...]})[0][0]
        return self.istft(pred, chunk.shape[-1])

    # -- الفصل الكامل ----------------------------------------------------
    def demix(self, mix: np.ndarray) -> np.ndarray:
        """mix: (2, N) float32 @44100 → vocals (2, N)."""
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        scaled = mix * (1.0 / peak) if peak > 1.0 else mix
        vocals = self._demix_core(np.ascontiguousarray(scaled, dtype=np.float32))
        if peak > 1.0:
            vocals *= peak
        return vocals

    def _demix_core(self, mix: np.ndarray) -> np.ndarray:
        chunk_size = self.hop * (self.dim_t - 1)
        trim = self.trim
        gen_size = chunk_size - 2 * trim
        if gen_size <= 0:
            raise RuntimeError("إعدادات النموذج غير متوافقة")

        pad = gen_size + trim - (mix.shape[-1] % gen_size)
        mixture = np.concatenate(
            (
                np.zeros((2, trim), dtype=np.float32),
                mix.astype(np.float32),
                np.zeros((2, pad), dtype=np.float32),
            ),
            axis=1,
        )

        step = int((1.0 - self.overlap) * chunk_size)
        total_len = mixture.shape[-1]
        result = np.zeros((2, total_len), dtype=np.float32)
        divider = np.zeros((2, total_len), dtype=np.float32)

        starts = list(range(0, total_len, step))
        for n, start in enumerate(starts, 1):
            end = min(start + chunk_size, total_len)
            actual = end - start
            part = mixture[:, start:end]
            if actual < chunk_size:
                part = np.concatenate(
                    (part, np.zeros((2, chunk_size - actual), dtype=np.float32)), axis=1
                )

            out = self.run_chunk(part)

            if self.overlap:
                window = np.hanning(actual).astype(np.float32)
                out[:, :actual] *= window
                divider[:, start:end] += window
            else:
                divider[:, start:end] += 1.0

            result[:, start:end] += out[:, :actual]
            if self.progress:
                self.progress(n, len(starts))

        result /= np.maximum(divider, 1e-8)
        vocals = result[:, trim:-trim]
        return np.ascontiguousarray(vocals[:, : mix.shape[-1]])


# ---------------------------------------------------------------------------
# التشغيل المستقل
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", default=str(Path(__file__).parent / "models" / "Kim_Vocal_2.onnx"))
    ap.add_argument("--inst", default=None)
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    mix, sr = read_wav(Path(args.input))
    if sr != SR:
        mix = resample_linear(mix, sr, SR)

    sep = MDXSeparator(args.model)
    print(f"model metadata: {sep.read_model_metadata()}", flush=True)
    print(f"mix: {mix.shape} @{SR}", flush=True)

    vocals = sep.demix(mix)
    write_wav(Path(args.output), vocals, SR)

    if args.inst:
        inst = mix[:, : vocals.shape[-1]] - vocals
        write_wav(Path(args.inst), inst, SR)

    print(f"OK {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
