"""
فصل سريع للكلام عن الموسيقى — numpy فقط، بلا ONNX، بلا GPU.

يعمل على مقاطع صغيرة (blocks) حتى لا تتضخم الذاكرة على السيرفرات الصغيرة
(512 MB)، لذلك يصلح لـ Render free.

الطريقة:
  1) طيف قصير (N_FFT=1024) لتقدير الطاقة لكل تردد.
  2) تقدير "الخلفية الموسيقية" بمُرشّح وسيط (median) على مدى زمني ~1.6 ثانية
     لكل تردد — الموسيقى تدوم، والكلام متقطّع.
  3) طرح طيفي بمُعامل زيادة + قناع ناعم (Wiener) مع تنعيم زمني/ترددي.
  4) تقليل الباص العميق والحفيف العالي (موسيقى/ضجيج).

الاستخدام:
    python fast_sep.py in.wav out.wav
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 44100
N_FFT = 1024
HOP = 256

# حجم الكتلة الواحدة (ثانية) + تراكب — يحدّد الذمّة القصوى
BLOCK_SEC = 8.0
OVERLAP_SEC = 0.6

MED_SEC = 1.6        # نافذة الوسيط لتقدير الخلفية
GROUP_FRAMES = 16    # تجميع الإطارات لتقدير سريع للخلفية (~93 ms)
OVER_SUB = 1.6       # معامل الطرح الزائد
MASK_FLOOR = 0.06    # أقل قيمة للقناع (يمنع الاختفاء الكامل)


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError("عرض عيّنة غير مدعوم")
    data = data.reshape(-1, ch).T
    if ch == 1:
        data = np.vstack([data, data])
    elif ch > 2:
        data = data[:2]
    return np.ascontiguousarray(data, dtype=np.float32), sr


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    data = np.clip(data, -1.0, 1.0)
    pcm = (data.T.reshape(-1) * 32767.0).astype("<i2")
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


def _frames(x: np.ndarray, win: np.ndarray) -> np.ndarray:
    """x: (ch, N) → إطارات (ch, T, N_FFT)."""
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    view = np.lib.stride_tricks.sliding_window_view(xp, N_FFT, axis=-1)[:, ::HOP, :]
    return np.ascontiguousarray(view * win)


def _istft(frames: np.ndarray, length: int) -> np.ndarray:
    """frames: (ch, T, N_FFT) → (ch, length)."""
    win = _hann(N_FFT)
    n_ch, n_frames = frames.shape[0], frames.shape[1]
    total = (n_frames - 1) * HOP + N_FFT
    out = np.zeros((n_ch, total), dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float32)
    win_sq = win * win
    block = frames * win[None, None, :]
    for i in range(n_frames):
        a = i * HOP
        out[:, a : a + N_FFT] += block[:, i, :]
        wsum[a : a + N_FFT] += win_sq
    out /= np.maximum(wsum, 1e-8)
    pad = N_FFT // 2
    return out[:, pad : pad + length]


def _band_limit() -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    band = np.ones_like(freqs, dtype=np.float32)
    band[freqs < 90] = np.clip(freqs[freqs < 90] / 90.0, 0.10, 1.0)
    band[freqs > 7500] = np.clip((11000.0 - freqs[freqs > 7500]) / 3500.0, 0.20, 1.0)
    return band.astype(np.float32)


def _coarse_floor(mag: np.ndarray) -> np.ndarray:
    """تقدير الخلفية المستمرة بكلفة صغيرة: أدنى قيمة في نوافذ قصيرة ثم وسيط.

    الموسيقى مستوى مستمر (أدنى قيمة ≈ مستواها)، والكلام متقطّع (أدنى قيمة منخفض).
    """
    bins, t_len = mag.shape
    g = int(max(1, min(GROUP_FRAMES, t_len)))
    n_g = t_len // g
    if n_g < 3:
        return np.full_like(mag, float(np.percentile(mag, 25)))

    coarse = mag[:, : n_g * g].reshape(bins, n_g, g).min(axis=2)   # (bins, n_g)
    span = max(3, int(round(MED_SEC * SR / HOP / g)) | 1)
    pad = span // 2
    padded = np.pad(coarse, ((0, 0), (pad, pad)), mode="edge")
    win_view = np.lib.stride_tricks.sliding_window_view(padded, span, axis=1)
    coarse = np.median(win_view, axis=-1)                          # (bins, n_g)

    floor = np.repeat(coarse, g, axis=1)
    if floor.shape[1] < t_len:
        floor = np.concatenate(
            [floor, np.repeat(coarse[:, -1:], t_len - floor.shape[1], axis=1)], axis=1
        )
    return floor.astype(np.float32)


def _process_block(
    seg: np.ndarray, win: np.ndarray, band: np.ndarray
) -> np.ndarray:
    """seg: (2, n) → كلام مقدّر (2, n)."""
    mono = seg.mean(axis=0, dtype=np.float32)
    wf = _frames(mono[None, :], win)[0]              # (T, N_FFT)
    mag = np.abs(np.fft.rfft(wf, axis=-1)).T.astype(np.float32)    # (bins, T)
    t_len = mag.shape[1]

    floor = _coarse_floor(mag)

    # قناع Wiener: نُبقي ما يزيد بوضوح عن الخلفية
    ratio = mag / (mag + OVER_SUB * floor + 1e-8)
    mask = np.clip(ratio, MASK_FLOOR, 1.0).astype(np.float32)

    # تنعيم: تردد (3) ثم زمن (5)
    padded = np.pad(mask, ((1, 1), (0, 0)), mode="edge")
    mask = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    k = 5
    padded = np.pad(mask, ((0, 0), (k // 2, k // 2)), mode="edge")
    acc = np.zeros_like(mask)
    for i in range(k):
        acc += padded[:, i : i + t_len]
    mask = acc / k

    mask = np.clip(mask * band[:, None], MASK_FLOOR, 1.0).astype(np.float32)

    sp = np.fft.rfft(_frames(seg, win), axis=-1)      # (2, T, bins)
    sp *= mask.T[None, :, :]
    out = np.fft.irfft(sp, n=N_FFT, axis=-1).astype(np.float32)
    return _istft(out, seg.shape[-1])


def separate(mix: np.ndarray, sr: int = SR) -> np.ndarray:
    """mix: (2, N) → كلام مقدّر (2, N). يعمل على كتل لتقليل الذاكرة."""
    x = mix
    if sr != SR:
        x = resample_linear(x, sr, SR)

    win = _hann(N_FFT)
    band = _band_limit()

    n = x.shape[-1]
    block = max(N_FFT * 8, int(BLOCK_SEC * SR))
    ramp = max(N_FFT, int(OVERLAP_SEC * SR))
    step = max(N_FFT * 4, block - ramp)

    out = np.zeros_like(x)
    wsum = np.zeros(n, dtype=np.float32)

    pos = 0
    while pos < n:
        end = min(n, pos + block)
        seg = x[:, pos:end]
        if seg.shape[-1] < N_FFT * 2:           # ذيل قصير جداً — نكمله
            seg = np.pad(seg, ((0, 0), (0, N_FFT * 2 - seg.shape[-1])), mode="constant")
        clean = _process_block(seg, win, band)
        clean = clean[:, : end - pos]

        w = np.ones(clean.shape[-1], dtype=np.float32)
        r = min(ramp, clean.shape[-1] // 2)
        if r > 0:
            if pos > 0:
                w[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
            if end < n:
                w[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
        out[:, pos:end] += clean * w
        wsum[pos:end] += w
        if end >= n:
            break
        pos += step

    out /= np.maximum(wsum, 1e-8)[None, :]
    return out.astype(np.float32)


def main() -> int:
    if len(sys.argv) < 3:
        print("USAGE: fast_sep.py in.wav out.wav", file=sys.stderr)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    x, sr = read_wav(src)
    y = separate(x, sr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_wav(dst, y, SR)
    print(f"OK {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
