"""
فصل سريع للكلام عن الموسيقى — numpy فقط، بلا ONNX، بلا GPU.

الفكرة: الموسيقى المستمرة (باص، أوتار، باد، إيقاع ثابت) تبقى بطيف شبه ثابت
مع الوقت، أما الكلام فمتقطّع وسريع التغيّر. لذلك نقدّر "مستوى الخلفية المستمر"
لكل تردد بمتتبّع بطيء (attack/release طويل)، ونطرحه من الطيف الحالي مع قناع ناعم.

النتيجة: أسرع ~100× من MDX (ثوانٍ بدل دقائق) بجودة أقل من نموذج AI لكن مقبولة
للكلام على خلفية موسيقية.

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


def separate(mix: np.ndarray, sr: int = SR) -> np.ndarray:
    """mix: (2, N) → كلام مقدّر (2, N)."""
    x = mix
    if sr != SR:
        x = resample_linear(x, sr, SR)

    win = _hann(N_FFT)
    # نقنع قناة واحدة (متوسط) لتقدير القناع، ثم نطبّقه على الاستيريو
    mono = x.mean(axis=0, dtype=np.float32)
    frames = _frames(mono[None, :], win)[0]          # (T, N_FFT)
    mag = np.abs(np.fft.rfft(frames, axis=-1)).T     # (bins, T)

    bins, n_frames = mag.shape
    hop_rate = SR / HOP                              # إطار/ثانية ≈ 172

    # متتبّع "الخلفية المستمرة": يصعد ببطء (0.8 ثانية) ويهبط ببطء (1.8 ثانية)
    a_up = float(1.0 - np.exp(-1.0 / (0.8 * hop_rate)))
    a_dn = float(1.0 - np.exp(-1.0 / (1.8 * hop_rate)))

    env = np.zeros(bins, dtype=np.float32)
    floor = np.zeros_like(mag)
    for t in range(n_frames):
        cur = mag[:, t]
        up = cur > env
        env = np.where(up, env + (cur - env) * a_up, env + (cur - env) * a_dn)
        floor[:, t] = env

    # قناع ناعم: نُبقي ما يزيد على الخلفية بشكل واضح
    ratio = np.clip((mag - 1.15 * floor) / (mag + 1e-8), 0.0, 1.0)

    # تنعيم زمني بسيط لتقليل "الطقطقة" المعدنية
    k = 3
    ker = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    padded = np.pad(ratio, ((0, 0), (pad, pad)), mode="edge")
    smooth = np.zeros_like(ratio)
    for i in range(k):
        smooth += padded[:, i : i + n_frames]
    ratio = smooth / k

    # حدّ ترددي: خفض الباص العميق (موسيقى) والحفيف العالي
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    band = np.ones_like(freqs, dtype=np.float32)
    band[freqs < 90] = np.clip(freqs[freqs < 90] / 90.0, 0.15, 1.0)
    band[freqs > 7500] = np.clip((11000 - freqs[freqs > 7500]) / 3500.0, 0.25, 1.0)

    mask = ratio * band[:, None]
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    # تطبيق القناع على القناتين
    wf = _frames(x, win)                             # (2, T, N_FFT)
    sp = np.fft.rfft(wf, axis=-1)                    # (2, T, bins)
    sp *= mask.T[None, :, :]
    out = np.fft.irfft(sp, n=N_FFT, axis=-1).astype(np.float32)
    return _istft(out, x.shape[-1])


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
