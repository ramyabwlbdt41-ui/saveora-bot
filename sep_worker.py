"""
عامل فصل الموسيقى — يعمل في بيئة Python 3.12 (venv312) حيث numpy + onnxruntime.

الاستخدام:
    python sep_worker.py <input.wav> <workdir> <output.wav> [model_path]

يطبع في آخر سطر:  OK <path>
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) < 4:
        print("USAGE: sep_worker.py input.wav workdir output.wav [model]", file=sys.stderr)
        return 2

    src = Path(sys.argv[1])
    workdir = Path(sys.argv[2])
    dest = Path(sys.argv[3])
    model = Path(sys.argv[4]) if len(sys.argv) > 4 else (HERE / "models" / "Kim_Vocal_2.onnx")

    if not model.exists():
        print(f"MODEL_MISSING {model}", file=sys.stderr)
        return 4

    workdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))

    from mdx_onnx import MDXSeparator, read_wav, write_wav, resample_linear, SR

    mix, sr = read_wav(src)
    if sr != SR:
        mix = resample_linear(mix, sr, SR)

    def progress(n: int, total: int) -> None:
        if n % 4 == 0 or n == total:
            print(f"PROGRESS {n}/{total}", flush=True)

    sep = MDXSeparator(model, progress=progress)
    vocals = sep.demix(mix)

    dest.parent.mkdir(parents=True, exist_ok=True)
    write_wav(dest, vocals, SR)
    print(f"OK {dest}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
