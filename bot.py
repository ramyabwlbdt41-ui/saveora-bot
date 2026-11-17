"""
SAVEORA — بوت تلغرام: أرسل رابطاً ← يصلك الملف مباشرة، ثم أزرار:
🎵 إزالة الموسيقى  ·  🎤 الكلام فقط  ·  🔇 بدون صوت

yt-dlp يدعم ~1800 موقع · ffmpeg للدمج والمعالجة
فصل الموسيقى: MDX-Net ONNX + onnxruntime (على المعالج، بلا كرت شاشة، بلا torch)

التشغيل:  python bot.py
"""

from __future__ import annotations

import asyncio
import html
import http.server
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
import yt_dlp

load_dotenv()

# على Windows قد يكون ترميز الطرفية cp1256 فيتعطّل عند طباعة العربية
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------
# الإعدادات
# ----------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BRAND = "SAVEORA"
HERE = Path(__file__).resolve().parent

ALLOWED_USERS: set[int] = {
    int(x) for x in re.split(r"[,\s]+", os.getenv("ALLOWED_USER_IDS", "")) if x.strip().isdigit()
}

MAX_MB = int(os.getenv("MAX_FILESIZE_MB", "50"))
MAX_DURATION_MIN = int(os.getenv("MAX_DURATION_MIN", "60"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
# أقصى مدة فيديو يسمح بفصل الموسيقى عنه (المعالجة على المعالج)
SEP_MAX_MINUTES = int(os.getenv("SEPARATOR_MAX_MINUTES", "12"))

COOKIE_FILE = os.getenv("COOKIES_FILE", "cookies.txt").strip()
COOKIES_BROWSER = os.getenv("COOKIES_FROM_BROWSER", "").strip()

# فصل الموسيقى
SEP_MODEL = os.getenv("SEPARATOR_MODEL", "Kim_Vocal_2.onnx")
MODEL_DIR = HERE / "models"
# المحرّك: fast = DSP سريع (ثوانٍ، ذاكرة صغيرة) · mdx = نموذج AI (جودة أعلى، بطيء جداً)
SEP_ENGINE = (os.getenv("SEPARATOR_ENGINE", "fast") or "fast").strip().lower()


def _find_sep_python() -> str | None:
    """بيئة Python منفصلة تعمل عليها حزم ML (numpy + onnxruntime)."""
    env = os.getenv("SEPARATOR_PYTHON", "").strip()
    if env and Path(env).exists():
        return env
    for cand in (
        HERE / "venv312" / "Scripts" / "python.exe",
        HERE / "venv312" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    if importlib.util.find_spec("audio_separator") is not None:
        return sys.executable
    # على السيرفر (Linux) لا يوجد venv منفصل — نستخدم نفس المفسّر إن توفرت الحزم
    if all(importlib.util.find_spec(m) is not None for m in ("numpy", "onnxruntime")):
        return sys.executable
    return None


SEP_PYTHON = _find_sep_python()
_SEP_OK: bool | None = None
_SEP_AT = 0.0

URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("audio_separator").setLevel(logging.WARNING)
log = logging.getLogger("saveora")

SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)
SEP_SEMAPHORE = asyncio.Semaphore(1)

LADDER = ["1080", "720", "480", "360", "audio"]
# الجودة الافتراضية: 720p (أسرع وأخف) — و1080 متاح بزر
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "720")

# ---------------------------------------------------------------------------
# لوحة المالك: معرّفات المدراء (من .env) + بيانات محفوظة محلياً
# ---------------------------------------------------------------------------
ADMIN_IDS: set[int] = {
    int(x) for x in re.split(r"[^\d]+", os.getenv("ADMIN_IDS", "")) if x.isdigit()
}

DATA_DIR = HERE / "_data"
USERS_FILE = DATA_DIR / "users.json"
CONF_FILE = DATA_DIR / "conf.json"
# انتظار إدخال من المدير: uid → "broadcast" | "channel"
PENDING: dict[int, str] = {}

# ملفات محفوظة مؤقتاً حتى تعمل الأزرار بلا إعادة تحميل
CACHE_DIR = HERE / "_cache"
JOBS: dict[str, dict[str, Any]] = {}
JOB_TTL = 3 * 60 * 60
MAX_JOBS = int(os.getenv("MAX_JOBS", "12"))
CACHE_MAX_BYTES = int(os.getenv("CACHE_MAX_MB", "350")) * 1024 * 1024

# ----------------------------------------------------------------------------
# نصوص الواجهة
# ----------------------------------------------------------------------------

def welcome_text(name: str) -> str:
    return (
        f"⚡️ <b>{BRAND}</b>\n"
        f"<i>محمّل الوسائط الذكي</i>\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"أهلاً <b>{html.escape(name)}</b> 👋\n"
        "أرسل الرابط — وسيصلك الملف <b>مباشرة</b>.\n\n"
        "▸ <b>يوتيوب · تيك توك · إنستغرام</b>\n"
        "▸ <b>تويتر/X · فيسبوك · سناب · بينترست</b>\n"
        "▸ روابط مباشرة <code>.mp4</code> و <code>.m3u8</code>\n\n"
        "بعد وصول الفيديو تظهر أزرار:\n"
        "🎵 <b>إزالة الموسيقى</b> — يبقى الكلام وحده\n"
        "🎤 الكلام فقط · 🔇 بدون صوت\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🔒 لا تسجيل · لا إعلانات · بلا وسطاء\n"
        f"📦 حتى {MAX_MB} MB  ·  ⏱ حتى {MAX_DURATION_MIN} دقيقة"
    )


HELP_TEXT = (
    "📖 <b>طريقة الاستخدام</b>\n"
    "━━━━━━━━━━━━━━━━━━\n\n"
    "<b>1.</b> انسخ رابط الفيديو والصقه هنا.\n"
    "<b>2.</b> سيصلك الملف تلقائياً بأعلى جودة تحت 50 MB.\n"
    "<b>3.</b> تحت الفيديو أزرار:\n"
    "   🎵 <b>إزالة الموسيقى</b> — يفصل الكلام عن الموسيقى والإيقاع ويُعيد الفيديو بصوت بشري فقط.\n"
    "   🎤 <b>الكلام فقط</b> — يرسل الصوت المفصول كملف صوتي.\n"
    "   🔇 <b>بدون صوت</b> — يحذف المسار الصوتي بالكامل.\n\n"
    "⚙️ <b>ملاحظات</b>\n"
    "• إزالة الموسيقى تعمل بالذكاء الاصطناعي: ممتازة، لكن ليست سحرية —\n"
    "  قد يبقى أثر خفيف أو يتغيّر طابع الصوت قليلاً.\n"
    "• المدة الطويلة تحتاج وقتاً: تقريباً دقيقة معالجة لكل دقيقة فيديو.\n"
    "• المحتوى المحمي DRM (نتفليكس/شاهد) غير مدعوم.\n"
    "• معرّفك: /id · الحالة: /stats"
)


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📖 طريقة الاستخدام", callback_data="ui|help"),
                InlineKeyboardButton("📊 الحالة", callback_data="ui|stats"),
            ]
        ]
    )


def media_keyboard(
    token: str, audio_only: bool = False, quality: str | None = None
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if audio_only:
        rows.append(
            [InlineKeyboardButton("🎤 الكلام فقط (بدون موسيقى)", callback_data=f"mx|voice|{token}")]
        )
    else:
        rows.append([InlineKeyboardButton("🎵 إزالة الموسيقى", callback_data=f"mx|music|{token}")])
        rows.append(
            [
                InlineKeyboardButton("🎤 الكلام فقط", callback_data=f"mx|voice|{token}"),
                InlineKeyboardButton("🔇 بدون صوت", callback_data=f"mx|mute|{token}"),
            ]
        )
    if quality and quality not in {"1080", "audio"}:
        rows.append([InlineKeyboardButton("⬆️ نسخة 1080p", callback_data=f"mx|hd|{token}")])
    return InlineKeyboardMarkup(rows)


async def safe_edit(q, text: str, **kw) -> bool:
    """تعديل رسالة الكولباك دون أن يوقف التنفيذ إن كانت مطابقة أو قديمة."""
    try:
        await q.edit_message_text(text, **kw)
        return True
    except BadRequest as exc:
        if "not modified" in str(exc).lower():
            return True
        log.warning("edit failed: %s", str(exc)[:160])
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("edit failed: %s", str(exc)[:160])
        return False


def admin_keyboard() -> InlineKeyboardMarkup:
    c = conf()
    gate = "✅ يعمل" if c.get("gate") else "⛔️ متوقف"
    ch = str(c.get("channel") or "") or "غير محددة"
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📢 نشر رسالة للجميع", callback_data="adm|bc")],
            [InlineKeyboardButton(f"📡 القناة: {ch}", callback_data="adm|ch")],
            [InlineKeyboardButton(f"🔐 اشتراك إجباري: {gate}", callback_data="adm|gate")],
            [InlineKeyboardButton("📊 إحصائيات المستخدمين", callback_data="adm|stats")],
        ]
    )


# ----------------------------------------------------------------------------
# أدوات
# ----------------------------------------------------------------------------

def human_size(num: float | None) -> str:
    if not num:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def fmt_duration(sec: float | None) -> str:
    if not sec:
        return "—"
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def progress_bar(pct: float, width: int = 12) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


_last_request: dict[int, float] = {}


def rate_limited(user_id: int, seconds: float = 3.0) -> bool:
    now = time.time()
    if now - _last_request.get(user_id, 0) < seconds:
        return True
    _last_request[user_id] = now
    return False


def cookie_opts() -> dict[str, Any]:
    if COOKIE_FILE and Path(COOKIE_FILE).exists():
        return {"cookiefile": COOKIE_FILE}
    if COOKIES_BROWSER:
        return {"cookiesfrombrowser": (COOKIES_BROWSER,)}
    return {}


def ffmpeg_path() -> str | None:
    env = os.getenv("FFMPEG_LOCATION", "").strip()
    if env:
        p = Path(env)
        if p.is_dir() and (p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")).exists():
            return str(p)
        if p.is_file():
            return str(p.parent)
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    local = HERE / "ffmpeg" / exe
    if local.exists():
        return str(local.parent)
    which = shutil.which("ffmpeg")
    if which:
        return str(Path(which).parent)

    # آخر خيار: بناء ffmpeg مرفق مع حزمة imageio-ffmpeg ← ننسخه باسم ffmpeg.exe
    try:
        import imageio_ffmpeg

        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        dst = HERE / "ffmpeg" / exe
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        return str(dst.parent)
    except Exception:
        return None


_FFMPEG_DIR: str | None = None
_FFMPEG_AT = 0.0


def ffmpeg_dir() -> str | None:
    """اكتشاف كسول: إن لم يوجد ffmpeg نعيد الفحص كل 60 ثانية بلا إعادة تشغيل."""
    global _FFMPEG_DIR, _FFMPEG_AT
    if _FFMPEG_DIR:
        return _FFMPEG_DIR
    if time.time() - _FFMPEG_AT > 60:
        _FFMPEG_AT = time.time()
        _FFMPEG_DIR = ffmpeg_path()
    return _FFMPEG_DIR


def ffmpeg_exe() -> str | None:
    d = ffmpeg_dir()
    if not d:
        return None
    exe = Path(d) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return str(exe) if exe.exists() else shutil.which("ffmpeg")


def separator_ready() -> bool:
    """هل محرّك الفصل متاح؟ (fast: numpy فقط · mdx: numpy + onnxruntime + النموذج)"""
    global _SEP_OK, _SEP_AT
    now = time.time()
    if _SEP_OK is True:
        return True
    if _SEP_OK is False and now - _SEP_AT < 45:
        return False
    _SEP_AT = now

    if not SEP_PYTHON:
        _SEP_OK = False
        return False
    if SEP_ENGINE != "mdx":
        _SEP_OK = True
        return True
    if not (MODEL_DIR / SEP_MODEL).exists():
        _SEP_OK = False
        return False
    try:
        proc = subprocess.run(
            [SEP_PYTHON, "-c", "import numpy, onnxruntime"],
            capture_output=True,
            timeout=180,
        )
        _SEP_OK = proc.returncode == 0
    except Exception:
        _SEP_OK = False
    return _SEP_OK


def run_ffmpeg(args: list[str], timeout: int = 900) -> None:
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg غير مثبّت")
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(((proc.stderr or "") + (proc.stdout or ""))[-400:] or "ffmpeg failed")


def compress_for_upload(src: Path, outdir: Path, max_height: int = 720) -> Path | None:
    """يضغط الفيديو قبل الرفع حين يكون الاتصال بطيئاً (أخف بكثير = يصل أسرع)."""
    out = outdir / f"{src.stem}_light.mp4"
    vf = f"scale='min({max_height},iw)':-2"
    try:
        run_ffmpeg(
            [
                "-y",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "32",
                "-maxrate",
                "900k",
                "-bufsize",
                "1800k",
                "-c:a",
                "aac",
                "-b:a",
                "80k",
                "-ac",
                "1",
                "-movflags",
                "+faststart",
                str(out),
            ],
            timeout=1200,
        )
    except Exception:
        log.exception("compress failed")
        return None
    if out.exists() and out.stat().st_size > 0:
        return out
    return None


# ----------------------------------------------------------------------------
# كاش الملفات + فصل الصوت
# ----------------------------------------------------------------------------

def register_job(src: Path, info: dict[str, Any], quality: str, url: str, user_id: int) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    token = os.urandom(4).hex()
    dest = CACHE_DIR / f"{token}{src.suffix.lower()}"
    shutil.move(str(src), str(dest))
    JOBS[token] = {
        "path": str(dest),
        "user": user_id,
        "ts": time.time(),
        "title": str(info.get("title") or src.stem)[:200],
        "duration": int(info.get("duration") or 0),
        "width": info.get("width"),
        "height": info.get("height"),
        "quality": quality,
        "url": url,
        "ext": src.suffix.lower(),
        "extractor": str(info.get("extractor_key") or info.get("extractor") or ""),
    }
    prune_jobs()
    return token


def prune_jobs() -> None:
    now = time.time()
    for t in [t for t, j in JOBS.items() if now - j["ts"] > JOB_TTL]:
        job = JOBS.pop(t)
        Path(job["path"]).unlink(missing_ok=True)
    while len(JOBS) > MAX_JOBS:
        t = min(JOBS, key=lambda k: JOBS[k]["ts"])
        job = JOBS.pop(t)
        Path(job["path"]).unlink(missing_ok=True)
    # لا نملأ القرص: نسقط الأقدم حتى ينزل الحجم تحت الحد
    def total() -> int:
        s = 0
        for j in JOBS.values():
            try:
                s += Path(j["path"]).stat().st_size
            except OSError:
                pass
        return s

    while JOBS and total() > CACHE_MAX_BYTES:
        t = min(JOBS, key=lambda k: JOBS[k]["ts"])
        job = JOBS.pop(t)
        Path(job["path"]).unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# المستخدمون + لوحة المالك
# ----------------------------------------------------------------------------

def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        tmp.replace(path)
    except Exception:
        log.exception("save failed: %s", path)


def track_user(uid: int, name: str) -> None:
    users = load_json(USERS_FILE, {})
    if not isinstance(users, dict):
        users = {}
    key = str(uid)
    if key not in users:
        users[key] = {"name": (name or "")[:60], "ts": int(time.time())}
        save_json(USERS_FILE, users)


def all_user_ids() -> list[int]:
    users = load_json(USERS_FILE, {})
    return sorted(int(k) for k in users if str(k).isdigit())


def conf() -> dict[str, Any]:
    c = load_json(CONF_FILE, {})
    if not isinstance(c, dict):
        c = {}
    c.setdefault("channel", "")
    c.setdefault("gate", False)
    return c


def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS


def channel_link(ch: str) -> str:
    ch = (ch or "").strip()
    if ch.startswith("http"):
        return ch
    return f"https://t.me/{ch.lstrip('@')}"


async def gate_ok(bot: Any, uid: int) -> bool:
    c = conf()
    ch = str(c.get("channel") or "").strip()
    if not ch or not c.get("gate"):
        return True
    try:
        m = await bot.get_chat_member(ch, uid)
        return m.status in {"member", "administrator", "creator"}
    except Exception:
        return True


def separate_fast(src_audio: Path, outdir: Path) -> Path:
    """فصل سريع بالـ DSP (ثوانٍ بدل دقائق) — جودة متوسطة."""
    py = SEP_PYTHON or sys.executable
    worker = HERE / "fast_sep.py"
    if not worker.exists():
        raise RuntimeError("fast_sep.py غير موجود")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "fast_vocals.wav"
    proc = subprocess.run(
        [py, "-u", str(worker), str(src_audio), str(dest)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(HERE),
        timeout=1800,
    )
    if proc.returncode != 0 or not dest.exists():
        if proc.returncode in (-9, 137):
            raise RuntimeError("الذاكرة غير كافية لمعالجة هذا المقطع — جرّب مقطعاً أقصر")
        raise RuntimeError(((proc.stderr or "") + (proc.stdout or ""))[-300:] or "فشل الفصل السريع")
    return dest


def separate_vocals(src_audio: Path, outdir: Path, prog: dict[str, Any] | None = None) -> Path:
    """يفصل الكلام/الغناء عن الموسيقى عبر عامل خارجي (بيئة 3.12)."""
    py = SEP_PYTHON
    worker = HERE / "sep_worker.py"
    if not py or not worker.exists():
        raise RuntimeError("محرّك الفصل غير مثبّت")

    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "vocals.wav"
    model_path = str(MODEL_DIR / SEP_MODEL)
    cmd = [py, "-u", str(worker), str(src_audio), str(outdir), str(dest), model_path]

    tail: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(HERE),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("PROGRESS ") and prog is not None:
            try:
                n, total = line.split(" ", 1)[1].split("/")
                prog["sep"] = (int(n), int(total))
            except Exception:
                pass
            continue
        tail.append(line)
        if len(tail) > 12:
            tail.pop(0)

    code = proc.wait(timeout=30)
    if code != 0 or not dest.exists():
        raise RuntimeError(" | ".join(tail[-6:])[-400:] or "فشل فصل الصوت")
    return dest


def build_stripped(
    job: dict[str, Any], workdir: Path, mode: str, prog: dict[str, Any] | None = None
) -> Path:
    """
    mode:
      music → فيديو بصورة أصلية + كلام فقط (موسيقى محذوفة)
      voice → ملف صوتي بالكلام فقط
      mute  → فيديو بلا صوت
    """
    src = Path(job["path"])
    wav = workdir / "audio.wav"

    if mode != "mute":
        run_ffmpeg(["-y", "-i", str(src), "-vn", "-ac", "2", "-ar", "44100", str(wav)])

    if mode == "mute":
        out = workdir / "mute.mp4"
        run_ffmpeg(["-y", "-i", str(src), "-an", "-c:v", "copy", str(out)])
        return out

    # الافتراضي على السيرفر: محرّك DSP السريع (ثوانٍ + ذاكرة صغيرة)
    # MDX (جودة أعلى بكثير في الوقت) يعمل فقط إذا ضبطت SEPARATOR_ENGINE=mdx
    if mode == "fast" or SEP_ENGINE != "mdx":
        vocals = separate_fast(wav, workdir)
    else:
        vocals = separate_vocals(wav, workdir, prog)

    if mode == "voice":
        out = workdir / "voice.mp3"
        run_ffmpeg(["-y", "-i", str(vocals), "-c:a", "libmp3lame", "-b:a", "192k", str(out)])
        return out

    # music: أعِد تركيب الكلام مع الفيديو نفسه
    container = ".mp4" if job["ext"] not in {".webm", ".mkv"} else ".mkv"
    out = workdir / f"clean{container}"
    vcodec = ["-c:v", "copy"]
    if container == ".mkv":
        vcodec = ["-c:v", "copy"]
    run_ffmpeg(
        [
            "-y",
            "-i", str(src),
            "-i", str(vocals),
            "-map", "0:v:0",
            "-map", "1:a:0",
            *vcodec,
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out),
        ]
    )
    return out


# ----------------------------------------------------------------------------
# اختيار الجودة
# ----------------------------------------------------------------------------

def _fmt_size(f: dict[str, Any]) -> int | None:
    return f.get("filesize") or f.get("filesize_approx")


def audio_size(info: dict[str, Any]) -> int | None:
    best = None
    for f in info.get("formats") or []:
        if f.get("acodec") in (None, "none") or f.get("vcodec") not in (None, "none"):
            continue
        size = _fmt_size(f)
        if size and (best is None or size > best):
            best = size
    return best


def video_size(info: dict[str, Any], height: int) -> int | None:
    """تقدير حجم التحميل: ملف واحد (صورة+صوت) إن وُجد، وإلا صورة + صوت منفصلين."""
    fmt = info.get("formats") or []
    progressive = [
        f
        for f in fmt
        if f.get("vcodec") not in (None, "none")
        and f.get("acodec") not in (None, "none")
        and (f.get("height") or 0) <= height
    ]
    if progressive:
        best_h = max((f.get("height") or 0) for f in progressive)
        cand = [f for f in progressive if (f.get("height") or 0) == best_h]
        p = max(cand, key=lambda f: f.get("tbr") or 0)
        size = _fmt_size(p)
        if size:
            return size

    vids = [
        f
        for f in fmt
        if f.get("vcodec") not in (None, "none") and (f.get("height") or 0) <= height
    ]
    if not vids:
        return None
    best_h = max((f.get("height") or 0) for f in vids)
    cand = [f for f in vids if (f.get("height") or 0) == best_h]
    v = max(cand, key=lambda f: f.get("tbr") or 0)
    v_size = _fmt_size(v)
    a_size = audio_size(info)
    if v_size and a_size:
        return v_size + a_size
    return v_size


def has_video(info: dict[str, Any]) -> bool:
    return any(f.get("vcodec") not in (None, "none") for f in (info.get("formats") or []))


def choose_quality(info: dict[str, Any]) -> str:
    if not has_video(info):
        return "audio"
    limit = MAX_MB * 1024 * 1024 * 0.95
    start = LADDER.index(DEFAULT_QUALITY) if DEFAULT_QUALITY in LADDER else 1
    unknown: str | None = None
    for h in LADDER[start:]:
        if h == "audio":
            break
        size = video_size(info, int(h))
        if size is None:
            if unknown is None:
                unknown = h
            continue
        if size <= limit:
            return h
    return unknown or "360"


def fallback_chain(quality: str) -> list[str]:
    idx = LADDER.index(quality) if quality in LADDER else 2
    chain = LADDER[idx:]
    first = chain[:2] if quality != "audio" else chain[:1]
    if "audio" not in first:
        first.append("audio")
    return first


def build_ydl_opts(
    quality: str, outdir: Path, hook, progressive_first: bool = True
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "outtmpl": str(outdir / "%(title).80s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "windowsfilenames": True,
        "trim_file_name": 80,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 8,
        "socket_timeout": 20,
        "cachedir": False,
        "overwrites": True,
    }
    opts.update(cookie_opts())
    ffdir = ffmpeg_dir()
    if ffdir:
        opts["ffmpeg_location"] = ffdir

    if quality == "audio":
        if ffdir:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ]
        else:
            opts["format"] = "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best"
        return opts

    h = int(quality)
    # يوتيوب يخنق الملفات المدمجة (progressive) — نستخدم DASH هناك، وغيرها ملف واحد أسرع
    if not ffdir:
        opts["format"] = f"best[height<={h}][ext=mp4]/best[height<={h}]/best[ext=mp4]/best"
        return opts

    if progressive_first:
        opts["format"] = (
            f"best[height<={h}][ext=mp4][acodec!=none]/"
            f"best[height<={h}][acodec!=none]/"
            f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best"
        )
    else:
        opts["format"] = (
            f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}][ext=mp4]/best[height<={h}]/best"
        )
    opts["merge_output_format"] = "mp4"
    return opts


def pick_output(outdir: Path) -> Path | None:
    files = [p for p in outdir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    files = [p for p in files if p.suffix.lower() not in {".part", ".ytdl", ".temp"}]
    return max(files, key=lambda p: p.stat().st_size) if files else None


# ----------------------------------------------------------------------------
# الأوامر
# ----------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "صديقي"
    await update.message.reply_text(
        welcome_text(name),
        parse_mode="HTML",
        reply_markup=main_keyboard(),
        disable_web_page_preview=True,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"🆔 معرّفك: <code>{update.effective_user.id}</code>", parse_mode="HTML"
    )


def stats_text() -> str:
    busy = 1 if SEMAPHORE.locked() else 0
    return (
        f"📊 <b>حالة {BRAND}</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ تحميلات جارية: <b>{busy}/{MAX_CONCURRENT}</b>\n"
        f"🗂 ملفات محفوظة: <b>{len(JOBS)}</b>\n"
        f"🧩 المحرّك: <code>yt-dlp {yt_dlp.version.__version__}</code>\n"
        f"🎞 ffmpeg: <b>{'جاهز' if ffmpeg_exe() else 'غير مثبّت'}</b>\n"
        f"🎵 فصل الموسيقى: <b>{'جاهز' if separator_ready() and ffmpeg_exe() else 'غير متاح'}</b>\n"
        f"🍪 كوكيز: <b>{'مفعّلة' if cookie_opts() else 'غير مستخدمة'}</b>"
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(stats_text(), parse_mode="HTML")


async def on_ui_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    if q.data == "ui|help":
        await q.message.reply_text(HELP_TEXT, parse_mode="HTML")
    else:
        await q.message.reply_text(stats_text(), parse_mode="HTML")


# ----------------------------------------------------------------------------
# لوحة المالك (نشر + قناة إجبارية)
# ----------------------------------------------------------------------------

async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_admin(user.id):
        return
    await update.message.reply_text(
        "🛠 <b>لوحة المالك</b>\n━━━━━━━━━━━━━━━━━━\n"
        f"👥 مستخدمون: <b>{len(all_user_ids())}</b>",
        parse_mode="HTML",
        reply_markup=admin_keyboard(),
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    if PENDING.pop(uid, None):
        await update.message.reply_text("تم الإلغاء.")
    else:
        await update.message.reply_text("لا يوجد شيء للإلغاء.")


async def save_channel(msg) -> None:
    raw = (msg.text or "").strip()
    ch = raw.rstrip("/")
    if "t.me/" in ch:
        ch = ch.split("t.me/", 1)[1].split("?")[0]
    if not ch:
        await msg.reply_text("❌ أرسل معرّفاً صالحاً مثل <code>@mychannel</code>", parse_mode="HTML")
        return
    if not ch.startswith("@") and not ch.startswith("-"):
        ch = "@" + ch
    c = conf()
    c["channel"] = ch
    save_json(CONF_FILE, c)
    try:
        chat = await msg.get_bot().get_chat(ch)
        title = chat.title or chat.username or ch
    except Exception:
        title = ch
    await msg.reply_text(
        f"✅ تم حفظ القناة: <b>{html.escape(str(title))}</b>\n"
        "لتشغيل الاشتراك الإجباري اضغط الزر 🔐 من /admin.",
        parse_mode="HTML",
        reply_markup=admin_keyboard(),
    )


async def do_broadcast(context: ContextTypes.DEFAULT_TYPE, msg) -> None:
    ids = all_user_ids()
    if not ids:
        await msg.reply_text("👥 لا يوجد مستخدمون مسجّلون بعد.")
        return
    note = await msg.reply_text(f"📢 جاري النشر إلى {len(ids)} مستخدم…")
    ok = fail = 0
    for i, uid in enumerate(ids, 1):
        try:
            await msg.copy(chat_id=uid)
            ok += 1
        except RetryAfter as exc:
            await asyncio.sleep(min(30, int(getattr(exc, "retry_after", 5)) + 1))
            try:
                await msg.copy(chat_id=uid)
                ok += 1
            except Exception:
                fail += 1
        except Exception:
            fail += 1
        await asyncio.sleep(0.05)
        if i % 25 == 0:
            try:
                await note.edit_text(f"📢 {i}/{len(ids)} …")
            except Exception:
                pass
    await note.edit_text(
        f"✅ <b>انتهى النشر</b>\nوصل: <b>{ok}</b> · فشل: <b>{fail}</b>",
        parse_mode="HTML",
        reply_markup=admin_keyboard(),
    )


async def on_admin_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    try:
        await q.answer()
    except Exception:
        pass
    uid = q.from_user.id
    if not is_admin(uid):
        await safe_edit(q, "⛔ هذا الزر للمالك فقط.")
        return

    act = q.data.split("|", 1)[1] if "|" in q.data else ""

    if act == "bc":
        PENDING[uid] = "broadcast"
        await safe_edit(
            q,
            "📢 <b>أرسل الرسالة الآن</b>\nنص أو صورة أو فيديو — وسأرسلها للجميع.\n"
            "للإلغاء: /cancel",
            parse_mode="HTML",
        )
        return

    if act == "ch":
        PENDING[uid] = "channel"
        c = conf()
        cur = c.get("channel") or "غير محددة"
        await safe_edit(
            q,
            f"📡 <b>القناة الحالية:</b> <code>{html.escape(str(cur))}</code>\n\n"
            "أرسل معرّف القناة الجديد مثل <code>@mychannel</code> أو رابطها.\n"
            "⚠️ يجب أن يكون البوت <b>مشرفاً</b> في القناة.\n"
            "للإلغاء: /cancel",
            parse_mode="HTML",
        )
        return

    if act == "gate":
        c = conf()
        if not c.get("channel"):
            await safe_edit(q, "⚠️ حدّد القناة أولاً من الزر 📡", reply_markup=admin_keyboard())
            return
        c["gate"] = not bool(c.get("gate"))
        save_json(CONF_FILE, c)
        await safe_edit(
            q,
            f"🔐 الاشتراك الإجباري: <b>{'يعمل ✅' if c['gate'] else 'متوقف ⛔️'}</b>",
            parse_mode="HTML",
            reply_markup=admin_keyboard(),
        )
        return

    ids = all_user_ids()
    stamp = time.strftime("%H:%M:%S")
    await safe_edit(
        q,
        "📊 <b>الإحصائيات</b>\n━━━━━━━━━━━━━━━━━━\n"
        f"👥 المستخدمون: <b>{len(ids)}</b>\n"
        f"📡 القناة: <code>{html.escape(str(conf().get('channel') or 'غير محددة'))}</code>\n"
        f"🔐 الإجباري: <b>{'يعمل' if conf().get('gate') else 'متوقف'}</b>\n"
        f"<i>{stamp}</i>",
        parse_mode="HTML",
        reply_markup=admin_keyboard(),
    )


async def on_admin_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """نشر صورة/فيديو/ملف من المدير."""
    msg = update.message
    user = update.effective_user
    if msg is None or not is_admin(user.id):
        return
    if PENDING.get(user.id) != "broadcast":
        return
    PENDING.pop(user.id, None)
    await do_broadcast(context, msg)


# ----------------------------------------------------------------------------
# استقبال الرابط → تحميل فوري
# ----------------------------------------------------------------------------

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if msg is None or not msg.text:
        return

    user = update.effective_user
    if not is_allowed(user.id):
        await msg.reply_text("⛔ هذا البوت خاص.")
        return

    track_user(user.id, user.first_name or "")

    # إدخال من لوحة المالك
    if is_admin(user.id) and PENDING.get(user.id):
        kind = PENDING.pop(user.id, "")
        if kind == "broadcast":
            await do_broadcast(context, msg)
            return
        if kind == "channel":
            await save_channel(msg)
            return

    if not await gate_ok(context.bot, user.id):
        ch = str(conf().get("channel") or "")
        await msg.reply_text(
            "🔐 <b>للاستخدام اشترك بالقناة أولاً</b>\nثم أرسل الرابط من جديد.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("📡 اشترك الآن", url=channel_link(ch))]]
            ),
        )
        return

    match = URL_RE.search(msg.text)
    if not match:
        await msg.reply_text(
            "🔗 أرسل رابطاً يبدأ بـ <code>http</code> أو <code>https</code>", parse_mode="HTML"
        )
        return

    if rate_limited(user.id):
        await msg.reply_text("⏳ لحظة… لا تكرر الطلب بسرعة.")
        return

    url = match.group(0).rstrip(".,)]»»")

    if SEMAPHORE.locked():
        status = await msg.reply_text("🕒 <b>في الطابور…</b> سيبدأ الآن.", parse_mode="HTML")
    else:
        status = await msg.reply_text("⚡️ <b>جاري الجلب…</b>", parse_mode="HTML")

    await context.bot.send_chat_action(msg.chat_id, ChatAction.UPLOAD_VIDEO)

    loop = asyncio.get_running_loop()

    def probe() -> dict[str, Any]:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": 25,
            "cachedir": False,
        }
        opts.update(cookie_opts())
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    try:
        info = await asyncio.wait_for(loop.run_in_executor(None, probe), timeout=40)
    except asyncio.TimeoutError:
        await status.edit_text("⌛️ الرابط لم يستجب. جرّب مرة أخرى.")
        return
    except Exception as exc:  # noqa: BLE001
        await status.edit_text(
            "❌ <b>تعذّر قراءة الرابط</b>\n\n"
            f"<code>{html.escape(str(exc)[:280])}</code>",
            parse_mode="HTML",
        )
        return

    duration = int(info.get("duration") or 0)
    if duration and duration > MAX_DURATION_MIN * 60:
        await status.edit_text(
            f"⛔ المدة {fmt_duration(duration)} تتجاوز الحد ({MAX_DURATION_MIN} دقيقة)."
        )
        return

    quality = choose_quality(info)
    is_youtube = "youtube" in str(info.get("extractor_key") or info.get("extractor") or "").lower()
    await download_and_send(
        context, msg.chat_id, status, url, quality, user.id, progressive_first=not is_youtube
    )


# ----------------------------------------------------------------------------
# التحميل والإرسال
# ----------------------------------------------------------------------------

async def download_and_send(
    context,
    chat_id: int,
    status,
    url: str,
    quality: str,
    user_id: int,
    edit: bool = False,
    progressive_first: bool = True,
) -> None:
    async with SEMAPHORE:
        for idx, q in enumerate(fallback_chain(quality)):
            outdir = Path(tempfile.mkdtemp(prefix="saveora_"))
            prog: dict[str, Any] = {"status": "بدء", "pct": 0.0, "speed": "", "eta": ""}
            loop = asyncio.get_running_loop()

            def hook(d: dict[str, Any], p: dict[str, Any] = prog) -> None:
                st = d.get("status")
                if st == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    got = d.get("downloaded_bytes") or 0
                    p["pct"] = (got / total * 100) if total else 0.0
                    p["speed"] = (d.get("_speed_str") or "").strip()
                    p["eta"] = (d.get("_eta_str") or "").strip()
                    p["status"] = "تحميل"
                elif st == "finished":
                    p["pct"] = 100.0
                    p["status"] = "دمج ومعالجة"

            def work(qq: str = q, od: Path = outdir) -> dict[str, Any]:
                opts = build_ydl_opts(qq, od, hook, progressive_first=progressive_first)
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(url, download=True) or {}

            task = loop.run_in_executor(None, work)
            last_edit = 0.0

            while not task.done():
                await asyncio.sleep(1.2)
                if time.time() - last_edit < 2.5:
                    continue
                last_edit = time.time()
                try:
                    label = "🎵 صوت" if q == "audio" else f"🎬 {q}p"
                    meta = (f"⚡ {html.escape(prog['speed'])}" if prog["speed"] else "") + (
                        f" · ⏱ {html.escape(prog['eta'])}" if prog["eta"] else ""
                    )
                    await status.edit_text(
                        f"⬇️ <b>جاري التحميل</b> {label}\n"
                        f"<code>{progress_bar(prog['pct'])}</code> {prog['pct']:.0f}%\n"
                        f"{meta}\n<i>{prog['status']}</i>",
                        parse_mode="HTML",
                    )
                    await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_VIDEO)
                except Exception:
                    pass

            try:
                info = await task
            except Exception as exc:  # noqa: BLE001
                shutil.rmtree(outdir, ignore_errors=True)
                log.exception("download failed: %s (%s)", url, q)
                if idx + 1 < len(fallback_chain(quality)):
                    continue
                await status.edit_text(
                    "❌ <b>تعذّر التحميل</b>\n\n"
                    f"<code>{html.escape(str(exc)[:300])}</code>\n\n"
                    "تأكد أن الرابط عام، أو أضف <code>cookies.txt</code> للمواقع التي تطلب دخول.",
                    parse_mode="HTML",
                )
                return

            file = pick_output(outdir)
            if file is None:
                shutil.rmtree(outdir, ignore_errors=True)
                await status.edit_text("❌ لم يُنتج أي ملف — قد يكون الرابط غير مدعوم.")
                return

            size = file.stat().st_size
            if size > MAX_MB * 1024 * 1024:
                shutil.rmtree(outdir, ignore_errors=True)
                if idx + 1 < len(fallback_chain(quality)):
                    await status.edit_text(
                        f"🔁 الحجم {human_size(size)} كبير — أجرّب جودة أقل…",
                        parse_mode="HTML",
                    )
                    continue
                await status.edit_text(
                    f"⚠️ <b>الملف {human_size(size)} أكبر من {MAX_MB} MB</b>\n"
                    "وهذه أقل جودة متاحة.",
                    parse_mode="HTML",
                )
                return

            duration = int(info.get("duration") or 0)
            title = (info.get("title") or file.stem)[:200]
            caption = f"✅ <b>{html.escape(title)}</b>\n📦 {human_size(size)}"

            token = ""
            try:
                token = register_job(file, info, q, url, user_id)
            except Exception:
                log.exception("cache move failed")

            send_path = Path(JOBS[token]["path"]) if token else file
            suffix = send_path.suffix.lower()
            is_audio = suffix in {".mp3", ".m4a", ".ogg", ".opus", ".wav"}
            is_video = suffix in {".mp4", ".mkv", ".webm", ".mov"}

            async def _deliver(path: Path) -> None:
                """إرسال مع 4 محاولات — الانقطاع المؤقت لا يُضيع الفيديو."""
                last: Exception | None = None
                for attempt in range(4):
                    try:
                        with open(path, "rb") as fh:
                            if is_audio:
                                await context.bot.send_audio(
                                    chat_id=chat_id, audio=fh, filename=path.name,
                                    caption=caption, parse_mode="HTML",
                                    duration=duration or None,
                                    read_timeout=600, write_timeout=900,
                                )
                            elif is_video:
                                try:
                                    await context.bot.send_video(
                                        chat_id=chat_id, video=fh, filename=path.name,
                                        caption=caption, parse_mode="HTML",
                                        duration=duration or None,
                                        width=info.get("width") or None,
                                        height=info.get("height") or None,
                                        supports_streaming=True,
                                        read_timeout=600, write_timeout=900,
                                    )
                                except (TimedOut, NetworkError):
                                    raise
                                except Exception:
                                    log.exception("send_video refused — retry as document")
                                    fh.seek(0)
                                    await context.bot.send_document(
                                        chat_id=chat_id, document=fh, filename=path.name,
                                        caption=caption, parse_mode="HTML",
                                        read_timeout=600, write_timeout=900,
                                    )
                            else:
                                await context.bot.send_document(
                                    chat_id=chat_id, document=fh, filename=path.name,
                                    caption=caption, parse_mode="HTML",
                                    read_timeout=600, write_timeout=900,
                                )
                        return
                    except (TimedOut, NetworkError) as exc:
                        last = exc
                        log.warning("send attempt %s/4: %r", attempt + 1, exc)
                        if attempt < 3:
                            await asyncio.sleep(3 * (attempt + 1))
                if last is not None:
                    raise last

            try:
                await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
                await _deliver(send_path)
                try:
                    await status.delete()
                except Exception:
                    pass
                if token:
                    await context.bot.send_message(
                        chat_id,
                        "✨ <b>تريد تعديل الصوت؟</b>",
                        parse_mode="HTML",
                        reply_markup=media_keyboard(token, audio_only=(q == "audio"), quality=q),
                    )
            except (TimedOut, NetworkError):
                # اتصال بطيء: نضغط الفيديو ونرسل نسخة أخف بدل الفشل
                light: Path | None = None
                if is_video:
                    try:
                        await status.edit_text(
                            "🐢 <b>الشبكة بطيئة…</b> أضغط الفيديو وأرسله.", parse_mode="HTML"
                        )
                    except Exception:
                        pass
                    light = await asyncio.to_thread(compress_for_upload, send_path, outdir)
                if light is not None:
                    try:
                        await _deliver(light)
                        try:
                            await status.delete()
                        except Exception:
                            pass
                        if token:
                            await context.bot.send_message(
                                chat_id,
                                "✨ <b>تريد تعديل الصوت؟</b>",
                                parse_mode="HTML",
                                reply_markup=media_keyboard(
                                    token, audio_only=(q == "audio"), quality=q
                                ),
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.exception("light send failed")
                        await status.edit_text(
                            f"⚠️ الشبكة ضعيفة جداً — جرّب مرة ثانية.\n<code>{html.escape(str(exc)[:150])}</code>",
                            parse_mode="HTML",
                        )
                else:
                    await status.edit_text(
                        "⚠️ الشبكة ضعيفة جداً — جرّب مرة ثانية بعد دقيقة.",
                    )
            except Exception as exc:  # noqa: BLE001
                log.exception("send failed")
                await status.edit_text(
                    f"⚠️ تم التحميل لكن فشل الإرسال:\n<code>{html.escape(str(exc)[:200])}</code>",
                    parse_mode="HTML",
                )
            finally:
                shutil.rmtree(outdir, ignore_errors=True)
            return


# ----------------------------------------------------------------------------
# أزرار تعديل الصوت
# ----------------------------------------------------------------------------

async def on_media_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    try:
        await q.answer("جارٍ التنفيذ…")
    except Exception:
        pass

    try:
        _, mode, token = q.data.split("|", 2)
    except ValueError:
        return

    uid = q.from_user.id
    job = JOBS.get(token)
    log.info(
        "media action: mode=%s user=%s job=%s jobs=%s",
        mode, uid, bool(job), len(JOBS),
    )

    if not job:
        await safe_edit(
            q,
            "⌛️ <b>انتهت صلاحية هذا الملف</b>\n"
            "أرسل الرابط من جديد وستصلك الأزرار مباشرة.",
            parse_mode="HTML",
        )
        return
    if job["user"] != uid:
        await safe_edit(q, "⛔ هذا الملف ليس لك.")
        return

    # ترقية إلى 1080p: يعيد الجلب بالجودة الأعلى ويرسل نسخة جديدة
    if mode == "hd":
        is_yt = "youtube" in str(job.get("extractor") or "").lower()
        await download_and_send(
            context,
            q.message.chat_id,
            q.message,
            job["url"],
            "1080",
            uid,
            edit=True,
            progressive_first=not is_yt,
        )
        return

    if not ffmpeg_exe():
        await safe_edit(
            q,
            "🎞 <b>ffmpeg غير متاح على السيرفر</b>\nأبلغ المالك.",
            parse_mode="HTML",
        )
        return

    if mode in {"music", "voice", "fast"} and not separator_ready():
        await safe_edit(
            q,
            "🎵 <b>محرّك الفصل غير جاهز بعد</b>\n\n"
            f"• المكتبات: <code>{'جاهزة' if SEP_PYTHON else 'تنزّل'}</code>\n\n"
            "أعد المحاولة بعد قليل.",
            parse_mode="HTML",
        )
        return

    # حدود المعالجة
    if mode in {"music", "voice", "fast"} and job["duration"] and job["duration"] > SEP_MAX_MINUTES * 60:
        await safe_edit(
            q,
            f"⏱ <b>المقطع {fmt_duration(job['duration'])} طويل للفصل</b>\n"
            f"الحد الحالي {SEP_MAX_MINUTES} دقيقة.\n"
            "استخدم 🔇 بدون صوت للمقاطع الطويلة.",
            parse_mode="HTML",
        )
        return

    await safe_edit(q, "🎧 <b>1/3 · استخراج الصوت…</b>", parse_mode="HTML")
    try:
        await context.bot.send_chat_action(q.message.chat_id, ChatAction.UPLOAD_VOICE)
    except Exception:
        pass

    loop = asyncio.get_running_loop()
    workdir = Path(tempfile.mkdtemp(prefix="saveora_edit_"))
    prog: dict[str, Any] = {}

    async with SEP_SEMAPHORE:
        try:
            if mode == "mute":
                await safe_edit(q, "🔇 <b>حذف المسار الصوتي…</b>", parse_mode="HTML")
            else:
                await safe_edit(
                    q,
                    "🎧 <b>2/3 · فصل الكلام عن الموسيقى…</b>\n"
                    "<i>ثوانٍ قليلة…</i>",
                    parse_mode="HTML",
                )

            log.info("build_stripped start: mode=%s dur=%s", mode, job.get("duration"))
            task = loop.run_in_executor(None, build_stripped, job, workdir, mode, prog)

            last_edit = 0.0
            while not task.done():
                await asyncio.sleep(2)
                sep = prog.get("sep")
                if not sep or mode == "mute" or time.time() - last_edit < 6:
                    continue
                last_edit = time.time()
                n, total = sep
                pct = n / max(1, total) * 100
                await safe_edit(
                    q,
                    f"🎧 <b>فصل الكلام عن الموسيقى</b>\n"
                    f"<code>{progress_bar(pct)}</code> {pct:.0f}%\n"
                    f"<i>الجزء {n} من {total}</i>",
                    parse_mode="HTML",
                )

            out = await asyncio.wait_for(task, timeout=7200)
            log.info("build_stripped done: %s (%.1f MB)", out.name, out.stat().st_size / 1048576)

            await safe_edit(q, "📤 <b>3/3 · جاري الإرسال…</b>", parse_mode="HTML")
            caption = {
                "music": "🎵 <b>بدون موسيقى</b> — الكلام فقط",
                "voice": "🎤 <b>الكلام فقط</b>",
                "mute": "🔇 <b>بدون صوت</b>",
                "fast": "⚡️ <b>إزالة سريعة</b> — كلام بدون موسيقى",
            }.get(mode, "✅ <b>جاهز</b>")

            async def _send() -> None:
                with open(out, "rb") as fh:
                    if mode == "voice":
                        await context.bot.send_audio(
                            chat_id=q.message.chat_id, audio=fh, filename=out.name,
                            caption=caption, parse_mode="HTML",
                            title=(job["title"] or "")[:60],
                            read_timeout=600, write_timeout=900,
                        )
                    else:
                        await context.bot.send_video(
                            chat_id=q.message.chat_id, video=fh, filename=out.name,
                            caption=caption, parse_mode="HTML",
                            width=job["width"], height=job["height"],
                            duration=job["duration"] or None,
                            supports_streaming=True,
                            read_timeout=600, write_timeout=900,
                        )

            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    await _send()
                    last_err = None
                    break
                except (TimedOut, NetworkError) as exc:
                    last_err = exc
                    log.warning("media send %s/3: %r", attempt + 1, exc)
                    if attempt < 2:
                        await asyncio.sleep(3 * (attempt + 1))
            if last_err is not None:
                raise last_err

            try:
                await q.delete_message()
            except Exception:
                pass
        except asyncio.TimeoutError:
            await safe_edit(q, "⌛️ المعالجة أخذت وقتاً طويلاً — جرّب مقطعاً أقصر.")
        except Exception as exc:  # noqa: BLE001
            log.exception("media action failed")
            await safe_edit(
                q,
                f"❌ <b>فشلت المعالجة</b>\n\n<code>{html.escape(str(exc)[:300])}</code>",
                parse_mode="HTML",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------------------
# الأخطاء
# ----------------------------------------------------------------------------

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("خطأ غير معالج", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                f"⚠️ صار خطأ غير متوقع:\n<code>{html.escape(str(context.error)[:200])}</code>",
                parse_mode="HTML",
            )
    except Exception:
        pass


# ----------------------------------------------------------------------------
# نقطة الدخول
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# الاستضافة 24/7 (Render / Koyeb / أي منصة مجانية)
# ----------------------------------------------------------------------------

def start_health_server() -> None:
    """منفذ HTTP صغير — المنصّات المجانية تعتبر التطبيق حياً إذا استجاب هنا."""
    port = int(os.getenv("PORT", "0") or "0")
    if not port:
        return

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("🌐 health server على المنفذ %s", port)


def start_keepalive() -> None:
    """يمنع الاستضافة المجانية من إيقاف البوت عند الخمول (كل 10 دقائق)."""
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEP_ALIVE_URL")
    if not url:
        return

    def loop() -> None:
        while True:
            time.sleep(600)
            try:
                urllib.request.urlopen(url, timeout=30).read()
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()
    log.info("🔁 keep-alive → %s", url)


def build_app() -> Application:
    app: Application = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .read_timeout(60)
        .write_timeout(180)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(on_ui_button, pattern=r"^ui\|"))
    app.add_handler(CallbackQueryHandler(on_admin_action, pattern=r"^adm\|"))
    app.add_handler(CallbackQueryHandler(on_media_action, pattern=r"^mx\|"))
    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.VIDEO | filters.Document.ALL | filters.VOICE | filters.AUDIO,
            on_admin_media,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)
    return app


async def serve(app: Application) -> None:
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True, bootstrap_retries=5)
    start_health_server()
    start_keepalive()
    log.info("%s يعمل…  ffmpeg=%s  separator=%s", BRAND, bool(ffmpeg_exe()), separator_ready())
    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN غير موجود — انسخ .env.example إلى .env وضع التوكن.")

    if not ALLOWED_USERS:
        log.warning("ALLOWED_USER_IDS فارغ — البوت مفتوح للجميع (استخدم /id).")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    app = build_app()
    try:
        asyncio.run(serve(app))
    except (KeyboardInterrupt, SystemExit):
        log.info("تم الإيقاف.")


if __name__ == "__main__":
    main()
