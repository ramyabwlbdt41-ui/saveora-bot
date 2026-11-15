#!/usr/bin/env bash
# تثبيت بوت تحميل الفيديو على Ubuntu (Oracle Cloud / أي VPS)
# التشغيل:  sudo bash install.sh
#
# يفترض أنك نسخت مجلد البوت إلى /opt/tg-video-bot أو شغّلت السكربت من داخله.

set -euo pipefail

APP_DIR="/opt/tg-video-bot"
SERVICE="tg-video-bot"
BOT_USER="tgbot"

echo "==> تحديث الحزم"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip ffmpeg ca-certificates

echo "==> إنشاء مستخدم الخدمة"
id -u "$BOT_USER" >/dev/null 2>&1 || useradd -r -m -d /home/$BOT_USER -s /bin/bash "$BOT_USER"

echo "==> نسخ الملفات إلى $APP_DIR"
mkdir -p "$APP_DIR"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$SRC" != "$APP_DIR" ]; then
  cp -f "$SRC"/bot.py "$SRC"/requirements.txt "$SRC"/.env.example "$APP_DIR"/ 2>/dev/null || true
  [ -f "$SRC/.env" ] && cp -f "$SRC/.env" "$APP_DIR"/.env
fi

echo "==> إنشاء البيئة الافتراضية"
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel >/dev/null
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo
  echo "!! لم يوجد ملف .env — أنشأته من المثال."
  echo "   عدّله الآن:  nano $APP_DIR/.env   (ضع BOT_TOKEN)"
  echo
fi

chown -R "$BOT_USER:$BOT_USER" "$APP_DIR"

echo "==> تسجيل خدمة systemd"
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Telegram video downloader bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$BOT_USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/bot.py
Restart=always
RestartSec=5
# حدود أمان
MemoryMax=1500M
CPUQuota=150%

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

echo
echo "==> تم. الحالة:"
systemctl --no-pager --full status "$SERVICE" | head -n 12 || true
echo
echo "متابعة السجل:   journalctl -u $SERVICE -f"
echo "إعادة التشغيل:  systemctl restart $SERVICE"
echo "تحديث yt-dlp:   $APP_DIR/venv/bin/pip install -U yt-dlp && systemctl restart $SERVICE"
