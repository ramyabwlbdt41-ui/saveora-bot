import json
import urllib.parse
import urllib.request

import bot

T = bot.BOT_TOKEN


def call(method, payload=None):
    data = urllib.parse.urlencode(payload or {}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{T}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


cmds = [
    {"command": "start", "description": "بدء الاستخدام"},
    {"command": "help", "description": "طريقة الاستخدام"},
    {"command": "id", "description": "معرّفي"},
    {"command": "stats", "description": "حالة البوت"},
]

print("commands:", call("setMyCommands", {"commands": json.dumps(cmds, ensure_ascii=False)})["ok"])
print("desc:", call("setMyDescription", {"description": "⚡️ محمّل الوسائط الذكي — أرسل أي رابط واستلم الملف مباشرة.\nيوتيوب · تيك توك · إنستغرام · X · فيسبوك والمزيد."})["ok"])
print("short:", call("setMyShortDescription", {"short_description": "حمّل أي فيديو من أي رابط بسرعة 🚀"})["ok"])
print("name:", call("setMyName", {"name": "SAVEORA ⚡️"})["ok"])
