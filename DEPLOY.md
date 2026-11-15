# 🚀 تشغيل SAVEORA 24 ساعة مجاناً (زي بوت الأسئلة)

البوت جاهز للنشر: `Dockerfile` + `render.yaml` + سيرفر صحة + keep-alive.
يعني يشتغل على سيرفر **Render** المجاني، 24/7، وأنت مطفي جهازك — وبنفس سرعة البوت الثاني (السيرفر ينزّل ويرفع بالنيابة عنك، ما يحتاج نت بيتك).

---

## ما تحتاجه (مجاناً — بلا بطاقة)

| الحساب | لماذا |
|---|---|
| GitHub | لحفظ الكود |
| Render | لتشغيل البوت 24/7 |

---

## 1) ارفع الكود على GitHub (من المتصفح، بلا Git)

1. افتح https://github.com/new
2. الاسم: `saveora-bot` → **Private** → Create repository
3. اضغط **uploading an existing file**
4. اسحب **محتويات** ملف `saveora-deploy.zip` (بعد فك الضغط) إلى الصفحة
5. **Commit changes**

> ⚠️ لا ترفع `.env` أبداً — فيه التوكن. (الملف مستثنى تلقائياً في الحزمة)

## 2) انشر على Render

1. https://dashboard.render.com → **Sign in with GitHub**
2. **New +** → **Web Service**
3. اختر مستودع `saveora-bot` → **Connect**
4. الإعدادات:
   - **Runtime**: `Docker`
   - **Instance Type**: `Free`
   - **Region**: `Frankfurt`
   - **Health Check Path**: `/`
5. **Environment** → أضف:

| Key | Value |
|---|---|
| `BOT_TOKEN` | توكن @BotFather |
| `DEFAULT_QUALITY` | `720` |
| `SEPARATOR_MAX_MINUTES` | `6` |

6. **Deploy Web Service** → انتظر 5–8 دقائق (يبني الحاوية وينزّل النموذج 63 MB مرة واحدة)

✅ لما يصير **Live** — البوت شغّال 24/7.

---

## 3) أوقف النسخة المحلية

تيليجرام يسمح بنسخة واحدة فقط. بعد ما يشتغل Render:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*bot.py*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

وإلا رح يطلع خطأ `Conflict: terminated by other getUpdates request`.

---

## ليش أسرع من جهازك؟

| | جهازك | Render |
|---|---|---|
| التنزيل من إنستغرام/تيك توك | 30–40 ثانية | 2–5 ثواني |
| الرفع لتلغرام | 30–40 ثانية | 2–4 ثواني |
| **الإجمالي** | **~70 ثانية** | **~8–12 ثانية** |

جهازك ينزّل بسرعة ~100 KB/s (تخنق من المزوّد)، والسيرفر عنده جيجابت.

---

## ملاحظات

| الموضوع | التفاصيل |
|---|---|
| الخمول | Render يوقف الخدمة بعد 15 دقيقة — البوت يضرب نفسه ping كل 10 دقائق فما ينام |
| 750 ساعة/شهر | كافية لخدمة واحدة تعمل الشهر كله (744 ساعة) |
| الرام | 512 MB — كافية للتحميل والدمج وفصل الموسيقى لمقطع قصير |
| فصل الموسيقى | يعمل، لكن المعالج المجاني بطيء (0.1 CPU) — خلّ `SEPARATOR_MAX_MINUTES=6` |
| الملفات المؤقتة | القرص مؤقت: تنمسح عند كل نشر — غير مؤثر (الأزرار تعمل 45 دقيقة) |
| تحديث البوت | عدّل على GitHub → Render ينشر تلقائياً |

---

## بديل أسرع (نفس المجانية)

- **Koyeb** — https://app.koyeb.com (نفس Dockerfile، منفذ 8080)
- **Fly.io** — https://fly.io (`fly launch` ثم `fly deploy`)

كلها تقرأ نفس المتغيرات، و`PORT` يُمرَّر تلقائياً.
