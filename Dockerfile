FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    PORT=8080

# ffmpeg للدمج + معالجة الصوت
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نموذج فصل الموسيقى (63 MB) — يُنزّل مرة واحدة وقت البناء
RUN mkdir -p models \
 && python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx', 'models/Kim_Vocal_2.onnx')" \
 && python -c "import os,sys; s=os.path.getsize('models/Kim_Vocal_2.onnx'); print('model bytes:', s); sys.exit(0 if s > 60000000 else 1)"

COPY . .

EXPOSE 8080

CMD ["python", "bot.py"]
