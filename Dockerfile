# CKDPredict 2.0 - container for Hugging Face Spaces (Docker SDK) or any Docker host.
FROM python:3.12-slim

# Tesseract = OCR for scanned lab reports; libgomp1 = needed by LightGBM/CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PORT=7860 COOKIE_SECURE=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Train inside the image so the saved model always matches the installed library versions (about 1 minute)
RUN python ml/train.py

EXPOSE 7860
# one worker: SQLite + a single in-memory model; threads handle concurrent requests
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
