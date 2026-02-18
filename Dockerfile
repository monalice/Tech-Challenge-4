FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["sh", "-c", "if [ ! -f models/lstm_btc_hourly.keras ] || [ ! -f models/scaler_btc.gz ]; then echo '[ERROR] Artefatos ausentes em models/. Necessário: lstm_btc_hourly.keras e scaler_btc.gz'; ls -la models || true; exit 1; fi; python -m uvicorn src.app:app --host 0.0.0.0 --port 8000"]
