import time
import json
import psutil
import joblib
import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# --- Schemas ---
class CryptoRequest(BaseModel):
    ticker: str = Field(default="BTC-USD", description="Ticker do Criptoativo (ex: BTC-USD)")

class PredictionResponse(BaseModel):
    ticker: str
    prediction_type: str
    predicted_price_usd: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    artifacts_ready: bool
    cpu_usage: float
    memory_usage: float
    details: str | None = None

# --- Lifecycle ---
ml_artifacts = {}
SUPPORTED_TICKER = "BTC-USD"
LOOKBACK = 60
MODEL_PATH = "models/lstm_btc_hourly.keras"
SCALER_PATH = "models/scaler_btc.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"
CACHE_TTL_SECONDS = 30
YFINANCE_TIMEOUT_SECONDS = 10
YFINANCE_MAX_RETRIES = 3
market_cache = {}

def remove_incomplete_hour_candle(series: pd.Series) -> pd.Series:
    if len(series) < 2:
        return series

    last_ts = pd.Timestamp(series.index[-1])
    now_utc = pd.Timestamp.utcnow()

    if last_ts.tzinfo is None:
        now_ref = now_utc.tz_localize(None)
    else:
        now_ref = now_utc.tz_convert(last_ts.tz)

    if last_ts >= now_ref.floor("h"):
        return series.iloc[:-1]
    return series

def get_cached_market_data(ticker: str) -> pd.DataFrame | None:
    cache_entry = market_cache.get(ticker)
    if not cache_entry:
        return None

    age_seconds = time.time() - cache_entry["cached_at"]
    if age_seconds > CACHE_TTL_SECONDS:
        return None

    return cache_entry["data"].copy()

def set_cached_market_data(ticker: str, data: pd.DataFrame):
    market_cache[ticker] = {
        "cached_at": time.time(),
        "data": data.copy()
    }

def download_with_retry(ticker: str) -> pd.DataFrame:
    cached = get_cached_market_data(ticker)
    if cached is not None:
        return cached

    last_error = None
    for attempt in range(1, YFINANCE_MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                period="1mo",
                interval="1h",
                progress=False,
                timeout=YFINANCE_TIMEOUT_SECONDS
            )
            if df is None or df.empty:
                raise ValueError("Resposta vazia do Yahoo Finance")

            set_cached_market_data(ticker, df)
            return df
        except Exception as error:
            last_error = error
            if attempt < YFINANCE_MAX_RETRIES:
                time.sleep(0.5 * attempt)

    raise HTTPException(
        status_code=503,
        detail=f"Falha ao consultar dados de mercado após {YFINANCE_MAX_RETRIES} tentativas"
    ) from last_error

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Carregando modelo LSTM Hourly e scaler...")
        ml_artifacts['model'] = tf.keras.models.load_model(MODEL_PATH)
        ml_artifacts['scaler'] = joblib.load(SCALER_PATH)

        try:
            with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
                ml_artifacts['metadata'] = json.load(meta_file)
        except FileNotFoundError:
            ml_artifacts['metadata'] = {
                "target": "log_return",
                "lookback": LOOKBACK,
                "ticker": SUPPORTED_TICKER
            }

        print("Artefatos carregados com sucesso.")
    except Exception as e:
        ml_artifacts.clear()
        raise RuntimeError(f"Falha crítica ao carregar artefatos do modelo: {e}") from e
    yield
    ml_artifacts.clear()

app = FastAPI(title="Bitcoin Hourly Forecaster", version="2.0.0", lifespan=lifespan)

# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    artifacts_ready = 'model' in ml_artifacts and 'scaler' in ml_artifacts
    return {
        "status": "healthy" if artifacts_ready else "degraded",
        "artifacts_ready": artifacts_ready,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "details": None if artifacts_ready else "Artefatos de modelo/scaler não carregados"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_next_hour(request: CryptoRequest):
    start_proc = time.perf_counter()
    ticker = request.ticker.upper()

    if ticker != SUPPORTED_TICKER:
        raise HTTPException(
            status_code=400,
            detail=f"Este modelo foi treinado apenas para {SUPPORTED_TICKER}."
        )
    
    if 'model' not in ml_artifacts or 'scaler' not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Modelo não disponível.")
        
    try:
        # 1. Coleta ativa de mercado com cache e retry
        df = download_with_retry(ticker)
            
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)

        if 'Close' not in df.columns:
            raise HTTPException(status_code=503, detail="Dados de mercado sem coluna Close")
            
        close_series = df['Close'].dropna()
        close_series = remove_incomplete_hour_candle(close_series)

        required_points = LOOKBACK + 1
        if len(close_series) < required_points:
            raise HTTPException(status_code=400, detail=f"Dados insuficientes para janela de retorno ({required_points} closes).")

        log_price_series = pd.Series(np.log(close_series.values), index=close_series.index)
        return_series = log_price_series.diff().dropna()
        
        if len(return_series) < LOOKBACK:
            raise HTTPException(status_code=400, detail=f"Dados insuficientes para gerar janela de retorno de {LOOKBACK}h.")

        # Pega as exatas últimas LOOKBACK janelas de retorno
        last_returns = np.asarray(return_series.to_numpy()[-LOOKBACK:], dtype=float).reshape(-1, 1)
        
        # 2. Pré-processamento
        scaler = ml_artifacts['scaler']
        model = ml_artifacts['model']
        
        scaled_input = scaler.transform(last_returns)
        X_input = np.reshape(scaled_input, (1, LOOKBACK, 1))
        
        # 3. Inferência
        predicted_scaled = model.predict(X_input, verbose=0)
        predicted_log_return = float(scaler.inverse_transform(predicted_scaled).reshape(-1)[0])
        last_close = float(close_series.iloc[-1])
        predicted_price = last_close * np.exp(predicted_log_return)
        
        proc_time = (time.perf_counter() - start_proc) * 1000
        
        return {
            "ticker": ticker,
            "prediction_type": "Next Hour Close",
            "predicted_price_usd": round(float(predicted_price), 2),
            "processing_time_ms": round(proc_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro interno em /predict: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Falha interna ao gerar previsão")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)