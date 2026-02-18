import time
import json
import psutil
import joblib
import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

# --- Schemas ---
class CryptoRequest(BaseModel):
    ticker: str = Field(default="BTC-USD", description="Ticker do criptoativo. Apenas BTC-USD é suportado")
    use_partial_candle: bool = Field(
        default=False,
        description="Se true, usa também a vela horária em formação. Se false (padrão), usa apenas velas fechadas"
    )

class ConfidenceIntervalResponse(BaseModel):
    low_usd: float = Field(description="Limite inferior em USD")
    high_usd: float = Field(description="Limite superior em USD")

class PredictionResponse(BaseModel):
    ticker: str = Field(description="Ticker previsto")
    prediction_type: str = Field(description="Tipo de previsão")
    input_mode: str = Field(description="Modo de entrada usado: closed_candles_only ou include_partial_candle")
    last_input_candle_utc: str = Field(description="Último candle usado como entrada em UTC (ISO-8601)")
    last_input_candle_brt: str = Field(description="Último candle usado como entrada em Brasília (ISO-8601)")
    predicted_price_usd: float = Field(description="Preço previsto para o fechamento da próxima hora")
    forecast_for_utc: str = Field(description="Início da hora prevista em UTC (ISO-8601)")
    forecast_for_brt: str = Field(description="Início da hora prevista em Brasília (ISO-8601)")
    forecast_close_utc: str = Field(description="Fechamento da hora prevista em UTC (ISO-8601)")
    forecast_close_brt: str = Field(description="Fechamento da hora prevista em Brasília (ISO-8601)")
    confidence_interval_95_usd: ConfidenceIntervalResponse | None = Field(default=None, description="Intervalo de confiança estimado de 95%")
    estimated_error_pct: float | None = Field(default=None, description="Erro percentual estimado com base nas métricas do modelo")
    processing_time_ms: float = Field(description="Tempo de processamento da requisição em milissegundos")

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(description="healthy quando todos os checks passam; caso contrário degraded")
    artifacts_ready: bool = Field(description="Modelo e scaler carregados")
    model_usable: bool = Field(description="Modelo responde a uma inferência de sanidade")
    market_data_accessible: bool = Field(description="Consulta de mercado disponível")
    last_market_timestamp_utc: str | None = Field(default=None, description="Último candle válido em UTC (ISO-8601)")
    last_market_timestamp_brt: str | None = Field(default=None, description="Último candle válido em Brasília (ISO-8601)")
    cpu_usage: float = Field(description="Uso atual de CPU (%)")
    memory_usage: float = Field(description="Uso atual de memória (%)")
    details: str | None = Field(default=None, description="Detalhes quando status=degraded")

class LiveResponse(BaseModel):
    status: str = Field(description="alive quando a API está respondendo")
    artifacts_ready: bool = Field(description="Modelo e scaler carregados")

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
BRASILIA_TZ = "America/Sao_Paulo"

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

def timestamp_to_utc_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")
    return ts_utc.isoformat()

def timestamp_to_brt_iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")
    return ts_utc.tz_convert(ZoneInfo(BRASILIA_TZ)).isoformat()

def estimate_uncertainty(predicted_price: float, metadata: dict) -> tuple[float | None, ConfidenceIntervalResponse | None]:
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}

    mape_price = metrics.get("mape_price")
    rmse_price = metrics.get("rmse_price")

    estimated_error_pct = None
    if mape_price is not None:
        estimated_error_pct = float(mape_price)
    elif rmse_price is not None and predicted_price > 0:
        estimated_error_pct = float((float(rmse_price) / predicted_price) * 100)

    if rmse_price is not None:
        margin = 1.96 * float(rmse_price)
    elif estimated_error_pct is not None:
        margin = predicted_price * (estimated_error_pct / 100)
    else:
        return estimated_error_pct, None

    ci = ConfidenceIntervalResponse(
        low_usd=round(max(0.0, predicted_price - margin), 2),
        high_usd=round(predicted_price + margin, 2)
    )
    return estimated_error_pct, ci

def load_trained_model(model_path: str):
    keras_module = getattr(tf, "keras", None)
    if keras_module is None or not hasattr(keras_module, "models"):
        raise RuntimeError("TensorFlow/Keras indisponível para carregar o modelo")
    return keras_module.models.load_model(model_path)

def perform_health_checks() -> dict:
    model = ml_artifacts.get("model")
    scaler = ml_artifacts.get("scaler")

    artifacts_ready = model is not None and scaler is not None
    model_usable = False
    market_data_accessible = False
    last_market_timestamp_utc = None
    last_market_timestamp_brt = None
    issues = []

    if artifacts_ready:
        try:
            if model is None:
                raise ValueError("Modelo indisponível")
            sample_input = np.zeros((1, LOOKBACK, 1), dtype=np.float32)
            prediction = model.predict(sample_input, verbose=0)
            if prediction is None or len(prediction) == 0:
                raise ValueError("Predição vazia do modelo")
            model_usable = True
        except Exception:
            issues.append("Modelo carregado, mas não respondeu a inferência de saúde")
    else:
        issues.append("Artefatos de modelo/scaler não carregados")

    try:
        df = download_with_retry(SUPPORTED_TICKER)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(SUPPORTED_TICKER, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)

        close_series = df["Close"].dropna()
        close_series = remove_incomplete_hour_candle(close_series)

        if len(close_series) == 0:
            raise ValueError("Sem candles válidos")

        market_data_accessible = True
        last_market_ts = pd.Timestamp(close_series.index[-1])
        last_market_timestamp_utc = timestamp_to_utc_iso(last_market_ts)
        last_market_timestamp_brt = timestamp_to_brt_iso(last_market_ts)
    except Exception:
        issues.append("Dados de mercado indisponíveis no momento")

    healthy = artifacts_ready and model_usable and market_data_accessible
    return {
        "status": "healthy" if healthy else "degraded",
        "artifacts_ready": artifacts_ready,
        "model_usable": model_usable,
        "market_data_accessible": market_data_accessible,
        "last_market_timestamp_utc": last_market_timestamp_utc,
        "last_market_timestamp_brt": last_market_timestamp_brt,
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "details": None if healthy else " | ".join(issues)
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
        ml_artifacts['model'] = load_trained_model(MODEL_PATH)
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
@app.get(
    "/live",
    response_model=LiveResponse,
    summary="Liveness da API",
    description="Endpoint leve para healthcheck de container, sem consulta externa."
)
def live_check():
    artifacts_ready = 'model' in ml_artifacts and 'scaler' in ml_artifacts
    return {
        "status": "alive",
        "artifacts_ready": artifacts_ready
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Saúde efetiva da API",
    description="Valida artefatos, inferência do modelo e acesso ao mercado. Retorna timestamps em UTC e Brasília."
)
def health_check():
    return perform_health_checks()

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prevê o próximo fechamento horário",
    description=(
        "Aceita apenas o ticker BTC-USD. "
        "O body é opcional: você pode omitir o body ou enviar {} para usar o padrão BTC-USD. "
        "Por padrão usa apenas velas fechadas; para incluir a vela em formação, use use_partial_candle=true. "
        "Retorna preço previsto, janela temporal da previsão em UTC/Brasília, intervalo de confiança e erro estimado."
    )
)
def predict_next_hour(
    request: CryptoRequest = Body(
        default_factory=CryptoRequest,
        openapi_examples={
            "sem_body_ou_vazio": {
                "summary": "Sem body ou body vazio",
                "description": "Pode omitir o body ou enviar {}. O ticker padrão será BTC-USD.",
                "value": {}
            },
            "body_explicito": {
                "summary": "Body explícito",
                "value": {"ticker": "BTC-USD"}
            },
            "com_vela_parcial": {
                "summary": "Com vela parcial",
                "description": "Inclui a vela horária em formação na entrada do modelo.",
                "value": {"ticker": "BTC-USD", "use_partial_candle": True}
            }
        }
    )
):
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
        if not request.use_partial_candle:
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
        last_observed_ts = pd.Timestamp(close_series.index[-1])
        forecast_for_ts = last_observed_ts + pd.Timedelta(hours=1)
        forecast_close_ts = forecast_for_ts + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        predicted_price = last_close * np.exp(predicted_log_return)

        metadata = ml_artifacts.get('metadata', {})
        estimated_error_pct, confidence_interval_95 = estimate_uncertainty(float(predicted_price), metadata)
        
        proc_time = (time.perf_counter() - start_proc) * 1000
        
        return {
            "ticker": ticker,
            "prediction_type": "Next Hour Close",
            "input_mode": "include_partial_candle" if request.use_partial_candle else "closed_candles_only",
            "last_input_candle_utc": timestamp_to_utc_iso(last_observed_ts),
            "last_input_candle_brt": timestamp_to_brt_iso(last_observed_ts),
            "predicted_price_usd": round(float(predicted_price), 2),
            "forecast_for_utc": timestamp_to_utc_iso(forecast_for_ts),
            "forecast_for_brt": timestamp_to_brt_iso(forecast_for_ts),
            "forecast_close_utc": timestamp_to_utc_iso(forecast_close_ts),
            "forecast_close_brt": timestamp_to_brt_iso(forecast_close_ts),
            "confidence_interval_95_usd": confidence_interval_95,
            "estimated_error_pct": None if estimated_error_pct is None else round(float(estimated_error_pct), 2),
            "processing_time_ms": round(proc_time, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro interno em /predict: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Falha interna ao gerar previsão")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)