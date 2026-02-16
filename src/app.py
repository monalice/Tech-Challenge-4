import time
import psutil
import joblib
import uvicorn
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional

# --- Schemas Pydantic (Validação de Dados) ---
class StockRequest(BaseModel):
    ticker: str = Field(..., description="Símbolo do ativo (ex: AAPL, BTC-USD, PETR4.SA)", min_length=1, max_length=15)

class StockResponse(BaseModel):
    ticker: str
    prediction_date: str
    predicted_price: float
    confidence_interval: dict  # {"lower": float, "upper": float}
    estimated_error_percentage: float
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    cpu_usage: float
    memory_usage: float

# --- Gestão de Ciclo de Vida da Aplicação ---
# Variável global para armazenar os artefatos carregados na memória
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carregamento de modelos na inicialização (Startup Event).
    Carrega 3 modelos separados: BTC, ETH e SOL.
    """
    try:
        print(" Carregando modelos LSTM para BTC, ETH e SOL...")
        
        # Carregar modelos para cada criptomoeda
        for ticker_key in ['btc', 'eth', 'sol']:
            model_path = f'models/{ticker_key}_model.keras'
            scaler_path = f'models/{ticker_key}_scaler.gz'
            
            try:
                ml_artifacts[f'{ticker_key}_model'] = tf.keras.models.load_model(model_path)
                ml_artifacts[f'{ticker_key}_scaler'] = joblib.load(scaler_path)
                print(f"   ✓ {ticker_key.upper()} modelo carregado")
            except FileNotFoundError:
                print(f"   ✗ {ticker_key.upper()} modelo não encontrado em {model_path}")
        
        if len(ml_artifacts) == 0:
            print(" ⚠ Nenhum modelo foi carregado. Treine os modelos primeiro!")
        else:
            print(f" Artefatos carregados: {len(ml_artifacts)//2} modelos prontos.")
    except Exception as e:
        print(f" Falha ao carregar modelos: {e}")
    yield
    # Cleanup (Shutdown Event)
    ml_artifacts.clear()
    print(" Memória limpa.")
    print(" Memória limpa.")

# Inicialização da Aplicação
app = FastAPI(
    title="StockCast AI API",
    description="API de Previsão Financeira baseada em LSTM",
    version="1.0.0",
    lifespan=lifespan
)

# --- Middleware de Observabilidade ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware para monitorar a latência de cada requisição.
    Adiciona o header X-Process-Time e loga a performance.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log estruturado (simulado)
    print(f" Path: {request.url.path} | Method: {request.method} | Time: {process_time:.4f}s")
    return response

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Endpoint para Liveness/Readiness Probes (Kubernetes/Docker).
    Monitora o consumo de recursos do container.
    """
    return {
        "status": "healthy",
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent
    }

@app.post("/predict", response_model=StockResponse, tags=["Inference"])
def predict_stock(request: StockRequest):
    """
    Realiza a inferência ativa do preço de fechamento.
    1. Baixa dados recentes do Yahoo Finance.
    2. Aplica pré-processamento (Janela 60 dias + Scaling).
    3. Executa o modelo LSTM.
    4. Desnormaliza e retorna o preço.
    """
    start_proc = time.perf_counter()
    ticker = request.ticker.upper()
    
    # Identificar qual modelo usar baseado no ticker
    ticker_key = None
    if 'BTC' in ticker:
        ticker_key = 'btc'
    elif 'ETH' in ticker:
        ticker_key = 'eth'
    elif 'SOL' in ticker:
        ticker_key = 'sol'
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Ticker '{ticker}' não suportado. Use BTC-USD, ETH-USD ou SOL-USD"
        )
    
    # Validação de Artefatos
    model_key = f'{ticker_key}_model'
    scaler_key = f'{ticker_key}_scaler'
    
    if model_key not in ml_artifacts or scaler_key not in ml_artifacts:
        raise HTTPException(
            status_code=503, 
            detail=f"Modelo para {ticker_key.upper()} não carregado. Treine o modelo primeiro."
        )
    
    try:
        # 1. Coleta de Dados Recentes (Active Inference)
        # Baixamos mais dias (100) para garantir 60 dias úteis após feriados
        df = yf.download(ticker, period="6mo", progress=False)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' não encontrado no Yahoo Finance.")
        
        # Tratamento MultiIndex e Seleção Adj Close
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)
        
        # Para criptomoedas usamos Close, para ações usamos Adj Close
        price_column = 'Close' if 'Close' in df.columns else 'Adj Close'
        
        if price_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Coluna de preço não encontrada para {ticker}")
        
        data_series = df[price_column].dropna()
        
        if len(data_series) < 60:
            raise HTTPException(status_code=400, detail="Histórico insuficiente para este ativo (mínimo 60 dias).")
        
        # Seleciona os últimos 60 dias exatos
        last_60_days = data_series.values[-60:].reshape(-1, 1)
        
        # 2. Pré-processamento - Usar o modelo e scaler específicos para este ticker
        scaler = ml_artifacts[scaler_key]
        model = ml_artifacts[model_key]
        
        # Normalização usando o scaler treinado (Crucial!)
        scaled_input = scaler.transform(last_60_days)
        
        # Reshape para (1, 60, 1) -> (Batch, Timesteps, Features)
        X_input = np.reshape(scaled_input, (1, 60, 1))
        
        # 3. Predição
        predicted_scaled = model.predict(X_input, verbose=0)
        
        # 4. Pós-processamento (Inverse Transform)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        final_price = float(predicted_price[0][0])
        
        # 5. Cálculo de Intervalo de Confiança e Erro baseado em volatilidade histórica
        # Calculamos o desvio padrão dos últimos 30 dias para estimar a incerteza
        recent_volatility = np.std(data_series.values[-30:])
        error_percentage = (recent_volatility / data_series.values[-1]) * 100
        
        # Intervalo de confiança de 95% (aproximadamente 2 desvios padrões)
        confidence_margin = 2 * recent_volatility
        lower_bound = final_price - confidence_margin
        upper_bound = final_price + confidence_margin
        
        processing_time = (time.perf_counter() - start_proc) * 1000 # ms
        
        return {
            "ticker": ticker,
            "prediction_date": "Next Market Close",
            "predicted_price": round(final_price, 2),
            "confidence_interval": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            "estimated_error_percentage": round(error_percentage, 2),
            "model_version": f"lstm_{ticker_key}",
            "processing_time_ms": round(processing_time, 2)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f" Erro na inferência: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no processamento da predição.")

@app.post("/predict/chart", tags=["Inference"])
def predict_with_chart(request: StockRequest):
    """
    Retorna a previsão junto com um gráfico em formato base64.
    O gráfico mostra os últimos 60 dias de histórico + previsão com intervalo de confiança.
    """
    ticker = request.ticker.upper()
    
    # Identificar qual modelo usar
    ticker_key = None
    if 'BTC' in ticker:
        ticker_key = 'btc'
    elif 'ETH' in ticker:
        ticker_key = 'eth'
    elif 'SOL' in ticker:
        ticker_key = 'sol'
    else:
        raise HTTPException(status_code=400, detail=f"Ticker '{ticker}' não suportado. Use BTC-USD, ETH-USD ou SOL-USD")
    
    model_key = f'{ticker_key}_model'
    scaler_key = f'{ticker_key}_scaler'
    
    if model_key not in ml_artifacts or scaler_key not in ml_artifacts:
        raise HTTPException(status_code=503, detail=f"Modelo para {ticker_key.upper()} não carregado.")
    
    try:
        # Baixar dados
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' não encontrado")
        
        # Processar dados
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except KeyError:
                df.columns = df.columns.get_level_values(0)
        
        price_column = 'Close' if 'Close' in df.columns else 'Adj Close'
        data_series = df[price_column].dropna()
        
        if len(data_series) < 60:
            raise HTTPException(status_code=400, detail="Histórico insuficiente")
        
        last_60_days = data_series.values[-60:].reshape(-1, 1)
        historical_dates = data_series.index[-60:]
        historical_prices = data_series.values[-60:]
        
        # Fazer predição
        scaler = ml_artifacts[scaler_key]
        model = ml_artifacts[model_key]
        
        scaled_input = scaler.transform(last_60_days)
        X_input = np.reshape(scaled_input, (1, 60, 1))
        predicted_scaled = model.predict(X_input, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        final_price = float(predicted_price[0][0])
        
        # Calcular intervalo de confiança
        recent_volatility = np.std(historical_prices[-30:])
        confidence_margin = 2 * recent_volatility
        lower_bound = final_price - confidence_margin
        upper_bound = final_price + confidence_margin
        error_percentage = (recent_volatility / historical_prices[-1]) * 100
        
        # Criar gráfico
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plotar histórico
        plt.plot(historical_dates, historical_prices, label='Histórico', color='#2E86AB', linewidth=2)
        
        # Plotar previsão
        next_date = historical_dates[-1] + pd.Timedelta(days=1)
        plt.scatter([next_date], [final_price], color='#A23B72', s=100, zorder=5, label='Previsão')
        
        # Plotar intervalo de confiança
        plt.fill_between([historical_dates[-1], next_date], 
                         [historical_prices[-1], lower_bound],
                         [historical_prices[-1], upper_bound],
                         alpha=0.3, color='#F18F01', label='Intervalo de Confiança (95%)')
        
        # Conectar último preço com previsão
        plt.plot([historical_dates[-1], next_date], [historical_prices[-1], final_price], 
                 '--', color='#A23B72', alpha=0.7, linewidth=2)
        
        plt.title(f'Previsão de Preço - {ticker}', fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço (USD)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return {
            "ticker": ticker,
            "predicted_price": round(final_price, 2),
            "confidence_interval": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            "estimated_error_percentage": round(error_percentage, 2),
            "chart_base64": image_base64,
            "chart_format": "png",
            "model_version": f"lstm_{ticker_key}"
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f" Erro ao gerar gráfico: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar gráfico: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)