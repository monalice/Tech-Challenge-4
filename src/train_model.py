import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Configurações Específicas para Bitcoin/Horário ---
TICKER = "BTC-USD"
# LIMITAÇÃO TÉCNICA YFINANCE:
# Dados horários (interval='1h') só estão disponíveis para os últimos 730 dias.
PERIOD = "730d"  
INTERVAL = "1h"

LOOKBACK = 60         # Janela de observação: 60 horas (2.5 dias)
BATCH_SIZE = 64       # Aumentado para 64 para estabilizar o gradiente em dados ruidosos
EPOCHS = 100
TEST_SIZE_PCT = 0.2   # 20% finais para teste (divisão cronológica)
VAL_SIZE_PCT = 0.1    # 10% finais do treino para validação temporal
WALK_FORWARD_SPLITS = 3
WALK_FORWARD_EPOCHS = 20
RANDOM_SEED = 42
EPSILON = 1e-8
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 10

# Caminhos
MODEL_PATH = "models/lstm_btc_hourly.keras"
SCALER_PATH = "models/scaler_btc.gz"
MODEL_META_PATH = "models/model_metadata_btc.json"
MODEL_CANDIDATE_PATH = "models/lstm_btc_hourly_candidate.keras"
SCALER_CANDIDATE_PATH = "models/scaler_btc_candidate.gz"
MODEL_CANDIDATE_META_PATH = "models/model_metadata_btc_candidate.json"

def ensure_directories():
    if not os.path.exists("models"):
        os.makedirs("models")

def download_crypto_data():
    """
    Baixa dados horários do Bitcoin respeitando o limite de 730 dias do Yahoo Finance.
    """
    print(f"[INFO] Baixando dados horários ({INTERVAL}) para {TICKER} (Últimos {PERIOD})...")
    
    # Download com retry para reduzir falhas transitórias da API
    df = None
    last_error = None
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            df = yf.download(
                TICKER,
                period=PERIOD,
                interval=INTERVAL,
                progress=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS
            )
            if df is not None and not df.empty:
                break
        except Exception as error:
            last_error = error

        if attempt < DOWNLOAD_MAX_RETRIES:
            time.sleep(0.5 * attempt)
    
    # Tratamento para MultiIndex (yfinance > 0.2)
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(TICKER, axis=1, level=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)

    # Verifica integridade
    if df is None or df.empty:
        raise ValueError(
            f"A API retornou um DataFrame vazio após {DOWNLOAD_MAX_RETRIES} tentativas. "
            f"Erro: {last_error}"
        )

    # Em cripto, Close e Adj Close são iguais. Usaremos Close.
    data = df[['Close']].copy()
    
    # Dropna é vital em dados horários pois podem haver gaps de manutenção da exchange
    data = data.dropna()
    
    print(f"[INFO] Total de registros (horas): {len(data)}")
    return data

def create_sliding_window(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def safe_mape(y_true, y_pred, eps=1e-8):
    denominator = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100

def run_walk_forward_backtest(X_train, y_train, scaler):
    if len(X_train) < (WALK_FORWARD_SPLITS + 1):
        print("[WARN] Dados insuficientes para walk-forward. Backtest pulado.")
        return

    print(f"[INFO] Iniciando walk-forward backtest com {WALK_FORWARD_SPLITS} splits...")
    tscv = TimeSeriesSplit(n_splits=WALK_FORWARD_SPLITS)
    model_maes = []
    baseline_maes = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        fold_model = build_lstm_architecture((X_train.shape[1], 1))
        fold_early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        fold_model.fit(
            X_tr,
            y_tr,
            batch_size=BATCH_SIZE,
            epochs=WALK_FORWARD_EPOCHS,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[fold_early_stop],
            verbose=0
        )

        y_pred_scaled = fold_model.predict(X_val_fold, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
        y_real = scaler.inverse_transform(y_val_fold.reshape(-1, 1)).reshape(-1)

        baseline_scaled = X_val_fold[:, -1, 0].reshape(-1, 1)
        baseline_pred = scaler.inverse_transform(baseline_scaled).reshape(-1)

        fold_mae = mean_absolute_error(y_real, y_pred)
        fold_baseline_mae = mean_absolute_error(y_real, baseline_pred)

        model_maes.append(fold_mae)
        baseline_maes.append(fold_baseline_mae)

        print(
            f"[WF][Fold {fold_idx}] MAE modelo (retorno): {fold_mae:.6f} | "
            f"MAE baseline (retorno): {fold_baseline_mae:.6f}"
        )

    print(
        f"[WF][Média] MAE modelo (retorno): {np.mean(model_maes):.6f} | "
        f"MAE baseline (retorno): {np.mean(baseline_maes):.6f}"
    )

def build_lstm_architecture(input_shape):
    model = Sequential()
    
    # Camada 1: LSTM robusta para capturar sequências
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) 
    
    # Camada 2: LSTM para condensar padrões
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1)) # Saída linear (Preço)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    ensure_directories()
    
    # 1. Pipeline de Dados
    df = download_crypto_data()
    
    # Modelagem em retornos logarítmicos para reduzir não-estacionaridade
    close_series = df['Close'].copy()
    log_price_series = pd.Series(np.log(close_series.values), index=close_series.index)
    returns_df = log_price_series.diff().dropna().to_frame(name='log_return')

    # Divisão cronológica em retornos
    split_idx = int(len(returns_df) * (1 - TEST_SIZE_PCT))
    train_data = returns_df.iloc[:split_idx]
    test_data = returns_df.iloc[split_idx:]

    if len(train_data) <= LOOKBACK:
        raise ValueError(
            f"Dados de treino insuficientes. Necessário mais que {LOOKBACK} registros, recebido: {len(train_data)}."
        )
    if len(test_data) == 0:
        raise ValueError("Conjunto de teste vazio. Ajuste TEST_SIZE_PCT.")
    
    print(f"[INFO] Treino (retornos): {len(train_data)} horas | Teste (retornos): {len(test_data)} horas")
    
    # 2. Normalização (Fit apenas no Treino!)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values)
    
    # Para o teste, precisamos das últimas LOOKBACK observações do treino
    dataset_total = pd.concat((train_data['log_return'], test_data['log_return']), axis=0)
    inputs = np.asarray(dataset_total.to_numpy(), dtype=float).reshape(-1, 1)
    scaled_all = scaler.transform(inputs)
    
    # 3. Janelamento
    X_train, y_train = create_sliding_window(scaled_train, LOOKBACK)

    # Gera janelas em toda a série escalada e recorta apenas alvos do período de teste
    X_all, y_all = create_sliding_window(scaled_all, LOOKBACK)
    test_start_idx_in_windows = split_idx - LOOKBACK
    if test_start_idx_in_windows < 0:
        raise ValueError("Split inválido para LOOKBACK atual. Ajuste TEST_SIZE_PCT ou LOOKBACK.")

    X_test = X_all[test_start_idx_in_windows:]
    y_test = y_all[test_start_idx_in_windows:]
    
    # Reshape para
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    run_walk_forward_backtest(X_train, y_train, scaler)

    if len(X_train) < 2:
        raise ValueError("Janelas de treino insuficientes para separar treino/validação.")

    # 4. Validação temporal explícita (sem mistura temporal)
    val_size = max(1, int(len(X_train) * VAL_SIZE_PCT))
    if val_size >= len(X_train):
        val_size = 1

    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    if len(X_train_fit) == 0:
        raise ValueError("Treino ficou vazio após split de validação. Ajuste VAL_SIZE_PCT.")
    
    # 5. Treinamento
    model = build_lstm_architecture((X_train.shape[1], 1))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Usa janela temporal final do treino como validação
    history = model.fit(
        X_train_fit, y_train_fit,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Avaliação
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions_return = scaler.inverse_transform(predictions_scaled).reshape(-1)
    y_test_return = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Converte previsão de retorno para preço no timestamp alvo
    target_indices = dataset_total.index[LOOKBACK + test_start_idx_in_windows: LOOKBACK + test_start_idx_in_windows + len(y_test)]
    prev_close = close_series.shift(1).reindex(target_indices).values
    y_test_real_price = close_series.reindex(target_indices).values

    valid_mask = (~np.isnan(prev_close)) & (~np.isnan(y_test_real_price))
    prev_close = prev_close[valid_mask]
    y_test_real_price = y_test_real_price[valid_mask]
    predictions_return = predictions_return[valid_mask]
    y_test_return = y_test_return[valid_mask]

    predictions_price = prev_close * np.exp(predictions_return)

    # Baseline ingênuo: próxima hora = último close observado na janela
    baseline_predictions_price = prev_close
    baseline_scaled = X_test[:, -1, 0].reshape(-1, 1)
    baseline_return = scaler.inverse_transform(baseline_scaled).reshape(-1)[valid_mask]
    
    mae = mean_absolute_error(y_test_real_price, predictions_price)
    rmse = np.sqrt(mean_squared_error(y_test_real_price, predictions_price))
    mape = safe_mape(y_test_real_price, predictions_price, EPSILON)

    baseline_mae = mean_absolute_error(y_test_real_price, baseline_predictions_price)
    baseline_rmse = np.sqrt(mean_squared_error(y_test_real_price, baseline_predictions_price))
    baseline_mape = safe_mape(y_test_real_price, baseline_predictions_price, EPSILON)

    model_return_mae = mean_absolute_error(y_test_return, predictions_return)
    baseline_return_mae = mean_absolute_error(y_test_return, baseline_return)

    model_direction = np.sign(predictions_return)
    real_direction = np.sign(y_test_return)
    direction_accuracy = np.mean(model_direction == real_direction) * 100

    model_beats_baseline = mae < baseline_mae and rmse < baseline_rmse

    metadata = {
        "ticker": TICKER,
        "target": "log_return",
        "lookback": LOOKBACK,
        "interval": INTERVAL,
        "period": PERIOD,
        "seed": RANDOM_SEED,
        "metrics": {
            "mae_price": float(mae),
            "rmse_price": float(rmse),
            "mape_price": float(mape),
            "mae_price_baseline": float(baseline_mae),
            "rmse_price_baseline": float(baseline_rmse),
            "mape_price_baseline": float(baseline_mape),
            "mae_return": float(model_return_mae),
            "mae_return_baseline": float(baseline_return_mae),
            "direction_accuracy_pct": float(direction_accuracy)
        },
        "model_promoted": bool(model_beats_baseline)
    }

    if model_beats_baseline:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(MODEL_META_PATH, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)
        print(f"[INFO] Modelo promovido e salvo em {MODEL_PATH}")
    else:
        model.save(MODEL_CANDIDATE_PATH)
        joblib.dump(scaler, SCALER_CANDIDATE_PATH)
        with open(MODEL_CANDIDATE_META_PATH, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)
        print(f"[WARN] Modelo não superou baseline. Salvo como candidato em {MODEL_CANDIDATE_PATH}")
    
    print("\n" + "="*40)
    print(f"RELATÓRIO DE PERFORMANCE ({TICKER} - HORÁRIO)")
    print("="*40)
    print(f"Erro Médio Absoluto (MAE): $ {mae:.2f}")
    print(f"RMSE: $ {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("-"*40)
    print("BASELINE INGÊNUO (y_hat = último close da janela)")
    print(f"MAE Baseline: $ {baseline_mae:.2f}")
    print(f"RMSE Baseline: $ {baseline_rmse:.2f}")
    print(f"MAPE Baseline: {baseline_mape:.2f}%")
    print("-"*40)
    print("MÉTRICAS DE RETORNO E DIREÇÃO")
    print(f"MAE Retorno (Modelo): {model_return_mae:.6f}")
    print(f"MAE Retorno (Baseline): {baseline_return_mae:.6f}")
    print(f"Acurácia Direcional: {direction_accuracy:.2f}%")
    print("-"*40)
    print(f"Modelo promovido para produção? {'SIM' if model_beats_baseline else 'NÃO'}")
    print("="*40)

if __name__ == "__main__":
    main()