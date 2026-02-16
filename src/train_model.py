import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configurações de Hiperparâmetros e Constantes
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]   # Treinar modelo separado para cada cripto
START_DATE = "2020-01-01"  # SOL só tem dados a partir de 2020
END_DATE = "2025-12-31"
LOOKBACK = 60         # Janela de observação temporal
BATCH_SIZE = 32       # Tamanho do lote para atualização de gradiente
EPOCHS = 1000          # Máximo de épocas (controlado por Early Stopping)
TEST_SPLIT_DATE = "2024-01-01" # Data de corte para validação cronológica

def ensure_directories():
    """Garante a existência dos diretórios necessários."""
    if not os.path.exists("models"):
        os.makedirs("models")

def download_financial_data(ticker, start, end):
    """
    Realiza o download dos dados históricos via yfinance.
    Trata o problema de MultiIndex nas versões recentes da biblioteca.
    """
    print(f"[INFO] Baixando dados para {ticker} de {start} a {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        raise ValueError(f"Não foi possível baixar dados para {ticker}. Verifique se o ticker está correto.")
    
    # Tratamento para garantir DataFrame plano (correção para yfinance > 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except KeyError:
            # Fallback caso a estrutura seja diferente
            df.columns = df.columns.get_level_values(0)

    # Para criptomoedas usamos Close, para ações usamos Adj Close
    price_column = 'Close' if 'Close' in df.columns else 'Adj Close'
    
    if price_column not in df.columns:
        raise ValueError(f"Coluna de preço não encontrada. Colunas disponíveis: {df.columns.tolist()}")
    
    data = df[[price_column]].copy()
    data.columns = ['Adj Close']  # Padronizar nome da coluna
    
    # Tratamento de Nulos: Forward Fill para propagar o último preço válido
    data = data.ffill().dropna()
    
    if len(data) == 0:
        raise ValueError(f"Nenhum dado válido encontrado para {ticker}.")
    
    print(f"[INFO] {len(data)} dias de dados carregados com sucesso")
    return data

def create_sliding_window(dataset, look_back=60):
    """
    Converte uma série temporal em um conjunto de dados supervisionado.
    Entrada: Vetor de preços normalizados.
    Saída: X (features de t-n a t-1) e y (target em t).
    """
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def build_lstm_architecture(input_shape):
    """
    Define a topologia da Rede Neural Profunda.
    Arquitetura: Stacked LSTM com Dropout.
    """
    model = Sequential()
    
    # 1ª Camada LSTM: return_sequences=True é vital para empilhar outra LSTM
    # 50 neurônios fornecem capacidade suficiente para capturar padrões complexos
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Desliga 20% dos neurônios para evitar overfitting
    
    # 2ª Camada LSTM: return_sequences=False pois a próxima é Densa
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Camada Densa para condensar as features em uma predição escalar
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1)) # Saída linear (regressão)
    
    # Compilação: Adam é o otimizador padrão para RNNs devido ao momento adaptativo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def main():
    ensure_directories()
    
    # Treinar um modelo para cada criptomoeda
    for ticker in TICKERS:
        print("\n" + "="*60)
        print(f"TREINANDO MODELO PARA {ticker}")
        print("="*60)
        
        # Gerar nomes de arquivos específicos para cada ticker
        ticker_name = ticker.split('-')[0].lower()  # btc, eth, sol
        model_path = f"models/{ticker_name}_model.keras"
        scaler_path = f"models/{ticker_name}_scaler.gz"
        
        # 1. Pipeline de Dados
        df = download_financial_data(ticker, START_DATE, END_DATE)
        
        # Divisão Cronológica (Treino vs Teste)
        train_data = df[df.index < TEST_SPLIT_DATE]
        test_data = df[df.index >= TEST_SPLIT_DATE]
        
        print(f"[INFO] Treino: {len(train_data)} amostras | Teste: {len(test_data)} amostras")
        
        # 2. Normalização
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.values)
        
        # Transform no teste
        dataset_total = pd.concat((train_data['Adj Close'], test_data['Adj Close']), axis=0)
        inputs = dataset_total.values.reshape(-1, 1)
        scaled_test = scaler.transform(inputs)
        
        # Persistência do Scaler
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler salvo em {scaler_path}")
        
        # 3. Preparação dos Tensores
        X_train, y_train = create_sliding_window(scaled_train, LOOKBACK)
        X_test, y_test = create_sliding_window(scaled_test, LOOKBACK)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 4. Construção e Treinamento do Modelo
        model = build_lstm_architecture((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        print(f"[INFO] Iniciando treinamento para {ticker}...")
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[early_stop],
            verbose=1
        )
        
        # 5. Persistência do Modelo
        model.save(model_path)
        print(f"[INFO] Modelo salvo em {model_path}")
        
        # 6. Avaliação e Métricas
        print("[INFO] Gerando predições...")
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = mean_absolute_error(y_test_real, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
        mape = np.mean(np.abs((y_test_real - predictions) / y_test_real)) * 100
        
        print("\n" + "="*40)
        print(f"RELATÓRIO DE PERFORMANCE ({ticker})")
        print("="*40)
        print(f"Erro Médio Absoluto (MAE): ${mae:.4f}")
        print(f"Raiz do Erro Quadrático (RMSE): ${rmse:.4f}")
        print(f"Erro Percentual Médio (MAPE): {mape:.2f}%")
        print("="*40)
    
    print("\n" + "="*60)
    print("TREINAMENTO COMPLETO! 3 modelos salvos:")
    print("  • models/btc_model.keras + btc_scaler.gz")
    print("  • models/eth_model.keras + eth_scaler.gz")
    print("  • models/sol_model.keras + sol_scaler.gz")
    print("="*60)

if __name__ == "__main__":
    main()