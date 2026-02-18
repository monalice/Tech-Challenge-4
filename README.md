# Tech-Challenge-4

API e pipeline de treino para previsão do próximo fechamento horário do Bitcoin (`BTC-USD`) com LSTM.

## Requisitos

- Docker (recomendado para execução da API)
- Python 3.11+
- Ambiente virtual (`.venv`)

## Executar API com Docker (recomendado)

### Com Docker Compose

```bash
docker-compose up --build
```

### Com Docker direto

```bash
docker build -t stockcast-api:latest .
docker run --rm -p 8000:8000 stockcast-api:latest
```

## Execução local (alternativa)

### Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Subir API localmente

```bash
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

## Documentação da API

Após subir a aplicação, acesse:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Endpoints

- `GET /live`
  - Endpoint leve para liveness (usado no healthcheck do Docker Compose).
  - Não consulta mercado nem aquece cache da previsão.
- `GET /health`
  - Endpoint de diagnóstico completo.
  - Retorna estado efetivo da API com validações de:
    - artefatos carregados
    - inferência real do modelo
    - acesso a dados de mercado
  - Inclui timestamp do último candle válido em:
    - `last_market_timestamp_utc`
    - `last_market_timestamp_brt`
- `POST /predict`
  - Aceita apenas `BTC-USD`.
  - Retorna, além do preço previsto:
    - `forecast_for_utc` (início da hora prevista em UTC)
    - `forecast_for_brt` (início da hora prevista em Brasília)
    - `forecast_close_utc` (fechamento da hora prevista em UTC)
    - `forecast_close_brt` (fechamento da hora prevista em Brasília)
    - `confidence_interval_95_usd` (intervalo de confiança estimado)
    - `estimated_error_pct` (erro percentual estimado)
  - Exemplo de body:

```json
{
  "ticker": "BTC-USD"
}
```

## Treinamento do modelo (opcional)

Para testar a API, não é necessário treinar o modelo localmente: os artefatos já estão versionados no repositório. Nesse caso, basta rodar a API com Docker.

Treine localmente apenas se quiser gerar novos artefatos:

```bash
python src/train_model.py
```

Saídas geradas em `models/`:

- `lstm_btc_hourly.keras`
- `scaler_btc.gz`
- `model_metadata_btc.json`

## Observações

- A API utiliza cache curto e retry para chamadas ao Yahoo Finance.
- O healthcheck do Docker Compose usa `GET /live` para evitar impacto em cache de mercado.
- O pipeline treina em `log-return` e converte previsão para preço final.
