# Tech-Challenge-4

API e pipeline de treino para previsão do próximo fechamento horário do Bitcoin (`BTC-USD`) com LSTM.

## Requisitos

- Python 3.11+
- Ambiente virtual (`.venv`)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Treinamento

```bash
python src/train_model.py
```

Saídas geradas em `models/`:

- Produção (quando supera baseline):
  - `lstm_btc_hourly.keras`
  - `scaler_btc.gz`
  - `model_metadata_btc.json`
- Candidato (quando não supera baseline):
  - `lstm_btc_hourly_candidate.keras`
  - `scaler_btc_candidate.gz`
  - `model_metadata_btc_candidate.json`

## Regras de promoção do modelo

O modelo só é promovido para produção quando supera o baseline ingênuo (`y_hat = último close`) em:

- MAE (preço)
- RMSE (preço)

Se não superar, é salvo como candidato e os artefatos de produção atuais não são sobrescritos.

## Executar API local

```bash
.venv\Scripts\python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
  - Retorna estado da aplicação e prontidão dos artefatos (`artifacts_ready`).
- `POST /predict`
  - Aceita apenas `BTC-USD`.
  - Exemplo de body:

```json
{
  "ticker": "BTC-USD"
}
```

## Build Docker

```bash
docker build -t stockcast-api:latest .
docker run --rm -p 8000:8000 stockcast-api:latest
```

## Observações

- A API utiliza cache curto e retry para chamadas ao Yahoo Finance.
- O pipeline treina em `log-return` e converte previsão para preço final.
