# Tech Challenge Fase 4 — LSTM Stock Price Prediction API

API REST com FastAPI que serve modelos LSTM para previsão de preço de fechamento de ações. Suporta múltiplos tickers.

Fluxo: **Colab (treino) → GitHub → Render (deploy)**

---

## Estrutura

```
├── app/                  # API FastAPI
├── src/training/         # Código de treino do modelo
├── models/<TICKER>/      # Artefatos por ticker (model.keras, scaler.pkl, metadata.json)
├── reports/<TICKER>/     # Métricas e gráficos gerados no treino
├── examples/             # JSONs de exemplo para testar a API
├── mlops/                # MLflow tracking + Evidently drift detection
├── Dockerfile
├── requirements.txt
```

Tickers treinados: AAPL, DIS, MSFT, NVDA, TSLA

---

## Rodando no Colab

```python
!git clone https://<TOKEN>@github.com/anibalssilva/tech-challenge-fase4-lstm.git
%cd tech-challenge-fase4-lstm
!pip install -r requirements.txt
```

Treinar:
```python
!python -m src.training.train --symbol DIS --start-date 2018-01-01 --end-date 2024-07-20 --sequence-length 60 --epochs 40 --batch-size 32 --model-dir models
```

Subir a API:
```python
!uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

---

## Rodando local

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docs em http://localhost:8000/docs

---

## Endpoints

| Método | Rota | O que faz |
|--------|------|-----------|
| GET | `/` | Info geral |
| GET | `/health` | Status da API |
| GET | `/models` | Tickers disponíveis |
| GET | `/models/{symbol}/health` | Status de um ticker |
| GET | `/models/{symbol}/info` | Metadados do modelo |
| GET | `/model-info` | Metadados do modelo padrão |
| GET | `/metrics` | Monitoramento (requests, erros, latência, CPU, memória) |
| POST | `/predict` | Previsão com lista de preços |
| POST | `/predict/from-yfinance` | Previsão via Yahoo Finance |
| POST | `/predict/{symbol}/from-yfinance` | Previsão por ticker via Yahoo Finance |

---

## Exemplos

```bash
curl http://localhost:8000/models

curl -X POST "http://localhost:8000/predict/AAPL/from-yfinance" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-07-20", "n_future": 5}'
```

---

## Deploy no Render

Conecta o repo no Render, seleciona Docker, deploy. Pronto.

---

## MLOps — Tracking e Drift Detection

Camada de observabilidade adicionada ao projeto. Stack: **MLflow** (tracking + Model Registry) e **Evidently AI** (data drift).

Permite versionar modelos, registrar métricas de cada treino e detectar mudanças na distribuição dos dados ao longo do tempo — fundamentos pra operar modelos de ML em produção.

### Setup

```bash
pip install -r requirements-monitoring.txt
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

UI em http://localhost:5000

### Notebooks

| Notebook | O que faz |
|----------|-----------|
| `mlops/notebooks/evaluate_and_track.ipynb` | Loga modelos `lstm_relu` e `lstm_tanh` no MLflow, registra o melhor como `stock-lstm-AAPL` v1 em Production |
| `mlops/notebooks/drift_detection.ipynb` | Compara distribuição do treino vs teste, gera HTML report do Evidently, loga no MLflow |

Ordem de execução: `evaluate_and_track` → `drift_detection`.

### Resultado (AAPL)

- 2 runs no experiment `stock-lstm-AAPL` com params, metrics e artifacts
- Modelo registrado em Production no Model Registry
- Data drift detectado entre período de treino (~$42-$170) e teste (~$170-$235)
- Métrica adicional `Directional Accuracy` calculada (% acerto da direção do movimento — relevante em trading)

---

## Obs

Projeto acadêmico. Não é recomendação financeira.
