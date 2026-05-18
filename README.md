# Tech Challenge Fase 4 — LSTM Stock Price Prediction API

Projeto para o **Tech Challenge Fase 4 — Machine Learning Engineering**.

Treina redes neurais **LSTM** para prever o preço de fechamento de ações e serve os modelos via **API REST com FastAPI**, com suporte a múltiplos tickers.

> Fluxo principal: **Google Colab → GitHub → Render**

---

## Arquitetura

```text
Google Colab
   |
   | 1. Treina modelo LSTM (por ticker)
   | 2. Gera artefatos em models/<TICKER>/
   | 3. Faz commit/push para GitHub
   v
GitHub Repository
   |
   | 4. Render lê o repositório
   | 5. Build via Dockerfile
   v
Render Web Service
   |
   | 6. API FastAPI em produção
   v
Usuário / Avaliador
```

---

## Estrutura do projeto

```text
tech-challenge-fase4-lstm/
├── app/
│   ├── main.py            # API FastAPI
│   ├── model_service.py   # Carregamento e inferência dos modelos
│   ├── monitoring.py      # Métricas de monitoramento
│   └── schemas.py         # Schemas Pydantic
├── src/
│   └── training/
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
├── models/
│   ├── AAPL/
│   ├── DIS/
│   ├── MSFT/
│   ├── NVDA/
│   └── TSLA/
│       ├── model.keras
│       ├── model_relu.keras
│       ├── model_tanh.keras
│       ├── scaler.pkl
│       └── metadata.json
├── reports/
│   └── <TICKER>/          # Métricas, gráficos e predições por ticker
├── examples/
│   ├── predict_request.json
│   └── predict_from_yfinance_request.json
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Como usar no Google Colab

```python
!git clone https://<SEU_TOKEN>@github.com/anibalssilva/tech-challenge-fase4-lstm.git
%cd tech-challenge-fase4-lstm
!pip install -r requirements.txt
```

### Treinar um modelo

```python
!python -m src.training.train --symbol DIS --start-date 2018-01-01 --end-date 2024-07-20 --sequence-length 60 --epochs 40 --batch-size 32 --model-dir models
```

Tickers disponíveis: `AAPL`, `DIS`, `MSFT`, `NVDA`, `TSLA` (ou qualquer ticker válido do Yahoo Finance).

### Rodar a API

```python
!uvicorn app.main:app --host 0.0.0.0 --port 8000 &
```

---

## Como rodar localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Acesse: http://localhost:8000/docs

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações gerais |
| GET | `/health` | Status da API e do modelo padrão |
| GET | `/model-info` | Metadados do modelo padrão |
| GET | `/metrics` | Métricas de monitoramento (requests, erros, latência, CPU, memória) |
| GET | `/models` | Lista tickers com modelos disponíveis |
| GET | `/models/{symbol}/health` | Status do modelo de um ticker específico |
| GET | `/models/{symbol}/info` | Metadados do modelo de um ticker |
| POST | `/predict` | Previsão a partir de lista de preços históricos |
| POST | `/predict/from-yfinance` | Previsão usando dados do Yahoo Finance |
| POST | `/predict/{symbol}/from-yfinance` | Previsão por ticker usando Yahoo Finance |

---

## Exemplos de chamada

```bash
# Health check
curl http://localhost:8000/health

# Listar modelos disponíveis
curl http://localhost:8000/models

# Previsão com Yahoo Finance para um ticker específico
curl -X POST "http://localhost:8000/predict/AAPL/from-yfinance" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-07-20", "n_future": 5}'

# Previsão com lista de preços
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

---

## Deploy no Render

1. Conecte o repositório GitHub no Render
2. Crie um novo **Web Service**
3. Selecione **Docker** como ambiente
4. Deploy

---

## Monitoramento

O endpoint `/metrics` retorna:
- Total de requisições
- Total de predições
- Quantidade de erros
- Tempo médio de resposta (ms)
- Uso de CPU e memória

---

## Observação

Este projeto tem finalidade acadêmica. Não deve ser usado como recomendação financeira.
