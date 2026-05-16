# Tech Challenge Fase 4 — LSTM Stock Price Prediction API

Projeto completo para o **Tech Challenge Fase 4 — Machine Learning Engineering**.

O objetivo é treinar uma rede neural **LSTM** para prever o preço de fechamento de uma ação e publicar o modelo em uma **API REST com FastAPI**, pronta para deploy no **Render** via repositório GitHub.

> Fluxo principal: **Google Colab → GitHub → Render**

---

## 1. O que este projeto entrega

- Coleta de dados históricos via `yfinance`
- Pré-processamento da série temporal de fechamento (`Close`)
- Treinamento de modelo LSTM
- Avaliação com MAE, RMSE e MAPE
- Salvamento dos artefatos de inferência:
  - `models/model.keras`
  - `models/scaler.pkl`
  - `models/metadata.json`
- API RESTful com FastAPI
- Endpoint para previsão usando lista de preços históricos
- Endpoint opcional para previsão usando Yahoo Finance
- Monitoramento simples de:
  - quantidade de requisições
  - quantidade de predições
  - erros
  - tempo médio de resposta
  - uso de CPU e memória
- Dockerfile para deploy
- `render.yaml` para deploy por Blueprint
- Notebook para desenvolvimento no Google Colab
- Scripts de treino, execução local, teste e GitHub push

---

## 2. Arquitetura

```text
Google Colab
   |
   | 1. Treina modelo LSTM
   | 2. Gera artefatos em /models
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

## 3. Estrutura do projeto

```text
tech-challenge-fase4-lstm-stock-api/
├── app/
│   ├── main.py
│   ├── model_service.py
│   ├── monitoring.py
│   └── schemas.py
├── src/
│   └── training/
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
├── notebooks/
│   └── 01_train_lstm_colab.ipynb
├── scripts/
│   ├── train_model.sh
│   ├── run_api_local.sh
│   ├── test_api.sh
│   └── git_push_from_colab.sh
├── docs/
│   ├── 01_colab_development.md
│   ├── 02_github_repository.md
│   ├── 03_render_deploy.md
│   ├── 04_api_usage.md
│   ├── 05_monitoring.md
│   └── 06_video_script.md
├── examples/
│   ├── predict_request.json
│   └── predict_from_yfinance_request.json
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── Dockerfile
├── render.yaml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 4. Como usar no Google Colab

Abra o notebook:

```text
notebooks/01_train_lstm_colab.ipynb
```

No Colab, execute as células para:

1. Clonar seu repositório GitHub
2. Instalar dependências
3. Treinar o modelo LSTM
4. Testar a API
5. Fazer commit e push dos artefatos gerados

Treino padrão:

```bash
python -m src.training.train \
  --symbol DIS \
  --start-date 2018-01-01 \
  --end-date 2024-07-20 \
  --sequence-length 60 \
  --epochs 40 \
  --batch-size 32 \
  --model-dir models
```

Você pode trocar `DIS` por outra ação, por exemplo:

- `AAPL`
- `MSFT`
- `GOOGL`
- `TSLA`
- `PETR4.SA`
- `VALE3.SA`

---

## 5. Como rodar a API localmente

Depois de treinar e gerar os arquivos em `models/`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ou use o script:

```bash
bash scripts/run_api_local.sh
```

Acesse:

```text
http://localhost:8000/docs
```

---

## 6. Exemplo de chamada da API

### Health check

```bash
curl http://localhost:8000/health
```

### Previsão usando lista de preços

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

### Previsão usando Yahoo Finance

```bash
curl -X POST "http://localhost:8000/predict/from-yfinance" \
  -H "Content-Type: application/json" \
  -d @examples/predict_from_yfinance_request.json
```

---

## 7. Deploy no Render

Opção recomendada:

1. Crie um repositório no GitHub
2. Faça push deste projeto
3. Treine o modelo no Colab
4. Faça commit/push dos arquivos gerados em `models/`
5. No Render, crie um novo **Web Service**
6. Conecte seu repositório GitHub
7. Use Docker
8. Deploy

O projeto também contém `render.yaml`, então você pode usar **Blueprint**.

---

## 8. Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/` | Informações gerais |
| GET | `/health` | Status da API e do modelo |
| GET | `/model-info` | Metadados do modelo treinado |
| GET | `/metrics` | Métricas simples de monitoramento |
| POST | `/predict` | Previsão a partir de preços históricos informados |
| POST | `/predict/from-yfinance` | Previsão a partir de dados baixados via Yahoo Finance |

---

## 9. Observação importante sobre previsão financeira

Este projeto tem finalidade acadêmica. A previsão de ações com LSTM é útil para demonstrar engenharia de machine learning, séries temporais e deploy de modelos, mas não deve ser usada como recomendação financeira.

---

## 10. Checklist de entrega

- [ ] Código-fonte no GitHub
- [ ] Notebook executado no Google Colab
- [ ] Modelo treinado salvo em `models/`
- [ ] API funcionando localmente ou no Colab
- [ ] API publicada no Render
- [ ] Link da API em produção
- [ ] Vídeo explicando coleta, treino, avaliação, API, deploy e monitoramento
