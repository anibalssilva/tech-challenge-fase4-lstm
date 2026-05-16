# Roteiro sugerido para o vídeo de entrega

## 1. Introdução

"Este é o projeto do Tech Challenge Fase 4. O objetivo foi desenvolver um modelo LSTM para prever o preço de fechamento de uma ação e disponibilizar esse modelo em uma API REST."

## 2. Explicar o problema

- Previsão de série temporal
- Ação escolhida
- Uso do preço de fechamento
- Coleta via Yahoo Finance usando `yfinance`

## 3. Mostrar o Google Colab

Mostre:

- clone do repositório
- instalação das dependências
- execução do treino
- geração dos artefatos em `models/`
- métricas MAE, RMSE e MAPE

## 4. Explicar o modelo LSTM

Explique de forma simples:

- LSTM é adequada para séries temporais
- o modelo usa janelas de preços anteriores
- `sequence_length=60` significa olhar os últimos 60 fechamentos
- a saída é o próximo preço previsto

## 5. Mostrar a estrutura do GitHub

Mostre:

- `app/`
- `src/training/`
- `models/`
- `Dockerfile`
- `render.yaml`
- `docs/`

## 6. Mostrar a API local ou no Render

Acesse:

```text
/docs
```

Mostre os endpoints:

- `/health`
- `/model-info`
- `/predict`
- `/predict/from-yfinance`
- `/metrics`

## 7. Fazer uma chamada real

Use `/predict` ou `/predict/from-yfinance` e mostre o retorno.

## 8. Mostrar monitoramento

Acesse:

```text
/metrics
```

Explique:

- latência média
- quantidade de requisições
- uso de CPU
- uso de memória

## 9. Conclusão

"Com isso, o projeto cobre coleta, pré-processamento, treinamento LSTM, avaliação, salvamento do modelo, criação da API, deploy com Docker no Render e monitoramento básico."
