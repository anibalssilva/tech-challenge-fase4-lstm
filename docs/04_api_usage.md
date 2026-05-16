# Uso da API

## Base URL local

```text
http://localhost:8000
```

## Base URL em produção

```text
https://SUA-API.onrender.com
```

## 1. Health check

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": false,
  "model_dir": "models",
  "model_file_exists": true,
  "scaler_file_exists": true,
  "metadata_file_exists": true
}
```

## 2. Informações do modelo

```bash
curl http://localhost:8000/model-info
```

## 3. Previsão informando preços históricos

Endpoint:

```text
POST /predict
```

Exemplo:

```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d @examples/predict_request.json
```

Payload:

```json
{
  "prices": [92.10, 92.80, 93.05],
  "n_future": 1
}
```

Observação: no uso real, envie pelo menos `sequence_length` preços. O padrão do projeto é 60.

## 4. Previsão usando Yahoo Finance

Endpoint:

```text
POST /predict/from-yfinance
```

Exemplo:

```bash
curl -X POST "http://localhost:8000/predict/from-yfinance"   -H "Content-Type: application/json"   -d @examples/predict_from_yfinance_request.json
```

Payload:

```json
{
  "symbol": "DIS",
  "start_date": "2018-01-01",
  "end_date": "2024-07-20",
  "n_future": 5
}
```

## 5. Métricas da API

```bash
curl http://localhost:8000/metrics
```

Exemplo de resposta:

```json
{
  "uptime_seconds": 120.5,
  "request_count": 10,
  "prediction_count": 3,
  "error_count": 0,
  "avg_latency_ms": 45.2,
  "last_latency_ms": 30.1,
  "cpu_percent": 2.5,
  "memory_mb": 350.8
}
```
