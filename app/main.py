import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.model_service import ModelNotReadyError, model_service
from app.monitoring import metrics_store
from app.schemas import (
    HealthResponse,
    PredictFromYFinanceRequest,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tech Challenge Fase 4 - LSTM Stock Price API",
    description=(
        "API REST para servir um modelo LSTM de previsão de preço de fechamento "
        "de ações."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    is_error = False

    try:
        response = await call_next(request)
        if response.status_code >= 400:
            is_error = True
        return response
    except Exception:
        is_error = True
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics_store.register_request(latency_ms=latency_ms, is_error=is_error)
        logger.info(
            "path=%s method=%s latency_ms=%.2f error=%s",
            request.url.path,
            request.method,
            latency_ms,
            is_error,
        )


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "hint": (
                "Execute o notebook de treino no Google Colab para gerar "
                "models/model.keras, models/scaler.pkl e models/metadata.json."
            ),
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Tech Challenge Fase 4 - LSTM Stock Price API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    files = model_service.files_status()
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "model_dir": str(model_service.model_dir),
        **files,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    model_service.load()
    return {
        "sequence_length": model_service.sequence_length,
        "metadata": model_service.metadata,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return metrics_store.snapshot()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        result = model_service.predict_from_prices(
            prices=payload.prices,
            n_future=payload.n_future,
        )
        metrics_store.register_prediction()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/from-yfinance")
def predict_from_yfinance(payload: PredictFromYFinanceRequest) -> dict[str, Any]:
    try:
        result = model_service.predict_from_yfinance(
            symbol=payload.symbol,
            start_date=payload.start_date,
            end_date=payload.end_date,
            n_future=payload.n_future,
        )
        metrics_store.register_prediction()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
