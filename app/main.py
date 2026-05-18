import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.model_service import ModelNotReadyError, model_service
from app.model_service import (
    get_model_service_for_symbol,
    list_available_model_symbols,
    normalize_symbol,
)
from app.monitoring import metrics_store
from app.schemas import PredictTickerFromYFinanceRequest
from app.schemas import (
    HealthResponse,
    PredictFromYFinanceRequest,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_SNAPSHOT_INTERVAL = int(os.getenv("METRICS_SNAPSHOT_INTERVAL", "30"))


async def _metrics_recorder() -> None:
    while True:
        metrics_store.record_snapshot()
        await asyncio.sleep(_SNAPSHOT_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_metrics_recorder())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="Tech Challenge Fase 4 - LSTM Stock Price API",
    description=(
        "API REST para servir um modelo LSTM de previsão de preço de fechamento "
        "de ações."
    ),
    version="1.0.0",
    lifespan=lifespan,
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Tech Challenge Fase 4 - LSTM Stock Price API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "dashboard": "/dashboard",
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


@app.get("/metrics/history")
def metrics_history() -> list[dict[str, Any]]:
    return metrics_store.history()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text(encoding="utf-8")


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



@app.get("/models")
def available_models() -> dict:
    return {
        "available_models": list_available_model_symbols()
    }


@app.get("/models/{symbol}/health")
def health_by_symbol(symbol: str) -> dict:
    try:
        clean_symbol = normalize_symbol(symbol)
        service = get_model_service_for_symbol(clean_symbol)
        files = service.files_status()

        return {
            "status": "ok",
            "symbol": clean_symbol,
            "model_loaded": service.is_loaded(),
            "model_dir": str(service.model_dir),
            **files,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/models/{symbol}/info")
def model_info_by_symbol(symbol: str) -> dict:
    try:
        clean_symbol = normalize_symbol(symbol)
        service = get_model_service_for_symbol(clean_symbol)
        service.load()

        return {
            "symbol": clean_symbol,
            "sequence_length": service.sequence_length,
            "metadata": service.metadata,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/{symbol}/from-yfinance")
def predict_by_symbol_from_yfinance(
    symbol: str,
    payload: PredictTickerFromYFinanceRequest,
) -> dict:
    try:
        clean_symbol = normalize_symbol(symbol)
        service = get_model_service_for_symbol(clean_symbol)

        result = service.predict_from_yfinance(
            symbol=clean_symbol,
            start_date=payload.start_date,
            end_date=payload.end_date,
            n_future=payload.n_future,
        )

        metrics_store.register_prediction()

        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

