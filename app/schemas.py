from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    prices: list[float] = Field(
        ...,
        description=(
            "Lista de preços históricos de fechamento. "
            "A lista precisa ter pelo menos o sequence_length usado no treino."
        ),
    )
    n_future: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Quantidade de dias futuros para prever.",
    )


class PredictFromYFinanceRequest(BaseModel):
    symbol: str = Field(
        default="DIS",
        min_length=1,
        description="Ticker da ação. Exemplos: DIS, AAPL, MSFT, PETR4.SA.",
    )
    start_date: str = Field(
        default="2018-01-01",
        description="Data inicial no formato YYYY-MM-DD.",
    )
    end_date: str | None = Field(
        default=None,
        description="Data final no formato YYYY-MM-DD. Se vazio, usa a data atual.",
    )
    n_future: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Quantidade de dias futuros para prever.",
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    n_future: int
    last_input_price: float
    sequence_length: int
    model_metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: str
    model_file_exists: bool
    scaler_file_exists: bool
    metadata_file_exists: bool
