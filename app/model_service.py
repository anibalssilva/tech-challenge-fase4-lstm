import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yfinance as yf
from tensorflow import keras


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(model_dir or os.getenv("MODEL_DIR", "models"))
        self.model_path = self.model_dir / "model.keras"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.metadata_path = self.model_dir / "metadata.json"

        self.model: keras.Model | None = None
        self.scaler: Any | None = None
        self.metadata: dict[str, Any] = {}

    def files_status(self) -> dict[str, bool]:
        return {
            "model_file_exists": self.model_path.exists(),
            "scaler_file_exists": self.scaler_path.exists(),
            "metadata_file_exists": self.metadata_path.exists(),
        }

    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        missing_files = [
            str(path)
            for path in [self.model_path, self.scaler_path, self.metadata_path]
            if not path.exists()
        ]
        if missing_files:
            raise ModelNotReadyError(
                "Artefatos do modelo não encontrados. "
                f"Arquivos ausentes: {missing_files}. "
                "Treine o modelo primeiro no Google Colab e faça commit/push da pasta models/."
            )

        self.model = keras.models.load_model(self.model_path, compile=False)
        self.scaler = joblib.load(self.scaler_path)

        with self.metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

    @property
    def sequence_length(self) -> int:
        value = self.metadata.get("sequence_length")
        if value is not None:
            return int(value)
        return int(os.getenv("DEFAULT_SEQUENCE_LENGTH", "60"))

    def predict_from_prices(
        self,
        prices: list[float],
        n_future: int = 1,
    ) -> dict[str, Any]:
        self.load()

        if self.model is None or self.scaler is None:
            raise ModelNotReadyError("Modelo não carregado.")

        if len(prices) < self.sequence_length:
            raise ValueError(
                f"Você enviou {len(prices)} preços, mas o modelo precisa de pelo "
                f"menos {self.sequence_length} preços históricos."
            )

        clean_prices = np.array(prices, dtype=float).reshape(-1, 1)
        if np.isnan(clean_prices).any():
            raise ValueError("A lista de preços contém valores inválidos ou NaN.")

        scaled_prices = self.scaler.transform(clean_prices)
        current_sequence = scaled_prices[-self.sequence_length :].reshape(
            1,
            self.sequence_length,
            1,
        )

        predictions: list[float] = []

        for _ in range(n_future):
            pred_scaled = self.model.predict(current_sequence, verbose=0)
            pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(round(float(pred_price), 4))

            next_scaled_value = pred_scaled.reshape(1, 1, 1)
            current_sequence = np.concatenate(
                [current_sequence[:, 1:, :], next_scaled_value],
                axis=1,
            )

        return {
            "predictions": predictions,
            "n_future": n_future,
            "last_input_price": round(float(clean_prices[-1][0]), 4),
            "sequence_length": self.sequence_length,
            "model_metadata": self.metadata,
        }

    def predict_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str | None,
        n_future: int,
    ) -> dict[str, Any]:
        final_date = end_date or date.today().isoformat()
        df = yf.download(
            symbol,
            start=start_date,
            end=final_date,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                "Nenhum dado foi encontrado no Yahoo Finance. "
                "Verifique ticker, start_date e end_date."
            )

        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            if "Close" in df.columns.get_level_values(0):
                close = df["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
            else:
                raise ValueError("Coluna Close não encontrada no dataset baixado.")
        else:
            if "Close" not in df.columns:
                raise ValueError("Coluna Close não encontrada no dataset baixado.")
            close = df["Close"]

        prices = close.dropna().astype(float).tolist()
        result = self.predict_from_prices(prices=prices, n_future=n_future)
        result["symbol"] = symbol
        result["downloaded_rows"] = len(prices)
        result["data_start_date"] = start_date
        result["data_end_date"] = final_date
        return result


model_service = ModelService()
