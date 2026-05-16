from __future__ import annotations

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"Nenhum dado encontrado para symbol={symbol}, "
            f"start_date={start_date}, end_date={end_date}."
        )

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            raise ValueError("Coluna Close não encontrada no DataFrame.")
    else:
        if "Close" not in df.columns:
            raise ValueError("Coluna Close não encontrada no DataFrame.")
        close = df["Close"]

    result = close.dropna().astype(float).to_frame(name="Close")
    if result.empty:
        raise ValueError("Após remover valores nulos, não restaram preços Close.")
    return result


def create_lstm_sequences(
    values: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []

    for index in range(sequence_length, len(values)):
        x_values.append(values[index - sequence_length : index, 0])
        y_values.append(values[index, 0])

    x_array = np.array(x_values)
    y_array = np.array(y_values)

    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1], 1))
    return x_array, y_array


def prepare_data(
    close_df: pd.DataFrame,
    sequence_length: int,
    train_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_df[["Close"]].values)

    x_values, y_values = create_lstm_sequences(
        values=scaled_values,
        sequence_length=sequence_length,
    )

    if len(x_values) == 0:
        raise ValueError(
            "Quantidade de dados insuficiente para criar sequências. "
            "Aumente o intervalo de datas ou reduza sequence_length."
        )

    split_index = int(len(x_values) * train_size)

    x_train = x_values[:split_index]
    y_train = y_values[:split_index]
    x_test = x_values[split_index:]
    y_test = y_values[split_index:]

    if len(x_test) == 0:
        raise ValueError(
            "O conjunto de teste ficou vazio. Aumente o período de dados "
            "ou reduza train_size."
        )

    return x_train, y_train, x_test, y_test, scaler
