import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denominator = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_percent": round(float(mape), 4),
    }
