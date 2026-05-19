from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.training.config import TrainingConfig
from src.training.data import (
    download_stock_data,
    extract_close_prices,
    prepare_data,
)
from src.training.metrics import regression_metrics
from src.training.model import build_lstm_model


def parse_args() -> argparse.Namespace:
    default = TrainingConfig()

    parser = argparse.ArgumentParser(
        description="Treina modelos LSTM para prever fechamento de ações."
    )
    parser.add_argument("--symbol", default=default.symbol)
    parser.add_argument("--start-date", default=default.start_date)
    parser.add_argument("--end-date", default=default.end_date)
    parser.add_argument("--sequence-length", type=int, default=default.sequence_length)
    parser.add_argument("--train-size", type=float, default=default.train_size)
    parser.add_argument("--epochs", type=int, default=default.epochs)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--lstm-units", type=int, default=default.lstm_units)
    parser.add_argument("--dropout", type=float, default=default.dropout)
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate)
    parser.add_argument("--model-dir", default=default.model_dir)
    parser.add_argument("--report-dir", default=default.report_dir)
    return parser.parse_args()


def create_callbacks() -> list:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        ),
    ]


def save_training_plot(history, report_dir: Path, model_name: str) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(history.history["loss"], label="train_loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")

    plt.title(f"Training Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / f"training_loss_{model_name}.png")
    plt.close()


def save_loss_comparison_plot(histories: dict[str, Any], report_dir: Path) -> None:
    plt.figure(figsize=(10, 5))

    for model_name, history in histories.items():
        plt.plot(
            history.history["loss"],
            label=f"{model_name} - train_loss",
        )

        if "val_loss" in history.history:
            plt.plot(
                history.history["val_loss"],
                label=f"{model_name} - val_loss",
            )

    plt.title("Comparação de Loss - LSTM tanh vs LSTM relu")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "loss_comparison.png")
    plt.close()


def save_metrics_comparison_plot(metrics_df: pd.DataFrame, report_dir: Path) -> None:
    plot_df = metrics_df.set_index("model")[["mae", "rmse", "mape"]]

    plot_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    plt.title("Comparação de Métricas - LSTM tanh vs LSTM relu")
    plt.xlabel("Modelo")
    plt.ylabel("Erro")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "metrics_comparison.png")
    plt.close()


def normalize_metric_key(key: str) -> str:
    return (
        str(key)
        .lower()
        .replace(" (%)", "")
        .replace("(%)", "")
        .replace("%", "")
        .replace(" ", "_")
        .strip()
    )


def get_metric_value(metrics: dict, metric_name: str) -> float:
    normalized_metrics = {
        normalize_metric_key(key): value
        for key, value in metrics.items()
    }

    normalized_name = normalize_metric_key(metric_name)

    if normalized_name not in normalized_metrics:
        raise KeyError(
            f"Métrica '{metric_name}' não encontrada. "
            f"Métricas disponíveis: {list(metrics.keys())}"
        )

    return float(normalized_metrics[normalized_name])


def standardize_regression_metrics(metrics: dict) -> dict:
    normalized_metrics = {
        normalize_metric_key(key): float(value)
        for key, value in metrics.items()
    }

    mae = normalized_metrics.get("mae")
    rmse = normalized_metrics.get("rmse")
    mape = normalized_metrics.get("mape")

    if mape is None:
        mape = normalized_metrics.get("mape_percent")

    if mae is None:
        raise KeyError(
            f"Métrica 'mae' não encontrada. Métricas disponíveis: {list(metrics.keys())}"
        )

    if rmse is None:
        raise KeyError(
            f"Métrica 'rmse' não encontrada. Métricas disponíveis: {list(metrics.keys())}"
        )

    if mape is None:
        raise KeyError(
            f"Métrica 'mape' ou 'mape_percent' não encontrada. "
            f"Métricas disponíveis: {list(metrics.keys())}"
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("1/6 - Baixando dados...")

    raw_df = download_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    close_df = extract_close_prices(raw_df)
    close_df.to_csv(report_dir / "close_prices.csv")

    print("2/6 - Preparando sequências...")

    x_train, y_train, x_test, y_test, scaler = prepare_data(
        close_df=close_df,
        sequence_length=args.sequence_length,
        train_size=args.train_size,
    )

    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    activations = ["tanh", "relu"]

    trained_models = {}
    histories = {}
    all_metrics = []

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test_real.reshape(-1),
        }
    )

    print("3/6 - Treinando e avaliando modelos...")

    for activation in activations:
        model_name = f"lstm_{activation}"

        print(f"\nConstruindo modelo: {model_name}")

        tf.keras.utils.set_random_seed(42)

        model = build_lstm_model(
            sequence_length=args.sequence_length,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            lstm_activation=activation,
        )

        model.summary()

        print(f"\nTreinando modelo: {model_name}")

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=create_callbacks(),
            verbose=0,
        )

        print(f"\nAvaliando modelo: {model_name}")

        y_pred_scaled = model.predict(x_test, verbose=0)
        y_pred_real = scaler.inverse_transform(y_pred_scaled)

        raw_metrics = regression_metrics(
            y_true=y_test_real,
            y_pred=y_pred_real,
        )

        standardized_metrics = standardize_regression_metrics(raw_metrics)

        model_metrics = {
            "model": model_name,
            "lstm_activation": activation,
            "mae": standardized_metrics["mae"],
            "rmse": standardized_metrics["rmse"],
            "mape": standardized_metrics["mape"],
            "epochs_executed": len(history.history["loss"]),
        }

        all_metrics.append(model_metrics)

        trained_models[activation] = model
        histories[model_name] = history

        predictions_df[f"y_pred_{activation}"] = y_pred_real.reshape(-1)

        save_training_plot(
            history=history,
            report_dir=report_dir,
            model_name=model_name,
        )

        model.save(model_dir / f"model_{activation}.keras")

        print(f"\nMétricas do modelo {model_name}:")
        print(model_metrics)

    print("\n4/6 - Comparando modelos...")

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(
        report_dir / "metrics_comparison.csv",
        index=False,
    )

    predictions_df.to_csv(
        report_dir / "predictions_comparison.csv",
        index=False,
    )

    save_loss_comparison_plot(
        histories=histories,
        report_dir=report_dir,
    )

    save_metrics_comparison_plot(
        metrics_df=metrics_df,
        report_dir=report_dir,
    )

    best_model_row = metrics_df.sort_values("rmse").iloc[0]
    best_activation = str(best_model_row["lstm_activation"])
    best_model = trained_models[best_activation]

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")
    
    print("\nComparação dos modelos:")
    print(metrics_df.to_string(index=False))

    print("\nMelhor modelo escolhido pelo menor RMSE:")
    print(best_model_row)

    print("\n5/6 - Salvando melhor modelo e artefatos...")

    best_model.save(model_dir / "model.keras")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    best_metrics = best_model_row.to_dict()

    metadata = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_length": args.sequence_length,
        "train_size": args.train_size,
        "epochs_requested": args.epochs,
        "epochs_executed": int(best_model_row["epochs_executed"]),
        "batch_size": args.batch_size,
        "lstm_units": args.lstm_units,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "features": ["Close"],
        "rows_downloaded": int(len(raw_df)),
        "rows_after_cleaning": int(len(close_df)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "last_close": round(float(close_df["Close"].iloc[-1]), 4),
        "models_tested": activations,
        "best_model": str(best_model_row["model"]),
        "best_activation": best_activation,
        "selection_metric": "rmse",
        "metrics": best_metrics,
        "all_models_metrics": metrics_df.to_dict(orient="records"),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(metadata),
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(best_metrics),
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (report_dir / "metrics_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(metrics_df.to_dict(orient="records")),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n6/6 - Treino finalizado com sucesso.")
    print(f"Melhor ativação LSTM: {best_activation}")
    print(f"Melhor modelo salvo em: {model_dir / 'model.keras'}")
    print(f"Scaler salvo em: {model_dir / 'scaler.pkl'}")
    print(f"Metadata salvo em: {model_dir / 'metadata.json'}")
    print(f"Relatórios salvos em: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
