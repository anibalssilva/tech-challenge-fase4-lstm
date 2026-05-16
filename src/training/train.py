from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
        description="Treina modelo LSTM para prever fechamento de ações."
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


def save_training_plot(history, report_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "training_loss.png")
    plt.close()


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

    print("3/6 - Construindo modelo...")
    model = build_lstm_model(
        sequence_length=args.sequence_length,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )
    model.summary()

    callbacks = [
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

    print("4/6 - Treinando modelo...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("5/6 - Avaliando modelo...")
    y_pred_scaled = model.predict(x_test, verbose=0)

    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred_scaled)

    metrics = regression_metrics(y_true=y_test_real, y_pred=y_pred_real)
    print("Métricas:", metrics)

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test_real.reshape(-1),
            "y_pred": y_pred_real.reshape(-1),
        }
    )
    predictions_df.to_csv(report_dir / "predictions.csv", index=False)

    save_training_plot(history, report_dir)

    print("6/6 - Salvando artefatos...")
    model.save(model_dir / "model.keras")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    metadata = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_length": args.sequence_length,
        "train_size": args.train_size,
        "epochs_requested": args.epochs,
        "epochs_executed": len(history.history["loss"]),
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
        "metrics": metrics,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\nTreino finalizado com sucesso.")
    print(f"Artefatos salvos em: {model_dir.resolve()}")
    print(f"Relatórios salvos em: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
