from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    symbol: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    sequence_length: int = 60
    train_size: float = 0.8
    epochs: int = 40
    batch_size: int = 32
    lstm_units: int = 128
    dropout: float = 0.2
    learning_rate: float = 0.001
    model_dir: str = "models"
    report_dir: str = "reports"
