from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    sequence_length: int,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=(sequence_length, 1)),
            LSTM(units=lstm_units, return_sequences=True),
            Dropout(dropout),
            LSTM(units=max(lstm_units // 2, 16), return_sequences=False),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
