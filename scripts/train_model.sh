#!/usr/bin/env bash
set -euo pipefail

SYMBOL="${SYMBOL:-DIS}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-2024-07-20}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-60}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"

python -m src.training.train \
  --symbol "$SYMBOL" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --model-dir models \
  --report-dir reports
