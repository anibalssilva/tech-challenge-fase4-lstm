#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR="${MODEL_DIR:-models}"
export PORT="${PORT:-8000}"

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
