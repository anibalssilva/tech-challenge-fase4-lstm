#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Health check"
curl -s "$BASE_URL/health" | python -m json.tool

echo ""
echo "Model info"
curl -s "$BASE_URL/model-info" | python -m json.tool

echo ""
echo "Prediction"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json | python -m json.tool

echo ""
echo "Metrics"
curl -s "$BASE_URL/metrics" | python -m json.tool
