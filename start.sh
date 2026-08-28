#!/bin/bash
set -e

PORT="${PORT:-8501}"

echo "Starting Streamlit on port $PORT..."
exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address=0.0.0.0 \
    --server.headless=true
