#!/bin/bash
# Start FastAPI (port 8000) in background, then Streamlit (port 8501) in foreground

echo "[start.sh] Starting FastAPI on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 &

echo "[start.sh] Starting Streamlit on port 8501..."
exec streamlit run app/Home.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base=dark
