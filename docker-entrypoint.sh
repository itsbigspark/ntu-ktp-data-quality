#!/bin/bash
# ============================================================================
# AI Powered DQ Investigator -- Docker Entrypoint
# ============================================================================
# Modes:
#   App mode (default):  docker run -p 8501:8501 dq-investigator
#   API mode:            docker run -p 8000:8000 dq-investigator --mode api
#   Both:                docker run -p 8501:8501 -p 8000:8000 dq-investigator --mode both
#   Pipeline mode:       docker run dq-investigator --input /data/file.csv
# ============================================================================

set -e

if [ "$1" = "--mode" ]; then
    case "$2" in
        app)
            echo "Starting DQ Investigator (Streamlit UI)..."
            exec streamlit run app/Home.py \
                --server.headless true \
                --server.port 8501 \
                --browser.gatherUsageStats false \
                --theme.base dark
            ;;
        api)
            echo "Starting DQ Investigator (REST API)..."
            exec uvicorn api.main:app --host 0.0.0.0 --port 8000
            ;;
        both)
            echo "Starting DQ Investigator (Streamlit + REST API)..."
            uvicorn api.main:app --host 0.0.0.0 --port 8000 &
            exec streamlit run app/Home.py \
                --server.headless true \
                --server.port 8501 \
                --browser.gatherUsageStats false \
                --theme.base dark
            ;;
        *)
            echo "Unknown mode: $2. Use: app, api, or both"
            exit 1
            ;;
    esac
elif [ "$1" = "--input" ] || [ "$1" = "-i" ]; then
    echo "Running DQ Investigator (Pipeline)..."
    exec python -m core.engine "$@"
else
    # Default: Streamlit app
    echo "Starting DQ Investigator (Streamlit UI)..."
    exec streamlit run app/Home.py \
        --server.headless true \
        --server.port 8501 \
        --browser.gatherUsageStats false \
        --theme.base dark
fi
