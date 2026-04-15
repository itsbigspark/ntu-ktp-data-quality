#!/bin/bash
# ============================================================================
# AI Powered DQ Investigator — Docker Entrypoint
# ============================================================================
# Usage:
#   Pipeline mode:  docker run dq-investigator --input /data/file.csv
#   App mode:       docker run -p 8501:8501 dq-investigator --mode app
# ============================================================================

set -e

if [ "$1" = "--mode" ] && [ "$2" = "app" ]; then
    echo "Starting AI Powered DQ Investigator (Streamlit)..."
    exec streamlit run data_quality_app.py \
        --server.headless true \
        --server.port 8501 \
        --browser.gatherUsageStats false
else
    echo "Running AI Powered DQ Investigator (Pipeline)..."
    exec python -m core.engine "$@"
fi
