#!/bin/bash
# =================================================================
# AI Powered DQ Investigator - One-Command Startup
# =================================================================
# Usage: ./start.sh
# =================================================================

set -e

echo ""
echo "  ██████╗  ██████╗    ██╗███╗   ██╗██╗   ██╗███████╗███████╗████████╗██╗ ██████╗  █████╗ ████████╗ ██████╗ ██████╗"
echo "  ██╔══██╗██╔═══██╗   ██║████╗  ██║██║   ██║██╔════╝██╔════╝╚══██╔══╝██║██╔════╝ ██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗"
echo "  ██║  ██║██║   ██║   ██║██╔██╗ ██║██║   ██║█████╗  ███████╗   ██║   ██║██║  ███╗███████║   ██║   ██║   ██║██████╔╝"
echo "  ██║  ██║██║▄▄ ██║   ██║██║╚██╗██║╚██╗ ██╔╝██╔══╝  ╚════██║   ██║   ██║██║   ██║██╔══██║   ██║   ██║   ██║██╔══██╗"
echo "  ██████╔╝╚██████╔╝   ██║██║ ╚████║ ╚████╔╝ ███████╗███████║   ██║   ██║╚██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║"
echo "  ╚═════╝  ╚══▀▀═╝    ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝"
echo ""
echo "  AI Powered Data Quality Investigator"
echo "  ======================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# -------------------------------------------------------------------
# Step 1: Check Python
# -------------------------------------------------------------------
echo "[1/4] Checking Python..."

if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.9+ from https://python.org"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "       Found Python $PY_VERSION"

# -------------------------------------------------------------------
# Step 2: Create virtual environment
# -------------------------------------------------------------------
echo "[2/4] Setting up virtual environment..."

if [ ! -d ".venv" ]; then
    echo "       Creating .venv..."
    $PYTHON -m venv .venv
else
    echo "       .venv already exists"
fi

# Activate
source .venv/bin/activate

# -------------------------------------------------------------------
# Step 3: Install dependencies
# -------------------------------------------------------------------
echo "[3/4] Installing dependencies..."

$PYTHON -m pip install --quiet --upgrade pip

if [ -f "requirements.txt" ]; then
    $PYTHON -m pip install --quiet -r requirements.txt
else
    echo "       WARNING: requirements.txt not found. Installing core packages..."
    $PYTHON -m pip install --quiet streamlit pandas numpy plotly scikit-learn sqlalchemy anthropic requests
fi

# Create output directory if needed
mkdir -p output

# -------------------------------------------------------------------
# Step 4: Launch the app
# -------------------------------------------------------------------
echo "[4/4] Launching DQ Investigator..."
echo ""
echo "  ================================================="
echo "  App URL: http://localhost:8501"
echo "  ================================================="
echo ""
echo "  Default login: admin / admin123"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

streamlit run app/Home.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark
