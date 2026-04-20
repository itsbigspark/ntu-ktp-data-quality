# ============================================================================
# AI Powered DQ Investigator -- Docker Image
# ============================================================================
# Usage:
#   docker-compose up        (recommended)
#   docker build -t dq-investigator . && docker run -p 8501:8501 dq-investigator
# ============================================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ ./core/
COPY dq_engine/ ./dq_engine/
COPY app/ ./app/
COPY api/ ./api/
COPY config.yaml* ./
COPY TEST2_DATA/ ./TEST2_DATA/
COPY configs/ ./configs/
COPY assets/ ./assets/
COPY lib/ ./lib/

# Create output directory
RUN mkdir -p output

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the multi-page app
CMD ["streamlit", "run", "app/Home.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--theme.base=dark"]
