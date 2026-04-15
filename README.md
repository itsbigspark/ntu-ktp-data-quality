# AI Powered DQ Investigator

Enterprise-grade data quality platform for banking and financial services. Upload data, validate it against 50+ rules, detect anomalies with ML, and get AI-powered insights that tell you exactly what's wrong and what to fix first.

## Quick Start

### Option 1: Shell Script (Recommended)

```bash
git clone <repo-url>
cd ntu_ktp_data_quality_APP_Rayane
./start.sh
```

Opens at `http://localhost:8501`. Login: `admin` / `admin123`.

### Option 2: Docker

```bash
git clone <repo-url>
cd ntu_ktp_data_quality_APP_Rayane
docker-compose up
```

### Option 3: Manual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
```

## What It Does

1. **Load Data** -- Upload CSV, Parquet, or Excel files
2. **Rules Engine** -- Auto-discovers validation rules from data patterns
3. **Validate** -- Rule-based, ML anomaly, corpus matching, and BERT-enhanced validation
4. **AI Enrichment** -- 5 targeted AI calls that generate:
   - Smart validation rules with confidence scores
   - Cross-column logical contradiction detection
   - Plain-English anomaly explanations with business impact
   - Priority-ranked triage with effort estimates
   - Executive summary for management
5. **Command Center** -- Real-time monitoring dashboard with quality trends, AI insights, and batch history
6. **Cleaning** -- Apply corrections, standardise formats, fix issues
7. **Deduplication** -- Exact and fuzzy duplicate detection with clustering
8. **Corpus Manager** -- Load and manage reference data corpora for standardisation
9. **Pipeline Manager** -- Build and execute multi-step data cleaning pipelines
10. **Entity Resolution** -- Match records across multiple datasets using vector similarity

## Architecture

```
app/                    UI layer (Streamlit multi-page app)
  Home.py               Entry point + login
  shared/               Auth, state, theme (shared across pages)
  pages/                One file per feature page

dq_engine/              Core engine (zero UI dependencies)
  orchestrators/
    validation.py       ValidationOrchestrator
    rules.py            RulesWorkflow
    ai_enrichment.py    AIEnrichment + LLM providers

core/                   Processing modules
  validator/            Rule-based validation
  anomaly.py            ML anomaly detection
  corpus_validation.py  Corpus matching
  enhanced_validator.py BERT-enhanced validation
  storage/database.py   SQLite/PostgreSQL persistence
```

The engine (`dq_engine/`) has zero UI dependencies. It can be wrapped in FastAPI, called from a Lambda function, or used as a Python library.

## AI Providers

| Provider | Setup | Data Privacy |
|----------|-------|-------------|
| Ollama (default) | Install [Ollama](https://ollama.ai), run `ollama pull phi3:mini` | Fully offline, data never leaves your machine |
| Anthropic (Claude) | Set API key in Settings page | Only column statistics sent, never raw data |
| AWS Bedrock | Configure AWS credentials | Enterprise-grade, runs in your VPC |

AI enrichment is optional. The validation engine works without it.

## Database

Results are stored in SQLite at `./output/dq_investigator.db`. Tables:

- `batch_runs` -- Pipeline execution metadata
- `issues` -- Every data quality issue found
- `audit_trail` -- Step-by-step pipeline timing
- `ai_smart_rules` -- AI-generated validation rules
- `ai_cross_column` -- Cross-column contradiction analysis
- `ai_explanations` -- Plain-English anomaly explanations
- `ai_triage` -- Priority-ranked action plan
- `ai_executive_summary` -- Management assessment

For enterprise deployment, configure PostgreSQL via environment variables.

## Configuration

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...    # For Claude AI enrichment
AWS_REGION=eu-west-2            # For Bedrock/S3
DATABASE_URL=postgresql://...    # For PostgreSQL (optional)
```

### Default Credentials

| Username | Password |
|----------|----------|
| admin | admin123 |
| analyst | dq2026 |
| demo | demo |

## Requirements

- Python 3.9+
- 4GB RAM minimum (8GB recommended for ML features)
- Ollama (optional, for offline AI)

## Project Structure

```
ntu_ktp_data_quality_APP_Rayane/
  app/                    Multi-page Streamlit application
  dq_engine/              Core engine package
  core/                   Processing modules
  output/                 Database and reports
  TEST2_DATA/             Demo dataset (500 rows, all error types)
  start.sh                One-command startup script
  Dockerfile              Docker containerisation
  docker-compose.yml      Docker Compose config
  requirements.txt        Python dependencies
  dashboard.py            Standalone Matrix dashboard (port 8503)
  data_quality_app.py     Legacy monolith (still functional)
```

## License

Proprietary. NTU KTP Project.
