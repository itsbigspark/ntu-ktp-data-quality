# DataQualify

Explainable, weakly-supervised data quality for tabular financial data.

DataQualify detects, explains, and suggests corrections for errors in tabular
datasets (KYC records, transactions, claims) by combining explicit domain rules,
a clean reference dataset, and unsupervised statistical learning — **without
requiring labelled training data**. Every decision is deterministic and carries a
human-readable rationale (the rule that fired, the expected value, a regulatory
citation), so the output is auditable for regulated use.

It ships with **four front doors** over one shared engine — a Python **library**, a
**CLI**, a **REST API**, and an interactive **Streamlit app** — plus a reference
**event-driven batch pipeline** on AWS.

Developed through a Knowledge Transfer Partnership between **bigspark Ltd** and
**Nottingham Trent University**, funded by Innovate UK.

---

## Install

```bash
pip install -e .                 # core engine only (pandas, numpy, scikit-learn)
pip install -e ".[api]"          # + REST API (FastAPI)
pip install -e ".[app]"          # + Streamlit application
pip install -e ".[ml]"           # + embeddings / vector search (torch, chromadb)
pip install -e ".[aws]"          # + S3 / batch pipeline (boto3)
pip install -e ".[all]"          # everything
```

The **core** install is deliberately light — no Streamlit, torch, chromadb, boto3
or redis. Install only the extras you need.

---

## 1. As a library

```python
import pandas as pd
import dataqualify as dq

df = pd.read_csv("customers.csv")

# Infer rules from the data, validate, return a cell-level issue report.
issues = dq.validate(df)

# Recommended: infer rules from a clean reference so bounds/allowed-values
# are not contaminated by the errors in df.
issues = dq.validate(df, reference=pd.read_csv("clean_reference.csv"))

# Full pipeline: six quality-dimension scores, corrected copy, audit trail.
result = dq.run_pipeline(df, dq.load_config())
print(result["overall_score"], result["issues_count"])
```

`issues` is a DataFrame with one row per finding: `row_id`, `column`, `issue`,
`detail`, `severity`, `value`, `expected`, `rule`, `description`.

## 2. As a CLI (pipeline step)

```bash
dataqualify validate data.csv
dataqualify validate data.csv --reference clean.csv --out issues.csv
dataqualify validate data.csv --rules rules.json --format json
```

Exits non-zero when issues are found (use `--no-fail` to override), so it can gate
a downstream job in Airflow, cron, or CI.

## 3. As a REST API

```bash
pip install -e ".[api]"
uvicorn api.main:app --port 8000        # interactive docs at /docs
```

Key endpoints (all require an `X-API-Key` header):

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/validate` | validate an uploaded file → scores, issues, audit trail |
| `POST /api/v1/s3/validate` | read from S3, validate, write the report back to S3 |
| `POST /api/v1/profile` | column-level data profiling |
| `POST /api/v1/rules/generate` | infer validation rules from a dataset |
| `GET  /api/v1/batches` | batch history |

## 4. As an interactive app

```bash
pip install -e ".[app]"
streamlit run app/Home.py                # opens at http://localhost:8501
```

Default logins (development): `admin` / `admin123`, `analyst` / `dq2026`.

---

## What it checks

Six quality dimensions — **Completeness, Uniqueness, Consistency, Validity,
Accuracy, Timeliness** — combined into an overall score against a configurable
pass threshold (default 85).

The engine runs: rule consolidation/inference → hybrid error detection
(rule-based + Levenshtein typo layer) → unsupervised anomaly scoring
(Isolation Forest + Local Outlier Factor, GMM-antimode threshold) → correction
suggestion (abstains rather than guessing) → quality scoring → an explainability
audit trail on every finding. Optional layers add regulatory citations (BCBS 239,
UK GDPR/DPA, FCA), entity resolution, and deduplication.

See `architecture.md` for the full pipeline and
`tests/ENGINE_FIXES_AND_TESTS.md` for precision behaviour and known limitations.

---

## Layout

```
dataqualify/        Public API package (library + CLI entry point)
core/               The engine: validation, anomaly, correction, corpus, RAG
  validator/          rule inference (discover.py) + validation (validate.py)
api/                FastAPI service (thin wrapper over the engine)
dq_engine/          Orchestrators (validation, rules, AI enrichment)
app/                Streamlit multi-page application
infra/              AWS SAM template, Lambda handlers, Step Functions
tests/              Engine + quality regression tests
```

The engine has no hard dependency on Streamlit, AWS, or a language model. LLMs are
used only for explanation, narration, and orchestration — never for the
data-quality decisions, which stay deterministic and auditable.

---

## Batch pipeline (AWS)

A reference event-driven deployment: a file dropped in an S3 inbox triggers
EventBridge → Step Functions → Lambda (thin orchestration) → the ECS engine
service (heavy compute) → a scored report back to S3 and a record in RDS. The
`[aws]` extra provides the S3 read/write-back helpers (`core/s3_writeback.py`).
The storage layer is being generalised behind an adapter so the same engine runs
against GCS / Azure Blob / local disk.

Optional AI enrichment (rule suggestions, plain-English explanations, executive
summaries) runs via local Ollama (default, fully offline), Anthropic Claude, or
AWS Bedrock. It is optional; the validation engine works without it.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

`tests/test_engine_quality.py` holds the precision, presence, and categorical
regression suite (see `tests/ENGINE_FIXES_AND_TESTS.md`). Requirements: Python
3.10+, 4 GB RAM (8 GB for ML features).

---

## License

Proprietary — a Knowledge Transfer Partnership output co-owned by bigspark Ltd,
Nottingham Trent University, and Innovate UK. Redistribution terms are subject to
the partners' agreement.
