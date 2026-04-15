"""
Lambda: run_dq_engine.py — Runs the DQ validation pipeline
============================================================
Called by Step Functions after the trigger.
Reads data from S3, runs the full DQ engine, writes results.

For datasets under ~50MB / 100K rows, Lambda is sufficient (15 min timeout).
For larger datasets, swap this for an ECS Fargate task.
"""

import os
import io
import json
import time
import logging
from datetime import datetime, timezone

import boto3
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", "")
CONFIG_KEY = os.environ.get("CONFIG_KEY", "config/config.yaml")
DB_HOST = os.environ.get("DQ_DB_HOST", "")

s3 = boto3.client("s3")


def handler(event, context):
    """
    Step Functions passes:
    {
        "batch_id": "batch_xxx",
        "source": {"bucket": "...", "key": "...", "format": "csv"},
        "timestamp": "..."
    }
    """
    logger.info(f"DQ Engine starting: {json.dumps(event)}")
    t_start = time.time()

    batch_id = event["batch_id"]
    source = event["source"]
    bucket = source["bucket"]
    key = source["key"]

    # ── 1. Load data from S3 ───────────────────────────────────────────────
    logger.info(f"Reading s3://{bucket}/{key}")
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()

    ext = source.get("format", "csv")
    if ext == "csv":
        df = pd.read_csv(io.BytesIO(body))
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(body))
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(body))
    elif ext == "json":
        df = pd.read_json(io.BytesIO(body))
    else:
        raise ValueError(f"Unsupported format: {ext}")

    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # ── 2. Load config (from S3 or use defaults) ──────────────────────────
    config = _load_config()

    # ── 3. Run the DQ pipeline ─────────────────────────────────────────────
    # Import the engine (bundled in the Lambda layer or container)
    from core.engine import run_pipeline
    result = run_pipeline(df=df, config=config, batch_id=batch_id)

    logger.info(f"Pipeline complete: {result['issues_count']} issues, score {result['overall_score']}%")

    # ── 4. Write results to S3 ─────────────────────────────────────────────
    output_paths = {}

    if OUTPUT_BUCKET:
        # Issue log
        if not result["issues"].empty:
            issue_key = f"reports/{batch_id}_issues.csv"
            _write_csv(result["issues"], OUTPUT_BUCKET, issue_key)
            output_paths["issue_log"] = f"s3://{OUTPUT_BUCKET}/{issue_key}"

        # Corrected dataset
        corrected_key = f"corrected/{batch_id}_corrected.csv"
        _write_csv(result["corrected_df"], OUTPUT_BUCKET, corrected_key)
        output_paths["corrected_dataset"] = f"s3://{OUTPUT_BUCKET}/{corrected_key}"

        # Quality report
        report_data = {
            "batch_id": result["batch_id"],
            "timestamp": result["timestamp"],
            "source_file": f"s3://{bucket}/{key}",
            "rows_processed": result["rows_processed"],
            "columns": result["columns"],
            "quality_scores": {k: v for k, v in result["quality_scores"].items() if k != "missing_by_column"},
            "overall_score": float(result["overall_score"]),
            "pass": bool(result["pass"]),
            "pass_threshold": float(result["pass_threshold"]),
            "issues_count": int(result["issues_count"]),
            "llm_calls": int(result.get("llm_calls", 0)),
            "data_sent_to_llm": bool(result.get("data_sent_to_llm", False)),
            "audit_trail": result.get("audit_trail", []),
        }
        report_key = f"reports/{batch_id}_report.json"
        _write_json(report_data, OUTPUT_BUCKET, report_key)
        output_paths["quality_report"] = f"s3://{OUTPUT_BUCKET}/{report_key}"

    # ── 5. Write to database (if configured) ───────────────────────────────
    if DB_HOST:
        try:
            from core.storage.database import save_results_to_db
            db_config = {
                "output": {
                    "database": {
                        "engine": "postgresql",
                        "host": DB_HOST,
                        "port": int(os.environ.get("DQ_DB_PORT", 5432)),
                        "name": os.environ.get("DQ_DB_NAME", "dq_investigator"),
                        "user": os.environ.get("DQ_DB_USER", ""),
                        "password": os.environ.get("DQ_DB_PASSWORD", ""),
                    }
                }
            }
            save_results_to_db(result, db_config, source_file=f"s3://{bucket}/{key}")
            output_paths["database"] = "written"
            logger.info("Database write complete")
        except Exception as e:
            logger.warning(f"Database write failed: {e}")
            output_paths["database"] = f"failed: {str(e)}"

    duration_ms = round((time.time() - t_start) * 1000)
    logger.info(f"Total duration: {duration_ms}ms")

    # ── Return to Step Functions ───────────────────────────────────────────
    return {
        "batch_id": batch_id,
        "rows_processed": result["rows_processed"],
        "issues_count": result["issues_count"],
        "overall_score": float(result["overall_score"]),
        "pass": bool(result["pass"]),
        "pass_threshold": float(result["pass_threshold"]),
        "quality_scores": {k: float(v) for k, v in result["quality_scores"].items() if k != "missing_by_column"},
        "audit_trail": result.get("audit_trail", []),
        "output_paths": output_paths,
        "duration_ms": duration_ms,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_config():
    """Load config from S3 or return defaults."""
    if CONFIG_BUCKET and CONFIG_KEY:
        try:
            import yaml
            response = s3.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_KEY)
            return yaml.safe_load(response["Body"].read())
        except Exception as e:
            logger.warning(f"Could not load config from S3: {e}, using defaults")

    # Defaults for Lambda execution
    return {
        "app": {"name": "AI Powered DQ Investigator", "mode": "enterprise"},
        "validation": {
            "rule_source": "infer",
            "enable_typo_detection": True,
            "enable_anomaly_detection": True,
            "enable_ml_anomaly": False,
            "corpus": {"enabled": False},
            "scoring": {"pass_threshold": 85.0},
        },
        "bert": {"enabled": False},
    }


def _write_csv(df, bucket, key):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info(f"Wrote s3://{bucket}/{key}")


def _write_json(data, bucket, key):
    import numpy as np

    def serializer(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    body = json.dumps(data, indent=2, default=serializer).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(f"Wrote s3://{bucket}/{key}")
