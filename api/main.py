"""
api/main.py -- FastAPI REST API for the DQ Engine
==================================================
Provides programmatic access to the same validation, profiling,
anomaly detection, and AI investigation capabilities as the
Streamlit UI.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs  (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import io
import json
import time
import logging
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import require_api_key, ensure_api_keys_table

from api.schemas import (
    HealthResponse,
    ValidateResponse,
    ProfileResponse,
    AnomalyResponse,
    RulesResponse,
    BatchListResponse,
    BatchDetailResponse,
    S3ValidateRequest,
    InvestigateRequest,
    InvestigateResponse,
)

logger = logging.getLogger("dq_api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DQ Investigator API",
    description=(
        "Enterprise data quality engine. Validate, profile, detect anomalies, "
        "and investigate data quality issues programmatically."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Create api_keys table on startup if it doesn't exist."""
    ensure_api_keys_table()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _numpy_serializer(obj):
    """Handle numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_json(data: Any) -> Any:
    """Round-trip through JSON to clean numpy types."""
    return json.loads(json.dumps(data, default=_numpy_serializer))


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame."""
    content = await file.read()
    filename = file.filename or "upload.csv"
    ext = filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "csv":
            return pd.read_csv(io.BytesIO(content))
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(io.BytesIO(content))
        elif ext == "parquet":
            return pd.read_parquet(io.BytesIO(content))
        elif ext == "json":
            return pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: .{ext}. Use CSV, Excel, Parquet, or JSON.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


def _get_config() -> Dict[str, Any]:
    """Load engine config."""
    from core.engine import load_config
    return load_config()


def _issues_to_records(issues_df: pd.DataFrame) -> List[Dict]:
    """Convert issues DataFrame to JSON-safe list of dicts."""
    if issues_df.empty:
        return []
    df = issues_df.copy()
    # Replace NaN with None for JSON
    df = df.where(df.notna(), None)
    records = df.to_dict(orient="records")
    return _safe_json(records)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/validate", tags=["Validation"])
async def validate(
    file: UploadFile = File(..., description="Data file (CSV, Excel, Parquet, JSON)"),
    rules_file: Optional[UploadFile] = File(None, description="Optional rules JSON file"),
    reference_file: Optional[UploadFile] = File(None, description="Optional reference/clean data file"),
    pass_threshold: float = Query(85.0, description="Quality score threshold for pass/fail"),
    enable_ml_anomaly: bool = Query(False, description="Enable ML-based anomaly detection"),
    client: str = Security(require_api_key),
):
    """
    Run full validation pipeline on an uploaded file.

    Returns quality scores, all issues found, and audit trail.
    """
    from core.engine import run_pipeline

    t0 = time.time()
    df = await _read_upload(file)

    # Optional rules
    rules = None
    if rules_file:
        rules_content = await rules_file.read()
        try:
            rules = json.loads(rules_content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid rules JSON file")

    # Optional reference data
    df_ref = None
    if reference_file:
        df_ref = await _read_upload(reference_file)

    # Build config
    config = _get_config()
    config.setdefault("validation", {}).setdefault("scoring", {})["pass_threshold"] = pass_threshold
    config["validation"]["enable_ml_anomaly"] = enable_ml_anomaly

    # Run pipeline
    result = run_pipeline(df=df, config=config, rules=rules, df_ref=df_ref)

    response = {
        "batch_id": result["batch_id"],
        "timestamp": result["timestamp"],
        "rows_processed": result["rows_processed"],
        "columns": result["columns"],
        "quality_scores": _safe_json(
            {k: v for k, v in result["quality_scores"].items() if k != "missing_by_column"}
        ),
        "overall_score": float(result["overall_score"]),
        "pass": result["pass"],
        "pass_threshold": pass_threshold,
        "issues_count": result["issues_count"],
        "issues": _issues_to_records(result["issues"]),
        "audit_trail": _safe_json(result["audit_trail"]),
        "processing_time_ms": round((time.time() - t0) * 1000),
    }
    return JSONResponse(content=response)


@app.post("/api/v1/profile", tags=["Profiling"])
async def profile(
    file: UploadFile = File(..., description="Data file to profile"),
    client: str = Security(require_api_key),
):
    """
    Profile a dataset: column types, distributions, nulls, outliers, correlations.
    """
    from core.data_profiler import profile_dataset

    t0 = time.time()
    df = await _read_upload(file)
    result = profile_dataset(df)

    return JSONResponse(content={
        "rows": len(df),
        "columns": len(df.columns),
        "profile": _safe_json(result),
        "processing_time_ms": round((time.time() - t0) * 1000),
    })


@app.post("/api/v1/anomalies", tags=["Anomaly Detection"])
async def detect_anomalies(
    file: UploadFile = File(..., description="Data file"),
    reference_file: Optional[UploadFile] = File(None, description="Optional reference data"),
    use_ml: bool = Query(True, description="Use ML models for anomaly detection"),
    client: str = Security(require_api_key),
):
    """
    Run anomaly detection on a dataset using statistical and ML methods.
    """
    from core.anomaly import ml_anomaly_report

    t0 = time.time()
    df = await _read_upload(file)
    df_ref = None
    if reference_file:
        df_ref = await _read_upload(reference_file)

    report = ml_anomaly_report(
        df_unclean=df,
        df_ref=df_ref,
        use_reference=(df_ref is not None),
        models=["zscore", "iqr", "isolation_forest"] if use_ml else ["zscore", "iqr"],
    )

    return JSONResponse(content={
        "rows_analyzed": len(df),
        "anomalies_found": len(report),
        "anomalies": _issues_to_records(report),
        "processing_time_ms": round((time.time() - t0) * 1000),
    })


@app.post("/api/v1/rules/generate", tags=["Rules"])
async def generate_rules(
    file: UploadFile = File(..., description="Data file to infer rules from"),
    client: str = Security(require_api_key),
):
    """
    Auto-generate validation rules from data patterns.
    """
    from core.validator.discover import infer_rules_from_unclean

    t0 = time.time()
    df = await _read_upload(file)
    rules = infer_rules_from_unclean(df)

    return JSONResponse(content={
        "columns_analyzed": len(df.columns),
        "rules_generated": len(rules),
        "rules": _safe_json(rules),
        "processing_time_ms": round((time.time() - t0) * 1000),
    })


@app.post("/api/v1/s3/validate", tags=["S3 Integration"])
async def validate_from_s3(request: S3ValidateRequest, client: str = Security(require_api_key)):
    """
    Validate a file directly from an S3 bucket.
    """
    from core.storage.s3 import read_from_s3
    from core.engine import run_pipeline

    t0 = time.time()

    try:
        df = read_from_s3(request.bucket, request.key, request.region)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read from S3: {str(e)}")

    df_ref = None
    if request.reference_key:
        try:
            df_ref = read_from_s3(request.bucket, request.reference_key, request.region)
        except Exception as e:
            logger.warning(f"Failed to load reference data from S3: {e}")

    rules = None
    if request.rules_key:
        try:
            import json as _json
            import boto3 as _boto3
            s3c = _boto3.client("s3", region_name=request.region)
            obj = s3c.get_object(Bucket=request.bucket, Key=request.rules_key)
            rules = _json.loads(obj["Body"].read().decode("utf-8"))
            logger.info(f"Loaded rules from s3://{request.bucket}/{request.rules_key} ({len(rules)} columns)")
        except Exception as e:
            logger.warning(f"Failed to load rules from S3: {e} — falling back to auto-infer")

    config = _get_config()
    config.setdefault("validation", {}).setdefault("scoring", {})["pass_threshold"] = request.pass_threshold

    result = run_pipeline(df=df, config=config, df_ref=df_ref, rules=rules)

    # Optionally save results back to S3
    output_paths = {}
    if request.output_bucket:
        from core.storage.s3 import write_csv_to_s3, write_json_to_s3
        batch_id = result["batch_id"]
        prefix = request.output_prefix or "reports"

        if not result["issues"].empty:
            output_paths["issues"] = write_csv_to_s3(
                result["issues"], request.output_bucket,
                f"{prefix}/{batch_id}_issues.csv", request.region,
            )
        report_data = {
            "batch_id": result["batch_id"],
            "timestamp": result["timestamp"],
            "rows_processed": result["rows_processed"],
            "overall_score": float(result["overall_score"]),
            "pass": result["pass"],
            "issues_count": result["issues_count"],
        }
        output_paths["report"] = write_json_to_s3(
            report_data, request.output_bucket,
            f"{prefix}/{batch_id}_report.json", request.region,
        )

    response = {
        "batch_id": result["batch_id"],
        "timestamp": result["timestamp"],
        "source": f"s3://{request.bucket}/{request.key}",
        "rows_processed": result["rows_processed"],
        "quality_scores": _safe_json(
            {k: v for k, v in result["quality_scores"].items() if k != "missing_by_column"}
        ),
        "overall_score": float(result["overall_score"]),
        "pass": result["pass"],
        "issues_count": result["issues_count"],
        "issues": _issues_to_records(result["issues"]),
        "output_paths": output_paths,
        "audit_trail": _safe_json(result["audit_trail"]),
        "processing_time_ms": round((time.time() - t0) * 1000),
    }
    return JSONResponse(content=response)


@app.post("/api/v1/investigate", tags=["Investigation"])
async def investigate(request: InvestigateRequest, client: str = Security(require_api_key)):
    """
    Run a deterministic investigation workflow over a dataset.

    Executes a fixed sequence: validate -> profile -> identify fixable issues
    -> summarise, returning a quality summary with a pass/fail recommendation.
    The steps and order are fixed and reproducible (this is not an autonomous
    LLM agent; for conversational, multi-step agent interaction use the
    Streamlit assistant).
    """
    from core.storage.s3 import read_from_s3

    t0 = time.time()

    # Load data from S3
    try:
        df = read_from_s3(request.bucket, request.key, request.region)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read from S3: {str(e)}")

    # Run the full pipeline first
    from core.engine import run_pipeline
    config = _get_config()
    result = run_pipeline(df=df, config=config)

    # Profile
    from core.data_profiler import profile_dataset
    profile = profile_dataset(df)

    # Build investigation summary
    issues_df = result["issues"]
    top_columns = []
    if not issues_df.empty and "column" in issues_df.columns:
        col_counts = issues_df["column"].value_counts().head(5)
        top_columns = [{"column": col, "issues": int(count)} for col, count in col_counts.items()]

    severity_breakdown = {}
    if not issues_df.empty and "severity" in issues_df.columns:
        severity_breakdown = _safe_json(issues_df["severity"].value_counts().to_dict())

    investigation = {
        "batch_id": result["batch_id"],
        "timestamp": result["timestamp"],
        "source": f"s3://{request.bucket}/{request.key}",
        "rows_processed": result["rows_processed"],
        "overall_score": float(result["overall_score"]),
        "pass": result["pass"],
        "issues_count": result["issues_count"],
        "top_problem_columns": top_columns,
        "severity_breakdown": severity_breakdown,
        "profile_summary": {
            "total_nulls": _safe_json(profile.get("dataset_summary", {}).get("total_nulls", 0)),
            "duplicate_rows": _safe_json(profile.get("dataset_summary", {}).get("duplicate_rows", 0)),
        },
        "recommendation": "PASS - Data quality meets threshold" if result["pass"]
            else "FAIL - Data quality below threshold, review top problem columns",
        "audit_trail": _safe_json(result["audit_trail"]),
        "processing_time_ms": round((time.time() - t0) * 1000),
    }
    return JSONResponse(content=investigation)


@app.get("/api/v1/batches", tags=["Batch History"])
async def list_batches(
    limit: int = Query(20, description="Max number of batches to return"),
    client: str = Security(require_api_key),
):
    """
    List past validation batch runs from the database.
    """
    try:
        from core.storage.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM batch_runs ORDER BY timestamp DESC LIMIT :limit"),
                {"limit": limit},
            ).fetchall()

            if not rows:
                return JSONResponse(content={"batches": [], "count": 0})

            columns = rows[0]._mapping.keys() if rows else []
            batches = [dict(zip(columns, row)) for row in rows]
            return JSONResponse(content={
                "batches": _safe_json(batches),
                "count": len(batches),
            })
    except Exception as e:
        logger.error(f"Batch history retrieval failed: {e}")
        return JSONResponse(content={
            "batches": [],
            "count": 0,
            "note": f"Could not retrieve batch history: {str(e)}",
        })


@app.get("/api/v1/batches/{batch_id}", tags=["Batch History"])
async def get_batch(batch_id: str, client: str = Security(require_api_key)):
    """
    Get detailed results for a specific batch run.
    """
    try:
        from core.storage.database import get_engine
        from sqlalchemy import text

        engine = get_engine()
        with engine.connect() as conn:
            # Batch metadata
            batch = conn.execute(
                text("SELECT * FROM batch_runs WHERE batch_id = :bid"),
                {"bid": batch_id},
            ).fetchone()

            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

            batch_dict = dict(batch._mapping)

            # Issues for this batch
            issues = conn.execute(
                text("SELECT * FROM issues WHERE batch_id = :bid"),
                {"bid": batch_id},
            ).fetchall()

            issues_list = []
            if issues:
                cols = issues[0]._mapping.keys()
                issues_list = [dict(zip(cols, row)) for row in issues]

            # Audit trail
            audit = conn.execute(
                text("SELECT * FROM audit_trail WHERE batch_id = :bid ORDER BY id"),
                {"bid": batch_id},
            ).fetchall()

            audit_list = []
            if audit:
                cols = audit[0]._mapping.keys()
                audit_list = [dict(zip(cols, row)) for row in audit]

            return JSONResponse(content={
                "batch": _safe_json(batch_dict),
                "issues": _safe_json(issues_list),
                "issues_count": len(issues_list),
                "audit_trail": _safe_json(audit_list),
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
