"""
core/storage/s3.py  --  AWS S3 Read/Write for the DQ Pipeline
=============================================================
Handles:
  - Reading incoming data files from S3 inbox
  - Writing results (reports, corrected data, logs) to S3 output
  - Archiving processed files
  - Listing pending files in inbox

Requires: boto3 (pip install boto3)
All S3 operations use the default credential chain:
  env vars > ~/.aws/credentials > IAM role (EC2/ECS/Lambda)
"""

import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger("dq_engine.s3")


def _get_client(region: str = "eu-west-2"):
    """Lazy boto3 client — only imported when S3 is actually used."""
    import boto3
    return boto3.client("s3", region_name=region)


# ---------------------------------------------------------------------------
# READ — Ingestion from S3
# ---------------------------------------------------------------------------

def list_inbox_files(
    bucket: str,
    prefix: str = "incoming/",
    region: str = "eu-west-2",
    supported_formats: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List data files waiting in the S3 inbox.
    Returns list of dicts: {key, size_kb, last_modified}
    """
    if supported_formats is None:
        supported_formats = ["csv", "xlsx", "parquet", "json"]

    s3 = _get_client(region)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    files = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
        if ext in supported_formats:
            files.append({
                "key": key,
                "size_kb": round(obj["Size"] / 1024, 1),
                "last_modified": obj["LastModified"].isoformat(),
            })

    logger.info(f"Found {len(files)} files in s3://{bucket}/{prefix}")
    return files


def read_from_s3(
    bucket: str,
    key: str,
    region: str = "eu-west-2",
) -> pd.DataFrame:
    """
    Read a data file from S3 into a DataFrame.
    Supports CSV, XLSX, Parquet, JSON.
    """
    s3 = _get_client(region)
    logger.info(f"Reading s3://{bucket}/{key}")

    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()
    size_kb = len(body) / 1024

    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(body))
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(body))
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(body))
    elif ext == "json":
        df = pd.read_json(io.BytesIO(body))
    else:
        raise ValueError(f"Unsupported S3 file format: {ext} (key: {key})")

    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns ({size_kb:.1f} KB)")
    return df


def archive_file(
    bucket: str,
    source_key: str,
    archive_prefix: str = "processed/",
    region: str = "eu-west-2",
) -> str:
    """
    Move a processed file from inbox to archive.
    Adds a timestamp prefix to avoid collisions.
    Returns the new archive key.
    """
    s3 = _get_client(region)

    filename = source_key.rsplit("/", 1)[-1]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_key = f"{archive_prefix}{ts}_{filename}"

    # Copy then delete (S3 has no native move)
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=archive_key,
    )
    s3.delete_object(Bucket=bucket, Key=source_key)

    logger.info(f"Archived: s3://{bucket}/{source_key} -> s3://{bucket}/{archive_key}")
    return archive_key


# ---------------------------------------------------------------------------
# WRITE — Results to S3
# ---------------------------------------------------------------------------

def write_csv_to_s3(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    region: str = "eu-west-2",
) -> str:
    """Write a DataFrame as CSV to S3. Returns the S3 URI."""
    s3 = _get_client(region)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    uri = f"s3://{bucket}/{key}"
    logger.info(f"Wrote CSV: {uri} ({len(df)} rows)")
    return uri


def write_json_to_s3(
    data: dict,
    bucket: str,
    key: str,
    region: str = "eu-west-2",
    json_serializer=None,
) -> str:
    """Write a dict as JSON to S3. Returns the S3 URI."""
    s3 = _get_client(region)
    body = json.dumps(data, indent=2, default=json_serializer).encode("utf-8")

    s3.put_object(Bucket=bucket, Key=key, Body=body)
    uri = f"s3://{bucket}/{key}"
    logger.info(f"Wrote JSON: {uri}")
    return uri


def save_results_to_s3(
    result: Dict[str, Any],
    config: Dict[str, Any],
    json_serializer=None,
) -> Dict[str, str]:
    """
    Save all pipeline outputs to S3.
    Mirrors the local save_results() structure but writes to S3.
    Returns dict of {output_type: s3_uri}.
    """
    out_cfg = config.get("output", {})
    s3_cfg = out_cfg.get("s3", {})
    gen_cfg = out_cfg.get("generate", {})
    region = config.get("ingestion", {}).get("s3", {}).get("region", "eu-west-2")

    bucket = s3_cfg.get("output_bucket", "")
    if not bucket:
        raise ValueError("output.s3.output_bucket must be set in config (or DQ_S3_OUTPUT_BUCKET env var)")

    reports_prefix = s3_cfg.get("reports_prefix", "reports/")
    corrected_prefix = s3_cfg.get("corrected_prefix", "corrected/")

    batch_id = result["batch_id"]
    paths: Dict[str, str] = {}

    # Issue log (CSV)
    if gen_cfg.get("issue_log", True) and not result["issues"].empty:
        key = f"{reports_prefix}{batch_id}_issues.csv"
        paths["issue_log"] = write_csv_to_s3(result["issues"], bucket, key, region)

    # Corrected dataset (CSV)
    if gen_cfg.get("corrected_dataset", True):
        key = f"{corrected_prefix}{batch_id}_corrected.csv"
        paths["corrected_dataset"] = write_csv_to_s3(result["corrected_df"], bucket, key, region)

    # Quality report (JSON)
    if gen_cfg.get("quality_report", True):
        report_data = {
            "batch_id": result["batch_id"],
            "timestamp": result["timestamp"],
            "rows_processed": result["rows_processed"],
            "columns": result["columns"],
            "quality_scores": {k: v for k, v in result["quality_scores"].items() if k != "missing_by_column"},
            "overall_score": result["overall_score"],
            "pass": result["pass"],
            "pass_threshold": result["pass_threshold"],
            "issues_count": result["issues_count"],
            "llm_calls": result["llm_calls"],
            "data_sent_to_llm": result["data_sent_to_llm"],
        }
        key = f"{reports_prefix}{batch_id}_report.json"
        paths["quality_report"] = write_json_to_s3(report_data, bucket, key, region, json_serializer)

    # Audit trail (JSON)
    if gen_cfg.get("audit_trail", True):
        audit_data = {
            "batch_id": result["batch_id"],
            "timestamp": result["timestamp"],
            "steps": result["audit_trail"],
        }
        key = f"{reports_prefix}{batch_id}_audit.json"
        paths["audit_trail"] = write_json_to_s3(audit_data, bucket, key, region, json_serializer)

    logger.info(f"All results saved to s3://{bucket}/ ({len(paths)} files)")
    return paths
