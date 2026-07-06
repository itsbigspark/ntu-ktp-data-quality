"""
core/s3_writeback.py — Write approved cleaned datasets back to S3
=================================================================

Used by the Cleaning tab to persist an analyst-approved cleaned dataset to
S3 after explicit human approval. Keeps all AWS concerns in one place so the
Streamlit app stays free of boto3 details.

Configuration (environment variables, all optional):
    S3_WRITEBACK_BUCKET   default bucket name
    S3_WRITEBACK_PREFIX   default key prefix (e.g. "cleaned/")
    AWS_DEFAULT_REGION    AWS region

Credentials are resolved by the standard boto3 chain (env vars, shared
config/credentials file, or an attached IAM role), so no secrets are handled
here directly.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def boto3_available() -> bool:
    """True if boto3 can be imported in the current environment."""
    try:
        import boto3  # noqa: F401
        return True
    except Exception:
        return False


def get_default_config() -> dict:
    """Read default S3 write-back settings from the environment."""
    return {
        "bucket": os.environ.get("S3_WRITEBACK_BUCKET", ""),
        "prefix": os.environ.get("S3_WRITEBACK_PREFIX", "cleaned/"),
        "region": os.environ.get("AWS_DEFAULT_REGION", ""),
    }


def build_key(prefix: str, filename: str) -> str:
    """
    Build a timestamped, collision-resistant S3 key.

    Example: cleaned/20260612_140530_dataset_cleaned.csv
    """
    prefix = (prefix or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = filename.rsplit("/", 1)[-1] or "dataset_cleaned.csv"
    return f"{prefix}{ts}_{filename}"


def write_df_to_s3(
    df: pd.DataFrame,
    bucket: str,
    prefix: str = "cleaned/",
    filename: str = "dataset_cleaned.csv",
    region: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Write a DataFrame to S3 as a UTF-8 CSV object.

    Returns a result dict:
        {"success": True,  "uri": "s3://bucket/key", "key": key, "bucket": bucket}
        {"success": False, "error": "<message>"}

    This function performs the upload immediately; the caller is responsible
    for gating it behind explicit human approval.
    """
    if not boto3_available():
        return {"success": False, "error": "boto3 is not installed in this environment."}

    if df is None or not isinstance(df, pd.DataFrame):
        return {"success": False, "error": "No valid DataFrame provided."}

    if not bucket or not str(bucket).strip():
        return {"success": False, "error": "No S3 bucket configured. Set S3_WRITEBACK_BUCKET or enter one."}

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    key = build_key(prefix, filename)

    try:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        body = buf.getvalue().encode("utf-8")

        client_kwargs = {}
        if region:
            client_kwargs["region_name"] = region
        s3 = boto3.client("s3", **client_kwargs)

        put_kwargs = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "text/csv",
        }
        # Attach audit metadata (S3 object metadata values must be strings)
        meta = {"source": "data_quality_app", "approved": "human"}
        if metadata:
            meta.update({str(k): str(v) for k, v in metadata.items()})
        put_kwargs["Metadata"] = meta

        s3.put_object(**put_kwargs)

        return {
            "success": True,
            "uri": f"s3://{bucket}/{key}",
            "bucket": bucket,
            "key": key,
            "rows": int(len(df)),
            "bytes": len(body),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": f"AWS error: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
