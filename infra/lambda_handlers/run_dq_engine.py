"""
Lambda: run_dq_engine.py — Runs the DQ validation pipeline via ECS API
=======================================================================
Calls POST /api/v1/s3/validate on the deployed ECS FastAPI rather than
importing core directly (core + ML deps are too large to bundle in Lambda).

The ECS API is already running; Lambda is a thin orchestration wrapper.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

OUTPUT_BUCKET    = os.environ.get("OUTPUT_BUCKET", "")
DQ_API_URL       = os.environ.get("DQ_API_URL", "")      # e.g. http://alb-host:8000
SSM_KEY_PATH     = os.environ.get("SSM_KEY_PATH", "/dq-investigator/master-api-key")
PASS_THRESHOLD   = float(os.environ.get("PASS_THRESHOLD", "85.0"))

s3  = boto3.client("s3")
ssm = boto3.client("ssm", region_name=os.environ.get("AWS_REGION", "us-east-1"))

_api_key_cache: str = ""


def _get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache:
        return _api_key_cache
    resp = ssm.get_parameter(Name=SSM_KEY_PATH, WithDecryption=True)
    _api_key_cache = resp["Parameter"]["Value"]
    return _api_key_cache


def handler(event, context):
    """
    Step Functions input:
    {
        "batch_id": "batch_xxx",
        "source": {"bucket": "...", "key": "...", "format": "csv"},
        "timestamp": "..."
    }
    """
    logger.info(f"DQ Engine starting: {json.dumps(event)}")
    t_start = time.time()

    batch_id = event["batch_id"]
    source   = event["source"]
    bucket   = source["bucket"]
    key      = source["key"]

    if not DQ_API_URL:
        raise RuntimeError("DQ_API_URL env var not set")

    api_key = _get_api_key()

    # ── Call ECS API: POST /api/v1/s3/validate ────────────────────────────
    payload = json.dumps({
        "bucket":         bucket,
        "key":            key,
        "region":         os.environ.get("AWS_REGION", "us-east-1"),
        "output_bucket":  OUTPUT_BUCKET,
        "output_prefix":  f"reports/{batch_id}",
        "pass_threshold": PASS_THRESHOLD,
    }).encode("utf-8")

    req = urllib.request.Request(
        url=f"{DQ_API_URL}/api/v1/s3/validate",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )

    logger.info(f"Calling ECS API: {DQ_API_URL}/api/v1/s3/validate")
    try:
        with urllib.request.urlopen(req, timeout=840) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        logger.error(f"API HTTP error {e.code}: {err_body}")
        raise RuntimeError(f"ECS API returned {e.code}: {err_body}")
    except urllib.error.URLError as e:
        logger.error(f"API connection error: {e}")
        raise RuntimeError(f"Could not reach ECS API at {DQ_API_URL}: {e}")

    logger.info(
        f"Pipeline complete: {result.get('issues_count', '?')} issues, "
        f"score {result.get('overall_score', '?')}%"
    )

    duration_ms = round((time.time() - t_start) * 1000)

    return {
        "batch_id":       result.get("batch_id", batch_id),
        "rows_processed": result.get("rows_processed", 0),
        "issues_count":   result.get("issues_count", 0),
        "overall_score":  float(result.get("overall_score", 0)),
        "pass":           bool(result.get("pass", False)),
        "pass_threshold": PASS_THRESHOLD,
        "quality_scores": result.get("quality_scores", {}),
        "output_paths":   result.get("output_paths", {}),
        "audit_trail":    result.get("audit_trail", []),
        "duration_ms":    duration_ms,
        "source":         source,
    }
