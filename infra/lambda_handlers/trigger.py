"""
Lambda: trigger.py — Entry point for S3 event
================================================
Triggered by EventBridge when a new file lands in the S3 inbox.
Validates the file metadata and starts the Step Functions execution.

Event source: EventBridge rule matching s3:ObjectCreated events.
"""

import os
import json
import logging
import boto3
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sfn_client = boto3.client("stepfunctions")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN", "")
SUPPORTED_FORMATS = {"csv", "xlsx", "parquet", "json"}


def handler(event, context):
    """
    Receives an EventBridge event for S3 ObjectCreated.
    Extracts bucket/key, validates format, starts Step Functions.
    """
    logger.info(f"Event received: {json.dumps(event)}")

    # Parse S3 event from EventBridge
    detail = event.get("detail", {})
    bucket = detail.get("bucket", {}).get("name", "")
    key = detail.get("object", {}).get("key", "")
    size = detail.get("object", {}).get("size", 0)

    if not bucket or not key:
        logger.error("Missing bucket or key in event")
        return {"status": "error", "message": "Missing bucket or key"}

    # Validate file format
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
    if ext not in SUPPORTED_FORMATS:
        logger.warning(f"Unsupported format: {ext} (key: {key})")
        return {"status": "skipped", "message": f"Unsupported format: {ext}"}

    # Generate batch ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    batch_id = f"batch_{filename}_{ts}"

    # Build Step Functions input
    sfn_input = {
        "batch_id": batch_id,
        "source": {
            "bucket": bucket,
            "key": key,
            "size_bytes": size,
            "format": ext,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Start execution
    logger.info(f"Starting Step Functions: {batch_id}")
    response = sfn_client.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=batch_id,
        input=json.dumps(sfn_input),
    )

    logger.info(f"Execution started: {response['executionArn']}")
    return {
        "status": "started",
        "batch_id": batch_id,
        "execution_arn": response["executionArn"],
    }
