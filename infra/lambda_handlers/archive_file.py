"""
Lambda: archive_file.py — Move processed file to archive
==========================================================
Called by Step Functions as the final step.
Moves the source file from inbox to processed/ prefix.
"""

import os
import json
import logging
import boto3
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

ARCHIVE_PREFIX = os.environ.get("ARCHIVE_PREFIX", "processed/")


def handler(event, context):
    """
    Receives summary from previous step.
    Archives the source file.
    """
    logger.info(f"Archiving: {json.dumps(event)}")

    source = event.get("source", {})
    bucket = source.get("bucket", "")
    key = source.get("key", "")
    batch_id = event.get("batch_id", "unknown")

    if not bucket or not key:
        logger.warning("No source bucket/key, skipping archive")
        return {"batch_id": batch_id, "archived": False}

    # Build archive key
    filename = key.rsplit("/", 1)[-1]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_key = f"{ARCHIVE_PREFIX}{ts}_{filename}"

    try:
        # Copy to archive
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": key},
            Key=archive_key,
        )
        # Delete original
        s3.delete_object(Bucket=bucket, Key=key)

        logger.info(f"Archived: s3://{bucket}/{key} -> s3://{bucket}/{archive_key}")
        return {
            "batch_id": batch_id,
            "archived": True,
            "archive_path": f"s3://{bucket}/{archive_key}",
        }
    except Exception as e:
        logger.error(f"Archive failed: {e}")
        return {
            "batch_id": batch_id,
            "archived": False,
            "error": str(e),
        }
