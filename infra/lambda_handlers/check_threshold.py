"""
Lambda: check_threshold.py — Evaluate results and decide next action
=====================================================================
Called by Step Functions after the DQ engine completes.
Checks if the quality score passes the threshold.
Routes to either "archive" (pass) or "alert" (fail) branch.
"""

import os
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    """
    Receives the DQ engine output from the previous step.
    Adds a 'route' field for Step Functions Choice state.
    """
    logger.info(f"Checking threshold: {json.dumps(event)}")

    batch_id = event.get("batch_id", "unknown")
    overall_score = event.get("overall_score", 0)
    pass_threshold = event.get("pass_threshold", 85.0)
    passed = event.get("pass", False)
    issues_count = event.get("issues_count", 0)

    # Build summary for downstream steps
    summary = {
        "batch_id": batch_id,
        "overall_score": overall_score,
        "pass_threshold": pass_threshold,
        "passed": passed,
        "issues_count": issues_count,
        "quality_scores": event.get("quality_scores", {}),
        "output_paths": event.get("output_paths", {}),
        "source": event.get("source", {}),
        "rows_processed": event.get("rows_processed", 0),
        "duration_ms": event.get("duration_ms", 0),
    }

    if passed:
        logger.info(f"PASS: {batch_id} scored {overall_score}% (threshold: {pass_threshold}%)")
        summary["route"] = "pass"
        summary["message"] = f"Batch {batch_id} passed quality check: {overall_score}%"
    else:
        logger.warning(f"FAIL: {batch_id} scored {overall_score}% (threshold: {pass_threshold}%)")
        summary["route"] = "fail"
        summary["message"] = (
            f"Batch {batch_id} FAILED quality check: {overall_score}% "
            f"(threshold: {pass_threshold}%). {issues_count} issues detected."
        )

    return summary
