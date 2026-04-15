"""
Lambda: send_alert.py — Send notification when quality check fails
===================================================================
Called by Step Functions when a batch fails the threshold.
Sends alerts via SNS (email/Slack) and optionally SES for rich email.
"""

import os
import json
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns_client = boto3.client("sns")

ALERT_TOPIC_ARN = os.environ.get("ALERT_TOPIC_ARN", "")
APP_URL = os.environ.get("APP_URL", "")  # Streamlit app URL for investigation link


def handler(event, context):
    """
    Receives summary from check_threshold step.
    Sends alert notification.
    """
    logger.info(f"Sending alert: {json.dumps(event)}")

    batch_id = event.get("batch_id", "unknown")
    overall_score = event.get("overall_score", 0)
    pass_threshold = event.get("pass_threshold", 85.0)
    issues_count = event.get("issues_count", 0)
    rows_processed = event.get("rows_processed", 0)
    quality_scores = event.get("quality_scores", {})
    source = event.get("source", {})
    output_paths = event.get("output_paths", {})

    # Build alert message
    subject = f"DQ Alert: {batch_id} scored {overall_score}% (threshold: {pass_threshold}%)"

    body_lines = [
        f"DATA QUALITY ALERT",
        f"==================",
        f"",
        f"Batch:           {batch_id}",
        f"Source:           s3://{source.get('bucket', '')}/{source.get('key', '')}",
        f"Rows processed:  {rows_processed}",
        f"Issues found:    {issues_count}",
        f"",
        f"QUALITY SCORES",
        f"--------------",
    ]

    for dimension, score in quality_scores.items():
        status = "PASS" if score >= pass_threshold else "FAIL"
        body_lines.append(f"  {dimension:15s}: {score:6.1f}%  [{status}]")

    body_lines.extend([
        f"",
        f"  OVERALL:          {overall_score:.1f}%  [FAIL]",
        f"  Threshold:        {pass_threshold:.1f}%",
        f"",
        f"OUTPUTS",
        f"-------",
    ])

    for name, path in output_paths.items():
        body_lines.append(f"  {name}: {path}")

    if APP_URL:
        body_lines.extend([
            f"",
            f"INVESTIGATE",
            f"-----------",
            f"Open the AI investigator to analyse this batch:",
            f"  {APP_URL}?batch={batch_id}",
        ])

    body_lines.extend([
        f"",
        f"---",
        f"AI Powered DQ Investigator | Automated Alert",
    ])

    body = "\n".join(body_lines)

    # Send via SNS
    if ALERT_TOPIC_ARN:
        try:
            sns_client.publish(
                TopicArn=ALERT_TOPIC_ARN,
                Subject=subject[:100],  # SNS subject limit
                Message=body,
            )
            logger.info(f"Alert sent to SNS topic: {ALERT_TOPIC_ARN}")
        except Exception as e:
            logger.error(f"SNS publish failed: {e}")
    else:
        logger.warning("No ALERT_TOPIC_ARN set, alert not sent")
        logger.info(f"Alert content:\n{body}")

    return {
        "batch_id": batch_id,
        "alert_sent": bool(ALERT_TOPIC_ARN),
        "subject": subject,
    }
