"""
core/alerts.py  --  Alert System for the DQ Pipeline
=====================================================
Sends notifications when batch quality drops below threshold.
Supports: console (always), email (SES), Slack (webhook).
Works in both standalone and enterprise (Lambda) modes.
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("dq_engine.alerts")


def check_and_alert(result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if an alert should be sent and dispatch it.
    Returns a summary of what was sent.
    """
    alert_cfg = config.get("alerts", {})
    if not alert_cfg.get("enabled", False):
        return {"alert_sent": False, "reason": "alerts disabled"}

    threshold = alert_cfg.get("on_score_below", 85.0)
    score = result.get("overall_score", 100)

    if score >= threshold:
        logger.info(f"Score {score}% >= threshold {threshold}% — no alert needed")
        return {"alert_sent": False, "reason": f"score {score}% above threshold {threshold}%"}

    # Build alert content
    alert = _build_alert(result, threshold)
    sent_to = []

    # Console (always)
    logger.warning(f"ALERT: {alert['subject']}")
    logger.warning(alert["body"][:500])

    # Email via SES
    email_cfg = alert_cfg.get("channels", {}).get("email", {})
    if email_cfg.get("enabled", False) and email_cfg.get("recipients"):
        try:
            _send_ses(alert, email_cfg)
            sent_to.append("email")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    # Slack webhook
    slack_cfg = alert_cfg.get("channels", {}).get("slack", {})
    if slack_cfg.get("enabled", False) and slack_cfg.get("webhook_url"):
        try:
            _send_slack(alert, slack_cfg)
            sent_to.append("slack")
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")

    return {"alert_sent": True, "sent_to": sent_to, "subject": alert["subject"]}


def _build_alert(result: Dict[str, Any], threshold: float) -> Dict[str, str]:
    """Build alert subject and body."""
    batch_id = result.get("batch_id", "unknown")
    score = result.get("overall_score", 0)
    issues = result.get("issues_count", 0)
    rows = result.get("rows_processed", 0)
    scores = result.get("quality_scores", {})

    subject = f"DQ Alert: {batch_id} scored {score}% (threshold: {threshold}%)"

    lines = [
        "DATA QUALITY ALERT",
        "=" * 40,
        "",
        f"Batch:           {batch_id}",
        f"Rows processed:  {rows}",
        f"Issues found:    {issues}",
        "",
        "QUALITY SCORES",
        "-" * 40,
    ]

    for dim, val in scores.items():
        if dim == "missing_by_column":
            continue
        status = "PASS" if val >= threshold else "FAIL"
        lines.append(f"  {dim:15s}: {val:6.1f}%  [{status}]")

    lines.extend([
        "",
        f"  OVERALL:          {score:.1f}%  [FAIL]",
        f"  Threshold:        {threshold:.1f}%",
        "",
        "---",
        "AI Powered DQ Investigator | Automated Alert",
    ])

    # Executive summary if available
    summary = result.get("executive_summary", "")
    if summary:
        lines.insert(4, f"\nSummary: {summary}\n")

    return {"subject": subject, "body": "\n".join(lines)}


def _send_ses(alert: Dict[str, str], email_cfg: Dict[str, Any]) -> None:
    """Send email via AWS SES."""
    import boto3
    ses = boto3.client("ses", region_name=email_cfg.get("ses_region", "eu-west-2"))

    ses.send_email(
        Source=email_cfg.get("sender", "dq-investigator@noreply.com"),
        Destination={"ToAddresses": email_cfg["recipients"]},
        Message={
            "Subject": {"Data": alert["subject"]},
            "Body": {"Text": {"Data": alert["body"]}},
        },
    )
    logger.info(f"Email sent to {email_cfg['recipients']}")


def _send_slack(alert: Dict[str, str], slack_cfg: Dict[str, Any]) -> None:
    """Send message to Slack webhook."""
    import requests
    payload = {
        "text": f"*{alert['subject']}*\n```{alert['body'][:2000]}```"
    }
    resp = requests.post(slack_cfg["webhook_url"], json=payload, timeout=10)
    resp.raise_for_status()
    logger.info("Slack alert sent")
