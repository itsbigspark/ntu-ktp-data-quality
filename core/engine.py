"""
core/engine.py  --  Standalone DQ Engine Entry Point
====================================================
Runs the full data quality pipeline WITHOUT Streamlit.
Can be called by:
  - CLI:           python -m core.engine --input data.csv
  - Scheduler:     Step Functions / Airflow / cron
  - API:           FastAPI wrapper (see api.py)
  - Streamlit:     imported as a module

This is the same validation logic used by the Streamlit app,
extracted into a clean, framework-agnostic interface.
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("dq_engine")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load config from YAML file, then override with environment variables.
    Env vars prefixed with DQ_ take precedence over file values.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return _default_config()

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Environment variable overrides
    env_map = {
        "DQ_S3_INBOX_BUCKET": ("ingestion", "s3", "inbox_bucket"),
        "DQ_S3_OUTPUT_BUCKET": ("output", "s3", "output_bucket"),
        "DQ_AWS_REGION": ("ingestion", "s3", "region"),
        "ANTHROPIC_API_KEY": ("llm", "anthropic", "api_key"),
        "DQ_DB_HOST": ("output", "database", "host"),
        "DQ_DB_PORT": ("output", "database", "port"),
        "DQ_DB_NAME": ("output", "database", "name"),
        "DQ_DB_USER": ("output", "database", "user"),
        "DQ_DB_PASSWORD": ("output", "database", "password"),
        "DQ_SLACK_WEBHOOK": ("alerts", "channels", "slack", "webhook_url"),
    }

    for env_var, key_path in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            _set_nested(cfg, key_path, val)
            logger.debug(f"Override from env: {env_var}")

    return cfg


def _set_nested(d: dict, keys: tuple, value: Any) -> None:
    """Set a value in a nested dict using a tuple of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _default_config() -> Dict[str, Any]:
    """Minimal defaults if no config file is found."""
    return {
        "app": {"name": "AI Powered DQ Investigator", "mode": "standalone", "log_level": "INFO"},
        "validation": {
            "rule_source": "infer",
            "enable_typo_detection": True,
            "enable_anomaly_detection": True,
            "enable_ml_anomaly": False,
            "corpus": {"enabled": True, "corpus_dir": "./TEST2_DATA/corpus", "jaro_winkler_threshold": 0.85},
            "column_detection": {"sample_size": 100, "confidence_threshold": 0.50},
            "scoring": {"pass_threshold": 85.0},
        },
        "output": {"storage": "local", "local": {"output_dir": "./output"}},
        "bert": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """Load a data file into a DataFrame. Supports CSV, XLSX, Parquet, JSON."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ext = path.suffix.lower()
    logger.info(f"Loading {ext} file: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    rules: Optional[Dict[str, Any]] = None,
    df_ref: Optional[pd.DataFrame] = None,
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full DQ pipeline on a DataFrame.

    Returns a dict with:
        - batch_id: str
        - timestamp: str (ISO format)
        - rows_processed: int
        - quality_scores: dict (6 dimensions)
        - overall_score: float
        - pass: bool
        - issues_count: int
        - issues: pd.DataFrame (the full issue report)
        - corrected_df: pd.DataFrame (auto-corrected data)
        - audit_trail: list of step dicts
    """
    # Lazy imports — only load heavy modules when pipeline actually runs
    from core.validator.validate import (
        validate_and_anomaly_report,
        attach_suggestions,
        attach_issue_descriptions,
    )
    from core.quality_scores import compute_quality_scores

    if batch_id is None:
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    val_cfg = config.get("validation", {})
    pass_threshold = val_cfg.get("scoring", {}).get("pass_threshold", 85.0)

    audit_trail: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows_processed": len(df),
        "columns": list(df.columns),
    }

    # ── Step 1: Rule inference or loading ──────────────────────────────────
    t0 = time.time()
    if rules is None:
        rule_source = val_cfg.get("rule_source", "infer")
        if rule_source == "file" and val_cfg.get("rules_file"):
            with open(val_cfg["rules_file"], "r") as f:
                rules = json.load(f)
            logger.info(f"Loaded rules from {val_cfg['rules_file']}")
        else:
            from core.validator.discover import infer_rules_from_unclean
            rules = infer_rules_from_unclean(df)
            logger.info(f"Inferred rules for {len(rules)} columns")

    audit_trail.append({
        "step": "rule_inference",
        "duration_ms": round((time.time() - t0) * 1000),
        "rules_count": len(rules),
    })

    # ── Step 2: Validation + anomaly detection ─────────────────────────────
    t0 = time.time()
    report = validate_and_anomaly_report(
        df=df,
        rules=rules,
        use_ml=val_cfg.get("enable_ml_anomaly", False),
        ml_models=val_cfg.get("ml_models", []),
        use_reference=(df_ref is not None),
        df_ref=df_ref,
    )
    audit_trail.append({
        "step": "validation",
        "duration_ms": round((time.time() - t0) * 1000),
        "issues_found": len(report),
    })
    logger.info(f"Validation complete: {len(report)} issues found")

    # ── Step 3: Attach suggestions ─────────────────────────────────────────
    t0 = time.time()
    corpus_dir = val_cfg.get("corpus", {}).get("corpus_dir", "")
    if val_cfg.get("corpus", {}).get("enabled", False) and corpus_dir:
        report = attach_suggestions(report, df=df, df_ref=df_ref, rules=rules)
    else:
        report = attach_suggestions(report, df=df, df_ref=df_ref, rules=rules)

    audit_trail.append({
        "step": "suggestions",
        "duration_ms": round((time.time() - t0) * 1000),
        "suggestions_generated": int(report["suggestion"].notna().sum()) if "suggestion" in report.columns else 0,
    })

    # ── Step 4: Attach issue descriptions ──────────────────────────────────
    report = attach_issue_descriptions(report)

    # ── Step 5: Quality scoring ────────────────────────────────────────────
    t0 = time.time()
    quality_scores = compute_quality_scores(df)
    overall = round(
        sum(v for k, v in quality_scores.items() if k != "missing_by_column") /
        max(len([k for k in quality_scores if k != "missing_by_column"]), 1),
        2,
    )
    audit_trail.append({
        "step": "scoring",
        "duration_ms": round((time.time() - t0) * 1000),
        "overall_score": overall,
    })
    logger.info(f"Quality score: {overall}%")

    # ── Step 6: BERT explanations (optional) ───────────────────────────────
    bert_cfg = config.get("bert", {})
    if bert_cfg.get("enabled", False) and not report.empty:
        t0 = time.time()
        try:
            from core.bert_explainer import explain_issues
            max_issues = bert_cfg.get("max_issues", 50)
            report = explain_issues(report, max_issues=max_issues)
            audit_trail.append({
                "step": "bert_explanations",
                "duration_ms": round((time.time() - t0) * 1000),
                "issues_explained": min(len(report), max_issues),
            })
        except Exception as e:
            logger.warning(f"BERT explanations skipped: {e}")

    # ── Assemble results ───────────────────────────────────────────────────
    result["quality_scores"] = quality_scores
    result["overall_score"] = overall
    result["pass"] = bool(overall >= pass_threshold)
    result["pass_threshold"] = pass_threshold
    result["issues_count"] = len(report)
    result["issues"] = report
    result["corrected_df"] = df.copy()  # placeholder — apply_fixes can be called separately
    result["rules"] = rules
    result["audit_trail"] = audit_trail
    result["llm_calls"] = 0
    result["data_sent_to_llm"] = False

    # ── Step 7: AI Enrichment (optional, 5 LLM calls) ────────────────────
    enrich_cfg = config.get("llm", {}).get("enrichment", {})
    if enrich_cfg.get("enabled", False):
        t0 = time.time()
        try:
            from core.ai_enrichment import run_all_enrichments
            enriched = run_all_enrichments(result, df, config)
            result["ai_enrichment"] = enriched
            result["llm_calls"] = enriched.get("llm_calls", 0)
            result["data_sent_to_llm"] = True  # samples were sent (masked)

            # Merge executive summary into result
            if enriched.get("executive_summary", {}).get("executive_summary"):
                result["executive_summary"] = enriched["executive_summary"]["executive_summary"]

            # Merge cross-column findings as additional issues
            cross = enriched.get("cross_column", {}).get("cross_column_findings", [])
            if cross:
                logger.info(f"Adding {len(cross)} cross-column findings to issues")

            audit_trail.append({
                "step": "ai_enrichment",
                "duration_ms": round((time.time() - t0) * 1000),
                "llm_calls": enriched.get("llm_calls", 0),
                "llm_provider": enriched.get("llm_provider", "unknown"),
            })
        except Exception as e:
            logger.warning(f"AI enrichment skipped: {e}")
            audit_trail.append({
                "step": "ai_enrichment",
                "duration_ms": round((time.time() - t0) * 1000),
                "error": str(e),
            })

    return result


def save_results_to_s3(
    result: Dict[str, Any],
    bucket: str,
    region: str = "us-east-1",
) -> Dict[str, str]:
    """
    Save pipeline results to S3 after a validation run.

    Writes two objects:
      - reports/<batch_id>_issues.csv   -- the full issue log (if any issues exist)
      - reports/<batch_id>_report.json  -- a summary JSON with key metrics

    Returns a dict with keys ``issues`` and/or ``report`` pointing at the S3 URIs.
    Returns an empty dict silently if boto3 is unavailable or the bucket is empty.
    """
    if not bucket:
        return {}

    try:
        from core.storage.s3 import write_csv_to_s3, write_json_to_s3
    except Exception:
        logger.warning("core.storage.s3 not available; S3 auto-save skipped")
        return {}

    try:
        batch_id = result.get("batch_id", f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        paths: Dict[str, str] = {}

        # Issue log (CSV) — only written when there are issues
        issues = result.get("issues")
        if issues is not None and hasattr(issues, "empty") and not issues.empty:
            key = f"reports/{batch_id}_issues.csv"
            paths["issues"] = write_csv_to_s3(issues, bucket, key, region)

        # Summary report (JSON)
        report_data = {
            "batch_id": batch_id,
            "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "overall_score": result.get("overall_score"),
            "pass": result.get("pass"),
            "issues_count": result.get("issues_count", 0),
            "quality_scores": {
                k: v
                for k, v in result.get("quality_scores", {}).items()
                if k != "missing_by_column"
            },
        }
        key = f"reports/{batch_id}_report.json"
        paths["report"] = write_json_to_s3(report_data, bucket, key, region, _json_serializer)

        logger.info(f"Auto-saved {len(paths)} result file(s) to s3://{bucket}/reports/")
        return paths

    except Exception as exc:
        logger.warning(f"S3 auto-save failed: {exc}")
        return {}


def _json_serializer(obj):
    """Handle numpy/pandas types that json.dump can't serialize."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def save_results(result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    """
    Persist pipeline results based on config.output settings.
    Returns a dict of output file paths.
    """
    out_cfg = config.get("output", {})
    gen_cfg = out_cfg.get("generate", {})
    storage = out_cfg.get("storage", "local")
    paths: Dict[str, str] = {}

    if storage == "local":
        local_cfg = out_cfg.get("local", {})
        output_dir = Path(local_cfg.get("output_dir", "./output"))
        batch_id = result["batch_id"]

        # Create directories
        for subdir in ["reports", "corrected", "logs"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Issue log
        if gen_cfg.get("issue_log", True) and not result["issues"].empty:
            p = output_dir / "reports" / f"{batch_id}_issues.csv"
            result["issues"].to_csv(p, index=False)
            paths["issue_log"] = str(p)
            logger.info(f"Issue log saved: {p}")

        # Corrected dataset
        if gen_cfg.get("corrected_dataset", True):
            p = output_dir / "corrected" / f"{batch_id}_corrected.csv"
            result["corrected_df"].to_csv(p, index=False)
            paths["corrected_dataset"] = str(p)
            logger.info(f"Corrected dataset saved: {p}")

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
            p = output_dir / "reports" / f"{batch_id}_report.json"
            with open(p, "w") as f:
                json.dump(report_data, f, indent=2, default=_json_serializer)
            paths["quality_report"] = str(p)
            logger.info(f"Quality report saved: {p}")

        # Audit trail
        if gen_cfg.get("audit_trail", True):
            p = output_dir / "logs" / f"{batch_id}_audit.json"
            with open(p, "w") as f:
                json.dump({
                    "batch_id": result["batch_id"],
                    "timestamp": result["timestamp"],
                    "steps": result["audit_trail"],
                }, f, indent=2, default=_json_serializer)
            paths["audit_trail"] = str(p)
            logger.info(f"Audit trail saved: {p}")

    elif storage == "s3":
        from core.storage.s3 import save_results_to_s3
        paths = save_results_to_s3(result, config, json_serializer=_json_serializer)

    elif storage == "database":
        from core.storage.database import save_results_to_db
        source_file = config.get("_source_file", "")
        counts = save_results_to_db(result, config, source_file=source_file)
        paths["database"] = f"Wrote {counts['batch_runs']} batch, {counts['issues']} issues, {counts['audit_trail']} audit steps"

    # Multi-output: if config says "all", write to local + database
    if storage == "all":
        # Local first
        config_local = {**config, "output": {**config.get("output", {}), "storage": "local"}}
        paths = save_results(result, config_local)
        # Then database
        try:
            from core.storage.database import save_results_to_db
            source_file = config.get("_source_file", "")
            counts = save_results_to_db(result, config, source_file=source_file)
            paths["database"] = f"Wrote {counts['batch_runs']} batch, {counts['issues']} issues, {counts['audit_trail']} audit steps"
        except Exception as e:
            logger.warning(f"Database write failed (local files still saved): {e}")

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AI Powered DQ Investigator — Standalone Pipeline",
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input data file, or s3://bucket/key")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file")
    parser.add_argument("--rules", "-r", default=None, help="Path to rules JSON file (optional)")
    parser.add_argument("--reference", default=None, help="Path to reference/clean data file (optional)")
    parser.add_argument("--batch-id", default=None, help="Custom batch ID (default: auto-generated)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--storage", default=None, help="Override output storage: local, s3, database, all")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    _setup_logging(config.get("app", {}).get("log_level", "INFO"))

    if args.output_dir:
        config.setdefault("output", {}).setdefault("local", {})["output_dir"] = args.output_dir
    if args.storage:
        config.setdefault("output", {})["storage"] = args.storage

    logger.info("=" * 60)
    logger.info(f"AI Powered DQ Investigator v{config.get('app', {}).get('version', '1.0.0')}")
    logger.info("=" * 60)

    # Load data — supports local paths and s3:// URIs
    input_path = args.input
    if input_path.startswith("s3://"):
        # Parse s3://bucket/key
        parts = input_path[5:].split("/", 1)
        bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
        region = config.get("ingestion", {}).get("s3", {}).get("region", "eu-west-2")
        from core.storage.s3 import read_from_s3
        df = read_from_s3(bucket, key, region)
    else:
        df = load_data(input_path)

    config["_source_file"] = input_path
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # Load rules if provided
    rules = None
    if args.rules:
        with open(args.rules, "r") as f:
            rules = json.load(f)
        logger.info(f"Loaded {len(rules)} rules from {args.rules}")

    # Load reference data if provided
    df_ref = None
    if args.reference:
        df_ref = load_data(args.reference)
        logger.info(f"Loaded reference data: {len(df_ref)} rows")

    # Run pipeline
    logger.info("Running pipeline...")
    result = run_pipeline(
        df=df,
        config=config,
        rules=rules,
        df_ref=df_ref,
        batch_id=args.batch_id,
    )

    # Save results
    paths = save_results(result, config)

    # Check alerts
    try:
        from core.alerts import check_and_alert
        alert_result = check_and_alert(result, config)
        if alert_result.get("alert_sent"):
            logger.warning(f"Alert sent: {alert_result.get('subject', '')}")
    except Exception as e:
        logger.warning(f"Alert check failed: {e}")

    # Summary
    logger.info("-" * 60)
    logger.info(f"Batch:          {result['batch_id']}")
    logger.info(f"Rows processed: {result['rows_processed']}")
    logger.info(f"Issues found:   {result['issues_count']}")
    logger.info(f"Quality score:  {result['overall_score']}%")
    logger.info(f"Pass:           {'YES' if result['pass'] else 'NO'} (threshold: {result['pass_threshold']}%)")
    logger.info(f"LLM calls:      {result['llm_calls']}")
    logger.info(f"Data sent to LLM: {result['data_sent_to_llm']}")
    logger.info("-" * 60)
    for name, path in paths.items():
        logger.info(f"Output [{name}]: {path}")
    logger.info("=" * 60)

    # Exit code: 0 = pass, 1 = fail (useful for CI/CD)
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
