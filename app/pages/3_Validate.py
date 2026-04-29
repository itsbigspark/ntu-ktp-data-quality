"""
Page 3: Validate -- Run the full validation pipeline.

Uses dq_engine.ValidationOrchestrator (not inline logic).
This is where the engine/UI separation pays off.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
import time
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header, terminal_block,
    kpi_card, MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("VALIDATE & FIX", "Run the full data quality pipeline")

df_raw = st.session_state.get("df_raw")
rules = st.session_state.get("rules_merged")

if df_raw is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

if rules is None:
    terminal_block(
        "// NO RULES AVAILABLE<br>"
        "<span style='color:#4a7a4f;'>Go to Rules page to generate validation rules first, "
        "or the system will use basic inferred rules.</span>"
    )

# ---------------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------------
section_header("// Validation Configuration")

col_cfg1, col_cfg2 = st.columns(2, gap="large")

with col_cfg1:
    use_rules = st.checkbox("Rule-based validation", value=True, key="v_use_rules")
    use_ml = st.checkbox("ML anomaly detection", value=False, key="v_use_ml")
    use_corpus = st.checkbox("Corpus validation", value=False, key="v_use_corpus")
    normalize = st.checkbox("Normalise data before validation", value=True, key="v_normalize")

with col_cfg2:
    use_ai = st.checkbox("AI Enrichment (5 AI calls)", value=False, key="v_use_ai")

    if use_ai:
        provider_type = st.session_state.get("ai_provider_type", "ollama")
        st.markdown(
            f'<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.72rem;">'
            f'AI Provider: {provider_type.upper()} -- Configure in Settings page</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------
section_header("// Execute Pipeline")

if st.button("RUN VALIDATION", key="run_validation", use_container_width=True):
    from dq_engine import ValidationOrchestrator, ValidationConfig

    # Build provider if AI enabled
    ai_provider = None
    if use_ai:
        provider_type = st.session_state.get("ai_provider_type", "ollama")
        try:
            if provider_type == "anthropic":
                from dq_engine import AnthropicProvider
                api_key = st.session_state.get("anthropic_api_key", "")
                if not api_key:
                    st.error("Anthropic API key not set. Go to Settings page.")
                    st.stop()
                ai_provider = AnthropicProvider(api_key=api_key)
            elif provider_type == "openai":
                from dq_engine.orchestrators.ai_enrichment import OpenAIProvider
                api_key = st.session_state.get("openai_api_key", "")
                if not api_key:
                    st.error("OpenAI API key not set. Go to Settings page.")
                    st.stop()
                ai_provider = OpenAIProvider(api_key=api_key)
            else:
                from dq_engine import OllamaProvider
                model = st.session_state.get("ollama_model", "phi3:mini")
                url = st.session_state.get("ollama_url", "http://localhost:11434")
                ai_provider = OllamaProvider(model=model, base_url=url)
        except Exception as e:
            st.error(f"Failed to initialise AI provider: {e}")
            ai_provider = None

    config = ValidationConfig(
        use_rules=use_rules,
        use_ml=use_ml,
        use_corpus=use_corpus,
        normalize=normalize,
        use_ai_enrichment=use_ai and ai_provider is not None,
        ai_provider=ai_provider,
    )

    # Prepare rules list
    if rules and isinstance(rules, dict):
        if "columns" in rules:
            # Standard format: convert to flat list
            rules_list = []
            for col, col_rules in rules.get("columns", {}).items():
                if isinstance(col_rules, dict):
                    rules_list.append({"column": col, **col_rules})
                elif isinstance(col_rules, list):
                    for r in col_rules:
                        rules_list.append({"column": col, **(r if isinstance(r, dict) else {})})
        else:
            rules_list = rules
    else:
        rules_list = rules or []

    # Run
    progress_bar = st.progress(0, text="Initialising pipeline...")
    status_text = st.empty()

    def on_progress(step, total, message):
        pct = step / max(total, 1)
        progress_bar.progress(pct, text=message)
        status_text.markdown(
            f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
            f'Step {step}/{total}: {message}</p>',
            unsafe_allow_html=True,
        )

    config.on_progress = on_progress

    orchestrator = ValidationOrchestrator()

    t0 = time.time()
    result = orchestrator.run(df_raw, rules_list, config)
    elapsed = time.time() - t0

    progress_bar.progress(1.0, text="Pipeline complete")

    # Store results
    st.session_state["validation_result"] = result
    st.session_state["unified_issues_report"] = result.report
    st.session_state["validation_completed"] = True
    if result.ai_enrichment:
        st.session_state["ai_enrichment_result"] = result.ai_enrichment

    st.success(f"Pipeline complete in {elapsed:.1f}s -- {result.total_issues} issues found")

    # Build batch metadata
    from datetime import datetime, timezone
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    overall_score = 100 - (result.total_issues / max(len(df_raw) * len(df_raw.columns), 1) * 100)

    # Save to database
    try:
        from core.storage.database import save_results_to_db, save_ai_enrichment

        db_config = {"output": {"database": {"engine": "sqlite", "path": "./output/dq_investigator.db"}}}

        db_result = {
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rows_processed": len(df_raw),
            "columns": list(df_raw.columns),
            "overall_score": overall_score,
            "pass": overall_score >= 85.0,
            "pass_threshold": 85.0,
            "issues_count": result.total_issues,
            "llm_calls": 5 if result.ai_enrichment else 0,
            "data_sent_to_llm": result.ai_enrichment is not None,
            "quality_scores": {},
            "issues": result.report,
            "audit_trail": [
                {"step": k, "duration_ms": int(v * 1000)}
                for k, v in result.step_timings.items()
            ],
        }
        save_results_to_db(db_result, db_config, source_file="app_upload")

        if result.ai_enrichment:
            save_ai_enrichment(batch_id, result.ai_enrichment, db_config)

        st.info(f"Results saved to database (batch: {batch_id})")
    except Exception as e:
        st.warning(f"Could not save to database: {e}")

    # ── Save results to S3 (if S3 output is configured) ──────────────────
    s3_out_bucket = st.session_state.get("s3_output_bucket", "")
    s3_out_region = st.session_state.get("s3_region", "us-east-1")

    if s3_out_bucket:
        try:
            from core.storage.s3 import write_csv_to_s3, write_json_to_s3
            import numpy as np

            def _np_serializer(obj):
                if isinstance(obj, (np.bool_,)): return bool(obj)
                if isinstance(obj, (np.integer,)): return int(obj)
                if isinstance(obj, (np.floating,)): return float(obj)
                if isinstance(obj, (np.ndarray,)): return obj.tolist()
                raise TypeError(f"Not serializable: {type(obj)}")

            s3_paths = {}

            # Issue log
            if isinstance(result.report, pd.DataFrame) and not result.report.empty:
                s3_paths["issues"] = write_csv_to_s3(
                    result.report, s3_out_bucket,
                    f"reports/{batch_id}_issues.csv", s3_out_region,
                )

            # Quality report
            report_data = {
                "batch_id": batch_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": st.session_state.get("s3_source_key", "file_upload"),
                "rows_processed": len(df_raw),
                "columns": list(df_raw.columns),
                "overall_score": round(overall_score, 2),
                "pass": overall_score >= 85.0,
                "pass_threshold": 85.0,
                "issues_count": result.total_issues,
                "step_timings": {k: round(v, 3) for k, v in result.step_timings.items()},
            }
            s3_paths["report"] = write_json_to_s3(
                report_data, s3_out_bucket,
                f"reports/{batch_id}_report.json", s3_out_region, _np_serializer,
            )

            st.session_state["last_s3_output_paths"] = s3_paths
            path_list = "<br>".join(f"&bull; {v}" for v in s3_paths.values())
            st.info(f"Results saved to S3:\n{chr(10).join(s3_paths.values())}")
        except Exception as e:
            st.warning(f"Could not save to S3: {e}")
    # Auto-save to S3 if data was loaded from S3 (and no dedicated output bucket is set)
    s3_bucket = st.session_state.get("s3_bucket")
    if s3_bucket and result:
        try:
            from core.engine import save_results_to_s3

            # Build a result dict compatible with save_results_to_s3
            _auto_result = {
                "batch_id": batch_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_score": overall_score,
                "pass": overall_score >= 85.0,
                "issues_count": result.total_issues,
                "quality_scores": {},
                "issues": result.report if isinstance(result.report, pd.DataFrame) else pd.DataFrame(),
            }
            s3_region = st.session_state.get("s3_region", "us-east-1")
            paths = save_results_to_s3(_auto_result, bucket=s3_bucket, region=s3_region)
            if paths:
                st.success(f"Results auto-saved to S3: {paths.get('report', '')}")
        except Exception as e:
            st.caption(f"S3 auto-save skipped: {e}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
result = st.session_state.get("validation_result")

if result is not None:
    section_header("// Validation Results")

    # KPI row
    kpi_cols = st.columns(4, gap="medium")
    issue_count = result.total_issues
    steps = len(result.steps_completed)
    total_time = result.total_time_seconds
    error_count = len(result.errors)

    kpi_data = [
        (str(issue_count), "Issues Found", "warn" if issue_count > 20 else ""),
        (str(steps), "Steps Completed", "cyan"),
        (f"{total_time:.1f}s", "Total Time", ""),
        (str(error_count), "Errors", "danger" if error_count > 0 else ""),
    ]
    for col, (val, label, cls) in zip(kpi_cols, kpi_data):
        col.markdown(kpi_card(val, label, cls), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Step timings
    with st.expander("Pipeline Step Timings", expanded=False):
        for step, secs in result.step_timings.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                f'border-bottom:1px solid rgba(0,255,65,0.08);font-family:Share Tech Mono;font-size:0.78rem;">'
                f'<span style="color:#00e5ff;">{step}</span>'
                f'<span style="color:#00ff41;">{secs:.2f}s</span></div>',
                unsafe_allow_html=True,
            )

    # Errors
    if result.errors:
        with st.expander("Errors", expanded=True):
            for err in result.errors:
                st.error(err)

    # Issues table
    if isinstance(result.report, pd.DataFrame) and not result.report.empty:
        section_header("// Issue Report")
        st.dataframe(result.report, use_container_width=True, hide_index=True, height=400)

        csv = result.report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "EXPORT ISSUES [CSV]",
            data=csv,
            file_name="validation_issues.csv",
            mime="text/csv",
        )

    # AI Enrichment summary
    if result.ai_enrichment:
        section_header("// AI Enrichment Summary")
        ai = result.ai_enrichment
        ai_cols = st.columns(4, gap="medium")
        ai_data = [
            (str(len(ai.smart_rules)), "Smart Rules", "cyan"),
            (str(len(ai.cross_column_issues)), "Cross-Column", ""),
            (str(len(ai.triage)), "Triage Items", "warn"),
            (f"{ai.total_time_seconds:.0f}s", "AI Time", ""),
        ]
        for col, (val, label, cls) in zip(ai_cols, ai_data):
            col.markdown(kpi_card(val, label, cls), unsafe_allow_html=True)

        if ai.executive_summary:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(0,15,2,0.85),rgba(0,30,40,0.6));'
                f'border:1px solid rgba(0,229,255,0.25);border-radius:12px;padding:20px 24px;'
                f'font-family:Share Tech Mono;font-size:0.85rem;color:#b0ffb8;line-height:1.7;">'
                f'{ai.executive_summary}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.7rem;margin-top:12px;">'
            'Full AI insights available on the Command Center dashboard and AI Investigation page.</p>',
            unsafe_allow_html=True,
        )

elif not st.session_state.get("validation_completed"):
    terminal_block(
        "// READY TO VALIDATE<br>"
        "<span style='color:#4a7a4f;'>Configure options above and click RUN VALIDATION.</span>"
    )
