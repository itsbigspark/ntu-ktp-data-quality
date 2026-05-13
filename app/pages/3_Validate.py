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
    kpi_card, workflow_breadcrumb,
    aggrid_issues,
    MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("VALIDATE & FIX", "Run the full data quality pipeline")

df_raw = st.session_state.get("df_raw")
rules = st.session_state.get("rules_merged")

workflow_breadcrumb([
    ("Load Data", df_raw is not None),
    ("Rules", rules is not None),
    ("Validate", st.session_state.get("validation_completed", False)),
    ("AI Investigate", st.session_state.get("ai_enrichment_result") is not None),
    ("Clean", st.session_state.get("df_corrected") is not None),
])

if df_raw is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

# Session status bar — shows what's ready
_corpus_mgr = st.session_state.get("corpus_manager")
_has_corpus = _corpus_mgr is not None and hasattr(_corpus_mgr, "list_corpora") and bool(_corpus_mgr.list_corpora() if callable(_corpus_mgr.list_corpora) else [])
_status_items = [
    ("Data", f"{len(df_raw):,} rows", "#00ff41"),
    ("Rules", f"{sum(len(v) for v in rules.values() if isinstance(v, list))} rules" if rules else "None (inferred)", "#00ff41" if rules else "#ff9100"),
    ("Corpus", "Loaded" if _has_corpus else "Not loaded", "#00ff41" if _has_corpus else "#4a7a4f"),
    ("AI", st.session_state.get("ai_provider_type", "ollama").upper(), "#00e5ff"),
]
st.markdown(
    '<div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap;">'
    + "".join(
        f'<div style="background:rgba(0,5,1,0.7);border:1px solid rgba(0,255,65,0.15);'
        f'border-radius:6px;padding:6px 14px;font-family:Share Tech Mono;font-size:0.72rem;">'
        f'<span style="color:#4a7a4f;">{label}: </span>'
        f'<span style="color:{color};">{value}</span></div>'
        for label, value, color in _status_items
    )
    + "</div>",
    unsafe_allow_html=True,
)

if rules is None:
    st.info("No rules loaded — validation will use inferred statistical rules. Go to Rules page to generate custom rules.")

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
            st.error("Could not connect to AI provider. Check your Settings — API key or Ollama URL may be wrong.")
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

    st.success(f"Pipeline complete in {elapsed:.1f}s — {result.total_issues} issues found")

    # Build batch metadata
    from datetime import datetime, timezone
    import uuid
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    st.session_state["current_batch_id"] = batch_id

    # ── Calculate proper quality dimension scores from issues report ──────
    def _calc_quality_scores(df, issues_df):
        """Derive 6-dimension scores from the issues DataFrame."""
        total_cells = max(len(df) * len(df.columns), 1)
        total_rows  = max(len(df), 1)
        scores = {"completeness": 100.0, "uniqueness": 100.0, "consistency": 100.0,
                  "validity": 100.0, "accuracy": 100.0, "timeliness": 100.0}
        if not isinstance(issues_df, pd.DataFrame) or issues_df.empty:
            return scores

        # Map issue_type → dimension
        _dim_map = {
            "missing_value": "completeness", "null_value": "completeness",
            "duplicate": "uniqueness", "near_duplicate": "uniqueness",
            "format_error": "validity", "regex_mismatch": "validity",
            "type_error": "validity", "invalid_value": "validity",
            "out_of_range": "validity", "constraint_violation": "validity",
            "corpus_mismatch": "accuracy", "unknown_value": "accuracy",
            "referential_integrity": "accuracy",
            "inconsistency": "consistency", "cross_column": "consistency",
            "stale_date": "timeliness", "future_date": "timeliness",
            "invalid_date": "timeliness",
        }
        issue_col = "issue_type" if "issue_type" in issues_df.columns else issues_df.columns[0]
        dim_counts = {d: 0 for d in scores}
        for itype in issues_df[issue_col].dropna():
            dim = _dim_map.get(str(itype).lower().replace(" ", "_"), "validity")
            dim_counts[dim] += 1

        # Score = 100 - (issues in dim / total_cells * 100), floored at 0
        for dim, count in dim_counts.items():
            penalty = (count / total_cells) * 100
            scores[dim] = max(0.0, round(100.0 - penalty, 1))

        # Completeness: also factor in actual null rate
        null_rate = df.isna().mean().mean() * 100
        scores["completeness"] = max(0.0, round(100.0 - null_rate, 1))

        # Uniqueness: factor in duplicate rows
        dup_rate = df.duplicated().mean() * 100
        scores["uniqueness"] = max(0.0, round(min(scores["uniqueness"], 100.0 - dup_rate), 1))

        return scores

    quality_scores = _calc_quality_scores(df_raw, result.report)
    overall_score  = round(sum(quality_scores.values()) / len(quality_scores), 1)

    # Work out source file name
    source_file = (
        st.session_state.get("s3_source_key")
        or st.session_state.get("db_last_query", "")[:60]
        or "file_upload"
    )

    # Save to database
    try:
        from core.storage.database import save_results_to_db, save_ai_enrichment, get_engine

        # Pass empty config — get_engine() inside will use DATABASE_URL → PostgreSQL
        db_config = {}

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
            "quality_scores": quality_scores,
            "issues": result.report,
            "audit_trail": [
                {"step": k, "duration_ms": int(v * 1000)}
                for k, v in result.step_timings.items()
            ],
        }
        save_results_to_db(db_result, db_config, source_file=source_file)
        st.info(f"Results saved — batch: `{batch_id}` | Score: {overall_score}%")
    except Exception as e:
        st.warning(f"Could not save results to database: {e}")

    # Save AI enrichment separately so a results-save failure doesn't lose AI data
    if result.ai_enrichment:
        try:
            from core.storage.database import save_ai_enrichment
            save_ai_enrichment(batch_id, result.ai_enrichment, {})
        except Exception as e:
            st.warning(f"Could not save AI enrichment to database: {e}")

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

    # Issues breakdown
    if isinstance(result.report, pd.DataFrame) and not result.report.empty:
        rpt = result.report

        # ── Severity breakdown ───────────────────────────────────────────────
        section_header("// Issue Breakdown")
        _sev_col = next((c for c in ["severity", "Severity"] if c in rpt.columns), None)
        _typ_col = next((c for c in ["issue_type", "issue", "Issue Type"] if c in rpt.columns), None)
        _col_col = next((c for c in ["column", "column_name", "Column"] if c in rpt.columns), None)

        breakdown_cols = st.columns(2, gap="large")

        with breakdown_cols[0]:
            st.markdown(
                '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.72rem;'
                'letter-spacing:1px;margin-bottom:8px;">BY SEVERITY</p>',
                unsafe_allow_html=True,
            )
            if _sev_col:
                sev_counts = rpt[_sev_col].value_counts()
                _sev_colors = {"critical": "#ff1744", "high": "#ff9100",
                               "medium": "#ffea00", "low": "#00e5ff", "info": "#4a7a4f"}
                for sev, count in sev_counts.items():
                    color = _sev_colors.get(str(sev).lower(), "#b0ffb8")
                    pct = count / len(rpt) * 100
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                        f'<span style="font-family:Share Tech Mono;font-size:0.72rem;'
                        f'color:{color};width:70px;text-transform:uppercase;">{sev}</span>'
                        f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:3px;height:8px;">'
                        f'<div style="width:{pct:.0f}%;background:{color};height:8px;border-radius:3px;"></div>'
                        f'</div>'
                        f'<span style="font-family:Share Tech Mono;font-size:0.72rem;color:#b0ffb8;width:40px;text-align:right;">{count}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No severity column in report.")

        with breakdown_cols[1]:
            st.markdown(
                '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.72rem;'
                'letter-spacing:1px;margin-bottom:8px;">TOP AFFECTED COLUMNS</p>',
                unsafe_allow_html=True,
            )
            if _col_col:
                col_counts = rpt[_col_col].value_counts().head(8)
                max_c = col_counts.max()
                for col_name, count in col_counts.items():
                    pct = count / max_c * 100
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                        f'<span style="font-family:Share Tech Mono;font-size:0.7rem;'
                        f'color:#00ff41;width:120px;white-space:nowrap;overflow:hidden;'
                        f'text-overflow:ellipsis;" title="{col_name}">{col_name}</span>'
                        f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:3px;height:8px;">'
                        f'<div style="width:{pct:.0f}%;background:#00ff41;height:8px;border-radius:3px;opacity:0.7;"></div>'
                        f'</div>'
                        f'<span style="font-family:Share Tech Mono;font-size:0.72rem;color:#b0ffb8;width:40px;text-align:right;">{count}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No column info in report.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Issue type summary ───────────────────────────────────────────────
        if _typ_col:
            with st.expander("Issue types breakdown", expanded=False):
                type_counts = rpt[_typ_col].value_counts().reset_index()
                type_counts.columns = ["Issue Type", "Count"]
                type_counts["% of Total"] = (type_counts["Count"] / len(rpt) * 100).round(1).astype(str) + "%"
                st.dataframe(type_counts, use_container_width=True, hide_index=True, height=min(300, 40 + 35 * len(type_counts)))

        # ── Full issue table ─────────────────────────────────────────────────
        section_header("// Full Issue Report")
        aggrid_issues(rpt)

        csv = rpt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "EXPORT ISSUES [CSV]",
            data=csv,
            file_name="validation_issues.csv",
            mime="text/csv",
        )

        # ── Next steps CTA ───────────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _has_fixes = "suggested_fix" in rpt.columns and rpt["suggested_fix"].notna().any()
        _has_ai    = result.ai_enrichment is not None
        st.markdown(
            '<div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);'
            'border-radius:10px;padding:16px 20px;font-family:Share Tech Mono;">'
            '<p style="color:#00e5ff;font-size:0.78rem;letter-spacing:1px;margin-bottom:8px;">// NEXT STEPS</p>'
            + (f'<p style="color:#b0ffb8;font-size:0.75rem;margin:4px 0;">→ <b>Cleaning</b> — '
               f'{rpt["suggested_fix"].notna().sum()} auto-fix suggestions ready to review and apply</p>'
               if _has_fixes else
               '<p style="color:#4a7a4f;font-size:0.75rem;margin:4px 0;">→ <b>Cleaning</b> — manually review and correct issues</p>')
            + ('<p style="color:#b0ffb8;font-size:0.75rem;margin:4px 0;">→ <b>AI Investigation</b> — AI enrichment results ready, including executive summary</p>'
               if _has_ai else
               '<p style="color:#4a7a4f;font-size:0.75rem;margin:4px 0;">→ <b>AI Investigation</b> — re-run with AI Enrichment enabled for deeper insights</p>')
            + '<p style="color:#b0ffb8;font-size:0.75rem;margin:4px 0;">→ <b>Command Center</b> — full dashboard with quality scores and trend charts</p>'
            + f'<p style="color:#4a7a4f;font-size:0.7rem;margin-top:10px;">Batch: {st.session_state.get("current_batch_id","—")}</p>'
            + '</div>',
            unsafe_allow_html=True,
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
