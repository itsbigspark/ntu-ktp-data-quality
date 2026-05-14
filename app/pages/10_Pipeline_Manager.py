"""
Page 10: Pipeline Manager -- Create, save, and execute reusable data quality pipelines.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("PIPELINE MANAGER", "Create reusable data quality pipelines and execute them")

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
try:
    from core.pipeline_manager import Pipeline, PipelineManager, PipelineExecutor
except ImportError:
    st.error("Could not import Pipeline Manager. Ensure core/pipeline_manager.py exists.")
    st.stop()

if st.session_state.get("pipeline_manager") is None:
    st.session_state["pipeline_manager"] = PipelineManager()

pm = st.session_state["pipeline_manager"]

# Cache loaded pipelines per page-render to avoid disk I/O storms inside
# the Library tab's expander loop. Streamlit reruns the whole script on
# every interaction, so this dict acts as a 1-render memo.
_pipeline_cache: dict = {}
def _cached_load(name: str):
    if name not in _pipeline_cache:
        _pipeline_cache[name] = pm.load_pipeline(name)
    return _pipeline_cache[name]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
create_tab, library_tab, execute_tab = st.tabs(["Create Pipeline", "Pipeline Library", "Execute Pipeline"])

# ======================== CREATE TAB ========================
with create_tab:
    section_header("// Create New Pipeline")

    # Pending step from other tabs
    if "pending_pipeline_step" in st.session_state:
        pending = st.session_state["pending_pipeline_step"]
        st.success(f"Configuration loaded: {pending['type']} -- {pending['description']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add This Step to Pipeline", type="primary", use_container_width=True):
                if "current_pipeline_steps" not in st.session_state:
                    st.session_state["current_pipeline_steps"] = []
                step_number = len(st.session_state["current_pipeline_steps"]) + 1
                st.session_state["current_pipeline_steps"].append({
                    "step_number": step_number,
                    "type": pending["type"],
                    "description": pending["description"],
                    "config": pending["config"],
                })
                del st.session_state["pending_pipeline_step"]
                st.rerun()
        with c2:
            if st.button("Discard", use_container_width=True):
                del st.session_state["pending_pipeline_step"]
                st.rerun()

    # Clear form keys BEFORE widgets are instantiated (avoids StreamlitAPIException)
    if st.session_state.pop("_reset_pipeline_form", False):
        for _k in ("new_pipeline_name", "new_pipeline_desc", "current_pipeline_steps"):
            st.session_state.pop(_k, None)

    # Pipeline metadata
    col1, col2 = st.columns(2)
    with col1:
        pipeline_name = st.text_input("Pipeline Name", placeholder="e.g., Banking Customer Data Quality", key="new_pipeline_name")
    with col2:
        pipeline_desc = st.text_input("Description", placeholder="Brief description", key="new_pipeline_desc")

    if "current_pipeline_steps" not in st.session_state:
        st.session_state["current_pipeline_steps"] = []

    # Add step interface
    with st.expander("Add Step", expanded=True):
        _STEP_LABELS = {
            "validate":          "Validation — rule-based issue detection",
            "clean":             "Cleaning — whitespace, case, punctuation",
            "corpus_standardize":"Corpus Standardisation — canonical value mapping",
            "deduplicate":       "Deduplication — fuzzy duplicate removal",
            "trim":              "Column Trimming — drop low-value columns",
            "ai_enrichment":     "AI Enrichment — smart rules, triage, executive summary",
            "pii_detection":     "PII Detection — scan for personal data",
        }

        step_type = st.selectbox(
            "Step Type",
            list(_STEP_LABELS.keys()),
            format_func=lambda x: _STEP_LABELS[x],
            key="step_type_select",
        )

        step_desc = st.text_input("Step Description", key="step_desc_input")

        step_config = {}

        if step_type == "validate":
            apply_fixes = st.checkbox("Apply fixes automatically", value=True)
            step_config = {"apply_fixes": apply_fixes}

        elif step_type == "clean":
            c1, c2, c3 = st.columns(3)
            with c1:
                lowercase = st.checkbox("Lowercase", value=True)
            with c2:
                strip_ws = st.checkbox("Strip whitespace", value=True)
            with c3:
                norm_punct = st.checkbox("Normalize punctuation", value=True)
            step_config = {"lowercase": lowercase, "strip_whitespace": strip_ws, "normalize_punctuation": norm_punct}

        elif step_type == "corpus_standardize":
            st.caption("Corpus mappings will be configured during execution based on available corpora.")
            step_config = {"mappings": {}}

        elif step_type == "deduplicate":
            c1, c2 = st.columns(2)
            with c1:
                dup_threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.8, 0.05)
            with c2:
                merge_strategy = st.selectbox("Merge strategy", ["fill_nulls", "keep_master", "most_common"])
            step_config = {"columns": [], "threshold": dup_threshold, "merge_strategy": merge_strategy}

        elif step_type == "trim":
            c1, c2 = st.columns(2)
            with c1:
                low_info = st.slider("Low info threshold", 0.0, 0.1, 0.01, 0.01)
            with c2:
                high_missing = st.slider("High missing threshold", 0.5, 1.0, 0.8, 0.05)
            step_config = {"low_info_threshold": low_info, "high_missing_threshold": high_missing, "protected_columns": []}

        elif step_type == "ai_enrichment":
            provider_type = st.session_state.get("ai_provider_type", "anthropic")
            st.caption(f"Uses AI provider from Settings: **{provider_type}**")
            if provider_type == "anthropic" and not st.session_state.get("anthropic_api_key"):
                st.warning("No Anthropic API key set — go to Settings first.")
            step_config = {"provider_type": provider_type}

        elif step_type == "pii_detection":
            pii_threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.7, 0.05, key="pii_step_threshold")
            st.caption("Scans all text columns for personal data (names, emails, phone numbers, etc.)")
            step_config = {"threshold": pii_threshold}

        if st.button("Add Step to Pipeline", type="primary"):
            st.session_state["current_pipeline_steps"].append({
                "step_number": len(st.session_state["current_pipeline_steps"]) + 1,
                "type": step_type,
                "description": step_desc or f"{step_type.title()} step",
                "config": step_config,
                "enabled": True,
            })
            st.rerun()

    # Display current steps
    steps = st.session_state["current_pipeline_steps"]
    if steps:
        section_header("// Current Pipeline Steps")
        for idx, step in enumerate(steps):
            c1, c2, c3, c4 = st.columns([0.5, 2, 2, 1])
            with c1:
                st.markdown(f"**{step['step_number']}.**")
            with c2:
                st.markdown(f"**{step['type'].title()}**")
            with c3:
                st.markdown(step["description"])
            with c4:
                if st.button("Remove", key=f"delete_step_{idx}"):
                    steps.pop(idx)
                    for i, s in enumerate(steps):
                        s["step_number"] = i + 1
                    st.rerun()

        if st.button("Save Pipeline", type="primary", use_container_width=True):
            if not pipeline_name:
                st.error("Please provide a pipeline name")
            elif not steps:
                st.error("Please add at least one step")
            else:
                pipeline = Pipeline(name=pipeline_name, description=pipeline_desc, steps=steps)
                filepath = pm.save_pipeline(pipeline)
                st.success(f"Pipeline '{pipeline_name}' saved to {filepath}")
                st.session_state["_reset_pipeline_form"] = True
                st.rerun()
    else:
        st.info("No steps added yet. Use the Add Step section above to build your pipeline.")

# ======================== LIBRARY TAB ========================
with library_tab:
    section_header("// Saved Pipelines")

    pipelines = pm.list_pipelines()

    if not pipelines:
        st.info("No saved pipelines yet. Create one in the Create Pipeline tab.")
    else:
        st.caption(f"{len(pipelines)} pipeline(s) available")

        for pipeline_info in pipelines:
            with st.expander(pipeline_info["name"], expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"**Description:** {pipeline_info['description'] or 'No description'}\n\n"
                        f"**Steps:** {pipeline_info['steps_count']} | **Version:** {pipeline_info['version']}"
                    )
                with c2:
                    created = pipeline_info["created_at"][:10] if pipeline_info["created_at"] else "Unknown"
                    modified = pipeline_info["last_modified"][:10] if pipeline_info["last_modified"] else "Unknown"
                    st.markdown(f"**Created:** {created} | **Modified:** {modified}")

                pipeline = _cached_load(pipeline_info["name"])
                if pipeline:
                    for step in pipeline.steps:
                        st.markdown(f"{step['step_number']}. **{step['type'].title()}**: {step['description']}")

                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button("Execute", key=f"exec_{pipeline_info['name']}"):
                        st.session_state["pipeline_to_execute"] = pipeline_info["name"]
                        st.session_state["active_tab_hint"] = "execute"
                        st.info("Pipeline selected. Switch to the **Execute Pipeline** tab to run it.")
                        st.success("Go to the Execute Pipeline tab above to start execution.")
                with a2:
                    if pipeline:
                        st.download_button(
                            "Export JSON", pipeline.to_json(),
                            file_name=f"{pipeline.name}.json", mime="application/json",
                            key=f"export_{pipeline_info['name']}",
                        )
                with a3:
                    if st.button("Delete", key=f"delete_{pipeline_info['name']}"):
                        pm.delete_pipeline(pipeline_info["name"])
                        st.success(f"Deleted: {pipeline_info['name']}")
                        st.rerun()

    st.markdown("---")
    section_header("// Import Pipeline")
    import_file = st.file_uploader("Upload pipeline JSON", type=["json"], key="import_pipeline_file")
    if import_file:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(import_file.read())
                tmp_path = tmp.name
            imported = pm.import_pipeline(tmp_path)
            if imported:
                st.success(f"Imported: {imported.name}")
                st.rerun()
            else:
                st.error("Failed to import pipeline")
        except Exception as e:
            st.error(f"Error importing: {e}")

# ======================== EXECUTE TAB ========================
with execute_tab:
    section_header("// Execute Pipeline")

    if not isinstance(st.session_state.get("df_raw"), pd.DataFrame):
        st.warning("Please upload data in the Load Data page first.")
        st.stop()

    df_raw = st.session_state["df_raw"]
    st.success(f"Data loaded: {len(df_raw)} rows x {len(df_raw.columns)} columns")

    pipelines = pm.list_pipelines()
    if not pipelines:
        st.info("No pipelines available. Create one in the Create Pipeline tab.")
        st.stop()

    pipeline_names = [p["name"] for p in pipelines]

    default_idx = 0
    if "pipeline_to_execute" in st.session_state and st.session_state["pipeline_to_execute"] in pipeline_names:
        default_idx = pipeline_names.index(st.session_state["pipeline_to_execute"])

    selected_name = st.selectbox("Select Pipeline", pipeline_names, index=default_idx, key="pipeline_execute_select")
    pipeline = _cached_load(selected_name)

    if pipeline:
        st.caption(f"Description: {pipeline.description or 'No description'}")
        with st.expander("Pipeline Steps", expanded=True):
            for step in pipeline.steps:
                st.markdown(f"{step['step_number']}. **{step['type'].title()}**: {step['description']}")

        # S3 save option
        s3_bucket = st.session_state.get("s3_bucket", "")
        save_to_s3 = False
        if s3_bucket:
            save_to_s3 = st.checkbox(
                f"Save each step output to S3 ({s3_bucket})",
                value=True,
                key="pipeline_save_s3",
            )

        if st.button("Execute Pipeline", type="primary", use_container_width=True):
            executor = PipelineExecutor(st.session_state)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, step):
                progress_bar.progress(current / total)
                status_text.markdown(f"**Step {current}/{total}:** {step['description']}...")

            with st.spinner("Executing pipeline..."):
                results = executor.execute_pipeline(
                    pipeline,
                    df_raw,
                    progress_callback=progress_callback,
                    s3_bucket=s3_bucket if save_to_s3 else None,
                    s3_region=st.session_state.get("s3_region", "us-east-1"),
                )

            progress_bar.progress(1.0)

            if results["success"]:
                status_text.success(f"Pipeline complete — Run ID: `{results.get('pipeline_run_id', '')}`")
                st.session_state["df_pipeline_output"] = results["df_output"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps Executed", len(results["steps_executed"]))
                m2.metric("Input Rows", len(df_raw))
                m3.metric("Output Rows", len(results["df_output"]), delta=len(results["df_output"]) - len(df_raw))
                m4.metric("Saved to DB", "Yes" if results.get("db_saved") else "No")

                section_header("// Step Results")
                for step in results["steps_executed"]:
                    details   = step.get("details", {})
                    rows_in   = step.get("rows_in", "—")
                    rows_out  = step.get("rows_out", "—")
                    s3_uri    = step.get("s3_uri", "")
                    stype     = step["type"]
                    df_snap   = step.get("df_snapshot")
                    diff      = step.get("diff_sample", [])

                    # Header label
                    _type_label = {
                        "validate":          "Validation",
                        "clean":             "Cleaning",
                        "corpus_standardize":"Corpus Standardisation",
                        "deduplicate":       "Deduplication",
                        "trim":              "Column Trimming",
                        "ai_enrichment":     "AI Enrichment",
                        "pii_detection":     "PII Detection",
                    }.get(stype, stype.title())

                    _skipped = details.get("skipped", False)
                    _header  = (
                        f"Step {step['step_number']}: {_type_label}"
                        + (" — Skipped" if _skipped else f" — {rows_in} → {rows_out} rows")
                    )

                    with st.expander(_header, expanded=not _skipped):
                        if _skipped:
                            st.caption(details.get("reason", ""))
                        else:
                            # ── Per-step structured display ────────────────
                            if stype == "validate":
                                issues_count = details.get("issues_found", 0)
                                st.markdown(f"**{issues_count}** issues found")
                                issues_records = details.get("issues", [])
                                if issues_records:
                                    # Fixed height + no use_container_width to avoid
                                    # React #185 measurement loop inside expander.
                                    st.dataframe(
                                        pd.DataFrame(issues_records),
                                        hide_index=True,
                                        height=300,
                                    )

                            elif stype in ("clean", "corpus_standardize"):
                                changed = details.get("values_changed", 0)
                                st.markdown(f"**{changed}** cell values changed")
                                if diff:
                                    st.caption(f"Sample of changes (up to 50 shown):")
                                    st.dataframe(
                                        pd.DataFrame(diff),
                                        hide_index=True,
                                        height=250,
                                    )

                            elif stype == "deduplicate":
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Duplicate pairs", details.get("duplicates_found", 0))
                                c2.metric("Rows removed",   details.get("duplicates_removed", 0))
                                c3.metric("Clusters",       details.get("clusters", 0))
                                if details.get("sampled_to"):
                                    st.caption(f"Dataset capped at {details['sampled_to']:,} rows for performance.")

                            elif stype == "trim":
                                c1, c2 = st.columns(2)
                                c1.metric("Columns removed", details.get("columns_removed", 0))
                                c2.metric("Columns kept",    details.get("columns_kept", 0))

                            elif stype == "ai_enrichment":
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Smart Rules",      details.get("smart_rules_count", 0))
                                c2.metric("Cross-Column",     details.get("cross_column_count", 0))
                                c3.metric("Triage Items",     details.get("triage_count", 0))
                                c4.metric("AI Time",          f"{details.get('ai_time_seconds', 0)}s")

                                summary = details.get("executive_summary", "")
                                if summary:
                                    st.markdown("**Executive Summary**")
                                    st.info(summary)

                                triage = details.get("triage", [])
                                if triage:
                                    st.markdown("**Triage — Priority Action Plan**")
                                    st.dataframe(pd.DataFrame(triage), hide_index=True, height=300)

                                smart_rules = details.get("smart_rules", [])
                                if smart_rules:
                                    show_rules = st.checkbox(f"Show Smart Rules ({len(smart_rules)})", key=f"show_rules_{step['step_number']}", value=False)
                                    if show_rules:
                                        st.dataframe(pd.DataFrame(smart_rules), hide_index=True, height=300)

                                cross = details.get("cross_column_issues", [])
                                if cross:
                                    show_cross = st.checkbox(f"Show Cross-Column Issues ({len(cross)})", key=f"show_cross_{step['step_number']}", value=False)
                                    if show_cross:
                                        st.dataframe(pd.DataFrame(cross), hide_index=True, height=300)

                            elif stype == "pii_detection":
                                c1, c2 = st.columns(2)
                                c1.metric("PII Findings",      details.get("pii_findings", 0))
                                c2.metric("Flagged Columns",   len(details.get("flagged_columns", [])))
                                flagged = details.get("flagged_columns", [])
                                if flagged:
                                    st.caption("Flagged columns: " + ", ".join(flagged))
                                entity_types = details.get("entity_types", {})
                                if entity_types:
                                    st.dataframe(
                                        pd.DataFrame(
                                            [{"Entity Type": k, "Count": v}
                                             for k, v in sorted(entity_types.items(), key=lambda x: -x[1])]
                                        ),
                                        hide_index=True,
                                        height=300,
                                    )

                            # ── Data preview at this step ──────────────────
                            if df_snap is not None:
                                st.markdown("---")
                                show_preview = st.checkbox(
                                    f"Show data at this point ({len(df_snap):,} rows × {len(df_snap.columns)} cols)",
                                    key=f"preview_step_{step['step_number']}",
                                    value=False,
                                )
                                if show_preview:
                                    st.dataframe(df_snap.head(100), hide_index=True, height=400)
                                st.download_button(
                                    f"Download step {step['step_number']} output (CSV)",
                                    df_snap.to_csv(index=False).encode("utf-8"),
                                    file_name=f"step_{step['step_number']}_{stype}.csv",
                                    mime="text/csv",
                                    key=f"dl_step_{step['step_number']}",
                                )

                            if s3_uri:
                                st.caption(f"Saved to S3: `{s3_uri}`")

                disabled_steps = [s for s in pipeline.steps if not s.get("enabled", True)]
                if disabled_steps:
                    st.caption(f"Skipped (disabled): {', '.join(s['description'] or s['type'] for s in disabled_steps)}")

                st.download_button(
                    "Download Final Output (CSV)",
                    results["df_output"].to_csv(index=False).encode("utf-8"),
                    f"{selected_name}_output.csv", "text/csv",
                    use_container_width=True,
                )
            else:
                status_text.error("Pipeline execution failed")
                for error in results["errors"]:
                    st.error(f"Step {error['step_number']} ({error['type']}): {error['error']}")
