"""
Page 6: Cleaning -- Apply corrections, standardise, normalise formats.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, terminal_block, workflow_breadcrumb
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("CLEANING & FIXES", "Apply corrections and standardise data")

df_raw = st.session_state.get("df_raw")
report = st.session_state.get("unified_issues_report")

workflow_breadcrumb([
    ("Load Data", df_raw is not None),
    ("Validate", st.session_state.get("validation_completed", False)),
    ("Cleaning", st.session_state.get("df_corrected") is not None),
    ("Dedupe", st.session_state.get("df_deduplicated") is not None),
])

if df_raw is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#5a9a5a;'>Go to Load Data first.</span>")
    st.stop()

# ---------------------------------------------------------------------------
# Basic text cleaning
# ---------------------------------------------------------------------------
section_header("// Basic Text Cleaning")

st.markdown(
    '<p style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;">'
    'Strip whitespace, fix encoding, normalise case.</p>',
    unsafe_allow_html=True,
)

text_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
selected_cols = st.multiselect("Select columns to clean", text_cols, default=text_cols[:5], key="clean_cols")

col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    strip_ws = st.checkbox("Strip whitespace", value=True, key="clean_strip")
    lower_case = st.checkbox("Lowercase", value=False, key="clean_lower")
with col_opts2:
    remove_dups_ws = st.checkbox("Remove double spaces", value=True, key="clean_dups")
    trim_special = st.checkbox("Remove leading/trailing special chars", value=False, key="clean_special")

if st.button("APPLY CLEANING", key="apply_clean", use_container_width=True):
    df_cleaned = df_raw.copy()
    for col in selected_cols:
        if col in df_cleaned.columns:
            s = df_cleaned[col].astype(str)
            if strip_ws:
                s = s.str.strip()
            if lower_case:
                s = s.str.lower()
            if remove_dups_ws:
                s = s.str.replace(r'\s+', ' ', regex=True)
            if trim_special:
                s = s.str.replace(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', regex=True)
            df_cleaned[col] = s

    st.session_state["df_cleaned"] = df_cleaned
    st.success(f"Cleaned {len(selected_cols)} columns across {len(df_cleaned)} rows")

# ---------------------------------------------------------------------------
# Review & Approve Fixes  (human-in-the-loop gate)
# ---------------------------------------------------------------------------
if report is not None and isinstance(report, pd.DataFrame) and not report.empty:
    section_header("// Review & Approve Fixes")

    st.markdown(
        '<div style="background:rgba(255,145,0,0.07);border:1px solid rgba(255,145,0,0.3);'
        'border-radius:8px;padding:12px 16px;font-family:Share Tech Mono;font-size:0.78rem;'
        'color:#ff9100;margin-bottom:16px;">'
        'ORIGINAL DATA IS NEVER MODIFIED — all corrections are written to a separate copy. '
        'Review and approve each fix individually before anything is applied.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Resolve suggestion column — handle both naming conventions
    _sug_col = next(
        (c for c in ["suggested_fix", "suggestion"] if c in report.columns),
        None,
    )
    _conf_col = next(
        (c for c in ["confidence", "score"] if c in report.columns),
        None,
    )

    if _sug_col is None:
        terminal_block("// NO FIX SUGGESTIONS AVAILABLE<br><span style='color:#5a9a5a;'>Run validation with rules or corpus enabled to generate suggestions.</span>")
    else:
        fixable = report[
            report[_sug_col].notna() & (report[_sug_col].astype(str).str.strip() != "")
        ].copy().reset_index(drop=True)

        if fixable.empty:
            terminal_block("// NO FIXABLE ISSUES FOUND<br><span style='color:#5a9a5a;'>All issues detected require manual review — no automatic suggestions available.</span>")
        else:
            st.markdown(
                f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
                f'{len(fixable)} suggested fixes out of {len(report)} total issues.</p>',
                unsafe_allow_html=True,
            )

            # Build review dataframe
            review = pd.DataFrame()
            review["Approve"] = True  # default all approved
            review["Row #"] = fixable.get("row_id", pd.Series(range(len(fixable)))).astype(str)
            review["Column"] = fixable.get("column", fixable.get("column_name", "")).astype(str)
            review["Issue Type"] = fixable.get("issue", fixable.get("issue_type", "")).astype(str)
            review["Severity"] = fixable.get("severity", "").astype(str)

            # Current value — look up from df_raw
            def _current_val(r):
                try:
                    rid = int(r["Row #"])
                    col = r["Column"]
                    if col in df_raw.columns and 0 <= rid < len(df_raw):
                        return str(df_raw.at[rid, col])
                except Exception:
                    pass
                return ""
            review["Current Value"] = review.apply(_current_val, axis=1)
            review["Suggested Fix"] = fixable[_sug_col].astype(str)

            if _conf_col:
                review["Confidence"] = fixable[_conf_col].apply(
                    lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "—"
                )

            # ── Bulk selection buttons ──────────────────────────────────────
            btn_col1, btn_col2, btn_col3 = st.columns(3, gap="small")
            with btn_col1:
                if st.button("SELECT ALL", key="fix_select_all", use_container_width=True):
                    st.session_state["_fix_approve_all"] = True
                    st.session_state["_fix_approve_none"] = False
                    st.session_state["_fix_approve_highconf"] = False
            with btn_col2:
                if st.button("HIGH CONFIDENCE ONLY (≥90%)", key="fix_select_hc", use_container_width=True):
                    st.session_state["_fix_approve_highconf"] = True
                    st.session_state["_fix_approve_all"] = False
                    st.session_state["_fix_approve_none"] = False
            with btn_col3:
                if st.button("DESELECT ALL", key="fix_deselect_all", use_container_width=True):
                    st.session_state["_fix_approve_none"] = True
                    st.session_state["_fix_approve_all"] = False
                    st.session_state["_fix_approve_highconf"] = False

            # Apply bulk selection to Approve column
            if st.session_state.get("_fix_approve_all"):
                review["Approve"] = True
            elif st.session_state.get("_fix_approve_none"):
                review["Approve"] = False
            elif st.session_state.get("_fix_approve_highconf") and _conf_col:
                review["Approve"] = fixable[_conf_col].apply(
                    lambda x: float(x) >= 0.9 if pd.notna(x) else False
                ).values

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Editable review table ───────────────────────────────────────
            st.markdown(
                '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.72rem;">'
                'Toggle the Approve checkbox on each row to include or exclude it from the fix batch.</p>',
                unsafe_allow_html=True,
            )

            edited = st.data_editor(
                review,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Approve": st.column_config.CheckboxColumn("Approve", width="small"),
                    "Row #": st.column_config.TextColumn("Row #", width="small"),
                    "Column": st.column_config.TextColumn("Column", width="small"),
                    "Issue Type": st.column_config.TextColumn("Issue Type", width="medium"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                    "Current Value": st.column_config.TextColumn("Current Value", width="medium"),
                    "Suggested Fix": st.column_config.TextColumn("Suggested Fix", width="medium"),
                },
                disabled=["Row #", "Column", "Issue Type", "Severity", "Current Value", "Suggested Fix"],
                key="fix_review_table",
            )

            approved_rows = edited[edited["Approve"] == True]
            st.markdown(
                f'<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;margin-top:8px;">'
                f'{len(approved_rows)} of {len(review)} fixes approved.</p>',
                unsafe_allow_html=True,
            )

            # ── Apply button ────────────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button(
                f"APPLY {len(approved_rows)} APPROVED FIXES",
                key="apply_approved_fixes",
                use_container_width=True,
                disabled=len(approved_rows) == 0,
                type="primary",
            ):
                _df_cleaned = st.session_state.get("df_cleaned")
                df_fixed = (_df_cleaned if _df_cleaned is not None else df_raw).copy()
                applied = 0
                skipped = 0
                for _, row in approved_rows.iterrows():
                    try:
                        rid = int(row["Row #"])
                        col = str(row["Column"])
                        fix = str(row["Suggested Fix"])
                        if col in df_fixed.columns and 0 <= rid < len(df_fixed) and fix:
                            df_fixed.at[rid, col] = fix
                            applied += 1
                    except Exception:
                        skipped += 1
                        continue

                st.session_state["df_corrected"] = df_fixed
                # Clear bulk selection flags after apply
                for flag in ["_fix_approve_all", "_fix_approve_none", "_fix_approve_highconf"]:
                    st.session_state.pop(flag, None)

                st.success(
                    f"{applied} fixes applied to corrected dataset. "
                    f"Original data unchanged. "
                    + (f"{skipped} skipped due to errors." if skipped else "")
                )

                # Show before/after comparison for approved columns
                changed_cols = approved_rows["Column"].unique().tolist()
                if changed_cols:
                    st.markdown(
                        '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;margin-top:12px;">'
                        'Before / After comparison (affected columns):</p>',
                        unsafe_allow_html=True,
                    )
                    compare = pd.DataFrame({
                        "Row #": approved_rows["Row #"].values,
                        "Column": approved_rows["Column"].values,
                        "Before": approved_rows["Current Value"].values,
                        "After": approved_rows["Suggested Fix"].values,
                    })
                    st.dataframe(compare, use_container_width=True, hide_index=True, height=min(300, 40 + 35 * len(compare)))

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
section_header("// Export")

_df_corrected = st.session_state.get("df_corrected")
_df_cleaned   = st.session_state.get("df_cleaned")
export_df = _df_corrected if _df_corrected is not None else _df_cleaned
if export_df is not None:
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("DOWNLOAD CLEANED DATA [CSV]", data=csv, file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.dataframe(export_df.head(50), use_container_width=True, hide_index=False, height=300)

    # ── S3 write-back (human approval gated) ──────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("// WRITE APPROVED DATASET TO S3", expanded=False):
        from core.s3_writeback import boto3_available, get_default_config, write_df_to_s3
        if not boto3_available():
            st.warning("boto3 is not installed in this environment, so S3 write-back is unavailable.")
        else:
            _cfg = get_default_config()
            st.caption(
                "Persist the cleaned dataset to S3. This requires explicit human "
                "approval below and uploads only when you click the button."
            )
            _c1, _c2 = st.columns(2)
            with _c1:
                _s3_bucket = st.text_input(
                    "S3 bucket", value=_cfg["bucket"],
                    placeholder="my-data-quality-bucket",
                    help="Defaults to the S3_WRITEBACK_BUCKET environment variable.",
                    key="s3_wb_bucket",
                )
            with _c2:
                _s3_prefix = st.text_input(
                    "Key prefix", value=_cfg["prefix"],
                    help="A timestamped filename is appended automatically.",
                    key="s3_wb_prefix",
                )
            _s3_filename = st.text_input("Base filename", value="cleaned_data.csv", key="s3_wb_filename")
            _approved = st.checkbox(
                "I have reviewed the cleaned dataset and approve writing it back to S3.",
                value=False, key="s3_wb_approved",
            )
            if st.button(
                "APPROVE & WRITE TO S3", type="primary",
                disabled=not _approved, use_container_width=True, key="s3_wb_button",
                help="Enabled only after you tick the approval checkbox above.",
            ):
                if not (_s3_bucket and _s3_bucket.strip()):
                    st.error("Enter an S3 bucket (or set S3_WRITEBACK_BUCKET).")
                else:
                    with st.spinner(f"Uploading {len(export_df)} rows to S3..."):
                        _res = write_df_to_s3(
                            export_df, bucket=_s3_bucket.strip(),
                            prefix=_s3_prefix, filename=_s3_filename or "cleaned_data.csv",
                            region=_cfg["region"] or None,
                            metadata={"rows": len(export_df), "columns": len(export_df.columns)},
                        )
                    if _res.get("success"):
                        st.success(f"Written to {_res['uri']}")
                        st.caption(f"{_res['rows']:,} rows · {_res['bytes']:,} bytes")
                    else:
                        st.error(f"Write-back failed: {_res.get('error')}")
else:
    terminal_block("// NO CLEANED DATA YET<br><span style='color:#5a9a5a;'>Apply cleaning or fixes above.</span>")
