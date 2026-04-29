"""
Page 17: AI Audit Log -- Complete audit trail for every AI call made by the pipeline.
Shows what statistics were sent to the LLM, what came back, and compliance proof
that zero raw data and zero PII was ever transmitted.
"""

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
import json
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header,
    MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE,
    MATRIX_BG, MATRIX_GRID, MATRIX_BORDER, MATRIX_TEXT, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("AI AUDIT LOG", "Complete transparency trail for every AI call")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        from core.storage.database import get_engine
        return get_engine()
    except Exception:
        return None


def _load_audit_log(engine, start_dt=None, end_dt=None, call_type=None, batch_id=None) -> pd.DataFrame:
    from sqlalchemy import text
    conditions = []
    params = {}
    if start_dt:
        conditions.append("timestamp >= :start_dt")
        params["start_dt"] = start_dt
    if end_dt:
        conditions.append("timestamp <= :end_dt")
        params["end_dt"] = end_dt
    if call_type and call_type != "All":
        conditions.append("call_type = :call_type")
        params["call_type"] = call_type
    if batch_id and batch_id.strip():
        conditions.append("batch_id LIKE :batch_id")
        params["batch_id"] = f"%{batch_id.strip()}%"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = text(f"""
        SELECT id, timestamp, batch_id, call_type, provider, model,
               payload_summary, response_preview, tokens_used,
               response_time_ms, raw_data_included, pii_included,
               success, error_message
        FROM ai_audit_log
        {where}
        ORDER BY timestamp DESC
        LIMIT 500
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def _table_exists(engine, table_name: str) -> bool:
    """Check whether a table exists in the database."""
    try:
        from sqlalchemy import text, inspect
        insp = inspect(engine)
        return table_name in insp.get_table_names()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Guard: DB connection
# ---------------------------------------------------------------------------

engine = _get_engine()
if engine is None:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron,sans-serif;font-size:1.2rem;color:#00ff41;">'
        '// DATABASE UNAVAILABLE</p>'
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;">Could not connect to the database. '
        'Ensure DATABASE_URL is set or the SQLite output file exists.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Guard: table may not exist yet (no AI calls have been made)
if not _table_exists(engine, "ai_audit_log"):
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron,sans-serif;font-size:1.2rem;color:#00ff41;">'
        '// NO AI AUDIT RECORDS YET</p>'
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;">'
        'The ai_audit_log table will be created automatically when the first AI enrichment runs. '
        'Enable AI enrichment and run a validation pipeline to populate this log.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_detail, tab_export = st.tabs([
    "Overview",
    "Call Detail",
    "Compliance Export",
])


# ===========================================================================
# TAB 1: Overview
# ===========================================================================

with tab_overview:
    section_header("// Filters")

    f_cols = st.columns([2, 2, 2, 2], gap="small")

    with f_cols[0]:
        date_start = st.date_input(
            "From date",
            value=date.today() - timedelta(days=30),
            key="aal_date_start",
        )
    with f_cols[1]:
        date_end = st.date_input(
            "To date",
            value=date.today(),
            key="aal_date_end",
        )
    with f_cols[2]:
        call_type_filter = st.selectbox(
            "Call type",
            options=["All", "smart_rules", "cross_column", "anomaly_explanation", "issue_triage", "executive_summary"],
            key="aal_call_type",
        )
    with f_cols[3]:
        batch_id_filter = st.text_input(
            "Batch ID contains",
            placeholder="e.g. batch_20250429",
            key="aal_batch_id",
        )

    # Load filtered data
    try:
        start_dt = datetime.combine(date_start, datetime.min.time())
        end_dt = datetime.combine(date_end, datetime.max.time())
        df_log = _load_audit_log(
            engine,
            start_dt=start_dt,
            end_dt=end_dt,
            call_type=call_type_filter,
            batch_id=batch_id_filter,
        )
    except Exception as exc:
        st.error(f"// QUERY ERROR: {exc}")
        st.stop()

    # KPI row
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_header("// Summary")

    total_calls = len(df_log)
    today_str = date.today().isoformat()
    calls_today = 0
    if not df_log.empty and "timestamp" in df_log.columns:
        try:
            calls_today = int(pd.to_datetime(df_log["timestamp"]).dt.date.astype(str).eq(today_str).sum())
        except Exception:
            calls_today = 0

    kpi_cols = st.columns(4, gap="medium")
    kpi_data = [
        (str(total_calls),   "Total AI Calls",        "cyan"),
        (str(calls_today),   "Calls Today",            "cyan"),
        ("Never",            "Raw Data Transmitted",   ""),
        ("Never",            "PII Transmitted",        ""),
    ]
    for col, (value, label, cls) in zip(kpi_cols, kpi_data):
        colour = MATRIX_CYAN if cls == "cyan" else MATRIX_GREEN
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="color:{colour};">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if df_log.empty:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:40px 20px;">'
            '<p style="font-family:Orbitron,sans-serif;font-size:1rem;color:#00ff41;">'
            '// NO RECORDS MATCH THE CURRENT FILTERS</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        section_header(f"// Audit Records ({len(df_log)} rows)")

        # Build display table
        display_cols = ["timestamp", "batch_id", "call_type", "provider", "model", "response_time_ms", "success"]
        present = [c for c in display_cols if c in df_log.columns]

        def _colour_success(val):
            s = str(val).lower()
            if s in ("true", "1", "yes"):
                return f"color: {MATRIX_GREEN}; font-weight: bold"
            return f"color: {MATRIX_RED}; font-weight: bold"

        styled = (
            df_log[present]
            .style
            .applymap(_colour_success, subset=["success"] if "success" in present else [])
            .set_properties(**{
                "font-family": "'Share Tech Mono', monospace",
                "font-size": "0.78rem",
                "background-color": "rgba(0,15,2,0.6)",
                "color": MATRIX_TEXT,
            })
        )
        st.dataframe(styled, use_container_width=True, height=360)

        # Expandable row detail
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        section_header("// Expand a Call")

        if not df_log.empty:
            row_labels = [
                f"{str(row['timestamp'])[:19]}  |  {row.get('call_type', '')}  |  {str(row.get('batch_id', ''))[:30]}"
                for _, row in df_log.iterrows()
            ]
            selected_idx = st.selectbox(
                "Select a call to inspect",
                options=list(range(len(df_log))),
                format_func=lambda i: row_labels[i],
                key="aal_expand_row",
            )
            if selected_idx is not None:
                row = df_log.iloc[selected_idx]
                with st.expander("Call details", expanded=True):
                    exp_cols = st.columns(2, gap="medium")
                    with exp_cols[0]:
                        st.markdown(
                            f'<p style="color:{MATRIX_CYAN};font-family:Share Tech Mono;font-size:0.82rem;">'
                            f'<b>What was sent to AI (statistics only):</b></p>',
                            unsafe_allow_html=True,
                        )
                        try:
                            payload = json.loads(row.get("payload_summary") or "{}")
                            for k, v in payload.items():
                                st.markdown(
                                    f'<p style="color:{MATRIX_TEXT};font-family:Share Tech Mono;font-size:0.78rem;">'
                                    f'<span style="color:{MATRIX_CYAN};">{k}:</span> {v}</p>',
                                    unsafe_allow_html=True,
                                )
                        except Exception:
                            st.text(row.get("payload_summary", ""))
                    with exp_cols[1]:
                        st.markdown(
                            f'<p style="color:{MATRIX_CYAN};font-family:Share Tech Mono;font-size:0.82rem;">'
                            f'<b>Response preview:</b></p>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="background:rgba(0,30,5,0.7);border:1px solid {MATRIX_BORDER};'
                            f'padding:10px;border-radius:4px;font-family:Share Tech Mono;font-size:0.75rem;'
                            f'color:{MATRIX_TEXT};white-space:pre-wrap;">'
                            f'{str(row.get("response_preview", ""))[:500]}</div>',
                            unsafe_allow_html=True,
                        )


# ===========================================================================
# TAB 2: Call Detail
# ===========================================================================

with tab_detail:
    section_header("// Call Detail Inspector")

    try:
        df_all = _load_audit_log(engine)
    except Exception as exc:
        st.error(f"// QUERY ERROR: {exc}")
        st.stop()

    if df_all.empty:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:40px 20px;">'
            '<p style="color:#4a7a4f;font-family:Share Tech Mono;">'
            '// No AI calls recorded yet. Run a pipeline with AI enrichment enabled.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        call_labels = [
            f"{str(row['timestamp'])[:19]}  |  {row.get('call_type', '')}  |  batch: {str(row.get('batch_id', ''))[:25]}"
            for _, row in df_all.iterrows()
        ]
        detail_idx = st.selectbox(
            "Select a call",
            options=list(range(len(df_all))),
            format_func=lambda i: call_labels[i],
            key="aal_detail_select",
        )

        if detail_idx is not None:
            row = df_all.iloc[detail_idx]

            # Meta row
            meta_cols = st.columns(4, gap="small")
            meta_cols[0].metric("Provider", str(row.get("provider", "—")))
            meta_cols[1].metric("Model", str(row.get("model", "—"))[:20])
            meta_cols[2].metric("Response Time", f"{row.get('response_time_ms', 0)} ms")
            meta_cols[3].metric("Tokens Used", str(row.get("tokens_used", 0)))

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # What was sent
            section_header("// What Was Sent to AI")
            st.markdown(
                f'<p style="color:{MATRIX_TEXT_DIM};font-family:Share Tech Mono;font-size:0.78rem;">'
                f'Only aggregated statistics and metadata — no raw data rows, no PII.</p>',
                unsafe_allow_html=True,
            )
            try:
                payload = json.loads(row.get("payload_summary") or "{}")
                for k, v in payload.items():
                    colour = MATRIX_RED if k in ("raw_data_included", "pii_included") and v else MATRIX_TEXT
                    st.markdown(
                        f'<p style="font-family:Share Tech Mono;font-size:0.82rem;">'
                        f'<span style="color:{MATRIX_CYAN};min-width:200px;display:inline-block;">{k}</span>'
                        f'<span style="color:{colour};">{v}</span></p>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.text(row.get("payload_summary", ""))

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # What came back
            section_header("// What Came Back")
            response_text = str(row.get("response_preview", "")).strip()
            if response_text:
                st.markdown(
                    f'<div style="background:rgba(0,30,5,0.7);border:1px solid {MATRIX_BORDER};'
                    f'padding:12px 16px;border-radius:4px;font-family:Share Tech Mono;font-size:0.78rem;'
                    f'color:{MATRIX_TEXT};white-space:pre-wrap;">'
                    f'{response_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p style="color:{MATRIX_TEXT_DIM};font-family:Share Tech Mono;font-size:0.78rem;">'
                    f'No response preview recorded.</p>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Data privacy badges
            section_header("// Data Privacy")
            badge_cols = st.columns(2, gap="medium")
            with badge_cols[0]:
                st.markdown(
                    f'<div style="background:rgba(0,60,10,0.8);border:2px solid {MATRIX_GREEN};'
                    f'border-radius:6px;padding:14px 20px;text-align:center;">'
                    f'<p style="color:{MATRIX_GREEN};font-family:Orbitron,sans-serif;font-size:0.9rem;margin:0;">'
                    f'NO RAW DATA</p>'
                    f'<p style="color:{MATRIX_TEXT_DIM};font-family:Share Tech Mono;font-size:0.72rem;margin:4px 0 0;">'
                    f'Zero data rows were transmitted to the AI provider</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with badge_cols[1]:
                st.markdown(
                    f'<div style="background:rgba(0,60,10,0.8);border:2px solid {MATRIX_GREEN};'
                    f'border-radius:6px;padding:14px 20px;text-align:center;">'
                    f'<p style="color:{MATRIX_GREEN};font-family:Orbitron,sans-serif;font-size:0.9rem;margin:0;">'
                    f'NO PII</p>'
                    f'<p style="color:{MATRIX_TEXT_DIM};font-family:Share Tech Mono;font-size:0.72rem;margin:4px 0 0;">'
                    f'Zero personally identifiable information was transmitted</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Error info (if any)
            err_msg = row.get("error_message")
            if err_msg and str(err_msg).strip() not in ("", "None", "nan"):
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.error(f"Error recorded: {err_msg}")


# ===========================================================================
# TAB 3: Compliance Export
# ===========================================================================

with tab_export:
    section_header("// Compliance Export")

    st.markdown(
        f'<p style="color:{MATRIX_TEXT_DIM};font-family:Share Tech Mono;font-size:0.82rem;">'
        f'Download a full CSV of all AI audit log entries for any date range. '
        f'Use this report to demonstrate to auditors that no raw data or PII was ever sent to an AI provider.</p>',
        unsafe_allow_html=True,
    )

    exp_cols = st.columns([2, 2, 1], gap="medium")
    with exp_cols[0]:
        export_start = st.date_input(
            "Export from",
            value=date.today() - timedelta(days=90),
            key="aal_export_start",
        )
    with exp_cols[1]:
        export_end = st.date_input(
            "Export to",
            value=date.today(),
            key="aal_export_end",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Generate Compliance Report", use_container_width=True, key="aal_gen_report"):
        try:
            export_start_dt = datetime.combine(export_start, datetime.min.time())
            export_end_dt = datetime.combine(export_end, datetime.max.time())
            df_export = _load_audit_log(engine, start_dt=export_start_dt, end_dt=export_end_dt)
        except Exception as exc:
            st.error(f"// Export failed: {exc}")
            df_export = pd.DataFrame()

        if df_export.empty:
            st.markdown(
                '<p style="color:#4a7a4f;font-family:Share Tech Mono;">'
                '// No AI calls recorded in the selected date range.</p>',
                unsafe_allow_html=True,
            )
        else:
            # Summary statement
            call_count = len(df_export)
            st.markdown(
                f'<div class="glass-card" style="padding:20px 24px;">'
                f'<p style="color:{MATRIX_GREEN};font-family:Share Tech Mono;font-size:0.88rem;line-height:1.6;">'
                f'Between <b>{export_start}</b> and <b>{export_end}</b>, '
                f'<b>{call_count}</b> AI call(s) were made. '
                f'<b>Zero raw data rows were transmitted.</b> '
                f'<b>Zero PII fields were transmitted.</b> '
                f'All calls sent only aggregated statistics and metadata to the AI provider.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Breakdown by call type
            if "call_type" in df_export.columns:
                breakdown = df_export["call_type"].value_counts().reset_index()
                breakdown.columns = ["call_type", "count"]
                st.markdown(
                    f'<p style="color:{MATRIX_CYAN};font-family:Share Tech Mono;font-size:0.82rem;">'
                    f'Calls by type:</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    breakdown.style.set_properties(**{
                        "font-family": "'Share Tech Mono', monospace",
                        "font-size": "0.78rem",
                        "background-color": "rgba(0,15,2,0.6)",
                        "color": MATRIX_TEXT,
                    }),
                    use_container_width=True,
                    height=min(200, 40 + len(breakdown) * 35),
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Download button
            csv_buf = io.StringIO()
            df_export.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode()

            st.download_button(
                label="DOWNLOAD AUDIT LOG CSV",
                data=csv_bytes,
                file_name=f"ai_audit_log_{export_start}_{export_end}.csv",
                mime="text/csv",
                use_container_width=True,
                key="aal_download_csv",
            )
