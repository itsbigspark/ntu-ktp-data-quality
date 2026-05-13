"""
Page 15: Batch History -- All past validation runs with download.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
import pandas as pd
import streamlit as st
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header, aggrid_plain,
    MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE,
    MATRIX_BG, MATRIX_GRID, MATRIX_BORDER, MATRIX_TEXT, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("BATCH HISTORY", "All past validation runs")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        from core.storage.database import get_engine
        return get_engine()
    except Exception:
        return None


def _load_batches(engine) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT batch_id, timestamp, source_file, rows_processed,
               overall_score, pass, issues_count,
               completeness, uniqueness, consistency,
               validity, accuracy, timeliness
        FROM batch_runs
        ORDER BY timestamp DESC
        LIMIT 100
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def _load_issues(engine, batch_id: str) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("SELECT * FROM issues WHERE batch_id = :bid")
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"bid": batch_id})


def _score_colour(score: float) -> str:
    if score >= 85:
        return MATRIX_GREEN
    if score >= 70:
        return MATRIX_ORANGE
    return MATRIX_RED


# ---------------------------------------------------------------------------
# Guard: DB
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

try:
    batches = _load_batches(engine)
except Exception as exc:
    st.error(f"// QUERY ERROR: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if batches.empty:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron,sans-serif;font-size:1.2rem;color:#00ff41;">'
        '// NO BATCH RECORDS FOUND</p>'
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;">'
        'Run a validation pipeline to record your first batch.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------

section_header("// Batch Summary")

total_batches = len(batches)
avg_score = batches["overall_score"].mean()
pass_col = batches["pass"].astype(bool)
pass_rate = pass_col.mean() * 100

score_class = "danger" if avg_score < 70 else ("warn" if avg_score < 85 else "")
pass_class = "danger" if pass_rate < 70 else ("warn" if pass_rate < 85 else "")

kpi_cols = st.columns(3, gap="medium")
kpi_data = [
    (str(total_batches), "Total Batches", "cyan"),
    (f"{avg_score:.1f}%", "Average Score", score_class),
    (f"{pass_rate:.0f}%", "Pass Rate", pass_class),
]
for col, (value, label, cls) in zip(kpi_cols, kpi_data):
    val_cls = f" {cls}" if cls else ""
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value{val_cls}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

section_header("// Filter Batches")

filter_cols = st.columns([2, 2, 1], gap="medium")
with filter_cols[0]:
    search_file = st.text_input(
        "Source file contains",
        placeholder="e.g. companies_house",
        key="bh_search_file",
    )
with filter_cols[1]:
    status_filter = st.selectbox(
        "Status",
        options=["All", "Pass", "Fail"],
        key="bh_status_filter",
    )
with filter_cols[2]:
    min_score = st.number_input(
        "Min score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        key="bh_min_score",
    )

# Apply filters
display_df = batches.copy()
if search_file:
    display_df = display_df[
        display_df["source_file"].fillna("").str.contains(search_file, case=False)
    ]
if status_filter == "Pass":
    display_df = display_df[display_df["pass"].astype(bool)]
elif status_filter == "Fail":
    display_df = display_df[~display_df["pass"].astype(bool)]
if min_score > 0:
    display_df = display_df[display_df["overall_score"] >= min_score]

st.markdown(
    f'<p style="color:{MATRIX_CYAN};font-family:Share Tech Mono;font-size:0.78rem;">'
    f'Showing {len(display_df)} of {total_batches} batches</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Colour-coded dataframe
# ---------------------------------------------------------------------------

section_header("// Batch Records")

DIMS = ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]

col_display = [
    "batch_id", "timestamp", "source_file", "rows_processed",
    "overall_score", "pass", "issues_count",
] + DIMS

present_cols = [c for c in col_display if c in display_df.columns]
aggrid_plain(display_df[present_cols], height=420)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Per-batch issue download
# ---------------------------------------------------------------------------

section_header("// Download Issues by Batch")

if display_df.empty:
    st.info("No batches match the current filters.")
else:
    batch_ids = display_df["batch_id"].tolist()
    selected_batch = st.selectbox(
        "Select a batch to download its issues",
        options=batch_ids,
        format_func=lambda bid: (
            f"{bid}  |  "
            + str(display_df.loc[display_df["batch_id"] == bid, "timestamp"].iloc[0])
            + "  |  "
            + str(display_df.loc[display_df["batch_id"] == bid, "source_file"].iloc[0])
        ),
        key="bh_select_batch",
    )

    if selected_batch:
        row = display_df[display_df["batch_id"] == selected_batch].iloc[0]
        info_cols = st.columns(4, gap="small")
        info_cols[0].metric("Score", f"{row['overall_score']:.1f}%")
        info_cols[1].metric("Rows", str(row["rows_processed"]))
        info_cols[2].metric("Issues", str(row["issues_count"]))
        info_cols[3].metric("Status", "PASS" if bool(row["pass"]) else "FAIL")

        try:
            issues_df = _load_issues(engine, selected_batch)
        except Exception as exc:
            st.error(f"// Could not load issues: {exc}")
            issues_df = pd.DataFrame()

        if issues_df.empty:
            st.markdown(
                '<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.82rem;">'
                '// No issues recorded for this batch.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="color:{MATRIX_CYAN};font-family:Share Tech Mono;font-size:0.78rem;">'
                f'{len(issues_df)} issues found in this batch</p>',
                unsafe_allow_html=True,
            )

            csv_buf = io.StringIO()
            issues_df.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode()

            st.download_button(
                label="DOWNLOAD ISSUES CSV",
                data=csv_bytes,
                file_name=f"issues_{selected_batch}.csv",
                mime="text/csv",
                use_container_width=True,
                key="bh_download_issues",
            )

            with st.expander("Preview issues (first 50 rows)"):
                aggrid_plain(issues_df.head(50), height=350)

# ---------------------------------------------------------------------------
# Export all visible batches
# ---------------------------------------------------------------------------

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
section_header("// Export Batch Table")

if not display_df.empty:
    csv_buf2 = io.StringIO()
    display_df.to_csv(csv_buf2, index=False)
    st.download_button(
        label="DOWNLOAD ALL VISIBLE BATCHES (CSV)",
        data=csv_buf2.getvalue().encode(),
        file_name="batch_history_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="bh_download_batches",
    )
