"""
Page 5: Command Center -- Matrix-themed monitoring dashboard.

Reads from SQLite database. No business logic -- pure visualization.
This is the dashboard.py content integrated as a multi-page app page.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header,
    MATRIX_GREEN, MATRIX_GREEN_DIM, MATRIX_CYAN, MATRIX_RED,
    MATRIX_ORANGE, MATRIX_YELLOW, MATRIX_BG, MATRIX_GRID,
    MATRIX_BORDER, MATRIX_TEXT, MATRIX_TEXT_DIM,
    SEVERITY_COLORS, SEVERITY_BG, severity_dot, heatmap_styler,
)
from shared.state import init_state

init_state()
apply_theme(show_rain=True)
require_auth()

# ---------------------------------------------------------------------------
# Database config
# ---------------------------------------------------------------------------
DB_CONFIG = {"output": {"database": {"engine": "sqlite", "path": "./output/dq_investigator.db"}}}


def _db_available() -> bool:
    # On ECS, DATABASE_URL env var is set → PostgreSQL is always available
    if os.environ.get("DATABASE_URL"):
        return True
    # Local dev: fall back to checking SQLite file
    return os.path.isfile(DB_CONFIG["output"]["database"]["path"])


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert #RRGGBB (or #RGB) to rgba(r,g,b,a) — Plotly fillcolor compatible."""
    h = (hex_color or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return f"rgba(0,255,65,{alpha})"
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return f"rgba(0,255,65,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def _chart_layout(height=350, show_legend=False):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0, 10, 2, 0.6)",
        font=dict(color=MATRIX_TEXT, family="'Courier New', monospace", size=12),
        xaxis=dict(gridcolor=MATRIX_GRID, linecolor=MATRIX_BORDER, zerolinecolor=MATRIX_GRID, tickfont=dict(color=MATRIX_TEXT_DIM)),
        yaxis=dict(gridcolor=MATRIX_GRID, linecolor=MATRIX_BORDER, zerolinecolor=MATRIX_GRID, tickfont=dict(color=MATRIX_TEXT_DIM)),
        margin=dict(l=50, r=20, t=40, b=50),
        height=height,
        showlegend=show_legend,
        hoverlabel=dict(bgcolor="rgba(0,15,2,0.95)", bordercolor=MATRIX_GREEN, font=dict(color=MATRIX_GREEN, family="'Courier New', monospace")),
    )


# ---------------------------------------------------------------------------
# Guard: database
# ---------------------------------------------------------------------------
page_header("COMMAND CENTER", "Enterprise Monitoring Dashboard")

if not _db_available():
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron,sans-serif;font-size:1.2rem;color:#00ff41;">'
        '// NO DATA FEED DETECTED</p>'
        '<p style="color:#5a9a5a;font-family:Share Tech Mono;">Run a validation pipeline to populate the database.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
from core.storage.database import (
    get_recent_batches, get_batch_issues, get_score_trend, get_batch_audit,
    get_ai_smart_rules, get_ai_cross_column, get_ai_explanations,
    get_ai_triage, get_ai_executive_summary, get_batches_with_ai,
    get_engine,
)

try:
    batches = get_recent_batches(DB_CONFIG, limit=100)
    trend = get_score_trend(DB_CONFIG, limit=100)
except Exception:
    st.error("Could not load dashboard data. The database may be initialising — try again in a moment.")
    st.stop()

if batches.empty:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron;font-size:1.2rem;color:#00ff41;">// DATABASE EMPTY</p>'
        '<p style="color:#5a9a5a;font-family:Share Tech Mono;">Execute pipeline to generate data stream.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

latest_batch_id = batches.iloc[0]["batch_id"]

# Real pipeline run time from audit trail
try:
    _audit_latest = get_batch_audit(DB_CONFIG, latest_batch_id)
    _total_ms = _audit_latest["duration_ms"].sum() if not _audit_latest.empty else 0
    _run_time_str = f"{_total_ms / 1000:.1f}s" if _total_ms else "N/A"
except Exception:
    _run_time_str = "N/A"

# ---------------------------------------------------------------------------
# Tab structure
# ---------------------------------------------------------------------------
dash_tab, research_tab = st.tabs(["Dashboard", "Investigate"])

# ===========================================================================
# DASHBOARD TAB
# ===========================================================================
with dash_tab:

    # Status bar
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f'<div style="background:rgba(0,15,2,0.6);border:1px solid rgba(0,255,65,0.15);border-radius:4px;'
        f'padding:6px 14px;margin-bottom:12px;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff41;'
        f'display:flex;gap:20px;align-items:center;">'
        f'<span><span style="display:inline-block;width:7px;height:7px;background:#00ff41;border-radius:50%;'
        f'box-shadow:0 0 8px #00ff41;margin-right:6px;"></span>SYSTEM ONLINE</span>'
        f'<span>LAST SCAN: {batches.iloc[0].get("timestamp", "N/A")}</span>'
        f'<span>BATCHES: {len(batches)}</span>'
        f'<span>LOCAL TIME: {now}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Drill-down breadcrumb (feature E)
    # State keys: cc_drill_batch, cc_drill_column, cc_drill_issue_row_id
    # Cleared by the "All Batches" link.
    # -----------------------------------------------------------------------
    _drill_batch  = st.session_state.get("cc_drill_batch")
    _drill_column = st.session_state.get("cc_drill_column")
    _drill_issue  = st.session_state.get("cc_drill_issue_row_id")

    def _clear_drill(reset_keys):
        for k in reset_keys:
            st.session_state.pop(k, None)

    bc_cols = st.columns([6, 1])
    with bc_cols[0]:
        crumbs = ['<span style="color:#5a9a5a;">All Batches</span>']
        if _drill_batch:
            crumbs[0] = '<a href="#" style="color:#00e5ff;text-decoration:none;">All Batches</a>'
            crumbs.append(f'<span style="color:#b0ffb8;">{_drill_batch}</span>')
        if _drill_column:
            crumbs.append(f'<span style="color:#00ff41;">{_drill_column}</span>')
        if _drill_issue is not None:
            crumbs.append(f'<span style="color:#ff9100;">row #{_drill_issue}</span>')

        st.markdown(
            '<div style="font-family:Share Tech Mono;font-size:0.78rem;'
            'padding:8px 14px;background:rgba(0,5,1,0.6);border:1px solid rgba(0,229,255,0.15);'
            'border-radius:4px;margin-bottom:16px;">'
            + ' <span style="color:#5a9a5a;">›</span> '.join(crumbs)
            + '</div>',
            unsafe_allow_html=True,
        )
    with bc_cols[1]:
        if _drill_batch or _drill_column or _drill_issue is not None:
            if st.button("← Back to all", key="cc_clear_drill", use_container_width=True):
                _clear_drill(["cc_drill_batch", "cc_drill_column", "cc_drill_issue_row_id"])
                st.rerun()

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    total_batches = len(batches)
    avg_score = batches["overall_score"].mean()
    pass_rate = batches["pass"].mean() * 100
    total_issues = int(batches["issues_count"].sum())

    score_class = "danger" if avg_score < 70 else ("warn" if avg_score < 85 else "")
    pass_class = "danger" if pass_rate < 70 else ("warn" if pass_rate < 85 else "")

    kpi_cols = st.columns(5, gap="medium")
    kpi_data = [
        (str(total_batches), "Total Batches", "cyan"),
        (f"{avg_score:.1f}%", "Avg Quality Score", score_class),
        (f"{pass_rate:.0f}%", "Pass Rate", pass_class),
        (str(total_issues), "Total Issues", "warn" if total_issues > 50 else ""),
        (_run_time_str, "Latest Run Time", "cyan"),
    ]
    for col, (value, label, cls) in zip(kpi_cols, kpi_data):
        val_class = f' {cls}' if cls else ''
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value{val_class}">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Quality Score Trend
    # -----------------------------------------------------------------------
    section_header("// Signal Trace : Quality Score Trend")

    if not trend.empty:
        layout = _chart_layout(height=320)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["timestamp"], y=trend["overall_score"],
            mode="lines", line=dict(color=MATRIX_GREEN, width=0),
            fill="tozeroy", fillcolor="rgba(0,255,65,0.06)", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=trend["timestamp"], y=trend["overall_score"],
            mode="lines+markers", line=dict(color=MATRIX_GREEN, width=2.5, shape="spline"),
            marker=dict(size=8, color=MATRIX_BG, line=dict(color=MATRIX_GREEN, width=2)),
            name="Quality Score", hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
        ))
        fig.add_hline(y=85, line_dash="dot", line_color=MATRIX_RED, line_width=1.5,
                      annotation_text="THRESHOLD 85%",
                      annotation_font=dict(color=MATRIX_RED, size=10, family="Share Tech Mono"))
        layout["yaxis"]["title"] = dict(text="Score (%)", font=dict(color=MATRIX_TEXT_DIM, size=11))
        fig.update_layout(**layout)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Radar + Severity
    # -----------------------------------------------------------------------
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        section_header("// Dimension Scan : Latest Batch")
        dims = ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]
        latest = batches.iloc[0]
        dim_values = [float(latest.get(d, 0)) for d in dims]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=dim_values + [dim_values[0]], theta=[d.upper() for d in dims] + [dims[0].upper()],
            fill="toself", fillcolor="rgba(0,255,65,0.08)",
            line=dict(color=MATRIX_GREEN, width=2),
            marker=dict(size=7, color=MATRIX_GREEN, line=dict(color=MATRIX_BG, width=1)),
            hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[85]*7, theta=[d.upper() for d in dims] + [dims[0].upper()],
            mode="lines", line=dict(color=MATRIX_RED, width=1, dash="dot"), showlegend=False, hoverinfo="skip",
        ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MATRIX_TEXT, family="'Share Tech Mono', monospace", size=11),
            height=380,
            polar=dict(
                bgcolor="rgba(0, 10, 2, 0.5)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(0,255,65,0.08)", tickfont=dict(color=MATRIX_TEXT_DIM, size=9)),
                angularaxis=dict(gridcolor="rgba(0,255,65,0.1)", tickfont=dict(color=MATRIX_GREEN, size=10, family="Share Tech Mono")),
            ),
            showlegend=False, margin=dict(l=60, r=60, t=30, b=30),
            hoverlabel=dict(bgcolor="rgba(0,15,2,0.95)", bordercolor=MATRIX_GREEN, font=dict(color=MATRIX_GREEN)),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        section_header("// Threat Level : Issues by Severity")
        try:
            issues_latest = get_batch_issues(DB_CONFIG, latest_batch_id)
        except Exception:
            issues_latest = pd.DataFrame()

        if issues_latest.empty:
            st.markdown('<div class="terminal-block">NO ISSUES DETECTED // ALL CLEAR</div>', unsafe_allow_html=True)
        else:
            sev_counts = issues_latest["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            color_map = {"critical": MATRIX_RED, "high": MATRIX_ORANGE, "medium": MATRIX_YELLOW, "low": MATRIX_CYAN}
            sev_counts["color"] = sev_counts["severity"].str.lower().map(color_map).fillna(MATRIX_GREEN)
            fig_sev = go.Figure(go.Bar(
                x=sev_counts["count"], y=sev_counts["severity"].str.upper(), orientation="h",
                marker=dict(color=sev_counts["color"], line=dict(color=sev_counts["color"], width=1)),
                hovertemplate="<b>%{y}</b>: %{x} issues<extra></extra>",
            ))
            layout_sev = _chart_layout(height=380)
            layout_sev["xaxis"]["title"] = dict(text="Issue Count", font=dict(color=MATRIX_TEXT_DIM, size=11))
            fig_sev.update_layout(**layout_sev)
            fig_sev.update_yaxes(categoryorder="array", categoryarray=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Batch History
    # -----------------------------------------------------------------------
    section_header("// Data Log : Batch History")

    _batch_cols = ["batch_id", "timestamp", "rows_processed", "overall_score", "pass", "issues_count"]
    if "source_file" in batches.columns:
        _batch_cols.append("source_file")
    display_df = batches[_batch_cols].copy()
    _display_names = ["Batch ID", "Timestamp", "Rows", "Score", "Pass", "Issues"]
    if "source_file" in batches.columns:
        _display_names.append("Source")
    display_df.columns = _display_names
    display_df["Pass"] = display_df["Pass"].map({True: "PASS", 1: "PASS", False: "FAIL", 0: "FAIL"})

    st.caption("Tip: click the empty space on the left edge of any row to select it, then press ‘Drill into selected batch’ below — or use the dropdown.")
    _bh_event = st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        height=min(350, 40 + 35 * len(display_df)),
        column_config={
            "Score": st.column_config.ProgressColumn("Score", format="%.1f%%", min_value=0, max_value=100),
            "Batch ID": st.column_config.TextColumn("Batch ID", width="medium"),
            "Pass": st.column_config.TextColumn("Status", width="small"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key="cc_batch_history_table",
    )
    # Wire row-click → drill into selected batch
    try:
        _sel_rows = _bh_event.selection.rows if hasattr(_bh_event, "selection") else []
    except Exception:
        _sel_rows = []

    _bh_pick_cols = st.columns([3, 2, 2])
    with _bh_pick_cols[0]:
        _bh_pick_drop = st.selectbox(
            "Or pick a batch from this list",
            options=display_df["Batch ID"].tolist(),
            index=(_sel_rows[0] if _sel_rows else 0),
            key="cc_bh_dropdown_pick",
        )
    with _bh_pick_cols[1]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Drill into selected batch", key="cc_bh_drill_btn", type="primary", use_container_width=True):
            _target = display_df.iloc[_sel_rows[0]]["Batch ID"] if _sel_rows else _bh_pick_drop
            if _target and st.session_state.get("cc_drill_batch") != _target:
                st.session_state["cc_drill_batch"] = _target
                st.session_state.pop("cc_drill_column", None)
                st.session_state.pop("cc_drill_issue_row_id", None)
                st.rerun()
    with _bh_pick_cols[2]:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.session_state.get("cc_drill_batch"):
            if st.button("Clear drill", key="cc_bh_clear_btn", use_container_width=True):
                st.session_state.pop("cc_drill_batch", None)
                st.session_state.pop("cc_drill_column", None)
                st.session_state.pop("cc_drill_issue_row_id", None)
                st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Batch Comparison + Pipeline Performance
    # -----------------------------------------------------------------------
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section_header("// Comparator : Batch Scores")
        if not trend.empty:
            comp = trend.copy()
            comp["color"] = comp["pass"].apply(lambda p: MATRIX_GREEN if p else MATRIX_RED)
            comp["label"] = comp["batch_id"].str.extract(r'(\d{8})')[0].apply(
                lambda x: pd.to_datetime(x, format="%Y%m%d").strftime("%b-%d") if pd.notna(x) else x
            )
            bar_colors = comp["color"].tolist()
            border_colors = [
                "rgba(0,255,65,0.5)" if c == MATRIX_GREEN else "rgba(255,23,68,0.5)"
                for c in bar_colors
            ]
            fig_comp = go.Figure(go.Bar(
                x=comp["label"], y=comp["overall_score"],
                marker=dict(color=bar_colors, line=dict(color=border_colors, width=1)),
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
            ))
            layout_comp = _chart_layout(height=340)
            layout_comp["yaxis"]["title"] = dict(text="Score (%)", font=dict(color=MATRIX_TEXT_DIM, size=11))
            layout_comp["xaxis"]["title"] = dict(text="Batch", font=dict(color=MATRIX_TEXT_DIM, size=11))
            layout_comp["xaxis"]["type"] = "category"
            fig_comp.update_layout(**layout_comp)
            fig_comp.add_hline(
                y=85, line_dash="dot", line_color=MATRIX_RED, line_width=1,
                annotation_text="PASS THRESHOLD",
                annotation_font=dict(color=MATRIX_RED, size=9, family="Share Tech Mono"),
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        section_header("// Performance : Pipeline Steps")
        try:
            audit_df = get_batch_audit(DB_CONFIG, latest_batch_id)
        except Exception:
            audit_df = pd.DataFrame()

        if audit_df.empty:
            st.markdown(
                '<div class="terminal-block">// NO AUDIT TRAIL — AWAITING PIPELINE EXECUTION</div>',
                unsafe_allow_html=True,
            )
        else:
            n = len(audit_df)
            colors = [
                f"rgba(0,{int(255 - (255 - 229) * i / max(n - 1, 1))},{int(65 + (255 - 65) * i / max(n - 1, 1))},0.85)"
                for i in range(n)
            ]
            fig_perf = go.Figure(go.Bar(
                x=audit_df["step_name"], y=audit_df["duration_ms"],
                marker=dict(color=colors, line=dict(color=MATRIX_GREEN, width=1)),
                hovertemplate="<b>%{x}</b><br>Duration: %{y:,}ms<extra></extra>",
            ))
            layout_perf = _chart_layout(height=340)
            layout_perf["yaxis"]["title"] = dict(text="Duration (ms)", font=dict(color=MATRIX_TEXT_DIM, size=11))
            layout_perf["xaxis"]["title"] = dict(text="Pipeline Step", font=dict(color=MATRIX_TEXT_DIM, size=11))
            fig_perf.update_layout(**layout_perf)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Batch Drill-Down — honours cc_drill_batch state from row-click above
    # -----------------------------------------------------------------------
    section_header("// Deep Scan : Batch Drill-Down")

    batch_ids = batches["batch_id"].tolist()
    _drill_state = st.session_state.get("cc_drill_batch")
    _default_idx = batch_ids.index(_drill_state) if (_drill_state in batch_ids) else 0
    selected_drill = st.selectbox(
        "SELECT BATCH FOR ANALYSIS", batch_ids,
        index=_default_idx, key="drill_batch",
    )
    # Keep state in sync with the dropdown choice
    if selected_drill != _drill_state:
        st.session_state["cc_drill_batch"] = selected_drill
        # Selecting a new batch invalidates deeper drills
        st.session_state.pop("cc_drill_column", None)
        st.session_state.pop("cc_drill_issue_row_id", None)

    if selected_drill:
        try:
            drill_issues = get_batch_issues(DB_CONFIG, selected_drill)
        except Exception as exc:
            st.error(f"// ERROR: {exc}")
            drill_issues = pd.DataFrame()

        if drill_issues.empty:
            st.markdown(
                '<div class="terminal-block">// SCAN COMPLETE — NO ANOMALIES DETECTED IN BATCH</div>',
                unsafe_allow_html=True,
            )
        else:
            drill_left, drill_right = st.columns(2, gap="large")

            with drill_left:
                by_col = drill_issues["column_name"].value_counts().reset_index()
                by_col.columns = ["Column", "Count"]
                fig_col = go.Figure(go.Bar(
                    y=by_col["Column"], x=by_col["Count"], orientation="h",
                    marker=dict(
                        color=[f"rgba(0,{min(255, 150 + i * 15)},{min(255, 65 + i * 25)},0.8)" for i in range(len(by_col))],
                        line=dict(color=MATRIX_GREEN, width=1),
                    ),
                    hovertemplate="<b>%{y}</b>: %{x} issues<extra></extra>",
                ))
                layout_col = _chart_layout(height=max(280, 35 * len(by_col)))
                layout_col["xaxis"]["title"] = dict(text="Issue Count", font=dict(color=MATRIX_TEXT_DIM, size=11))
                fig_col.update_layout(**layout_col, title=dict(
                    text="ISSUES BY COLUMN",
                    font=dict(size=13, color=MATRIX_GREEN, family="Orbitron"), x=0.5,
                ))
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_col, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
                # Click-through table beneath the chart (feature A)
                st.caption("Click a column to view its health profile →")
                _bc_event = st.dataframe(
                    by_col, hide_index=True, height=min(220, 40 + 32 * len(by_col)),
                    on_select="rerun", selection_mode="single-row",
                    key="cc_by_col_click",
                )
                try:
                    _sel = _bc_event.selection.rows if hasattr(_bc_event, "selection") else []
                except Exception:
                    _sel = []
                if _sel:
                    _picked_col = by_col.iloc[_sel[0]]["Column"]
                    if st.session_state.get("cc_drill_column") != _picked_col:
                        st.session_state["cc_drill_column"] = _picked_col
                        st.rerun()

            with drill_right:
                by_type = drill_issues["issue_type"].value_counts().reset_index()
                by_type.columns = ["Type", "Count"]
                fig_type = go.Figure(go.Bar(
                    y=by_type["Type"], x=by_type["Count"], orientation="h",
                    marker=dict(
                        color=[f"rgba(0,{min(255, 200 + i * 10)},{min(255, 200 + i * 10)},0.8)" for i in range(len(by_type))],
                        line=dict(color=MATRIX_CYAN, width=1),
                    ),
                    hovertemplate="<b>%{y}</b>: %{x} issues<extra></extra>",
                ))
                layout_type = _chart_layout(height=max(280, 35 * len(by_type)))
                layout_type["xaxis"]["title"] = dict(text="Issue Count", font=dict(color=MATRIX_TEXT_DIM, size=11))
                fig_type.update_layout(**layout_type, title=dict(
                    text="ISSUES BY TYPE",
                    font=dict(size=13, color=MATRIX_CYAN, family="Orbitron"), x=0.5,
                ))
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_type, use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.82rem;'
                'letter-spacing:2px;">// FULL ISSUE LOG — click a row to see surrounding data context</p>',
                unsafe_allow_html=True,
            )
            # Optional pre-filter by column when user has drilled into one
            _drill_col_active = st.session_state.get("cc_drill_column")
            _issues_view = (
                drill_issues[drill_issues["column_name"] == _drill_col_active]
                if _drill_col_active and "column_name" in drill_issues.columns
                else drill_issues
            )
            if _drill_col_active:
                st.caption(f"Filtered to column: **{_drill_col_active}**")

            # Style severity column inline
            def _sev_cell_style(v):
                c = SEVERITY_COLORS.get(str(v).lower(), "#b0ffb8")
                bg = SEVERITY_BG.get(str(v).lower(), "")
                return f"color: {c}; background-color: {bg}; font-weight: 600;"
            try:
                _issues_view_styled = _issues_view.style.map(_sev_cell_style, subset=["severity"]) \
                    if "severity" in _issues_view.columns else _issues_view
            except Exception:
                _issues_view_styled = _issues_view

            _issue_event = st.dataframe(
                _issues_view_styled, hide_index=True, height=360,
                on_select="rerun", selection_mode="single-row",
                key="cc_issue_log_click",
            )
            try:
                _isel = _issue_event.selection.rows if hasattr(_issue_event, "selection") else []
            except Exception:
                _isel = []
            if _isel:
                _picked_row = _issues_view.iloc[_isel[0]]
                _row_id = _picked_row.get("row_id")
                if _row_id is not None:
                    st.session_state["cc_drill_issue_row_id"] = int(_row_id)

            # ── Issue-row context drill (feature C) ────────────────────────
            _row_id_ctx = st.session_state.get("cc_drill_issue_row_id")
            if _row_id_ctx is not None:
                with st.expander(f"// Row Context — row #{_row_id_ctx} in batch {selected_drill}", expanded=True):
                    # Find all issues for this row
                    _row_issues = drill_issues[drill_issues["row_id"] == _row_id_ctx] \
                        if "row_id" in drill_issues.columns else pd.DataFrame()
                    if _row_issues.empty:
                        st.caption("No issues recorded for this row id.")
                    else:
                        st.markdown(
                            f'<p style="color:#ff9100;font-family:Share Tech Mono;font-size:0.78rem;">'
                            f'{len(_row_issues)} issue(s) on this row:</p>',
                            unsafe_allow_html=True,
                        )
                        for _, _ri in _row_issues.iterrows():
                            _sev = str(_ri.get("severity", "medium")).lower()
                            _col_n = _ri.get("column_name", "?")
                            _itype = _ri.get("issue_type", "?")
                            _det = _ri.get("description", "") or _ri.get("detail", "")
                            st.markdown(
                                f'<div style="background:{SEVERITY_BG.get(_sev,"")};border-left:3px solid {SEVERITY_COLORS.get(_sev,"#b0ffb8")};'
                                f'padding:8px 12px;margin:4px 0;font-family:Share Tech Mono;font-size:0.78rem;">'
                                f'{severity_dot(_sev)}<b style="color:{SEVERITY_COLORS.get(_sev,"#b0ffb8")};">{_sev.upper()}</b> '
                                f'<span style="color:#00e5ff;">{_col_n}</span> '
                                f'<span style="color:#5a9a5a;">· {_itype}</span><br>'
                                f'<span style="color:#b0ffb8;">{_det}</span></div>',
                                unsafe_allow_html=True,
                            )
                    if st.button("Clear row context", key="cc_clear_row_ctx"):
                        st.session_state.pop("cc_drill_issue_row_id", None)
                        st.rerun()

            csv_export = _issues_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="EXPORT ISSUES [CSV]",
                data=csv_export,
                file_name=f"issues_{selected_drill}.csv",
                mime="text/csv",
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # AI Enrichment Intelligence
    # -----------------------------------------------------------------------
    st.markdown(
        '<div class="section-header" style="border-image:linear-gradient(90deg, #00e5ff, #00ff41, transparent) 1;">'
        '// Neural Net Intelligence '
        '<span style="display:inline-block;background:rgba(0,229,255,0.12);border:1px solid rgba(0,229,255,0.35);'
        'color:#00e5ff;font-size:0.58rem;padding:3px 10px;border-radius:12px;letter-spacing:2px;margin-left:10px;'
        'vertical-align:middle;">AI POWERED</span></div>',
        unsafe_allow_html=True,
    )

    _ai_batches_err = None
    try:
        ai_batches = get_batches_with_ai(DB_CONFIG)
    except Exception as _e:
        ai_batches = []
        _ai_batches_err = str(_e)
    if _ai_batches_err:
        st.error(
            f"Could not load AI batches: {_ai_batches_err}. "
            "This is likely a DB-engine SQL-dialect difference. "
            "Open the diagnostic below to see the underlying tables."
        )

    if not ai_batches:
        st.markdown(
            '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(0,229,255,0.2);border-radius:8px;'
            'padding:30px;text-align:center;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;letter-spacing:2px;">'
            '// NO AI ENRICHMENT DATA<br>'
            '<span style="font-size:0.72rem;">Run validation with AI enrichment enabled to generate intelligence feeds.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Diagnostic peek — shows whether the AI tables exist and how many
        # rows each holds. Critical for debugging silent-save failures.
        with st.expander("🔧 Diagnostic: peek at AI tables", expanded=False):
            try:
                _diag_eng = get_engine()
                _ai_tables = [
                    "ai_smart_rules", "ai_cross_column", "ai_explanations",
                    "ai_triage", "ai_executive_summary",
                ]
                _rows = []
                with _diag_eng.connect() as _c:
                    for _t in _ai_tables:
                        try:
                            _n = _c.execute(text(f"SELECT COUNT(*) FROM {_t}")).scalar() or 0
                            _last = _c.execute(text(f"SELECT MAX(batch_id) FROM {_t}")).scalar() or "—"
                        except Exception as _e:
                            _n = f"table missing? {_e}"
                            _last = "—"
                        _rows.append({"Table": _t, "Rows": _n, "Latest batch_id": _last})
                _diag_df = pd.DataFrame(_rows)
                st.dataframe(_diag_df, hide_index=True, height=240)
                st.caption(
                    "If all tables have 0 rows: AI enrichment never wrote to the DB "
                    "(provider unreachable or returned empty). If only "
                    "ai_executive_summary is 0 but others have rows: the provider "
                    "skipped the summary step but produced the others — Command "
                    "Center keys on ai_executive_summary, so still no batches "
                    "appear in the dropdown above."
                )
            except Exception as _diag_e:
                st.warning(f"Diagnostic query failed: {_diag_e}")
    else:
        ai_batch = st.selectbox("SELECT AI-ENRICHED BATCH", ai_batches, index=0, key="ai_batch")

        # Executive Summary
        try:
            exec_summary = get_ai_executive_summary(DB_CONFIG, ai_batch)
        except Exception:
            exec_summary = None

        if exec_summary:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,rgba(0,15,2,0.85),rgba(0,30,40,0.6));'
                f'border:1px solid rgba(0,229,255,0.25);border-radius:12px;padding:24px 28px;'
                f'font-family:Share Tech Mono;font-size:0.88rem;color:#b0ffb8;line-height:1.7;'
                f'position:relative;overflow:hidden;box-shadow:0 0 25px rgba(0,229,255,0.06);">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
                f'background:linear-gradient(90deg,transparent,#00e5ff,#00ff41,transparent);opacity:0.7;"></div>'
                f'{exec_summary}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Triage + Smart Rules
        ai_left, ai_right = st.columns([6, 4], gap="large")

        with ai_left:
            st.markdown(
                '<div class="section-header" style="border-image:linear-gradient(90deg,#ff1744,#ff9100,transparent) 1;">'
                '// Priority Triage : AI Action Plan</div>',
                unsafe_allow_html=True,
            )
            try:
                triage_df = get_ai_triage(DB_CONFIG, ai_batch)
            except Exception:
                triage_df = pd.DataFrame()

            if triage_df.empty:
                st.markdown('<div class="terminal-block">// NO TRIAGE DATA</div>', unsafe_allow_html=True)
            else:
                for _, row in triage_df.iterrows():
                    sev = str(row.get("severity", "medium")).lower()
                    pri = int(row.get("priority", 0))
                    col_name = str(row.get("column_name", ""))
                    issue_t = str(row.get("issue_type", ""))
                    count = int(row.get("issue_count", 0))
                    effort = str(row.get("effort", ""))
                    reason = str(row.get("reason", ""))
                    rec = str(row.get("recommendation", ""))
                    sev_color = {"critical": MATRIX_RED, "high": MATRIX_ORANGE, "medium": MATRIX_YELLOW, "low": MATRIX_CYAN}.get(sev, MATRIX_TEXT_DIM)
                    effort_color = {"quick_fix": MATRIX_GREEN, "moderate": MATRIX_YELLOW, "complex": MATRIX_RED}.get(effort.lower().replace(" ", "_"), MATRIX_TEXT_DIM)

                    st.markdown(
                        f'<div style="background:rgba(0,5,1,0.85);border-left:3px solid {sev_color};'
                        f'border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:8px;font-family:Share Tech Mono;font-size:0.82rem;color:#b0ffb8;">'
                        f'<span style="font-family:Orbitron;font-size:1.4rem;font-weight:900;color:{sev_color};'
                        f'text-shadow:0 0 10px {sev_color}80;float:left;margin-right:16px;">#{pri}</span>'
                        f'<strong style="color:{MATRIX_GREEN};">{col_name}</strong>'
                        f' -- <span style="color:{MATRIX_CYAN};">{issue_t}</span>'
                        f' <span style="font-size:0.7rem;color:{MATRIX_TEXT_DIM};">({count} rows)</span>'
                        f'<br><span style="font-size:0.78rem;">{reason}</span>'
                        f'<div style="font-size:0.7rem;color:{MATRIX_TEXT_DIM};margin-top:4px;">'
                        f'EFFORT: <span style="color:{effort_color};">{effort.upper()}</span>'
                        f' | SEVERITY: <span style="color:{sev_color};">{sev.upper()}</span></div>'
                        f'<div style="font-size:0.72rem;color:#5a9a5a;margin-top:4px;">REC: {rec}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with ai_right:
            st.markdown(
                '<div class="section-header" style="border-image:linear-gradient(90deg,#00e5ff,#00ff41,transparent) 1;">'
                '// Smart Rules : AI Generated</div>',
                unsafe_allow_html=True,
            )
            try:
                rules_df = get_ai_smart_rules(DB_CONFIG, ai_batch)
            except Exception:
                rules_df = pd.DataFrame()

            if rules_df.empty:
                st.markdown('<div class="terminal-block">// NO AI RULES</div>', unsafe_allow_html=True)
            else:
                for _, row in rules_df.iterrows():
                    col_name = str(row.get("column_name", ""))
                    rule_type = str(row.get("rule_type", ""))
                    desc = str(row.get("description", ""))
                    conf = float(row.get("confidence", 0))
                    conf_pct = int(conf * 100)
                    type_color = {"format": MATRIX_CYAN, "range": MATRIX_GREEN, "required": MATRIX_RED,
                                  "pattern": MATRIX_YELLOW, "allowed_values": MATRIX_ORANGE}.get(rule_type.lower(), MATRIX_TEXT)

                    st.markdown(
                        f'<div style="background:rgba(0,5,1,0.8);border:1px solid rgba(0,229,255,0.15);'
                        f'border-radius:8px;padding:12px 14px;margin-bottom:6px;font-family:Share Tech Mono;">'
                        f'<span style="color:{type_color};font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;'
                        f'font-weight:bold;">{rule_type}</span>'
                        f' <span style="color:{MATRIX_GREEN};font-size:0.82rem;">{col_name}</span>'
                        f'<br><span style="color:#b0ffb8;font-size:0.78rem;">{desc}</span>'
                        f'<div style="height:4px;background:rgba(0,255,65,0.1);border-radius:2px;margin-top:6px;overflow:hidden;">'
                        f'<div style="height:100%;width:{conf_pct}%;background:linear-gradient(90deg,#00e5ff,#00ff41);border-radius:2px;"></div></div>'
                        f'<span style="color:#5a9a5a;font-size:0.65rem;">CONFIDENCE: {conf_pct}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # -----------------------------------------------------------------------
    # Pending Fixes — Human Review Gate (session-level)
    # -----------------------------------------------------------------------
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header" style="border-image:linear-gradient(90deg,#ff9100,#ff1744,transparent) 1;">'
        '// Pending Fixes : Human Review Required</div>',
        unsafe_allow_html=True,
    )

    _report = st.session_state.get("unified_issues_report")
    _df_raw = st.session_state.get("df_raw")
    _df_corrected = st.session_state.get("df_corrected")

    if _df_raw is None or _report is None or not isinstance(_report, pd.DataFrame) or _report.empty:
        st.markdown(
            '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(255,145,0,0.2);border-radius:8px;'
            'padding:24px;text-align:center;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
            '// NO ACTIVE SESSION DATA<br>'
            '<span style="font-size:0.72rem;">Load a dataset and run validation to see pending fix suggestions here.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        _sug_col = next((c for c in ["suggested_fix", "suggestion"] if c in _report.columns), None)
        _conf_col = next((c for c in ["confidence", "score"] if c in _report.columns), None)

        if _sug_col is None:
            st.markdown(
                '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(255,145,0,0.2);border-radius:8px;'
                'padding:20px;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.78rem;">'
                '// No fix suggestions in current validation report. Re-run validation with rules or corpus enabled.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            _fixable = _report[
                _report[_sug_col].notna() & (_report[_sug_col].astype(str).str.strip() != "")
            ].copy().reset_index(drop=True)

            if _fixable.empty:
                st.markdown(
                    '<div style="background:rgba(0,255,65,0.05);border:1px solid rgba(0,255,65,0.2);'
                    'border-radius:8px;padding:20px;font-family:Share Tech Mono;color:#00ff41;font-size:0.82rem;">'
                    '// ALL CLEAR — No pending fix suggestions in current session.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                _already_applied = _df_corrected is not None
                _status_color = "#5a9a5a" if _already_applied else "#ff9100"
                _status_text = "FIXES APPLIED" if _already_applied else "AWAITING REVIEW"

                st.markdown(
                    f'<div style="display:flex;gap:20px;align-items:center;margin-bottom:16px;'
                    f'font-family:Share Tech Mono;font-size:0.82rem;">'
                    f'<span style="color:{_status_color};border:1px solid {_status_color};'
                    f'padding:3px 12px;border-radius:4px;letter-spacing:2px;">{_status_text}</span>'
                    f'<span style="color:#b0ffb8;">{len(_fixable)} suggested fixes in current session</span>'
                    f'<span style="color:#5a9a5a;">Dataset: {len(_df_raw):,} rows</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Build compact review table
                _review = pd.DataFrame()
                _review["Approve"] = not _already_applied  # default True if not yet applied
                _review["Row #"] = _fixable.get("row_id", pd.Series(range(len(_fixable)))).astype(str)
                _review["Column"] = _fixable.get("column", _fixable.get("column_name", "")).astype(str)
                _review["Issue Type"] = _fixable.get("issue", _fixable.get("issue_type", "")).astype(str)
                _review["Severity"] = _fixable.get("severity", "").astype(str)

                def _cur_val(r):
                    try:
                        rid = int(r["Row #"])
                        col = r["Column"]
                        if col in _df_raw.columns and 0 <= rid < len(_df_raw):
                            return str(_df_raw.at[rid, col])
                    except Exception:
                        pass
                    return ""

                _review["Current Value"] = _review.apply(_cur_val, axis=1)
                _review["Suggested Fix"] = _fixable[_sug_col].astype(str)
                if _conf_col:
                    _review["Confidence"] = _fixable[_conf_col].apply(
                        lambda x: f"{float(x)*100:.0f}%" if pd.notna(x) else "—"
                    )

                # Bulk select buttons
                _bc1, _bc2, _bc3 = st.columns(3, gap="small")
                with _bc1:
                    if st.button("SELECT ALL", key="cc_fix_all", use_container_width=True):
                        st.session_state["_cc_fix_all"] = True
                        st.session_state["_cc_fix_none"] = False
                        st.session_state["_cc_fix_hc"] = False
                with _bc2:
                    if st.button("HIGH CONFIDENCE (≥90%)", key="cc_fix_hc", use_container_width=True):
                        st.session_state["_cc_fix_hc"] = True
                        st.session_state["_cc_fix_all"] = False
                        st.session_state["_cc_fix_none"] = False
                with _bc3:
                    if st.button("DESELECT ALL", key="cc_fix_none", use_container_width=True):
                        st.session_state["_cc_fix_none"] = True
                        st.session_state["_cc_fix_all"] = False
                        st.session_state["_cc_fix_hc"] = False

                if st.session_state.get("_cc_fix_all"):
                    _review["Approve"] = True
                elif st.session_state.get("_cc_fix_none"):
                    _review["Approve"] = False
                elif st.session_state.get("_cc_fix_hc") and _conf_col:
                    _review["Approve"] = _fixable[_conf_col].apply(
                        lambda x: float(x) >= 0.9 if pd.notna(x) else False
                    ).values

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                _edited = st.data_editor(
                    _review,
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
                    key="cc_fix_review_table",
                )

                _approved = _edited[_edited["Approve"] == True]
                st.markdown(
                    f'<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;margin-top:8px;">'
                    f'{len(_approved)} of {len(_review)} fixes approved.</p>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"APPLY {len(_approved)} APPROVED FIXES",
                    key="cc_apply_fixes",
                    use_container_width=True,
                    disabled=len(_approved) == 0 or _already_applied,
                    type="primary",
                ):
                    _df_fixed = _df_raw.copy()
                    _applied = 0
                    for _, _row in _approved.iterrows():
                        try:
                            _rid = int(_row["Row #"])
                            _col = str(_row["Column"])
                            _fix = str(_row["Suggested Fix"])
                            if _col in _df_fixed.columns and 0 <= _rid < len(_df_fixed) and _fix:
                                _df_fixed.at[_rid, _col] = _fix
                                _applied += 1
                        except Exception:
                            continue

                    st.session_state["df_corrected"] = _df_fixed
                    for _flag in ["_cc_fix_all", "_cc_fix_none", "_cc_fix_hc"]:
                        st.session_state.pop(_flag, None)

                    st.success(
                        f"{_applied} fixes applied. Original data untouched. "
                        f"Go to the Cleaning page to download the corrected dataset."
                    )

                if _already_applied:
                    st.info("Fixes have already been applied this session. Go to Cleaning to download the corrected dataset.")

    # -----------------------------------------------------------------------
    # Column Health Profile (feature B) — appears when a column is drilled
    # -----------------------------------------------------------------------
    _drill_col = st.session_state.get("cc_drill_column")
    if _drill_col:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        section_header(f"// Column Health Profile : {_drill_col}")
        try:
            _eng = get_engine()
            _q = text("""
                SELECT i.batch_id, b.timestamp, COUNT(*) AS issue_count,
                       SUM(CASE WHEN LOWER(i.severity)='critical' THEN 1 ELSE 0 END) AS critical_n,
                       SUM(CASE WHEN LOWER(i.severity)='high'     THEN 1 ELSE 0 END) AS high_n,
                       SUM(CASE WHEN LOWER(i.severity)='medium'   THEN 1 ELSE 0 END) AS medium_n,
                       SUM(CASE WHEN LOWER(i.severity)='low'      THEN 1 ELSE 0 END) AS low_n
                FROM issues i
                JOIN batch_runs b ON b.batch_id = i.batch_id
                WHERE i.column_name = :col
                GROUP BY i.batch_id, b.timestamp
                ORDER BY b.timestamp DESC
                LIMIT 40
            """)
            with _eng.connect() as conn:
                _col_hist = pd.read_sql(_q, conn, params={"col": _drill_col})
        except Exception as exc:
            _col_hist = pd.DataFrame()
            st.warning(f"Could not load column history: {exc}")

        if _col_hist.empty:
            st.info(f"No historical issues recorded for column `{_drill_col}` yet.")
        else:
            # KPI strip for the column
            _ck1, _ck2, _ck3, _ck4 = st.columns(4, gap="medium")
            _ck1.metric("Batches with issues", len(_col_hist))
            _ck2.metric("Total issues all-time", int(_col_hist["issue_count"].sum()))
            _ck3.metric("Avg issues per batch", f"{_col_hist['issue_count'].mean():.1f}")
            _ck4.metric("Worst batch (count)", int(_col_hist["issue_count"].max()))

            # Trend chart: stacked severity bars over time
            _col_hist_chart = _col_hist.sort_values("timestamp")
            fig_colhist = go.Figure()
            for _sev_name, _color in [
                ("critical_n", SEVERITY_COLORS["critical"]),
                ("high_n",     SEVERITY_COLORS["high"]),
                ("medium_n",   SEVERITY_COLORS["medium"]),
                ("low_n",      SEVERITY_COLORS["low"]),
            ]:
                if _col_hist_chart[_sev_name].sum() > 0:
                    fig_colhist.add_trace(go.Bar(
                        x=_col_hist_chart["batch_id"], y=_col_hist_chart[_sev_name],
                        name=_sev_name.replace("_n", "").upper(),
                        marker=dict(color=_color, line=dict(color=_color, width=1)),
                        hovertemplate="<b>%{x}</b><br>%{y} issues<extra></extra>",
                    ))
            _layout_ch = _chart_layout(height=320, show_legend=True)
            _layout_ch["barmode"] = "stack"
            _layout_ch["yaxis"]["title"] = dict(text="Issue count", font=dict(color=MATRIX_TEXT_DIM, size=11))
            fig_colhist.update_layout(**_layout_ch)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_colhist, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Trend slope text
            if len(_col_hist_chart) >= 3:
                _first_half = _col_hist_chart.iloc[:len(_col_hist_chart)//2]["issue_count"].mean()
                _second_half = _col_hist_chart.iloc[len(_col_hist_chart)//2:]["issue_count"].mean()
                _delta = _second_half - _first_half
                if _delta > 0.5:
                    _verdict = f'<span style="color:{SEVERITY_COLORS["high"]};">↗ Getting worse: +{_delta:.1f} issues/batch over time</span>'
                elif _delta < -0.5:
                    _verdict = f'<span style="color:{SEVERITY_COLORS["info"]};">↘ Improving: −{abs(_delta):.1f} issues/batch over time</span>'
                else:
                    _verdict = f'<span style="color:{MATRIX_CYAN};">→ Stable: no clear trend</span>'
                st.markdown(
                    f'<p style="font-family:Share Tech Mono;font-size:0.82rem;padding:8px 0;">{_verdict}</p>',
                    unsafe_allow_html=True,
                )

            with st.expander("Recent batches affecting this column"):
                st.dataframe(_col_hist, hide_index=True, height=300)

        if st.button("← Clear column drill", key="cc_clear_col_drill"):
            st.session_state.pop("cc_drill_column", None)
            st.rerun()

    # -----------------------------------------------------------------------
    # Compare Batches (feature D)
    # -----------------------------------------------------------------------
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    section_header("// Compare Batches")
    st.caption("Pick 2–3 batches to see their dimension scores and issue counts side-by-side.")

    _cmp_picks = st.multiselect(
        "Select batches to compare", batch_ids,
        default=batch_ids[:min(2, len(batch_ids))],
        max_selections=3, key="cc_cmp_picks",
    )
    if len(_cmp_picks) >= 2:
        _cmp_df = batches[batches["batch_id"].isin(_cmp_picks)].copy()
        _dims = ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]
        _present_dims = [d for d in _dims if d in _cmp_df.columns]

        # Side-by-side metric cards per batch
        _cmp_cols = st.columns(len(_cmp_picks), gap="medium")
        for _i, _bid in enumerate(_cmp_picks):
            _r = _cmp_df[_cmp_df["batch_id"] == _bid].iloc[0]
            _passed = bool(_r.get("pass", False))
            _border = SEVERITY_COLORS["info"] if _passed else SEVERITY_COLORS["high"]
            with _cmp_cols[_i]:
                st.markdown(
                    f'<div style="border:1px solid {_border}40;background:rgba(0,5,1,0.7);'
                    f'border-radius:8px;padding:14px 16px;border-left:3px solid {_border};">'
                    f'<p style="color:{_border};font-family:Share Tech Mono;font-size:0.7rem;'
                    f'letter-spacing:1px;margin:0;">{_bid}</p>'
                    f'<p style="color:#b0ffb8;font-family:Orbitron;font-size:1.6rem;margin:6px 0 0 0;">'
                    f'{float(_r.get("overall_score", 0)):.1f}%</p>'
                    f'<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.7rem;margin:0;">'
                    f'{int(_r.get("issues_count", 0))} issues · {int(_r.get("rows_processed", 0))} rows · '
                    f'{"PASS" if _passed else "FAIL"}</p></div>',
                    unsafe_allow_html=True,
                )

        # Dimension comparison radar (overlay)
        if _present_dims:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            _palette = [MATRIX_GREEN, MATRIX_CYAN, MATRIX_ORANGE]
            fig_cmp = go.Figure()
            for _idx, _bid in enumerate(_cmp_picks):
                _r = _cmp_df[_cmp_df["batch_id"] == _bid].iloc[0]
                _vals = [float(_r.get(d, 0)) for d in _present_dims]
                _c = _palette[_idx % len(_palette)]
                fig_cmp.add_trace(go.Scatterpolar(
                    r=_vals + [_vals[0]],
                    theta=[d.upper() for d in _present_dims] + [_present_dims[0].upper()],
                    fill="toself", name=_bid,
                    line=dict(color=_c, width=2),
                    fillcolor=_hex_to_rgba(_c, 0.18),
                    hovertemplate=f"<b>{_bid}</b><br>%{{theta}}: %{{r:.1f}}%<extra></extra>",
                ))
            fig_cmp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=MATRIX_TEXT, family="'Share Tech Mono', monospace", size=11),
                height=420, showlegend=True,
                polar=dict(
                    bgcolor="rgba(0, 10, 2, 0.5)",
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor="rgba(0,255,65,0.08)",
                                    tickfont=dict(color=MATRIX_TEXT_DIM, size=9)),
                    angularaxis=dict(gridcolor="rgba(0,255,65,0.1)",
                                     tickfont=dict(color=MATRIX_GREEN, size=10, family="Share Tech Mono")),
                ),
                margin=dict(l=70, r=70, t=30, b=30),
                legend=dict(font=dict(color=MATRIX_TEXT, size=10), orientation="h",
                            yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Tabular delta if exactly 2 selected
        if len(_cmp_picks) == 2:
            _a = _cmp_df[_cmp_df["batch_id"] == _cmp_picks[0]].iloc[0]
            _b = _cmp_df[_cmp_df["batch_id"] == _cmp_picks[1]].iloc[0]
            _delta_rows = []
            for _d in _present_dims + ["overall_score", "issues_count"]:
                _av = float(_a.get(_d, 0))
                _bv = float(_b.get(_d, 0))
                _delta_rows.append({
                    "Metric": _d.replace("_", " ").title(),
                    _cmp_picks[0]: _av, _cmp_picks[1]: _bv,
                    "Δ": _bv - _av,
                })
            _delta_df = pd.DataFrame(_delta_rows)
            def _color_delta(v):
                try:
                    v = float(v)
                except Exception:
                    return ""
                if v > 0:   return f"color: {SEVERITY_COLORS['info']};"
                if v < 0:   return f"color: {SEVERITY_COLORS['high']};"
                return ""
            try:
                _styled_delta = _delta_df.style.map(_color_delta, subset=["Δ"])
                st.dataframe(_styled_delta, hide_index=True, height=300)
            except Exception:
                st.dataframe(_delta_df, hide_index=True, height=300)
    else:
        st.caption("Select at least 2 batches to compare.")

    # Footer
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px 0;">'
        '<p style="color:#00ff4120;font-family:Share Tech Mono;font-size:0.6rem;letter-spacing:3px;">'
        'AI POWERED DQ INVESTIGATOR // COMMAND CENTER // v2.0 // THERE IS NO SPOON</p></div>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# RESEARCH / INVESTIGATE TAB
# ===========================================================================
with research_tab:

    # -----------------------------------------------------------------------
    # Helper: load latest batch issues from DB directly via get_engine
    # -----------------------------------------------------------------------
    def _research_load_latest_batch():
        """Return (batch_row, issues_df) for the most recent batch_run."""
        try:
            engine = get_engine()  # uses DATABASE_URL env var → PostgreSQL on ECS
            with engine.connect() as conn:
                batch_row = conn.execute(
                    text("SELECT * FROM batch_runs ORDER BY timestamp DESC LIMIT 1")
                ).mappings().fetchone()
                if batch_row is None:
                    return None, pd.DataFrame()
                bid = batch_row["batch_id"]
                issues_df = pd.read_sql(
                    text("SELECT * FROM issues WHERE batch_id = :bid"),
                    conn,
                    params={"bid": bid},
                )
            return dict(batch_row), issues_df
        except Exception:
            return None, pd.DataFrame()

    def _research_column_options(issues_df):
        """Return column names ordered by issue count DESC."""
        if issues_df.empty or "column_name" not in issues_df.columns:
            return []
        return (
            issues_df["column_name"]
            .value_counts()
            .index.tolist()
        )

    # -----------------------------------------------------------------------
    # Section 1 — Issue Explorer
    # -----------------------------------------------------------------------
    section_header("// Issue Explorer")

    _res_batch, _res_issues = _research_load_latest_batch()

    if _res_batch is None or _res_issues.empty:
        st.markdown(
            '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(0,255,65,0.2);border-radius:8px;'
            'padding:30px;text-align:center;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
            '// NO BATCH DATA FOUND<br>'
            '<span style="font-size:0.72rem;">Run a validation pipeline to populate the issues feed.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Batch info banner
        _bid = _res_batch.get("batch_id", "N/A")
        _bts = _res_batch.get("timestamp", "N/A")
        _bsc = _res_batch.get("overall_score", 0)
        _bsc_color = MATRIX_RED if _bsc < 70 else (MATRIX_YELLOW if _bsc < 85 else MATRIX_GREEN)
        st.markdown(
            f'<div style="background:rgba(0,10,2,0.7);border:1px solid rgba(0,255,65,0.2);border-radius:6px;'
            f'padding:10px 18px;margin-bottom:18px;font-family:Share Tech Mono;font-size:0.78rem;color:#b0ffb8;'
            f'display:flex;gap:28px;align-items:center;flex-wrap:wrap;">'
            f'<span style="color:{MATRIX_GREEN};">BATCH: <strong>{_bid}</strong></span>'
            f'<span style="color:{MATRIX_TEXT_DIM};">TIMESTAMP: {_bts}</span>'
            f'<span>SCORE: <strong style="color:{_bsc_color};">{_bsc:.1f}%</strong></span>'
            f'<span style="color:{MATRIX_TEXT_DIM};">TOTAL ISSUES: {len(_res_issues)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Column selector
        _col_options = _research_column_options(_res_issues)
        if not _col_options:
            st.info("No column data found in issues table.")
        else:
            _sel_col = st.selectbox(
                "Select a column to investigate",
                _col_options,
                key="research_col_select",
            )

            if _sel_col:
                _col_issues = _res_issues[_res_issues["column_name"] == _sel_col]
                _total_col_issues = len(_col_issues)

                # Metrics row
                _m1, _m2, _m3, _m4, _m5 = st.columns(5, gap="small")
                _sev_map = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                if "severity" in _col_issues.columns:
                    for _s in _col_issues["severity"].str.lower():
                        if _s in _sev_map:
                            _sev_map[_s] += 1
                _m1.markdown(
                    f'<div class="kpi-card"><div class="kpi-value">{_total_col_issues}</div>'
                    f'<div class="kpi-label">Total Issues</div></div>',
                    unsafe_allow_html=True,
                )
                _m2.markdown(
                    f'<div class="kpi-card"><div class="kpi-value danger">{_sev_map["critical"]}</div>'
                    f'<div class="kpi-label">Critical</div></div>',
                    unsafe_allow_html=True,
                )
                _m3.markdown(
                    f'<div class="kpi-card"><div class="kpi-value warn">{_sev_map["high"]}</div>'
                    f'<div class="kpi-label">High</div></div>',
                    unsafe_allow_html=True,
                )
                _m4.markdown(
                    f'<div class="kpi-card"><div class="kpi-value cyan">{_sev_map["medium"]}</div>'
                    f'<div class="kpi-label">Medium</div></div>',
                    unsafe_allow_html=True,
                )
                _m5.markdown(
                    f'<div class="kpi-card"><div class="kpi-value">{_sev_map["low"]}</div>'
                    f'<div class="kpi-label">Low</div></div>',
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                # Issue type breakdown chart
                if "issue_type" in _col_issues.columns and not _col_issues.empty:
                    _itype_counts = _col_issues["issue_type"].value_counts().reset_index()
                    _itype_counts.columns = ["Issue Type", "Count"]
                    _itype_colors = [
                        f"rgba(0,{min(255, 180 + i * 12)},{min(255, 100 + i * 20)},0.85)"
                        for i in range(len(_itype_counts))
                    ]
                    _fig_itype = go.Figure(go.Bar(
                        y=_itype_counts["Issue Type"],
                        x=_itype_counts["Count"],
                        orientation="h",
                        marker=dict(color=_itype_colors, line=dict(color=MATRIX_CYAN, width=1)),
                        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
                    ))
                    _layout_itype = _chart_layout(height=max(200, 40 * len(_itype_counts)))
                    _layout_itype["xaxis"]["title"] = dict(text="Count", font=dict(color=MATRIX_TEXT_DIM, size=11))
                    _fig_itype.update_layout(
                        **_layout_itype,
                        title=dict(
                            text=f"ISSUE TYPES — {_sel_col.upper()}",
                            font=dict(size=12, color=MATRIX_CYAN, family="Orbitron"), x=0.5,
                        ),
                    )
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(_fig_itype, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

                # Sample affected rows
                st.markdown(
                    '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.78rem;'
                    'letter-spacing:2px;margin-top:8px;">// SAMPLE AFFECTED ROWS (first 20)</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    _col_issues.head(20),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 45 + 35 * min(20, len(_col_issues))),
                )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 2 — Root Cause Analysis
    # -----------------------------------------------------------------------
    section_header("// Root Cause")

    # Only render if we have a batch and a selected column
    if _res_batch is None:
        st.markdown(
            '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(0,229,255,0.15);border-radius:8px;'
            'padding:20px;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
            '// Run a validation pipeline first to see root cause analysis.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        _rc_bid = _res_batch.get("batch_id", "")
        _rc_col = st.session_state.get("research_col_select", "")

        # AI Explanations
        _ai_expl = pd.DataFrame()
        try:
            engine = get_engine()  # uses DATABASE_URL → PostgreSQL on ECS
            with engine.connect() as conn:
                if _rc_col:
                    _ai_expl = pd.read_sql(
                        text(
                            "SELECT * FROM ai_explanations "
                            "WHERE batch_id = :bid AND column_name = :col"
                        ),
                        conn,
                        params={"bid": _rc_bid, "col": _rc_col},
                    )
                else:
                    _ai_expl = pd.read_sql(
                        text("SELECT * FROM ai_explanations WHERE batch_id = :bid"),
                        conn,
                        params={"bid": _rc_bid},
                    )
        except Exception:
            _ai_expl = pd.DataFrame()

        if _ai_expl.empty:
            st.markdown(
                '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(0,229,255,0.2);border-radius:8px;'
                'padding:20px;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
                '// No AI explanations found for this batch/column.<br>'
                '<span style="font-size:0.72rem;">Run AI Investigation (page 4) to generate root cause analysis for this batch.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for _, _expl_row in _ai_expl.iterrows():
                _expl_col = str(_expl_row.get("column_name", ""))
                _expl_itype = str(_expl_row.get("issue_type", ""))
                _expl_text = str(_expl_row.get("explanation", ""))
                _expl_impact = str(_expl_row.get("business_impact", ""))
                _expl_action = str(_expl_row.get("suggested_action", ""))
                st.markdown(
                    f'<div style="background:rgba(0,10,2,0.75);border:1px solid rgba(0,255,65,0.35);'
                    f'border-left:4px solid {MATRIX_GREEN};border-radius:0 10px 10px 0;'
                    f'padding:16px 20px;margin-bottom:12px;font-family:Share Tech Mono;font-size:0.82rem;color:#b0ffb8;">'
                    f'<div style="font-family:Orbitron;font-size:0.75rem;color:{MATRIX_GREEN};'
                    f'letter-spacing:2px;margin-bottom:8px;">'
                    f'{_expl_col.upper()} &mdash; <span style="color:{MATRIX_CYAN};">{_expl_itype}</span>'
                    f'</div>'
                    f'<p style="margin:0 0 8px 0;line-height:1.6;">{_expl_text}</p>'
                    f'<div style="border-top:1px solid rgba(0,255,65,0.1);padding-top:8px;margin-top:8px;">'
                    f'<span style="color:{MATRIX_ORANGE};font-size:0.72rem;letter-spacing:1px;">BUSINESS IMPACT: </span>'
                    f'<span style="font-size:0.78rem;">{_expl_impact}</span></div>'
                    f'<div style="margin-top:6px;">'
                    f'<span style="color:{MATRIX_CYAN};font-size:0.72rem;letter-spacing:1px;">SUGGESTED ACTION: </span>'
                    f'<span style="font-size:0.78rem;">{_expl_action}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Cross-column issues involving this column
        _cc_issues = pd.DataFrame()
        try:
            engine = get_engine()  # uses DATABASE_URL → PostgreSQL on ECS
            with engine.connect() as conn:
                _cc_all = pd.read_sql(
                    text("SELECT * FROM ai_cross_column WHERE batch_id = :bid"),
                    conn,
                    params={"bid": _rc_bid},
                )
            if not _cc_all.empty and _rc_col:
                _cc_issues = _cc_all[
                    _cc_all["columns_involved"].str.contains(_rc_col, case=False, na=False)
                ]
            elif not _cc_all.empty:
                _cc_issues = _cc_all
        except Exception:
            _cc_issues = pd.DataFrame()

        if not _cc_issues.empty:
            st.markdown(
                '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.78rem;'
                'letter-spacing:2px;margin-top:16px;">// CROSS-COLUMN ISSUES</p>',
                unsafe_allow_html=True,
            )
            for _, _cc_row in _cc_issues.iterrows():
                _cc_cols = str(_cc_row.get("columns_involved", ""))
                _cc_type = str(_cc_row.get("check_type", ""))
                _cc_desc = str(_cc_row.get("description", ""))
                _cc_sev = str(_cc_row.get("severity", "medium")).lower()
                _cc_query = str(_cc_row.get("suggested_query", ""))
                _cc_sev_color = {"critical": MATRIX_RED, "high": MATRIX_ORANGE, "medium": MATRIX_YELLOW, "low": MATRIX_CYAN}.get(_cc_sev, MATRIX_TEXT_DIM)
                st.markdown(
                    f'<div style="background:rgba(0,5,2,0.7);border:1px solid {_cc_sev_color}40;'
                    f'border-left:3px solid {_cc_sev_color};border-radius:0 8px 8px 0;'
                    f'padding:12px 16px;margin-bottom:8px;font-family:Share Tech Mono;font-size:0.78rem;color:#b0ffb8;">'
                    f'<span style="color:{MATRIX_CYAN};font-size:0.68rem;letter-spacing:2px;">{_cc_type}</span>'
                    f' &nbsp;|&nbsp; <span style="color:{_cc_sev_color};font-size:0.68rem;">{_cc_sev.upper()}</span>'
                    f'<br><strong style="color:{MATRIX_GREEN};">{_cc_cols}</strong>'
                    f'<br><span style="font-size:0.76rem;">{_cc_desc}</span>'
                    + (
                        f'<div style="margin-top:6px;padding:6px 10px;background:rgba(0,0,0,0.4);'
                        f'border-radius:4px;font-size:0.7rem;color:#5a9a5a;font-family:Courier New,monospace;">'
                        f'{_cc_query}</div>'
                        if _cc_query and _cc_query != "None" else ""
                    )
                    + f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 3 — Action Plan
    # -----------------------------------------------------------------------
    section_header("// Action Plan")

    if _res_batch is None:
        st.markdown(
            '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(255,145,0,0.15);border-radius:8px;'
            'padding:20px;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
            '// No batch loaded. Run a validation pipeline to generate an action plan.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        _ap_bid = _res_batch.get("batch_id", "")

        # Load triage
        _ap_triage = pd.DataFrame()
        try:
            engine = get_engine()  # uses DATABASE_URL → PostgreSQL on ECS
            with engine.connect() as conn:
                _ap_triage = pd.read_sql(
                    text("SELECT * FROM ai_triage WHERE batch_id = :bid ORDER BY priority ASC"),
                    conn,
                    params={"bid": _ap_bid},
                )
        except Exception:
            _ap_triage = pd.DataFrame()

        if _ap_triage.empty:
            st.markdown(
                '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(255,145,0,0.2);border-radius:8px;'
                'padding:20px;font-family:Share Tech Mono;color:#5a9a5a;font-size:0.82rem;">'
                '// No triage data found for this batch.<br>'
                '<span style="font-size:0.72rem;">Run AI Investigation (page 4) to generate an action plan.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            # Group by priority bucket
            _pri_groups = {1: [], 2: [], 3: []}
            for _, _tr in _ap_triage.iterrows():
                _pri = int(_tr.get("priority", 3))
                _bucket = 1 if _pri == 1 else (2 if _pri == 2 else 3)
                _pri_groups[_bucket].append(_tr)

            _pri_config = {
                1: (MATRIX_RED, "PRIORITY 1 — CRITICAL", "rgba(255,23,68,0.08)"),
                2: (MATRIX_ORANGE, "PRIORITY 2 — HIGH", "rgba(255,145,0,0.07)"),
                3: (MATRIX_CYAN, "PRIORITY 3 — MEDIUM / LOW", "rgba(0,229,255,0.06)"),
            }

            for _bucket, (_bcolor, _blabel, _bbg) in _pri_config.items():
                if not _pri_groups[_bucket]:
                    continue
                st.markdown(
                    f'<div style="background:{_bbg};border:1px solid {_bcolor}30;'
                    f'border-left:4px solid {_bcolor};border-radius:0 10px 10px 0;'
                    f'padding:14px 18px;margin-bottom:14px;">'
                    f'<div style="font-family:Orbitron;font-size:0.72rem;color:{_bcolor};'
                    f'letter-spacing:2px;margin-bottom:10px;">{_blabel}</div>',
                    unsafe_allow_html=True,
                )
                for _tr in _pri_groups[_bucket]:
                    _tr_cat = str(_tr.get("issue_category", _tr.get("column_name", "")))
                    _tr_desc = str(_tr.get("description", _tr.get("reason", "")))
                    _tr_action = str(_tr.get("recommended_action", _tr.get("recommendation", "")))
                    _tr_effort = str(_tr.get("estimated_effort", _tr.get("effort", "")))
                    _tr_impact = str(_tr.get("business_impact", ""))
                    st.markdown(
                        f'<div style="font-family:Share Tech Mono;font-size:0.80rem;color:#b0ffb8;'
                        f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<strong style="color:{_bcolor};">{_tr_cat}</strong>'
                        f'<br><span style="font-size:0.76rem;">{_tr_desc}</span>'
                        f'<div style="margin-top:5px;font-size:0.72rem;color:#5a9a5a;">'
                        f'ACTION: <span style="color:#b0ffb8;">{_tr_action}</span>'
                        + (f' &nbsp;|&nbsp; EFFORT: <span style="color:{MATRIX_YELLOW};">{_tr_effort}</span>' if _tr_effort and _tr_effort != "None" else "")
                        + (f'<br>IMPACT: {_tr_impact}' if _tr_impact and _tr_impact != "None" else "")
                        + f'</div></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Quick action buttons
        st.markdown(
            '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.78rem;'
            'letter-spacing:2px;margin-bottom:10px;">// QUICK ACTIONS</p>',
            unsafe_allow_html=True,
        )
        _qa1, _qa2 = st.columns(2, gap="medium")
        with _qa1:
            st.markdown(
                '<div style="background:rgba(0,10,2,0.6);border:1px solid rgba(0,255,65,0.2);'
                'border-radius:8px;padding:14px 16px;font-family:Share Tech Mono;font-size:0.8rem;color:#b0ffb8;">'
                '<span style="color:#00ff41;font-weight:bold;">GO TO CLEANING PAGE</span><br>'
                '<span style="color:#5a9a5a;font-size:0.72rem;">Navigate to page 3 (Cleaning) in the sidebar to apply corrections.</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        with _qa2:
            if _res_issues is not None and not _res_issues.empty:
                _dl_csv = _res_issues.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="DOWNLOAD ISSUES CSV",
                    data=_dl_csv,
                    file_name=f"issues_{_ap_bid}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div style="background:rgba(0,5,1,0.5);border:1px dashed rgba(0,255,65,0.15);'
                    'border-radius:8px;padding:14px 16px;font-family:Share Tech Mono;font-size:0.78rem;color:#5a9a5a;">'
                    '// No issues data available to download.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # Research tab footer
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px 0;">'
        '<p style="color:#00ff4120;font-family:Share Tech Mono;font-size:0.6rem;letter-spacing:3px;">'
        'AI POWERED DQ INVESTIGATOR // RESEARCH // v2.0 // FOLLOW THE WHITE RABBIT</p></div>',
        unsafe_allow_html=True,
    )
