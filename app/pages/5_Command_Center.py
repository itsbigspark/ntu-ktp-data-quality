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
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header,
    MATRIX_GREEN, MATRIX_GREEN_DIM, MATRIX_CYAN, MATRIX_RED,
    MATRIX_ORANGE, MATRIX_YELLOW, MATRIX_BG, MATRIX_GRID,
    MATRIX_BORDER, MATRIX_TEXT, MATRIX_TEXT_DIM,
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
    return os.path.isfile(DB_CONFIG["output"]["database"]["path"])


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
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;">Run a validation pipeline to populate the database.</p>'
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
)

try:
    batches = get_recent_batches(DB_CONFIG, limit=100)
    trend = get_score_trend(DB_CONFIG, limit=100)
except Exception as exc:
    st.error(f"// SYSTEM ERROR: {exc}")
    st.stop()

if batches.empty:
    st.markdown(
        '<div class="glass-card" style="text-align:center;padding:60px 20px;">'
        '<p style="font-family:Orbitron;font-size:1.2rem;color:#00ff41;">// DATABASE EMPTY</p>'
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;">Execute pipeline to generate data stream.</p>'
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

# Status bar
now = time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f'<div style="background:rgba(0,15,2,0.6);border:1px solid rgba(0,255,65,0.15);border-radius:4px;'
    f'padding:6px 14px;margin-bottom:20px;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff41;'
    f'display:flex;gap:20px;align-items:center;">'
    f'<span><span style="display:inline-block;width:7px;height:7px;background:#00ff41;border-radius:50%;'
    f'box-shadow:0 0 8px #00ff41;margin-right:6px;"></span>SYSTEM ONLINE</span>'
    f'<span>LAST SCAN: {batches.iloc[0].get("timestamp", "N/A")}</span>'
    f'<span>BATCHES: {len(batches)}</span>'
    f'<span>LOCAL TIME: {now}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Quality Score Trend
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Radar + Severity
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Batch History
# ---------------------------------------------------------------------------
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

st.dataframe(
    display_df, use_container_width=True, hide_index=True,
    height=min(350, 40 + 35 * len(display_df)),
    column_config={
        "Score": st.column_config.ProgressColumn("Score", format="%.1f%%", min_value=0, max_value=100),
        "Batch ID": st.column_config.TextColumn("Batch ID", width="medium"),
        "Pass": st.column_config.TextColumn("Status", width="small"),
    },
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Batch Comparison + Pipeline Performance
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Batch Drill-Down
# ---------------------------------------------------------------------------
section_header("// Deep Scan : Batch Drill-Down")

batch_ids = batches["batch_id"].tolist()
selected_drill = st.selectbox("SELECT BATCH FOR ANALYSIS", batch_ids, index=0, key="drill_batch")

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
            '<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.82rem;'
            'letter-spacing:2px;">// FULL ISSUE LOG</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(drill_issues, use_container_width=True, hide_index=True, height=360)

        csv_export = drill_issues.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="EXPORT ISSUES [CSV]",
            data=csv_export,
            file_name=f"issues_{selected_drill}.csv",
            mime="text/csv",
        )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AI Enrichment Intelligence
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-header" style="border-image:linear-gradient(90deg, #00e5ff, #00ff41, transparent) 1;">'
    '// Neural Net Intelligence '
    '<span style="display:inline-block;background:rgba(0,229,255,0.12);border:1px solid rgba(0,229,255,0.35);'
    'color:#00e5ff;font-size:0.58rem;padding:3px 10px;border-radius:12px;letter-spacing:2px;margin-left:10px;'
    'vertical-align:middle;">AI POWERED</span></div>',
    unsafe_allow_html=True,
)

try:
    ai_batches = get_batches_with_ai(DB_CONFIG)
except Exception:
    ai_batches = []

if not ai_batches:
    st.markdown(
        '<div style="background:rgba(0,5,1,0.6);border:1px dashed rgba(0,229,255,0.2);border-radius:8px;'
        'padding:30px;text-align:center;font-family:Share Tech Mono;color:#4a7a4f;font-size:0.82rem;letter-spacing:2px;">'
        '// NO AI ENRICHMENT DATA<br>'
        '<span style="font-size:0.72rem;">Run validation with AI enrichment enabled to generate intelligence feeds.</span>'
        '</div>',
        unsafe_allow_html=True,
    )
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
                    f'<div style="font-size:0.72rem;color:#4a7a4f;margin-top:4px;">REC: {rec}</div>'
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
                    f'<span style="color:#4a7a4f;font-size:0.65rem;">CONFIDENCE: {conf_pct}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Pending Fixes — Human Review Gate (session-level)
# ---------------------------------------------------------------------------
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
        'padding:24px;text-align:center;font-family:Share Tech Mono;color:#4a7a4f;font-size:0.82rem;">'
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
            'padding:20px;font-family:Share Tech Mono;color:#4a7a4f;font-size:0.78rem;">'
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
            _status_color = "#4a7a4f" if _already_applied else "#ff9100"
            _status_text = "FIXES APPLIED" if _already_applied else "AWAITING REVIEW"

            st.markdown(
                f'<div style="display:flex;gap:20px;align-items:center;margin-bottom:16px;'
                f'font-family:Share Tech Mono;font-size:0.82rem;">'
                f'<span style="color:{_status_color};border:1px solid {_status_color};'
                f'padding:3px 12px;border-radius:4px;letter-spacing:2px;">{_status_text}</span>'
                f'<span style="color:#b0ffb8;">{len(_fixable)} suggested fixes in current session</span>'
                f'<span style="color:#4a7a4f;">Dataset: {len(_df_raw):,} rows</span>'
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

# Footer
st.markdown(
    '<div style="text-align:center;padding:30px 0 10px 0;">'
    '<p style="color:#00ff4120;font-family:Share Tech Mono;font-size:0.6rem;letter-spacing:3px;">'
    'AI POWERED DQ INVESTIGATOR // COMMAND CENTER // v2.0 // THERE IS NO SPOON</p></div>',
    unsafe_allow_html=True,
)
