"""
Page 16: Cross-Batch Trending -- Quality scores and dimensions over time.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header,
    MATRIX_GREEN, MATRIX_GREEN_DIM, MATRIX_CYAN, MATRIX_CYAN_DIM,
    MATRIX_RED, MATRIX_ORANGE, MATRIX_YELLOW,
    MATRIX_BG, MATRIX_GRID, MATRIX_BORDER, MATRIX_TEXT, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("QUALITY TRENDING", "Cross-batch quality scores over time")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMS = ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]

DIM_COLOURS = [
    MATRIX_GREEN,    # completeness
    MATRIX_CYAN,     # uniqueness
    MATRIX_ORANGE,   # consistency
    MATRIX_YELLOW,   # validity
    "#b388ff",       # accuracy  (soft purple)
    "#40c4ff",       # timeliness (light blue)
]

ROLLING_WINDOW = 7

# ---------------------------------------------------------------------------
# Chart layout helper (matches Command Center style)
# ---------------------------------------------------------------------------

def _chart_layout(height=350, show_legend=False, legend_pos=None):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0, 10, 2, 0.6)",
        font=dict(color=MATRIX_TEXT, family="'Courier New', monospace", size=12),
        xaxis=dict(
            gridcolor=MATRIX_GRID,
            linecolor=MATRIX_BORDER,
            zerolinecolor=MATRIX_GRID,
            tickfont=dict(color=MATRIX_TEXT_DIM),
        ),
        yaxis=dict(
            gridcolor=MATRIX_GRID,
            linecolor=MATRIX_BORDER,
            zerolinecolor=MATRIX_GRID,
            tickfont=dict(color=MATRIX_TEXT_DIM),
        ),
        margin=dict(l=55, r=20, t=40, b=55),
        height=height,
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor="rgba(0,15,2,0.95)",
            bordercolor=MATRIX_GREEN,
            font=dict(color=MATRIX_GREEN, family="'Courier New', monospace"),
        ),
    )
    if show_legend and legend_pos:
        layout["legend"] = legend_pos
    return layout


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_engine():
    try:
        from core.storage.database import get_engine
        return get_engine()
    except Exception:
        return None


def _load_trend(engine, limit: int = 30) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT batch_id, timestamp, source_file, rows_processed,
               overall_score, pass, issues_count,
               completeness, uniqueness, consistency,
               validity, accuracy, timeliness
        FROM batch_runs
        ORDER BY timestamp DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"lim": limit})
    # Reverse so oldest first for charts
    return df.iloc[::-1].reset_index(drop=True)


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
    trend = _load_trend(engine, limit=30)
except Exception as exc:
    st.error(f"// QUERY ERROR: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Fewer-than-2-batches guard
# ---------------------------------------------------------------------------

if len(trend) < 2:
    st.info(
        "// Not enough data to display trends. "
        f"At least 2 batch runs are required — currently {len(trend)} found. "
        "Run more validation pipelines and return here."
    )
    if not trend.empty:
        st.dataframe(trend, use_container_width=True)
    st.stop()

# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

import time as _time
now = _time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    f'<div style="background:rgba(0,15,2,0.6);border:1px solid rgba(0,255,65,0.15);border-radius:4px;'
    f'padding:6px 14px;margin-bottom:20px;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff41;'
    f'display:flex;gap:20px;align-items:center;">'
    f'<span><span style="display:inline-block;width:7px;height:7px;background:#00ff41;border-radius:50%;'
    f'box-shadow:0 0 8px #00ff41;margin-right:6px;"></span>LIVE FEED</span>'
    f'<span>BATCHES LOADED: {len(trend)}</span>'
    f'<span>FIRST: {trend.iloc[0].get("timestamp", "N/A")}</span>'
    f'<span>LATEST: {trend.iloc[-1].get("timestamp", "N/A")}</span>'
    f'<span>LOCAL TIME: {now}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# X-axis labels: use timestamp if available, otherwise batch index
x_labels = trend["timestamp"].astype(str).tolist() if "timestamp" in trend.columns else list(range(len(trend)))

# ---------------------------------------------------------------------------
# 1. Overall Score Trend
# ---------------------------------------------------------------------------

section_header("// Signal Trace : Overall Quality Score")

layout1 = _chart_layout(height=320)
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=x_labels, y=trend["overall_score"],
    mode="lines", line=dict(color=MATRIX_GREEN, width=0),
    fill="tozeroy", fillcolor="rgba(0,255,65,0.05)",
    showlegend=False, hoverinfo="skip",
))
fig1.add_trace(go.Scatter(
    x=x_labels, y=trend["overall_score"],
    mode="lines+markers",
    line=dict(color=MATRIX_GREEN, width=2.5, shape="spline"),
    marker=dict(size=8, color=MATRIX_BG, line=dict(color=MATRIX_GREEN, width=2)),
    name="Overall Score",
    hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
))
fig1.add_hline(
    y=85, line_dash="dot", line_color=MATRIX_RED, line_width=1.5,
    annotation_text="THRESHOLD 85%",
    annotation_font=dict(color=MATRIX_RED, size=10, family="Share Tech Mono"),
)
layout1["yaxis"]["title"] = dict(text="Score (%)", font=dict(color=MATRIX_TEXT_DIM, size=11))
layout1["yaxis"]["range"] = [0, 105]
fig1.update_layout(**layout1)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Quality Dimensions Multi-line
# ---------------------------------------------------------------------------

section_header("// Dimension Scan : All 6 Quality Dimensions")

layout2 = _chart_layout(
    height=380, show_legend=True,
    legend_pos=dict(
        x=1.01, y=1, bgcolor="rgba(0,10,2,0.8)",
        bordercolor=MATRIX_BORDER, borderwidth=1,
        font=dict(color=MATRIX_TEXT, size=11),
    ),
)
fig2 = go.Figure()

for dim, colour in zip(DIMS, DIM_COLOURS):
    if dim not in trend.columns:
        continue
    fig2.add_trace(go.Scatter(
        x=x_labels,
        y=trend[dim],
        mode="lines+markers",
        name=dim.upper(),
        line=dict(color=colour, width=2, shape="spline"),
        marker=dict(size=6, color=colour),
        hovertemplate=f"<b>%{{x}}</b><br>{dim.upper()}: %{{y:.1f}}%<extra></extra>",
    ))

fig2.add_hline(
    y=85, line_dash="dot", line_color=MATRIX_RED, line_width=1,
    annotation_text="85% threshold",
    annotation_font=dict(color=MATRIX_RED, size=9, family="Share Tech Mono"),
)
layout2["yaxis"]["title"] = dict(text="Score (%)", font=dict(color=MATRIX_TEXT_DIM, size=11))
layout2["yaxis"]["range"] = [0, 105]
fig2.update_layout(**layout2)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. Issues count bar chart
# ---------------------------------------------------------------------------

section_header("// Issue Volume : Issues per Batch")

layout3 = _chart_layout(height=280)
fig3 = go.Figure()

bar_colours = [
    MATRIX_RED if v > 50 else (MATRIX_ORANGE if v > 20 else MATRIX_GREEN_DIM)
    for v in trend["issues_count"].fillna(0)
]
fig3.add_trace(go.Bar(
    x=x_labels,
    y=trend["issues_count"].fillna(0),
    marker=dict(
        color=bar_colours,
        line=dict(color="rgba(0,0,0,0)", width=0),
        opacity=0.85,
    ),
    name="Issues",
    hovertemplate="<b>%{x}</b><br>Issues: %{y}<extra></extra>",
))
layout3["yaxis"]["title"] = dict(text="Issue Count", font=dict(color=MATRIX_TEXT_DIM, size=11))
fig3.update_layout(**layout3)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. Rolling pass/fail rate (7-batch window)
# ---------------------------------------------------------------------------

section_header(f"// Pass/Fail Rate : Rolling {ROLLING_WINDOW}-Batch Window")

trend["pass_bool"] = trend["pass"].astype(bool).astype(int)
roll_pass = trend["pass_bool"].rolling(window=ROLLING_WINDOW, min_periods=1).mean() * 100
roll_fail = 100 - roll_pass

layout4 = _chart_layout(height=280)
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=x_labels, y=roll_pass,
    mode="lines+markers",
    line=dict(color=MATRIX_GREEN, width=2.5, shape="spline"),
    marker=dict(size=7, color=MATRIX_GREEN),
    fill="tozeroy", fillcolor="rgba(0,255,65,0.07)",
    name="Pass Rate %",
    hovertemplate="<b>%{x}</b><br>Pass: %{y:.0f}%<extra></extra>",
))
fig4.add_trace(go.Scatter(
    x=x_labels, y=roll_fail,
    mode="lines",
    line=dict(color=MATRIX_RED, width=1.5, dash="dot", shape="spline"),
    name="Fail Rate %",
    hovertemplate="<b>%{x}</b><br>Fail: %{y:.0f}%<extra></extra>",
))
fig4.add_hline(y=85, line_dash="dot", line_color=MATRIX_CYAN, line_width=1,
               annotation_text="85% target",
               annotation_font=dict(color=MATRIX_CYAN, size=9, family="Share Tech Mono"))

legend_pos4 = dict(
    x=1.01, y=1, bgcolor="rgba(0,10,2,0.8)",
    bordercolor=MATRIX_BORDER, borderwidth=1,
    font=dict(color=MATRIX_TEXT, size=11),
)
layout4["showlegend"] = True
layout4["legend"] = legend_pos4
layout4["yaxis"]["title"] = dict(text="Rate (%)", font=dict(color=MATRIX_TEXT_DIM, size=11))
layout4["yaxis"]["range"] = [0, 105]
fig4.update_layout(**layout4)

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Score table with colour coding
# ---------------------------------------------------------------------------

section_header("// Score Matrix : Colour-Coded by Threshold")

score_cols = ["timestamp", "source_file", "overall_score"] + [d for d in DIMS if d in trend.columns]
table_df = trend[score_cols].copy()
table_df = table_df.iloc[::-1].reset_index(drop=True)  # newest first in table


def _colour_cell(val):
    try:
        v = float(val)
        if v >= 85:
            return f"color: {MATRIX_GREEN}; font-weight: bold"
        if v >= 70:
            return f"color: {MATRIX_ORANGE}"
        return f"color: {MATRIX_RED}"
    except (TypeError, ValueError):
        return ""


numeric_score_cols = [c for c in ["overall_score"] + DIMS if c in table_df.columns]

styled_table = (
    table_df.style
    .applymap(_colour_cell, subset=numeric_score_cols)
    .format({c: "{:.1f}" for c in numeric_score_cols}, na_rep="N/A")
    .set_properties(**{
        "font-family": "'Share Tech Mono', monospace",
        "font-size": "0.78rem",
        "background-color": "rgba(0,15,2,0.6)",
        "color": MATRIX_TEXT,
    })
)

st.dataframe(styled_table, use_container_width=True)

# ---------------------------------------------------------------------------
# Legend note
# ---------------------------------------------------------------------------

st.markdown(
    '<div style="margin-top:16px;font-family:Share Tech Mono;font-size:0.72rem;'
    'color:#4a7a4f;display:flex;gap:20px;align-items:center;">'
    f'<span style="color:{MATRIX_GREEN};">&#9632; &gt;= 85% PASS</span>'
    f'<span style="color:{MATRIX_ORANGE};">&#9632; 70-84% WARN</span>'
    f'<span style="color:{MATRIX_RED};">&#9632; &lt; 70% FAIL</span>'
    '</div>',
    unsafe_allow_html=True,
)
