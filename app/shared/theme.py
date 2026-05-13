"""
Matrix theme for the DQ Investigator.

Shared across all pages. Call apply_theme() at the top of each page
to get consistent styling.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------
MATRIX_GREEN = "#00ff41"
MATRIX_GREEN_DIM = "#00cc33"
MATRIX_GREEN_GLOW = "#00ff4180"
MATRIX_CYAN = "#00e5ff"
MATRIX_CYAN_DIM = "#00b8d4"
MATRIX_RED = "#ff1744"
MATRIX_ORANGE = "#ff9100"
MATRIX_YELLOW = "#ffea00"
MATRIX_BG = "#000000"
MATRIX_BG_CARD = "rgba(0, 15, 2, 0.85)"
MATRIX_BORDER = "rgba(0, 255, 65, 0.2)"
MATRIX_BORDER_BRIGHT = "rgba(0, 255, 65, 0.5)"
MATRIX_GRID = "rgba(0, 255, 65, 0.08)"
MATRIX_TEXT = "#b0ffb8"
MATRIX_TEXT_DIM = "#4a7a4f"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ---- global dark background ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #b0ffb8 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] {
    background-color: #000a02 !important;
    border-right: 1px solid rgba(0,255,65,0.1);
}
.stApp > header { background: transparent !important; }

/* ---- scrollbar ---- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #00ff4140; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00ff4180; }

/* ---- sidebar nav links ---- */
[data-testid="stSidebarNav"] a {
    color: #4a7a4f !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
    font-size: 0.85rem;
}
[data-testid="stSidebarNav"] a:hover {
    color: #00ff41 !important;
}
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    color: #00ff41 !important;
    font-weight: bold;
    background: rgba(0,255,65,0.05) !important;
}

/* ---- page title ---- */
.page-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #00ff41;
    text-shadow: 0 0 15px #00ff4180, 0 0 30px #00ff4140;
    letter-spacing: 3px;
    margin: 0 0 4px 0;
    line-height: 1.2;
}
.page-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #4a7a4f;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

/* ---- section headers ---- */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #00ff41;
    text-shadow: 0 0 10px #00ff4160;
    padding-bottom: 6px;
    margin-bottom: 14px;
    border-bottom: 1px solid;
    border-image: linear-gradient(90deg, #00ff41, #00e5ff, transparent) 1;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ---- glass card ---- */
.glass-card {
    background: rgba(0, 15, 2, 0.6);
    border: 1px solid rgba(0, 255, 65, 0.12);
    border-radius: 10px;
    padding: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 12px;
}

/* ---- KPI cards ---- */
.kpi-card {
    background: rgba(0, 15, 2, 0.75);
    border: 1px solid rgba(0, 255, 65, 0.2);
    border-radius: 10px;
    padding: 20px 14px 16px 14px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.05), inset 0 1px 0 rgba(0,255,65,0.1);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ff41, transparent);
    opacity: 0.6;
}
.kpi-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    color: #00ff41;
    text-shadow: 0 0 15px #00ff4180, 0 0 30px #00ff4140;
    line-height: 1.2;
}
.kpi-value.warn { color: #ff9100; text-shadow: 0 0 15px #ff910080; }
.kpi-value.danger { color: #ff1744; text-shadow: 0 0 15px #ff174480; }
.kpi-value.cyan { color: #00e5ff; text-shadow: 0 0 15px #00e5ff80; }
.kpi-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #4a7a4f;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 8px;
}

/* ---- terminal block ---- */
.terminal-block {
    background: rgba(0,5,1,0.9);
    border: 1px solid rgba(0,255,65,0.2);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: #00ff41;
    margin: 10px 0;
}

/* ---- chart container ---- */
@keyframes border-glow {
    0%, 100% { border-color: rgba(0,255,65,0.15); }
    50% { border-color: rgba(0,255,65,0.35); }
}
.chart-container {
    border: 1px solid rgba(0,255,65,0.15);
    border-radius: 10px;
    padding: 8px;
    animation: border-glow 4s ease-in-out infinite;
    background: rgba(0,5,1,0.4);
}

/* ---- buttons ---- */
.stButton > button {
    background: rgba(0,255,65,0.08) !important;
    color: #00ff41 !important;
    border: 1px solid rgba(0,255,65,0.25) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: rgba(0,255,65,0.2) !important;
    box-shadow: 0 0 15px rgba(0,255,65,0.15);
    border-color: rgba(0,255,65,0.5) !important;
}

/* ---- file uploader ---- */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,255,65,0.2);
    border-radius: 8px;
    padding: 10px;
}
[data-testid="stFileUploader"] label {
    color: #4a7a4f !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ---- text inputs ---- */
.stTextInput input, .stSelectbox select, .stNumberInput input {
    background: rgba(0,5,1,0.8) !important;
    color: #00ff41 !important;
    border: 1px solid rgba(0,255,65,0.2) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ---- dataframe styling ---- */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(0,255,65,0.12);
}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(0,5,1,0.6);
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a7a4f !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    color: #00ff41 !important;
    background: rgba(0,255,65,0.1) !important;
}

/* ---- expander ---- */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,255,65,0.12) !important;
    border-radius: 8px;
    background: rgba(0,5,1,0.4);
}

/* ---- info/warning/error ---- */
[data-testid="stAlert"] {
    background: rgba(0,15,2,0.6) !important;
    border: 1px solid rgba(0,255,65,0.2) !important;
    color: #00ff41 !important;
}
</style>
"""

# Digital rain background (optional, for landing/dashboard pages)
_MATRIX_RAIN = """
<canvas id="matrix-rain" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;opacity:0.10;"></canvas>
<script>
(function() {
    const c = document.getElementById('matrix-rain');
    if (!c) return;
    const ctx = c.getContext('2d');
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*';
    const fontSize = 13;
    const columns = Math.floor(c.width / fontSize);
    const drops = Array(columns).fill(1);
    function draw() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.06)';
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.fillStyle = '#00ff41';
        ctx.font = fontSize + 'px "Courier New"';
        for (let i = 0; i < drops.length; i++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            ctx.globalAlpha = Math.random() * 0.5 + 0.1;
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
            if (drops[i] * fontSize > c.height && Math.random() > 0.975) drops[i] = 0;
            drops[i]++;
        }
        ctx.globalAlpha = 1;
    }
    setInterval(draw, 50);
    window.addEventListener('resize', () => { c.width = window.innerWidth; c.height = window.innerHeight; });
})();
</script>
"""


def apply_theme(show_rain: bool = False):
    """Apply the Matrix theme and render the sidebar footer (user + logout)."""
    css = _BASE_CSS
    if show_rain:
        css += _MATRIX_RAIN
    st.markdown(css, unsafe_allow_html=True)

    # Sidebar: show logged-in user and logout button on every page
    user = st.session_state.get("auth_user")
    if user:
        with st.sidebar:
            st.markdown(
                f'<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.7rem;'
                f'letter-spacing:1px;margin:0;padding:4px 0;">USER: {user.upper()}</p>',
                unsafe_allow_html=True,
            )
            if st.button("Logout", key="sidebar_logout", use_container_width=True):
                st.session_state["authenticated"] = False
                st.session_state["auth_user"] = None
                st.rerun()


def page_header(title: str, subtitle: str = ""):
    """Render a consistent page header."""
    html = f'<p class="page-title">{title}</p>'
    if subtitle:
        html += f'<p class="page-subtitle">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


def section_header(text: str):
    """Render a section header."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def kpi_card(value: str, label: str, accent: str = ""):
    """Return HTML for a KPI card."""
    cls = f" {accent}" if accent else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value{cls}">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def terminal_block(text: str):
    """Render a terminal-style message."""
    st.markdown(f'<div class="terminal-block">{text}</div>', unsafe_allow_html=True)


def aggrid_issues(df, height: int = 400):
    """
    Render an issues DataFrame using AG Grid with severity colour coding.
    Falls back to st.dataframe if streamlit-aggrid is not installed.
    """
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
        from st_aggrid.shared import GridUpdateMode

        _sev_col = next((c for c in ["severity", "Severity"] if c in df.columns), None)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            resizable=True,
            sortable=True,
            filter=True,
            wrapText=False,
            autoHeight=False,
        )
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_side_bar(filters_panel=True, columns_panel=False)

        if _sev_col:
            sev_style = JsCode("""
            function(params) {
                const sev = (params.value || '').toLowerCase();
                if (sev === 'critical') return {'color': '#ff1744', 'fontWeight': 'bold', 'background': 'rgba(255,23,68,0.12)'};
                if (sev === 'high')     return {'color': '#ff9100', 'fontWeight': 'bold', 'background': 'rgba(255,145,0,0.10)'};
                if (sev === 'medium')   return {'color': '#ffea00', 'background': 'rgba(255,234,0,0.08)'};
                if (sev === 'low')      return {'color': '#00e5ff', 'background': 'rgba(0,229,255,0.06)'};
                return {};
            }
            """)
            gb.configure_column(_sev_col, cellStyle=sev_style)

        row_style = JsCode("""
        function(params) {
            const sev = (params.data.severity || params.data.Severity || '').toLowerCase();
            if (sev === 'critical') return {'borderLeft': '3px solid #ff1744'};
            if (sev === 'high')     return {'borderLeft': '3px solid #ff9100'};
            if (sev === 'medium')   return {'borderLeft': '3px solid #ffea00'};
            if (sev === 'low')      return {'borderLeft': '3px solid #00e5ff'};
            return {};
        }
        """)

        go = gb.build()
        go["getRowStyle"] = row_style
        go["rowHeight"] = 32
        go["headerHeight"] = 36

        AgGrid(
            df,
            gridOptions=go,
            height=height,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="alpine",
            custom_css={
                ".ag-root-wrapper": {"background": "#000f02", "border": "1px solid rgba(0,255,65,0.2)", "border-radius": "8px"},
                ".ag-header": {"background": "#000f02", "border-bottom": "1px solid rgba(0,255,65,0.3)"},
                ".ag-header-cell-label": {"color": "#00e5ff", "font-family": "Share Tech Mono", "font-size": "0.72rem", "letter-spacing": "1px"},
                ".ag-row": {"background": "#000f02", "border-bottom": "1px solid rgba(0,255,65,0.06)", "font-family": "Share Tech Mono", "font-size": "0.75rem", "color": "#b0ffb8"},
                ".ag-row-hover": {"background": "rgba(0,255,65,0.05) !important"},
                ".ag-cell": {"border-right": "1px solid rgba(0,255,65,0.04)"},
                ".ag-paging-panel": {"background": "#000f02", "color": "#4a7a4f", "font-family": "Share Tech Mono", "font-size": "0.7rem"},
                ".ag-filter-toolpanel": {"background": "#000f02"},
                ".ag-side-bar": {"background": "#000f02", "border-left": "1px solid rgba(0,255,65,0.15)"},
            },
        )
    except ImportError:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def aggrid_plain(df, height: int = 350, page_size: int = 20):
    """
    Render a plain DataFrame using AG Grid (no severity colouring).
    Falls back to st.dataframe if streamlit-aggrid is not installed.
    """
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder
        from st_aggrid.shared import GridUpdateMode

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
        go = gb.build()
        go["rowHeight"] = 32
        go["headerHeight"] = 36

        AgGrid(
            df,
            gridOptions=go,
            height=height,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="alpine",
            custom_css={
                ".ag-root-wrapper": {"background": "#000f02", "border": "1px solid rgba(0,255,65,0.2)", "border-radius": "8px"},
                ".ag-header": {"background": "#000f02", "border-bottom": "1px solid rgba(0,255,65,0.3)"},
                ".ag-header-cell-label": {"color": "#00e5ff", "font-family": "Share Tech Mono", "font-size": "0.72rem", "letter-spacing": "1px"},
                ".ag-row": {"background": "#000f02", "border-bottom": "1px solid rgba(0,255,65,0.06)", "font-family": "Share Tech Mono", "font-size": "0.75rem", "color": "#b0ffb8"},
                ".ag-row-hover": {"background": "rgba(0,255,65,0.05) !important"},
                ".ag-paging-panel": {"background": "#000f02", "color": "#4a7a4f", "font-family": "Share Tech Mono", "font-size": "0.7rem"},
            },
        )
    except ImportError:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def workflow_breadcrumb(steps: list[tuple[str, bool]]):
    """
    Render a horizontal pipeline breadcrumb.

    steps: list of (label, is_complete) tuples in order.
    """
    parts = []
    for i, (label, done) in enumerate(steps):
        color = "#00ff41" if done else "#4a7a4f"
        border = "rgba(0,255,65,0.4)" if done else "rgba(74,122,79,0.3)"
        bg = "rgba(0,255,65,0.06)" if done else "rgba(0,5,1,0.4)"
        tick = " ✓" if done else ""
        parts.append(
            f'<div style="background:{bg};border:1px solid {border};border-radius:6px;'
            f'padding:5px 12px;font-family:Share Tech Mono;font-size:0.7rem;color:{color};">'
            f'{label}{tick}</div>'
        )
        if i < len(steps) - 1:
            parts.append(
                '<div style="color:#4a7a4f;font-size:0.8rem;padding:0 2px;align-self:center;">→</div>'
            )
    html = (
        '<div style="display:flex;gap:6px;align-items:center;'
        'flex-wrap:wrap;margin-bottom:16px;">'
        + "".join(parts)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)
