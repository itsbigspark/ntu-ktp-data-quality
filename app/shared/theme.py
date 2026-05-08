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
