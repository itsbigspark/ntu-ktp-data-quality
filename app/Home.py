"""
Home.py -- Entry point for the AI Powered DQ Investigator.

Run: streamlit run app/Home.py
"""

import sys
import os

# Ensure project root is on the path so core/ and dq_engine/ are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="AI Powered DQ Investigator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared.state import init_state
from shared.auth import check_auth, login_page

# Initialize all session state
init_state()

# ---------------------------------------------------------------------------
# Auth gate — show login, then redirect to Command Center
# ---------------------------------------------------------------------------
if not check_auth():
    from shared.theme import apply_theme
    apply_theme(show_rain=True)
    login_page()
    st.stop()

# Authenticated: redirect straight to Command Center (the dashboard)
st.switch_page("pages/5_Command_Center.py")
