"""
Home.py -- Entry point for the AI Powered DQ Investigator.

Run: streamlit run app/Home.py
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="DQ Investigator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from shared.state import init_state
from shared.auth import check_auth, login_page

init_state()

if not check_auth():
    from shared.theme import apply_theme
    apply_theme(show_rain=True)
    login_page()
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — grouped navigation sections
# ---------------------------------------------------------------------------
pages = st.navigation(
    {
        "Core Workflow": [
            st.Page("pages/1_Load_Data.py",           title="Load Data",            icon="📂"),
            st.Page("pages/2_Rules.py",               title="Rules",                icon="📋"),
            st.Page("pages/3_Validate.py",            title="Validate",             icon="✅"),
            st.Page("pages/6_Cleaning.py",            title="Cleaning",             icon="🧹"),
            st.Page("pages/7_Dedupe.py",              title="Dedupe",               icon="🔁"),
        ],
        "AI & Intelligence": [
            st.Page("pages/4_AI_Investigation.py",    title="AI Investigation",     icon="🤖"),
            st.Page("pages/5_Command_Center.py",      title="Command Center",       icon="📡"),
            st.Page("pages/12_Entity_Resolution.py",  title="Entity Resolution",    icon="🔗"),
            st.Page("pages/13_Entity_Graph.py",       title="Entity Graph",         icon="🕸️"),
        ],
        "Data & Analytics": [
            st.Page("pages/14_External_Validation.py",title="External Validation",  icon="🌐"),
            st.Page("pages/15_Batch_History.py",      title="Batch History",        icon="🕒"),
            st.Page("pages/16_Trending.py",           title="Trending",             icon="📈"),
            st.Page("pages/17_AI_Audit_Log.py",       title="AI Audit Log",         icon="🔍"),
        ],
        "Advanced": [
            st.Page("pages/9_Corpus_Manager.py",      title="Corpus Manager",       icon="📚"),
            st.Page("pages/10_Pipeline_Manager.py",   title="Pipeline Manager",     icon="🔧"),
            st.Page("pages/11_Vector_Index.py",       title="Vector Index",         icon="🗂️"),
        ],
        "Configuration": [
            st.Page("pages/8_Settings.py",            title="Settings",             icon="⚙️"),
        ],
    },
    position="sidebar",
)
pages.run()
