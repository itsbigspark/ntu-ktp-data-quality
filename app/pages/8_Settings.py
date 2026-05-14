"""
Page 8: Settings -- Configure AI providers, data sources, and system options.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, terminal_block
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("SETTINGS", "Configure AI providers and system options")

# ---------------------------------------------------------------------------
# AI Provider Configuration
# ---------------------------------------------------------------------------
section_header("// AI Provider")

provider = st.selectbox(
    "Select AI Provider",
    ["ollama", "anthropic", "openai", "gemini"],
    index=["ollama", "anthropic", "openai", "gemini"].index(
        st.session_state.get("ai_provider_type", "ollama")
    ),
    key="settings_provider",
    format_func=lambda x: {
        "ollama": "Ollama (Local / Offline)",
        "anthropic": "Anthropic (Claude API)",
        "openai": "OpenAI (GPT API)",
        "gemini": "Google Gemini API",
    }.get(x, x),
)
st.session_state["ai_provider_type"] = provider

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if provider == "ollama":
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00ff41;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">OLLAMA (LOCAL)</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">Runs entirely on your machine. No API key needed. '
        'No data leaves your computer. Requires Ollama installed and running.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        model = st.text_input(
            "Model name",
            value=st.session_state.get("ollama_model", "phi3:mini"),
            key="ollama_model_input",
        )
        st.session_state["ollama_model"] = model
    with col2:
        url = st.text_input(
            "Ollama URL",
            value=st.session_state.get("ollama_url", "http://localhost:11434"),
            key="ollama_url_input",
        )
        st.session_state["ollama_url"] = url

    if st.button("Test Ollama Connection", key="test_ollama"):
        try:
            import requests
            resp = requests.get(f"{url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                st.success(f"Connected. Available models: {', '.join(models)}")
            else:
                st.warning("Connected but no models found. Run: ollama pull phi3:mini")
        except Exception as e:
            st.error(f"Cannot reach Ollama at {url}. Is the Ollama service running? (ollama serve)")

elif provider == "anthropic":
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">ANTHROPIC (CLAUDE)</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">Uses Claude API for high-quality AI enrichment. '
        'Requires an API key from console.anthropic.com. Data statistics (not raw data) are sent to the API.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.get("anthropic_api_key", ""),
        type="password",
        key="anthropic_key_input",
    )
    st.session_state["anthropic_api_key"] = api_key

    _anthropic_models = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]
    _current_model = st.session_state.get("anthropic_model", "claude-opus-4-6")
    _model_idx = _anthropic_models.index(_current_model) if _current_model in _anthropic_models else 0
    anthropic_model = st.selectbox(
        "Claude Model",
        _anthropic_models,
        index=_model_idx,
        key="anthropic_model_select",
    )
    st.session_state["anthropic_model"] = anthropic_model

    if api_key and st.button("Test Anthropic Connection", key="test_anthropic"):
        try:
            from dq_engine import AnthropicProvider
            p = AnthropicProvider(api_key=api_key)
            resp = p.call("Reply with exactly: CONNECTION_OK", temperature=0)
            if "CONNECTION_OK" in resp or len(resp) > 0:
                st.success(f"Connected to Anthropic API. Response: {resp[:50]}")
            else:
                st.warning("Connected but unexpected response")
        except Exception as e:
            st.error("Anthropic connection failed. Check your API key is correct and has credits.")

elif provider == "openai":
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">OPENAI (GPT)</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">Uses OpenAI GPT API. '
        'Requires an API key from platform.openai.com.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        key="openai_key_input",
    )
    st.session_state["openai_api_key"] = api_key

    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        key="openai_model_select",
    )
    st.session_state["openai_model"] = model

elif provider == "gemini":
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">GOOGLE GEMINI</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">Uses Google Gemini API. '
        'Requires an API key from ai.google.dev.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        key="gemini_key_input",
    )
    st.session_state["gemini_api_key"] = api_key

# ---------------------------------------------------------------------------
# Data Source Connectors
# ---------------------------------------------------------------------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("// Data Source Connectors")

connector_tab1, connector_tab2, connector_tab3 = st.tabs(["S3 Bucket", "Database", "API Endpoint"])

with connector_tab1:
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">AWS S3</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">'
        'Configure S3 buckets for data input and output. '
        'Data loading is done on the Load Data page. '
        'Requires AWS credentials (via environment variables or IAM role).</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;letter-spacing:1px;">'
        'S3 OUTPUT BUCKET (for saving results)</p>',
        unsafe_allow_html=True,
    )

    s3o_col1, s3o_col2 = st.columns(2)
    with s3o_col1:
        s3_out_bucket = st.text_input("Output Bucket Name", value=st.session_state.get("s3_output_bucket", ""), key="s3_out_bucket_settings", placeholder="e.g. dq-investigator-output")
        st.session_state["s3_output_bucket"] = s3_out_bucket
    with s3o_col2:
        s3_out_region = st.text_input("Output Region", value=st.session_state.get("s3_region", "us-east-1"), key="s3_out_region_settings")
        st.session_state["s3_region"] = s3_out_region

    if s3_out_bucket and st.button("Test S3 Output Connection", key="test_s3_out"):
        try:
            import boto3
            s3 = boto3.client("s3", region_name=s3_out_region)
            s3.head_bucket(Bucket=s3_out_bucket)
            st.success(f"Connected to s3://{s3_out_bucket}/")
        except Exception as e:
            st.error("S3 connection failed. Check your bucket name, region, and AWS credentials.")

    st.markdown(
        '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.72rem;margin-top:8px;">'
        'When set, validation results (issue log, quality report) are automatically saved to this bucket '
        'under reports/ and corrected/ prefixes.</p>',
        unsafe_allow_html=True,
    )

with connector_tab2:
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">DATABASE</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">'
        'Connect to PostgreSQL, MySQL, or any SQLAlchemy-compatible database to load data via SQL query.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    db_url = st.text_input(
        "Database URL",
        value=st.session_state.get("source_db_url", ""),
        key="source_db_url_input",
        placeholder="postgresql://user:pass@host:5432/dbname",
        type="password",
    )
    st.session_state["source_db_url"] = db_url

    db_query = st.text_area(
        "SQL Query",
        value=st.session_state.get("source_db_query", "SELECT * FROM transactions LIMIT 1000"),
        key="source_db_query_input",
        height=80,
    )
    st.session_state["source_db_query"] = db_query

    if db_url and st.button("Test Database Connection", key="test_db"):
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                st.success("Database connection successful.")
        except Exception as e:
            st.error("Database connection failed. Check the URL format and that the host is reachable.")

    if db_url and db_query and st.button("Load Data from Database", key="load_db"):
        try:
            import pandas as pd
            from sqlalchemy import create_engine
            engine = create_engine(db_url)
            df = pd.read_sql(db_query, engine)
            st.session_state["df_raw_full"] = df
            st.session_state["df_raw"] = df
            st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from database")
        except Exception as e:
            st.error("Failed to load data from database. Check your SQL query and connection URL.")

with connector_tab3:
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#00e5ff;font-family:Orbitron;font-size:0.8rem;letter-spacing:2px;">REST API</p>'
        '<p style="color:#b0ffb8;font-size:0.78rem;">'
        'Fetch data from a REST API endpoint that returns JSON. '
        'The response should be a JSON array of objects.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    api_url = st.text_input("API URL", value=st.session_state.get("source_api_url", ""), key="source_api_url_input", placeholder="https://api.example.com/data")
    st.session_state["source_api_url"] = api_url

    api_headers = st.text_input(
        "Authorization Header (optional)",
        value=st.session_state.get("source_api_header", ""),
        key="source_api_header_input",
        placeholder="Bearer your-token-here",
        type="password",
    )
    st.session_state["source_api_header"] = api_headers

    if api_url and st.button("Fetch Data from API", key="load_api"):
        try:
            import requests
            import pandas as pd

            headers = {}
            if api_headers:
                headers["Authorization"] = api_headers

            resp = requests.get(api_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            elif isinstance(data, dict) and "results" in data:
                df = pd.DataFrame(data["results"])
            else:
                df = pd.DataFrame([data])

            st.session_state["df_raw_full"] = df
            st.session_state["df_raw"] = df
            st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from API")
        except Exception as e:
            st.error("API request failed. Check the URL is reachable and the response is JSON.")

# ---------------------------------------------------------------------------
# Database Configuration (results storage)
# ---------------------------------------------------------------------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("// Results Database")

st.markdown(
    '<div class="glass-card">'
    '<p style="color:#b0ffb8;font-size:0.78rem;">'
    'Validation results are stored in a local SQLite database at <code style="color:#00e5ff;">'
    './output/dq_investigator.db</code>. '
    'For enterprise deployment, configure PostgreSQL via DATABASE_URL environment variable.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Check DB status
db_path = "./output/dq_investigator.db"
if os.path.exists(db_path):
    size_kb = os.path.getsize(db_path) / 1024
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM batch_runs")
        batch_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM issues")
        issue_count = cursor.fetchone()[0]
        conn.close()
        st.markdown(
            f'<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;">'
            f'Database: ONLINE ({size_kb:.0f} KB) | {batch_count} batches | {issue_count} issues</p>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            f'<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;">'
            f'Database: ONLINE ({size_kb:.0f} KB)</p>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<p style="color:#ff9100;font-family:Share Tech Mono;font-size:0.78rem;">'
        'Database: NOT FOUND (will be created on first validation run)</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# System Info
# ---------------------------------------------------------------------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("// System Info")

import platform

info_items = [
    ("Python", platform.python_version()),
    ("Platform", platform.platform()),
    ("Engine", "dq_engine v1.0.0"),
    ("AI Provider", provider),
]

for label, value in info_items:
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
        f'border-bottom:1px solid rgba(0,255,65,0.06);font-family:Share Tech Mono;font-size:0.78rem;">'
        f'<span style="color:#5a9a5a;">{label}</span>'
        f'<span style="color:#b0ffb8;">{value}</span></div>',
        unsafe_allow_html=True,
    )
