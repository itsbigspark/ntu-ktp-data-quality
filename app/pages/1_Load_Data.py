"""
Page 1: Load Data -- Upload CSV/Parquet/Excel files for analysis.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, terminal_block
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("LOAD DATA", "Upload datasets for quality analysis")

# ---------------------------------------------------------------------------
# Data source selector
# ---------------------------------------------------------------------------
section_header("// Data Ingestion")

source_mode = st.radio(
    "Data Source",
    ["File Upload", "AWS S3 Bucket"],
    horizontal=True,
    key="data_source_mode",
)

raw_file = None
ref_file = None

if source_mode == "File Upload":
    col_raw, col_ref = st.columns(2, gap="large")

    with col_raw:
        st.markdown(
            '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
            'letter-spacing:1px;">PRIMARY DATASET (required)</p>',
            unsafe_allow_html=True,
        )
        raw_file = st.file_uploader(
            "Upload raw/unclean data",
            type=["csv", "parquet", "xlsx", "xls", "json"],
            key="raw_upload",
        )

    with col_ref:
        st.markdown(
            '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.82rem;'
            'letter-spacing:1px;">REFERENCE DATASET (optional)</p>',
            unsafe_allow_html=True,
        )
        ref_file = st.file_uploader(
            "Upload clean reference data",
            type=["csv", "parquet", "xlsx", "xls", "json"],
            key="ref_upload",
        )

else:
    # ── S3 Source ──────────────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
        'letter-spacing:1px;">CONNECT TO S3 BUCKET</p>',
        unsafe_allow_html=True,
    )

    s3_col1, s3_col2 = st.columns(2, gap="large")
    with s3_col1:
        s3_bucket = st.text_input("S3 Bucket Name", value=st.session_state.get("s3_bucket", ""), key="s3_bucket_input", placeholder="e.g. my-data-bucket")
        s3_region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1", "eu-west-2", "eu-north-1", "ap-southeast-1"], key="s3_region_input")
    with s3_col2:
        s3_prefix = st.text_input("File Path / Prefix", value=st.session_state.get("s3_prefix", "incoming/"), key="s3_prefix_input", placeholder="e.g. incoming/ or data/file.csv")
        s3_ref_key = st.text_input("Reference Data Key (optional)", value="", key="s3_ref_key_input", placeholder="e.g. reference/clean_data.csv")

    s3_action_col1, s3_action_col2 = st.columns(2, gap="large")

    with s3_action_col1:
        if st.button("LIST FILES IN BUCKET", key="s3_list_files", use_container_width=True):
            try:
                from core.storage.s3 import list_inbox_files
                files = list_inbox_files(bucket=s3_bucket, prefix=s3_prefix, region=s3_region)
                if files:
                    st.session_state["s3_file_list"] = files
                    st.success(f"Found {len(files)} files in s3://{s3_bucket}/{s3_prefix}")
                else:
                    st.warning(f"No supported files found in s3://{s3_bucket}/{s3_prefix}")
            except Exception as e:
                st.error(f"Failed to list S3 files: {e}")

    # Show file list and let user select
    if st.session_state.get("s3_file_list"):
        file_options = [f"{f['key']}  ({f['size_kb']} KB)" for f in st.session_state["s3_file_list"]]
        selected_file = st.selectbox("Select a file to load", file_options, key="s3_selected_file")
        selected_key = st.session_state["s3_file_list"][file_options.index(selected_file)]["key"] if selected_file else None
    else:
        selected_key = None
        # Allow direct key entry
        direct_key = s3_prefix if s3_prefix and not s3_prefix.endswith("/") else None
        if direct_key:
            selected_key = direct_key

    with s3_action_col2:
        if st.button("LOAD FROM S3", key="s3_load_file", use_container_width=True):
            if not s3_bucket:
                st.error("Enter a bucket name.")
            elif not selected_key:
                st.error("Select a file or enter a direct file path (not ending with /).")
            else:
                try:
                    from core.storage.s3 import read_from_s3
                    with st.spinner(f"Loading s3://{s3_bucket}/{selected_key}..."):
                        df = read_from_s3(bucket=s3_bucket, key=selected_key, region=s3_region)
                    st.session_state["df_raw_full"] = df
                    st.session_state["df_raw"] = df
                    st.session_state["s3_bucket"] = s3_bucket
                    st.session_state["s3_source_key"] = selected_key
                    st.session_state["s3_region"] = s3_region
                    st.success(f"Loaded from S3: {selected_key} -- {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Failed to load from S3: {e}")
                else:
                    st.rerun()

            # Load reference data from S3 if specified
            if s3_ref_key and s3_bucket:
                try:
                    from core.storage.s3 import read_from_s3
                    df_ref = read_from_s3(bucket=s3_bucket, key=s3_ref_key, region=s3_region)
                    st.session_state["df_ref"] = df_ref
                    st.success(f"Reference loaded from S3: {s3_ref_key} -- {df_ref.shape[0]} rows")
                except Exception as e:
                    import streamlit.runtime.scriptrunner as _sr
                    if isinstance(e, _sr.StopException):
                        raise
                    st.warning(f"Could not load reference from S3: {e}")

# Rules JSON upload
with st.expander("Import Rules JSON (optional)", expanded=False):
    rules_file = st.file_uploader(
        "Upload pre-defined validation rules",
        type=["json"],
        key="rules_upload",
    )
    if rules_file is not None:
        import json
        try:
            st.session_state["rules_json"] = json.load(rules_file)
            st.success(f"Rules loaded: {len(st.session_state['rules_json'].get('columns', {}))} columns")
        except Exception as e:
            st.error(f"Failed to parse rules JSON: {e}")

# Quick-load demo data from disk
_demo_path = os.path.join(_ROOT, "TEST2_DATA", "main_data", "customer_transactions.csv")
if os.path.exists(_demo_path) and st.session_state.get("df_raw") is None:
    if st.button("Load Demo Dataset (TEST2_DATA)", type="secondary"):
        df_demo = pd.read_csv(_demo_path)
        st.session_state["df_raw_full"] = df_demo
        st.session_state["df_raw"] = df_demo
        st.rerun()

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load files
# ---------------------------------------------------------------------------
def _load_file(uploaded_file) -> pd.DataFrame:
    """Load an uploaded file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}")


if raw_file is not None:
    try:
        df = _load_file(raw_file)
        st.session_state["df_raw_full"] = df
        st.session_state["df_raw"] = df
        st.success(f"Loaded: {raw_file.name} -- {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to load file: {e}")

if ref_file is not None:
    try:
        df_ref = _load_file(ref_file)
        st.session_state["df_ref"] = df_ref
        st.success(f"Reference loaded: {ref_file.name} -- {df_ref.shape[0]} rows, {df_ref.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to load reference: {e}")

# ---------------------------------------------------------------------------
# Sampling controls
# ---------------------------------------------------------------------------
df_raw = st.session_state.get("df_raw")

if df_raw is not None:
    section_header("// Dataset Controls")

    col_sample, col_info = st.columns([1, 2], gap="large")

    with col_sample:
        total_rows = len(st.session_state["df_raw_full"])
        if total_rows < 1:
            st.warning("Uploaded file has 0 rows.")
            st.stop()
        use_rows = st.slider(
            "Rows to use",
            min_value=min(10, total_rows),
            max_value=max(10, total_rows),
            value=min(500, total_rows),
            step=10,
            key="sample_rows",
        )
        sample_method = st.radio(
            "Sampling method",
            ["Head (first N rows)", "Random sample"],
            key="sample_method",
        )

        if st.button("Apply Sampling", key="apply_sample"):
            full = st.session_state["df_raw_full"]
            if sample_method == "Head (first N rows)":
                st.session_state["df_raw"] = full.head(use_rows)
            else:
                st.session_state["df_raw"] = full.sample(n=use_rows, random_state=42)
            st.success(f"Working dataset: {len(st.session_state['df_raw'])} rows")
            st.rerun()

    with col_info:
        df_show = st.session_state["df_raw"]
        st.markdown(
            f'<div class="glass-card">'
            f'<p style="color:#00ff41;font-family:Orbitron;font-size:0.85rem;letter-spacing:2px;">DATASET SUMMARY</p>'
            f'<table style="font-family:Share Tech Mono;font-size:0.8rem;color:#b0ffb8;width:100%;">'
            f'<tr><td style="color:#4a7a4f;">Rows:</td><td>{len(df_show):,}</td>'
            f'<td style="color:#4a7a4f;">Columns:</td><td>{len(df_show.columns)}</td></tr>'
            f'<tr><td style="color:#4a7a4f;">Missing cells:</td><td>{int(df_show.isna().sum().sum()):,}</td>'
            f'<td style="color:#4a7a4f;">Duplicates:</td><td>{int(df_show.duplicated().sum()):,}</td></tr>'
            f'<tr><td style="color:#4a7a4f;">Memory:</td><td>{df_show.memory_usage(deep=True).sum() / 1024:.1f} KB</td>'
            f'<td style="color:#4a7a4f;">Full dataset:</td><td>{total_rows:,} rows</td></tr>'
            f'</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Data preview
    section_header("// Data Preview")
    st.dataframe(df_show.head(50), use_container_width=True, hide_index=False, height=400)

    # Column types
    with st.expander("Column Types", expanded=False):
        dtype_df = pd.DataFrame({
            "Column": df_show.columns,
            "Type": df_show.dtypes.astype(str).values,
            "Non-Null": df_show.notna().sum().values,
            "Null %": (df_show.isna().mean() * 100).round(1).values,
            "Unique": df_show.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

else:
    terminal_block(
        "// AWAITING DATA UPLOAD<br>"
        "<span style='color:#4a7a4f;'>Upload a CSV, Parquet, or Excel file to begin analysis.</span>"
    )
