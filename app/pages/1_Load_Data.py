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
    ["File Upload", "AWS S3 Bucket", "Database", "REST API", "Google Sheets"],
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

elif source_mode == "AWS S3 Bucket":
    # ── S3 Source ──────────────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
        'letter-spacing:1px;">CONNECT TO S3 BUCKET</p>',
        unsafe_allow_html=True,
    )

    s3_col1, s3_col2 = st.columns(2, gap="large")
    with s3_col1:
        s3_bucket = st.text_input(
            "S3 Bucket Name",
            value=st.session_state.get("s3_bucket", "ntu-dq-investigator-data"),
            key="s3_bucket_input",
            placeholder="e.g. my-data-bucket",
        )
    with s3_col2:
        s3_region = st.selectbox(
            "AWS Region",
            ["us-east-1", "us-west-2", "eu-west-1", "eu-west-2", "eu-north-1", "ap-southeast-1"],
            key="s3_region_input",
        )

    # ── Browse button ──────────────────────────────────────────────────────
    s3_prefix = st.text_input(
        "Prefix / folder to browse (optional)",
        value=st.session_state.get("s3_prefix", ""),
        key="s3_prefix_input",
        placeholder="e.g. TEST2_DATA/  or leave blank to list everything",
    )

    if st.button("BROWSE BUCKET", key="s3_list_files", use_container_width=False):
        if not s3_bucket:
            st.error("Enter a bucket name.")
        else:
            try:
                from core.storage.s3 import list_inbox_files
                files = list_inbox_files(bucket=s3_bucket, prefix=s3_prefix, region=s3_region)
                if files:
                    st.session_state["s3_file_list"] = files
                    st.session_state["s3_bucket"] = s3_bucket
                    st.session_state["s3_region"] = s3_region
                    st.success(f"Found {len(files)} files")
                else:
                    st.warning(f"No supported files found. Try a different prefix.")
            except Exception as e:
                st.error(f"Failed to browse bucket: {e}")

    # ── File selectors ─────────────────────────────────────────────────────
    file_list = st.session_state.get("s3_file_list", [])
    file_options = ["(none)"] + [f"{f['key']}  ({f['size_kb']} KB)" for f in file_list]

    def _key_from_option(opt):
        if not opt or opt == "(none)":
            return None
        # strip the size suffix
        return file_list[[f"{f['key']}  ({f['size_kb']} KB)" for f in file_list].index(opt)]["key"] if opt in [f"{f['key']}  ({f['size_kb']} KB)" for f in file_list] else None

    if file_list:
        st.markdown(
            '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.78rem;margin:12px 0 4px 0;">'
            'SELECT FILES TO LOAD</p>',
            unsafe_allow_html=True,
        )
        sel_col1, sel_col2, sel_col3 = st.columns(3, gap="medium")

        with sel_col1:
            st.markdown('<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.72rem;">PRIMARY DATA (required)</p>', unsafe_allow_html=True)
            sel_main = st.selectbox("Main dataset", file_options, key="s3_sel_main", label_visibility="collapsed")

        with sel_col2:
            st.markdown('<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.72rem;">REFERENCE DATA (optional)</p>', unsafe_allow_html=True)
            sel_ref = st.selectbox("Reference dataset", file_options, key="s3_sel_ref", label_visibility="collapsed")

        with sel_col3:
            st.markdown('<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.72rem;">RULES JSON (optional)</p>', unsafe_allow_html=True)
            # Filter to only JSON files for rules
            json_options = ["(none)"] + [f"{f['key']}  ({f['size_kb']} KB)" for f in file_list if f["key"].endswith(".json")]
            sel_rules = st.selectbox("Rules JSON", json_options, key="s3_sel_rules", label_visibility="collapsed")

        if st.button("LOAD SELECTED FILES FROM S3", key="s3_load_file", use_container_width=True, type="primary"):
            main_key = _key_from_option(sel_main)
            ref_key  = _key_from_option(sel_ref)
            rules_key = _key_from_option(sel_rules)

            if not main_key:
                st.error("Select a primary dataset.")
            else:
                from core.storage.s3 import read_from_s3
                loaded = []

                # Main data
                try:
                    with st.spinner(f"Loading main data: {main_key}..."):
                        df = read_from_s3(bucket=s3_bucket, key=main_key, region=s3_region)
                    st.session_state["df_raw_full"] = df
                    st.session_state["df_raw"] = df
                    st.session_state["s3_bucket"] = s3_bucket
                    st.session_state["s3_source_key"] = main_key
                    st.session_state["s3_region"] = s3_region
                    loaded.append(f"Main data: {main_key} ({df.shape[0]:,} rows x {df.shape[1]} cols)")
                except Exception as e:
                    st.error(f"Failed to load main data: {e}")

                # Reference data
                if ref_key:
                    try:
                        with st.spinner(f"Loading reference data: {ref_key}..."):
                            df_ref = read_from_s3(bucket=s3_bucket, key=ref_key, region=s3_region)
                        st.session_state["df_ref"] = df_ref
                        loaded.append(f"Reference: {ref_key} ({df_ref.shape[0]:,} rows)")
                    except Exception as e:
                        st.warning(f"Could not load reference data: {e}")

                # Rules JSON
                if rules_key:
                    try:
                        with st.spinner(f"Loading rules: {rules_key}..."):
                            import boto3, json as _json
                            s3c = boto3.client("s3", region_name=s3_region)
                            obj = s3c.get_object(Bucket=s3_bucket, Key=rules_key)
                            rules = _json.loads(obj["Body"].read().decode("utf-8"))
                        st.session_state["rules_json"] = rules
                        loaded.append(f"Rules JSON: {rules_key} ({len(rules.get('columns', {}))} columns)")
                    except Exception as e:
                        st.warning(f"Could not load rules JSON: {e}")

                if loaded:
                    for msg in loaded:
                        st.success(f"Loaded — {msg}")
                    st.rerun()
    else:
        st.info("Click BROWSE BUCKET to list available files, then select what to load.")

# ---------------------------------------------------------------------------
# Database connector
# ---------------------------------------------------------------------------
if source_mode == "Database":
    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
        'letter-spacing:1px;">CONNECT TO DATABASE</p>',
        unsafe_allow_html=True,
    )

    DB_ENGINES = {
        "PostgreSQL":  "postgresql+psycopg2://user:password@host:5432/dbname",
        "MySQL":       "mysql+pymysql://user:password@host:3306/dbname",
        "MSSQL":       "mssql+pyodbc://user:password@host:1433/dbname?driver=ODBC+Driver+17+for+SQL+Server",
        "Snowflake":   "snowflake://user:password@account/dbname/schema?warehouse=WH",
        "BigQuery":    None,   # handled separately — needs service account JSON
        "SQLite":      "sqlite:///path/to/file.db",
        "Other (paste full URL)": "",
    }

    db_col1, db_col2 = st.columns(2, gap="large")

    with db_col1:
        db_engine_type = st.selectbox(
            "Database Type",
            list(DB_ENGINES.keys()),
            key="db_engine_type",
        )

    with db_col2:
        db_limit = st.number_input(
            "Row limit (0 = no limit)",
            min_value=0, max_value=10_000_000, value=10_000, step=1_000,
            key="db_row_limit",
            help="Limit rows fetched to avoid loading huge tables into memory.",
        )

    # ── BigQuery: service account JSON upload ──────────────────────────────
    bq_credentials = None
    if db_engine_type == "BigQuery":
        st.markdown(
            '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
            'BigQuery uses a Service Account JSON key for authentication.</p>',
            unsafe_allow_html=True,
        )
        bq_col1, bq_col2 = st.columns(2, gap="large")
        with bq_col1:
            bq_project = st.text_input(
                "GCP Project ID",
                key="bq_project",
                placeholder="e.g. my-gcp-project",
            )
            bq_dataset = st.text_input(
                "Dataset (optional)",
                key="bq_dataset",
                placeholder="e.g. analytics",
                help="Leave blank to query across all datasets.",
            )
        with bq_col2:
            bq_key_file = st.file_uploader(
                "Service Account JSON Key",
                type=["json"],
                key="bq_key_upload",
                help="Download from GCP Console → IAM → Service Accounts → Keys. "
                     "Key is used only for this session and never stored.",
            )
            if bq_key_file:
                import json as _json
                bq_credentials = _json.load(bq_key_file)
                st.success("Service account key loaded")

        conn_str = None  # BigQuery uses credentials object, not a URL
        db_schema = bq_dataset

    else:
        # ── All other DBs: connection string ──────────────────────────────
        placeholder = DB_ENGINES.get(db_engine_type, "")
        conn_str = st.text_input(
            "Connection String",
            value=st.session_state.get("db_conn_str", placeholder or ""),
            type="password",
            help="Your credentials are used only for this session and never stored.",
            key="db_conn_str_input",
            placeholder=placeholder or "dialect+driver://user:password@host/dbname",
        )
        db_schema = st.text_input(
            "Schema (optional)",
            value="",
            key="db_schema_input",
            placeholder="e.g. public, dbo, sales",
        )

    # SQL query box
    section_header("// SQL Query")
    _default_query = "SELECT * FROM your_table"
    if db_engine_type == "BigQuery":
        _default_query = "SELECT * FROM `your_dataset.your_table` LIMIT 10000"
    db_query = st.text_area(
        "SELECT query",
        value=st.session_state.get("db_last_query", _default_query),
        height=100,
        key="db_query_input",
        help="Write any SELECT query. Use LIMIT in your query or set Row Limit above.",
    )

    # ── Helper: build engine ───────────────────────────────────────────────
    def _build_engine(engine_type, conn_string, bq_project_id, bq_creds):
        from sqlalchemy import create_engine as _ce
        if engine_type == "BigQuery":
            from google.oauth2 import service_account
            from sqlalchemy_bigquery import BigQueryDialect  # noqa — ensures dialect registered
            creds = service_account.Credentials.from_service_account_info(bq_creds)
            url = f"bigquery://{bq_project_id}"
            return _ce(url, credentials_base=creds)
        return _ce(conn_string)

    # Helper: list tables
    db_act1, db_act2 = st.columns(2, gap="large")

    with db_act1:
        if st.button("LIST TABLES", use_container_width=True, key="db_list_tables"):
            _ready = (bq_credentials is not None) if db_engine_type == "BigQuery" else bool(conn_str)
            if not _ready:
                st.error("Enter connection details first.")
            else:
                try:
                    from sqlalchemy import inspect as _inspect
                    _eng = _build_engine(
                        db_engine_type, conn_str,
                        st.session_state.get("bq_project", ""),
                        bq_credentials,
                    )
                    _insp = _inspect(_eng)
                    schemas = [db_schema] if db_schema else [None]
                    tables = []
                    for sch in schemas:
                        try:
                            tables += _insp.get_table_names(schema=sch)
                        except Exception:
                            pass
                    if tables:
                        st.session_state["db_table_list"] = tables
                        st.success(f"Found {len(tables)} tables")
                    else:
                        st.warning("No tables found — check schema/dataset name or permissions.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    if st.session_state.get("db_table_list"):
        selected_table = st.selectbox(
            "Quick-load table",
            st.session_state["db_table_list"],
            key="db_table_select",
        )
        if db_engine_type == "BigQuery":
            ds = f"{db_schema}." if db_schema else ""
            proj = st.session_state.get("bq_project", "project")
            st.session_state["db_last_query"] = f"SELECT * FROM `{proj}.{ds}{selected_table}` LIMIT {db_limit or 10000}"
        else:
            limit_clause = f" LIMIT {db_limit}" if db_limit > 0 else ""
            schema_prefix = f"{db_schema}." if db_schema else ""
            st.session_state["db_last_query"] = f'SELECT * FROM {schema_prefix}"{selected_table}"{limit_clause}'

    with db_act2:
        if st.button("LOAD FROM DATABASE", type="primary", use_container_width=True, key="db_load"):
            _bq_ready = db_engine_type == "BigQuery" and bq_credentials and st.session_state.get("bq_project")
            _other_ready = db_engine_type != "BigQuery" and conn_str
            if not (_bq_ready or _other_ready):
                if db_engine_type == "BigQuery":
                    st.error("Upload a service account JSON key and enter a GCP Project ID.")
                else:
                    st.error("Enter a connection string.")
            elif not db_query.strip():
                st.error("Enter a SQL query.")
            else:
                try:
                    from sqlalchemy import text as _text
                    with st.spinner("Connecting and fetching data..."):
                        _eng = _build_engine(
                            db_engine_type, conn_str,
                            st.session_state.get("bq_project", ""),
                            bq_credentials,
                        )
                        _q = db_query.strip()
                        # Auto-inject LIMIT for non-BigQuery (BQ uses its own syntax)
                        if db_limit > 0 and "limit" not in _q.lower() and db_engine_type != "BigQuery":
                            _q = f"{_q} LIMIT {db_limit}"
                        with _eng.connect() as _conn:
                            df = pd.read_sql(_text(_q), _conn)
                    st.session_state["df_raw_full"] = df
                    st.session_state["df_raw"] = df
                    st.session_state["db_conn_str"] = conn_str or ""
                    st.session_state["db_last_query"] = db_query
                    st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {db_engine_type}")
                except Exception as e:
                    st.error(f"Failed to load from database: {e}")
                else:
                    st.rerun()

    st.divider()
    st.caption(
        "Supported drivers must be installed in the environment: "
        "`psycopg2` (PostgreSQL), `pymysql` (MySQL), `pyodbc` (MSSQL), "
        "`snowflake-sqlalchemy` (Snowflake), `sqlalchemy-bigquery` (BigQuery)."
    )

# ---------------------------------------------------------------------------
# REST API connector
# ---------------------------------------------------------------------------
elif source_mode == "REST API":
    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
        'letter-spacing:1px;">CONNECT TO REST API</p>',
        unsafe_allow_html=True,
    )

    api_col1, api_col2 = st.columns([3, 1], gap="large")
    with api_col1:
        api_url = st.text_input(
            "API Endpoint URL",
            key="api_url",
            placeholder="https://api.example.com/v1/data",
            help="Any JSON-returning GET endpoint. Supports pagination.",
        )
        st.caption(
            "Test endpoints: "
            "`https://jsonplaceholder.typicode.com/posts` (100 posts) | "
            "`https://jsonplaceholder.typicode.com/users` (10 users) | "
            "`https://restcountries.com/v3.1/all?fields=name,population,region` (countries)"
        )
    with api_col2:
        api_method = st.selectbox("Method", ["GET", "POST"], key="api_method")

    # Auth
    auth_col1, auth_col2 = st.columns(2, gap="large")
    with auth_col1:
        api_auth_type = st.selectbox(
            "Authentication",
            ["None", "Bearer Token", "API Key Header", "Basic Auth"],
            key="api_auth_type",
        )
    with auth_col2:
        api_auth_value = ""
        if api_auth_type == "Bearer Token":
            api_auth_value = st.text_input("Bearer Token", type="password", key="api_bearer")
        elif api_auth_type == "API Key Header":
            hdr_col1, hdr_col2 = st.columns(2)
            api_key_header = hdr_col1.text_input("Header Name", value="X-API-Key", key="api_key_header")
            api_auth_value = hdr_col2.text_input("API Key", type="password", key="api_key_value")
        elif api_auth_type == "Basic Auth":
            b_col1, b_col2 = st.columns(2)
            api_basic_user = b_col1.text_input("Username", key="api_basic_user")
            api_basic_pass = b_col2.text_input("Password", type="password", key="api_basic_pass")

    # Advanced
    with st.expander("Advanced Options", expanded=False):
        adv1, adv2 = st.columns(2)
        with adv1:
            api_data_path = st.text_input(
                "JSON data path (dot notation)",
                value="",
                key="api_data_path",
                placeholder="e.g. data.results  or  items",
                help="Path to the array within the JSON response. Leave blank if the root is an array.",
            )
            api_limit = st.number_input("Max rows", min_value=0, max_value=1_000_000, value=10_000, step=1_000, key="api_max_rows")
        with adv2:
            api_extra_headers = st.text_area(
                "Extra headers (JSON)",
                value="{}",
                key="api_extra_headers",
                height=80,
                placeholder='{"Content-Type": "application/json"}',
            )
            api_body = st.text_area(
                "Request body (POST only, JSON)",
                value="{}",
                key="api_body",
                height=80,
            )

    if st.button("FETCH FROM API", key="api_fetch", use_container_width=True, type="primary"):
        if not api_url.strip():
            st.error("Enter an API URL.")
        else:
            try:
                import requests as _req
                import json as _json

                # Build headers
                headers = {}
                try:
                    headers = _json.loads(st.session_state.get("api_extra_headers", "{}") or "{}")
                except Exception:
                    pass

                if api_auth_type == "Bearer Token":
                    headers["Authorization"] = f"Bearer {api_auth_value}"
                elif api_auth_type == "API Key Header":
                    headers[st.session_state.get("api_key_header", "X-API-Key")] = api_auth_value

                # Auth tuple for Basic
                auth = None
                if api_auth_type == "Basic Auth":
                    auth = (st.session_state.get("api_basic_user", ""), st.session_state.get("api_basic_pass", ""))

                with st.spinner(f"Fetching {api_url}..."):
                    if api_method == "POST":
                        body = {}
                        try:
                            body = _json.loads(st.session_state.get("api_body", "{}") or "{}")
                        except Exception:
                            pass
                        resp = _req.post(api_url.strip(), headers=headers, json=body, auth=auth, timeout=30)
                    else:
                        resp = _req.get(api_url.strip(), headers=headers, auth=auth, timeout=30)

                resp.raise_for_status()
                raw_json = resp.json()

                # Navigate to data path
                data_path = st.session_state.get("api_data_path", "").strip()
                if data_path:
                    for key in data_path.split("."):
                        if isinstance(raw_json, dict):
                            raw_json = raw_json.get(key, raw_json)
                        else:
                            break

                # Normalise to DataFrame
                if isinstance(raw_json, list):
                    df = pd.json_normalize(raw_json)
                elif isinstance(raw_json, dict):
                    # Try values if it looks like a records dict
                    vals = list(raw_json.values())
                    if vals and isinstance(vals[0], dict):
                        df = pd.json_normalize(vals)
                    else:
                        df = pd.json_normalize([raw_json])
                else:
                    st.error(f"Unexpected response type: {type(raw_json)}. Expected array or object.")
                    st.stop()

                if api_limit > 0:
                    df = df.head(api_limit)

                st.session_state["df_raw_full"] = df
                st.session_state["df_raw"] = df
                st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns from API")
                st.rerun()

            except Exception as e:
                # Don't swallow Streamlit's internal rerun signal
                try:
                    from streamlit.runtime.scriptrunner import StopException
                    if isinstance(e, StopException):
                        raise
                except ImportError:
                    pass
                st.error(f"API fetch failed: {e}")

# ---------------------------------------------------------------------------
# Google Sheets connector
# ---------------------------------------------------------------------------
elif source_mode == "Google Sheets":
    st.markdown(
        '<p style="color:#00ff41;font-family:Share Tech Mono;font-size:0.82rem;'
        'letter-spacing:1px;">CONNECT TO GOOGLE SHEETS</p>',
        unsafe_allow_html=True,
    )

    gs_col1, gs_col2 = st.columns(2, gap="large")
    with gs_col1:
        gs_url = st.text_input(
            "Google Sheets URL",
            key="gs_url",
            placeholder="https://docs.google.com/spreadsheets/d/SHEET_ID/...",
            help="Paste the full URL from your browser. The sheet must be shared.",
        )
        gs_sheet_name = st.text_input(
            "Sheet / Tab name (optional)",
            value="",
            key="gs_sheet_name",
            placeholder="Sheet1",
            help="Leave blank to load the first sheet.",
        )
    with gs_col2:
        gs_auth_type = st.selectbox(
            "Authentication",
            ["Service Account JSON", "Public sheet (no auth)"],
            key="gs_auth_type",
        )
        gs_creds = None
        if gs_auth_type == "Service Account JSON":
            gs_key_file = st.file_uploader(
                "Service Account JSON Key",
                type=["json"],
                key="gs_key_upload",
                help="GCP Console → IAM → Service Accounts → Keys. "
                     "Share the sheet with the service account email.",
            )
            if gs_key_file:
                import json as _json
                gs_creds = _json.load(gs_key_file)
                st.success(f"Key loaded: {gs_creds.get('client_email', 'ok')}")

    if st.button("LOAD FROM GOOGLE SHEETS", key="gs_load", use_container_width=True, type="primary"):
        if not gs_url.strip():
            st.error("Enter a Google Sheets URL.")
        elif gs_auth_type == "Service Account JSON" and not gs_creds:
            st.error("Upload a service account JSON key.")
        else:
            try:
                import gspread
                from gspread.utils import extract_id_from_url

                with st.spinner("Connecting to Google Sheets..."):
                    sheet_id = extract_id_from_url(gs_url.strip())

                    if gs_auth_type == "Service Account JSON":
                        gc = gspread.service_account_from_dict(gs_creds)
                    else:
                        # Public sheet — use anonymous access
                        gc = gspread.service_account_from_dict({
                            "type": "service_account",
                            "project_id": "",
                            "private_key_id": "",
                            "private_key": "",
                            "client_email": "",
                            "client_id": "",
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                        }) if False else None  # fallback to export URL

                    if gs_auth_type == "Public sheet (no auth)":
                        # Use the CSV export URL — works for publicly shared sheets
                        import requests as _req
                        _gid = 0  # first sheet
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={_gid}"
                        resp = _req.get(csv_url, timeout=30)
                        resp.raise_for_status()
                        import io
                        df = pd.read_csv(io.StringIO(resp.text))
                    else:
                        spreadsheet = gc.open_by_key(sheet_id)
                        if gs_sheet_name.strip():
                            worksheet = spreadsheet.worksheet(gs_sheet_name.strip())
                        else:
                            worksheet = spreadsheet.sheet1
                        records = worksheet.get_all_records()
                        df = pd.DataFrame(records)

                st.session_state["df_raw_full"] = df
                st.session_state["df_raw"] = df
                st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns from Google Sheets")
                st.rerun()

            except Exception as e:
                try:
                    from streamlit.runtime.scriptrunner import StopException
                    if isinstance(e, StopException):
                        raise
                except ImportError:
                    pass
                st.error(f"Google Sheets load failed: {e}")
                st.caption("Make sure the sheet is shared with your service account email, or set to 'Anyone with link can view'.")

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
            st.error(f"Failed to parse rules JSON — check the file is valid JSON. ({type(e).__name__})")

@st.cache_data(show_spinner=False)
def _load_demo_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Demo Data Loader
# ---------------------------------------------------------------------------
_DEMO_MAIN  = os.path.join(_ROOT, "TEST2_DATA", "main_data",  "customer_transactions.csv")
_DEMO_REF   = os.path.join(_ROOT, "TEST2_DATA", "reference",  "known_fraudulent_merchants.csv")
_DEMO_RULES = os.path.join(_ROOT, "TEST2_DATA", "rules",      "validation_rules.json")
_DEMO_CORPUS_DIR = os.path.join(_ROOT, "TEST2_DATA", "corpus")

_DEMO_CORPORA = [
    ("corpus_bank_names.csv",        "bank_name",        "alias",      "alias", "canonical"),
    ("corpus_company_names.csv",     "merchant_name",    "alias",      "alias", "canonical"),
    ("corpus_country_codes.csv",     "country",          "alias",      "alias", "canonical"),
    ("corpus_transaction_types.csv", "transaction_type", "validation", "value", "value"),
]

_demo_available = os.path.exists(_DEMO_MAIN)

if _demo_available:
    section_header("// Quick Start — Demo Dataset")

    already_loaded = (
        st.session_state.get("df_raw") is not None
        and st.session_state.get("_demo_loaded", False)
    )

    if already_loaded:
        st.success(
            "Demo dataset loaded — 500 rows · reference data · validation rules · 4 corpora. "
            "Scroll down or head to Validate to continue."
        )
        if st.button("Reload Demo Data", key="reload_demo", type="secondary"):
            st.session_state["_demo_loaded"] = False
            st.rerun()
    else:
        col_demo, col_desc = st.columns([1, 2], gap="large")
        with col_demo:
            st.markdown(
                '<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.8rem;'
                'letter-spacing:1px;">ONE-CLICK DEMO</p>',
                unsafe_allow_html=True,
            )
            load_demo = st.button(
                "LOAD DEMO DATA",
                key="load_demo_all",
                type="primary",
                use_container_width=True,
            )
        with col_desc:
            st.markdown(
                '<p style="color:#5a9a5a;font-family:monospace;font-size:0.78rem;line-height:1.7;">'
                "Loads all demo assets in one click:<br>"
                "&nbsp;&nbsp;• <b style='color:#00ff41'>500-row</b> financial transactions dataset (with intentional errors)<br>"
                "&nbsp;&nbsp;• <b style='color:#00ff41'>Reference</b> known-fraudulent merchants list<br>"
                "&nbsp;&nbsp;• <b style='color:#00ff41'>Validation rules</b> JSON (email, postcode, amounts &amp; more)<br>"
                "&nbsp;&nbsp;• <b style='color:#00ff41'>4 corpora</b> — bank names, merchant names, country codes, transaction types"
                "</p>",
                unsafe_allow_html=True,
            )

        if load_demo:
            with st.spinner("Loading demo data..."):
                # 1. Main dataset
                df_demo = _load_demo_csv(_DEMO_MAIN)
                st.session_state["df_raw_full"] = df_demo
                st.session_state["df_raw"] = df_demo

                # 2. Reference dataset
                if os.path.exists(_DEMO_REF):
                    st.session_state["df_ref"] = _load_demo_csv(_DEMO_REF)

                # 3. Validation rules JSON
                if os.path.exists(_DEMO_RULES):
                    import json as _json
                    with open(_DEMO_RULES) as _f:
                        st.session_state["rules_json"] = _json.load(_f)

                # 4. Corpus — load into Redis via CorpusManager
                cm = st.session_state.get("corpus_manager")
                if cm is not None:
                    for _fname, _name, _ctype, _key, _val in _DEMO_CORPORA:
                        _fpath = os.path.join(_DEMO_CORPUS_DIR, _fname)
                        if not os.path.exists(_fpath):
                            continue
                        try:
                            _df_c = _load_demo_csv(_fpath)
                            cm.load_corpus_from_dataframe(
                                _df_c, corpus_name=_name, corpus_type=_ctype,
                                key_column=_key, value_column=_val,
                            )
                        except Exception:
                            pass

                st.session_state["_demo_loaded"] = True
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
        _slider_min = min(10, total_rows)
        _slider_max = max(_slider_min + 1, total_rows)  # ensure max > min always
        use_rows = st.slider(
            "Rows to use",
            min_value=_slider_min,
            max_value=_slider_max,
            value=min(500, _slider_max),
            step=max(1, min(10, _slider_max - _slider_min)),
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
            f'<tr><td style="color:#5a9a5a;">Rows:</td><td>{len(df_show):,}</td>'
            f'<td style="color:#5a9a5a;">Columns:</td><td>{len(df_show.columns)}</td></tr>'
            f'<tr><td style="color:#5a9a5a;">Missing cells:</td><td>{int(df_show.isna().sum().sum()):,}</td>'
            f'<td style="color:#5a9a5a;">Duplicates:</td><td>{int(df_show.duplicated().sum()):,}</td></tr>'
            f'<tr><td style="color:#5a9a5a;">Memory:</td><td>{df_show.memory_usage(deep=True).sum() / 1024:.1f} KB</td>'
            f'<td style="color:#5a9a5a;">Full dataset:</td><td>{total_rows:,} rows</td></tr>'
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
        "<span style='color:#5a9a5a;'>Upload a CSV, Parquet, or Excel file to begin analysis.</span>"
    )
