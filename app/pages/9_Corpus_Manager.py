"""
Page 9: Corpus Manager -- Upload and manage custom corpora (Redis-backed).
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("CORPUS MANAGER", "Upload and manage custom corpora for data standardisation")

# ---------------------------------------------------------------------------
# Import corpus manager
# ---------------------------------------------------------------------------
try:
    from core.corpus_manager import CorpusManager
except ImportError:
    st.error("Could not import CorpusManager. Ensure core/corpus_manager.py exists.")
    st.stop()

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
if "corpus_manager" not in st.session_state:
    try:
        st.session_state["corpus_manager"] = CorpusManager()
    except ConnectionError as e:
        st.session_state["corpus_manager"] = None
        st.error(str(e))

corpus_mgr = st.session_state.get("corpus_manager")
if corpus_mgr is None:
    st.warning("Redis is not available. Start Redis to enable the Corpus Manager.")
    st.code("brew services start redis", language="bash")
    st.stop()

st.markdown(
    '<div class="glass-card">'
    '<p style="color:#b0ffb8;font-size:0.78rem;">'
    'Upload corpus files to standardise and validate data using domain-specific knowledge. '
    'Supported formats: Excel, CSV, TSV, JSON, TXT. '
    'Corpus types: <b>Alias</b> (map variations to canonical), '
    '<b>Lookup</b> (key-value with full record), '
    '<b>Validation</b> (list of valid values).</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
upload_tab, manage_tab = st.tabs(["Upload Corpus", "Manage Corpora"])

# ======================== UPLOAD TAB ========================
with upload_tab:
    section_header("// Upload Corpus Files")

    uploaded_files = st.file_uploader(
        "Choose corpus file(s)",
        type=["xlsx", "xls", "csv", "tsv", "json", "txt"],
        key="corpus_upload",
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")

        for file_idx, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"{uploaded_file.name}", expanded=(len(uploaded_files) == 1)):
                try:
                    df_corpus = CorpusManager.parse_corpus_file(uploaded_file)

                    st.caption(f"{df_corpus.shape[0]} rows, {df_corpus.shape[1]} columns")
                    st.dataframe(df_corpus.head(10), use_container_width=True, height=200)

                    section_header("// Configure Corpus")

                    col1, col2 = st.columns(2)
                    with col1:
                        corpus_name = st.text_input(
                            "Corpus Name",
                            value=uploaded_file.name.split(".")[0].lower().replace(" ", "_"),
                            help="Unique name for this corpus",
                            key=f"corpus_name_{file_idx}",
                        )
                        corpus_type = st.selectbox(
                            "Corpus Type",
                            ["alias", "lookup", "validation"],
                            key=f"corpus_type_{file_idx}",
                        )
                    with col2:
                        normalize_keys = st.checkbox(
                            "Normalize keys",
                            value=True,
                            help="Lowercase and strip whitespace from keys",
                            key=f"normalize_{file_idx}",
                        )

                    # Column mapping
                    columns = list(df_corpus.columns)
                    lookup_column = None

                    if corpus_type == "alias":
                        ca, cb, cc = st.columns(3)
                        with ca:
                            key_column = st.selectbox("Key Column (alias/variation)", columns, key=f"key_col_{file_idx}")
                        with cb:
                            value_column = st.selectbox(
                                "Value Column (canonical)", columns,
                                index=min(1, len(columns) - 1), key=f"val_col_{file_idx}",
                            )
                        with cc:
                            use_lookup = st.checkbox("Use lookup column?", key=f"use_lookup_{file_idx}")
                            if use_lookup and len(columns) > 2:
                                lookup_column = st.selectbox(
                                    "Lookup Column", columns,
                                    index=min(2, len(columns) - 1), key=f"lookup_col_{file_idx}",
                                )

                    elif corpus_type == "lookup":
                        key_column = st.selectbox("Key Column", columns, key=f"key_col_{file_idx}")
                        value_column = columns[0]
                        st.caption(f"All {len(columns)} columns will be stored for each key")

                    elif corpus_type == "validation":
                        key_column = st.selectbox("Values Column", columns, key=f"key_col_{file_idx}")
                        value_column = key_column

                    # Load button
                    if st.button(f"Load '{corpus_name}' into Redis", type="primary", key=f"load_btn_{file_idx}"):
                        with st.spinner(f"Loading {corpus_name} into Redis..."):
                            result = corpus_mgr.load_corpus_from_dataframe(
                                df=df_corpus,
                                corpus_name=corpus_name,
                                corpus_type=corpus_type,
                                key_column=key_column,
                                value_column=value_column,
                                lookup_column=lookup_column,
                                normalize_keys=normalize_keys,
                            )

                        if result["status"] == "success":
                            st.success(
                                f"'{corpus_name}' loaded: {result['loaded']} entries, "
                                f"{result['skipped']} skipped, {result['errors']} errors"
                            )
                        else:
                            st.error(result["message"])

                except Exception as e:
                    st.error(f"Error processing file: {e}")

# ======================== MANAGE TAB ========================
with manage_tab:
    section_header("// Loaded Corpora")

    corpora = corpus_mgr.list_corpora()

    if not corpora:
        st.info("No corpora loaded. Upload a corpus in the Upload tab.")
    else:
        for idx, corpus in enumerate(corpora):
            with st.expander(f"{corpus.get('corpus_name', 'Unknown')}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f"**Type:** `{corpus.get('corpus_type', 'unknown')}` | "
                        f"**Key Column:** `{corpus.get('key_column', 'N/A')}` | "
                        f"**Rows:** {corpus.get('total_rows', 'N/A')} | "
                        f"**Loaded:** {corpus.get('loaded_at', 'N/A')}"
                    )
                    if st.button("View Stats", key=f"stats_{idx}"):
                        stats = corpus_mgr.get_corpus_stats(corpus.get("corpus_name"))
                        if stats["status"] == "success":
                            st.json(stats)

                with col2:
                    if st.button("Delete", key=f"delete_{idx}", type="secondary"):
                        result = corpus_mgr.delete_corpus(corpus.get("corpus_name"))
                        if result["status"] == "success":
                            st.success(f"Deleted {result['keys_deleted']} keys")
                            st.rerun()
                        else:
                            st.error(result["message"])

        # Redis info
        st.markdown("---")
        section_header("// Redis Statistics")
        try:
            info = corpus_mgr.redis.info()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Keys", info.get("db0", {}).get("keys", 0) if "db0" in info else 0)
            c2.metric("Used Memory", info.get("used_memory_human", "N/A"))
            c3.metric("Connected Clients", info.get("connected_clients", 0))
        except Exception as e:
            st.warning(f"Could not fetch Redis stats: {e}")
