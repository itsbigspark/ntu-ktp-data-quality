"""
Session state initialization for the DQ Investigator.

Centralizes all session_state keys so pages don't have to
worry about KeyError. Call init_state() once from Home.py.

Also handles:
  - Redis auto-init (tries to connect, then start redis-server if missing)
  - Corpus auto-load from TEST2_DATA/corpus/ when Redis is empty
"""

import os
import subprocess
import time
import streamlit as st


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _init_redis():
    """
    Try to connect to a running Redis instance.
    If not available, attempt to start redis-server automatically.
    Returns a connected redis.Redis client, or None if unavailable.
    """
    def _try_connect():
        try:
            import redis as _redis
            r = _redis.Redis(
                host="localhost", port=6379,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            r.ping()
            return r
        except Exception:
            return None

    # First attempt — maybe Redis is already running
    r = _try_connect()
    if r is not None:
        return r

    # Second attempt — try to start redis-server as a background daemon
    for redis_bin in ["redis-server", "/opt/homebrew/bin/redis-server"]:
        try:
            subprocess.Popen(
                [redis_bin, "--daemonize", "yes"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.5)
            break
        except FileNotFoundError:
            continue

    return _try_connect()  # None if still unreachable


def _autoload_corpora(corpus_manager):
    """
    Load the demo corpus files from TEST2_DATA/corpus/ into Redis
    if Redis is empty. Runs once per session.
    """
    import pandas as pd

    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _corpus_dir = os.path.join(_project_root, "TEST2_DATA", "corpus")

    try:
        _existing = corpus_manager.list_corpora() if hasattr(corpus_manager, "list_corpora") else []
    except Exception:
        _existing = []

    if _existing or not os.path.isdir(_corpus_dir):
        return  # Already populated or no demo data

    _SAMPLE_CORPORA = [
        # (filename,                    corpus_name,        type,         key_col,  value_col)
        ("corpus_bank_names.csv",        "bank_name",        "alias",      "alias",  "canonical"),
        ("corpus_company_names.csv",     "merchant_name",    "alias",      "alias",  "canonical"),
        ("corpus_country_codes.csv",     "country",          "alias",      "alias",  "canonical"),
        ("corpus_transaction_types.csv", "transaction_type", "validation", "value",  "value"),
    ]

    for _fname, _name, _ctype, _key, _val in _SAMPLE_CORPORA:
        _fpath = os.path.join(_corpus_dir, _fname)
        if not os.path.exists(_fpath):
            continue
        try:
            _df = pd.read_csv(_fpath)
            if _ctype == "validation":
                corpus_manager.load_corpus_from_dataframe(
                    _df, corpus_name=_name, corpus_type=_ctype,
                    key_column=_key, value_column=_key,
                )
            else:
                corpus_manager.load_corpus_from_dataframe(
                    _df, corpus_name=_name, corpus_type=_ctype,
                    key_column=_key, value_column=_val,
                )
        except Exception:
            pass  # Silently skip files that fail to load


# ---------------------------------------------------------------------------
# State initializer
# ---------------------------------------------------------------------------

def init_state():
    """Initialize all session state keys with defaults. Safe to call multiple times."""

    defaults = {
        # Auth
        "authenticated": False,
        "auth_user": None,

        # Data
        "df_raw_full": None,
        "df_raw": None,
        "df_ref": None,
        "df_corrected": None,
        "df_precleaned": None,
        "df_cleaned": None,
        "df_standardized": None,
        "df_enriched": None,
        "df_trimmed": None,
        "df_validated": None,
        "df_deduplicated": None,
        "df_auto_after": None,
        "df_pipeline_output": None,

        # Rules
        "rules_json": None,
        "rules_merged": None,
        "generated_rules": None,
        "rule_selections": {},

        # Validation
        "unified_issues_report": None,
        "validation_completed": False,

        # AI / Chat
        "chat_history": [],
        "agent_chat_history": [],
        "ai_provider_type": "ollama",  # ollama, anthropic, openai, gemini
        "ai_provider_config": {},
        "chat_result": None,
        "saved_results": [],

        # Deduplication
        "dedup_pairs": None,
        "dedup_clusters": None,
        "dedup_cluster_summary": None,
        "column_duplicates": None,

        # Entity resolution
        "er_pairs": None,
        "er_datasets": None,

        # Managers
        "corpus_manager": None,
        "pipeline_manager": None,

        # Auto report
        "auto_scores": None,

        # Data profile
        "data_profile": None,

        # AI enrichment results (from dq_engine)
        "ai_enrichment_result": None,

        # Settings
        "ollama_model": "phi3:mini",
        "ollama_url": "http://localhost:11434",
        "anthropic_api_key": "",
        "anthropic_model": "claude-opus-4-6",
        "openai_api_key": "",
        "gemini_api_key": "",

        # Internal flags
        "_redis_init_done": False,
        "_sample_corpora_loaded": False,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # -----------------------------------------------------------------------
    # Redis + Corpus Manager — runs once per session after first login
    # -----------------------------------------------------------------------
    if st.session_state.get("authenticated") and not st.session_state["_redis_init_done"]:
        st.session_state["_redis_init_done"] = True

        _redis_client = _init_redis()
        if _redis_client is not None:
            try:
                from core.corpus_manager import CorpusManager
                st.session_state["corpus_manager"] = CorpusManager(redis_client=_redis_client)
            except Exception:
                st.session_state["corpus_manager"] = None
        else:
            st.session_state["corpus_manager"] = None

    # -----------------------------------------------------------------------
    # Auto-load demo corpora into Redis if empty — runs once per session
    # -----------------------------------------------------------------------
    cm = st.session_state.get("corpus_manager")
    if cm is not None and not st.session_state["_sample_corpora_loaded"]:
        st.session_state["_sample_corpora_loaded"] = True
        _autoload_corpora(cm)
