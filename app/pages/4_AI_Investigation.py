"""
Page 4: AI Investigation -- View AI enrichment results and chat.

Displays cross-column analysis, anomaly explanations, and allows
interactive AI chat about data quality issues.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header, terminal_block, kpi_card, workflow_breadcrumb,
    MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE, MATRIX_YELLOW, MATRIX_TEXT, MATRIX_TEXT_DIM,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("AI INVESTIGATION", "AI-powered data quality insights")

workflow_breadcrumb([
    ("Load Data", st.session_state.get("df_raw") is not None),
    ("Validate", st.session_state.get("validation_completed", False)),
    ("AI Enrichment", st.session_state.get("ai_enrichment_result") is not None),
    ("Clean", st.session_state.get("df_corrected") is not None),
])

DB_CONFIG = {"output": {"database": {"engine": "sqlite", "path": "./output/dq_investigator.db"}}}

# ---------------------------------------------------------------------------
# Source: session state (just ran) or database (historical)
# ---------------------------------------------------------------------------
section_header("// Intelligence Source")

source = st.radio(
    "View AI insights from:",
    ["Current session (latest validation run)", "Historical (from database)"],
    key="ai_source",
    horizontal=True,
)

enrichment = None
if source.startswith("Current") and st.session_state.get("ai_enrichment_result"):
    enrichment = st.session_state["ai_enrichment_result"]
    st.markdown(
        f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
        f'Loaded from current session -- Provider: {enrichment.provider}</p>',
        unsafe_allow_html=True,
    )
elif source.startswith("Historical"):
    try:
        from core.storage.database import get_batches_with_ai
        ai_batches = get_batches_with_ai(DB_CONFIG)
        if ai_batches:
            selected_batch = st.selectbox("Select batch", ai_batches, key="ai_inv_batch")
        else:
            st.markdown('<div class="terminal-block">// NO AI-ENRICHED BATCHES IN DATABASE</div>', unsafe_allow_html=True)
            selected_batch = None
    except Exception as e:
        st.warning(f"Cannot read database: {e}")
        selected_batch = None
else:
    if not st.session_state.get("ai_enrichment_result"):
        terminal_block(
            "// NO AI RESULTS IN CURRENT SESSION<br>"
            "<span style='color:#5a9a5a;'>Run validation with AI enrichment enabled, or switch to Historical view.</span>"
        )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper to load from DB
# ---------------------------------------------------------------------------
def _load_from_db(batch_id):
    from core.storage.database import (
        get_ai_smart_rules, get_ai_cross_column, get_ai_explanations,
        get_ai_triage, get_ai_executive_summary,
    )
    return {
        "smart_rules": get_ai_smart_rules(DB_CONFIG, batch_id),
        "cross_column": get_ai_cross_column(DB_CONFIG, batch_id),
        "explanations": get_ai_explanations(DB_CONFIG, batch_id),
        "triage": get_ai_triage(DB_CONFIG, batch_id),
        "executive_summary": get_ai_executive_summary(DB_CONFIG, batch_id),
    }


# Determine what to display
db_data = None
if source.startswith("Historical") and 'selected_batch' in dir() and selected_batch:
    try:
        db_data = _load_from_db(selected_batch)
    except Exception as e:
        st.error(f"Failed to load AI data: {e}")

# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------
exec_text = None
if enrichment:
    exec_text = enrichment.executive_summary
elif db_data:
    exec_text = db_data["executive_summary"]

if exec_text:
    section_header("// Executive Summary")
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(0,15,2,0.85),rgba(0,30,40,0.6));'
        f'border:1px solid rgba(0,229,255,0.25);border-radius:12px;padding:24px 28px;'
        f'font-family:Share Tech Mono;font-size:0.88rem;color:#b0ffb8;line-height:1.7;">'
        f'{exec_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cross-Column Analysis
# ---------------------------------------------------------------------------
cross_data = None
if enrichment:
    cross_data = enrichment.cross_column_issues
elif db_data and not db_data["cross_column"].empty:
    cross_data = db_data["cross_column"].to_dict("records")

if cross_data:
    section_header("// Cross-Column Logic Analysis")

    for item in cross_data:
        if isinstance(item, dict):
            cols_raw = item.get("columns_involved", item.get("columns", []))
            if isinstance(cols_raw, str):
                try:
                    cols_list = json.loads(cols_raw)
                except Exception:
                    cols_list = [cols_raw]
            else:
                cols_list = cols_raw
            cols_str = " + ".join(str(c) for c in cols_list)
            desc = str(item.get("description", ""))
            sev = str(item.get("severity", "medium")).lower()
            check_t = str(item.get("check_type", ""))
            query = str(item.get("suggested_query", ""))

            sev_color = {"critical": MATRIX_RED, "high": MATRIX_ORANGE, "medium": MATRIX_YELLOW, "low": MATRIX_CYAN}.get(sev, MATRIX_TEXT)

            st.markdown(
                f'<div style="background:rgba(0,5,1,0.8);border:1px solid rgba(192,80,255,0.15);'
                f'border-radius:8px;padding:14px 16px;margin-bottom:8px;font-family:Share Tech Mono;">'
                f'<span style="color:#c050ff;font-size:0.72rem;font-weight:bold;">{cols_str}</span>'
                f' <span style="font-size:0.62rem;color:{sev_color};letter-spacing:1px;text-transform:uppercase;'
                f'border:1px solid {sev_color};padding:1px 6px;border-radius:4px;">{sev}</span>'
                f'<br><span style="color:#b0ffb8;font-size:0.78rem;">{desc}</span>'
                f'<div style="color:#5a9a5a;font-size:0.7rem;margin-top:6px;">'
                f'CHECK: {check_t} | QUERY: {query}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Anomaly Explanations
# ---------------------------------------------------------------------------
explain_data = None
if enrichment:
    explain_data = enrichment.explanations
elif db_data and not db_data["explanations"].empty:
    explain_data = db_data["explanations"].to_dict("records")

if explain_data:
    section_header("// Anomaly Explanations")

    for item in explain_data:
        if isinstance(item, dict):
            col_name = str(item.get("column_name", item.get("column", "")))
            issue_t = str(item.get("issue_type", ""))
            explanation = str(item.get("explanation", ""))
            impact = str(item.get("business_impact", ""))
            action = str(item.get("suggested_action", ""))

            st.markdown(
                f'<div style="background:rgba(0,5,1,0.8);border:1px solid rgba(0,255,65,0.12);'
                f'border-radius:8px;padding:14px 16px;margin-bottom:8px;font-family:Share Tech Mono;">'
                f'<span style="color:{MATRIX_GREEN};font-size:0.82rem;font-weight:bold;">{col_name}</span>'
                f' -- <span style="color:{MATRIX_CYAN};font-size:0.72rem;">{issue_t}</span>'
                f'<br><span style="color:#b0ffb8;font-size:0.78rem;">{explanation}</span>'
                f'<div style="margin-top:6px;font-size:0.72rem;">'
                f'<span style="color:{MATRIX_ORANGE};">IMPACT:</span> <span style="color:#b0ffb8;">{impact}</span></div>'
                f'<div style="font-size:0.72rem;">'
                f'<span style="color:{MATRIX_CYAN};">ACTION:</span> <span style="color:#b0ffb8;">{action}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# AI Agent Chat (agentic, tool-calling)
# ---------------------------------------------------------------------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("// AI Agent Chat")

st.markdown(
    '<p style="color:#5a9a5a;font-family:Share Tech Mono;font-size:0.75rem;">'
    'Agentic chat — the AI can run validation, count issues, explain rows, apply fixes, and more. '
    'Configure your AI provider in Settings first.</p>',
    unsafe_allow_html=True,
)

# ── Chat options ────────────────────────────────────────────────────────────
provider_type = st.session_state.get("ai_provider_type", "ollama")
_local_only = provider_type != "anthropic"

col_opt1, col_opt2 = st.columns([1, 1])
with col_opt1:
    include_data = st.toggle(
        "Include data rows in context",
        value=False,
        key="agent_include_data",
        disabled=_local_only,
        help="Sends actual data rows to the LLM. Only available with Anthropic. Ollama keeps all data local.",
    )
    if _local_only and include_data:
        st.session_state["agent_include_data"] = False
        include_data = False
with col_opt2:
    n_rows = st.number_input(
        "Rows to include in context",
        min_value=5, max_value=100, value=20, step=5,
        key="agent_n_rows",
        disabled=not include_data,
    )

# ── Resolve model string ────────────────────────────────────────────────────
if provider_type == "anthropic":
    _model = st.session_state.get("anthropic_model", "claude-opus-4-6")
    _api_key = st.session_state.get("anthropic_api_key", "")
else:
    _model = st.session_state.get("ollama_model", "llama3.1")
    _api_key = ""

# ── Session chat history ────────────────────────────────────────────────────
if "agent_chat_history" not in st.session_state:
    st.session_state["agent_chat_history"] = []

chat_history = st.session_state["agent_chat_history"]

# ── Render history ──────────────────────────────────────────────────────────
for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about your data, run validation, review fixes..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        if provider_type == "anthropic" and not _api_key:
            response = "Anthropic API key not set. Go to Settings and add your API key, or switch to Ollama."
            tools_called = []
            open_tool = None
        else:
            try:
                from core.chat_agent import answer as agent_answer
                with st.spinner("Agent thinking..."):
                    raw = agent_answer(
                        user_message=user_input,
                        state=st.session_state,
                        history=chat_history,
                        model=_model,
                        include_data=include_data,
                        n_rows=int(n_rows),
                        api_key=_api_key,
                    )
                response = raw.get("response", "")
                tools_called = raw.get("tools_called", [])
                open_tool = raw.get("open_tool")

                # ── Audit log ──────────────────────────────────────────────
                try:
                    from core.storage.database import log_ai_call
                    _tokens = raw.get("usage", {})
                    _total_tokens = (
                        _tokens.get("input_tokens", 0) + _tokens.get("output_tokens", 0)
                        if isinstance(_tokens, dict) else 0
                    )
                    log_ai_call(
                        batch_id=st.session_state.get("current_batch_id", "interactive"),
                        call_type="chat_agent",
                        provider=provider_type,
                        model=_model,
                        payload_summary=user_input[:500],
                        response_preview=response[:500],
                        tokens_used=_total_tokens,
                        response_time_ms=int(raw.get("latency_ms", 0)),
                        success=True,
                    )
                except Exception:
                    pass  # audit log failure must never break the chat
                # ───────────────────────────────────────────────────────────

            except Exception as e:
                response = f"Agent error: {e}"
                tools_called = []
                open_tool = None

        st.markdown(response)

        # Render any chart the agent generated (stored as PNG bytes in session state)
        _chart_bytes = st.session_state.pop("_chat_chart", None)
        if _chart_bytes:
            try:
                st.image(_chart_bytes, use_column_width=True)
            except TypeError:
                # Very old Streamlit — fall back to no width hint
                st.image(_chart_bytes)

        if tools_called:
            st.caption("Tools executed: " + " > ".join(f"`{t}`" for t in tools_called))
        else:
            st.caption("No tool called — general response")

        # Navigation hint when agent recommends opening a page
        _PAGE_MAP = {
            "Load":          "1_Load_Data",
            "Rules":         "2_Rules",
            "Validate":      "3_Validate",
            "AI":            "4_AI_Investigation",
            "Command":       "5_Command_Center",
            "Clean":         "6_Cleaning",
            "Dedupe":        "7_Dedupe",
            "Settings":      "8_Settings",
            "Corpus":        "9_Corpus_Manager",
            "Pipeline":      "10_Pipeline_Manager",
            "Vector":        "11_Vector_Index",
            "Entity":        "12_Entity_Resolution",
            "Graph":         "13_Entity_Graph",
        }
        if open_tool:
            matched = next((v for k, v in _PAGE_MAP.items() if k.lower() in open_tool.lower()), None)
            hint = f"pages/{matched}" if matched else open_tool
            st.info(f"Agent suggests navigating to: **{open_tool}** — use the sidebar to go there.")

    chat_history.append({"role": "assistant", "content": response})
    st.session_state["agent_chat_history"] = chat_history

# ── Clear chat button ───────────────────────────────────────────────────────
if chat_history:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Clear Chat", key="clear_agent_chat"):
        st.session_state["agent_chat_history"] = []
        st.rerun()

if not enrichment and not db_data:
    terminal_block(
        "// TIP: AI enrichment results will appear above once you run validation with AI enabled.<br>"
        "<span style='color:#5a9a5a;'>The agent chat works independently — you can use it even without AI enrichment results.</span>"
    )
