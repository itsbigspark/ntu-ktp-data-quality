"""
Page 19: Pipeline Monitor -- connect a source, run a batch, track results.

Drives the headless source-connector + batch runner (dataqualify.batch.run_batch)
from the UI: point it at a file, an S3 object, a JSON HTTP API, or the Companies
House API, and every run is recorded and shown here. This is the interactive face
of the "connect a source -> it runs itself -> each batch is tracked" flow.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, kpi_card, terminal_block
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("PIPELINE MONITOR", "Connect a source, run a batch, track every result")

from dataqualify.sources import parse_source
from dataqualify.batch import run_batch, SQLiteBatchStore

_store = SQLiteBatchStore()

# ---------------------------------------------------------------------------
# Run a batch from any source
# ---------------------------------------------------------------------------
section_header("// Run a Batch From a Source")

st.markdown(
    '<p style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;">'
    'A source can be a local file, an S3 object, a JSON HTTP API, or Companies House. '
    'It is fetched, validated, and recorded as a tracked batch.</p>',
    unsafe_allow_html=True,
)

_examples = {
    "Local file": "companies_house_clean.csv",
    "S3 object": "s3://my-bucket/incoming/data.csv",
    "HTTP JSON API": "https://api.example.com/records",
    "Companies House": "companies-house:12345678",
}
_c1, _c2 = st.columns([3, 1])
with _c2:
    _kind = st.selectbox("Example", list(_examples.keys()), key="pm_kind")
with _c1:
    _source_spec = st.text_input(
        "Source", value=_examples[_kind],
        help="file path · s3://bucket/key · https://... · companies-house:NUMBER",
        key="pm_source",
    )

_o1, _o2 = st.columns(2)
with _o1:
    _sink = st.text_input("Sink directory (optional)", value="",
                          placeholder="./out", key="pm_sink",
                          help="If set, issues / corrected / report are written here.")
with _o2:
    _ref = st.text_input("Reference file (optional)", value="",
                         placeholder="clean_reference.csv", key="pm_ref",
                         help="Infer rules from a clean reference instead of the data itself.")

if st.button("RUN BATCH", key="pm_run", use_container_width=True, type="primary"):
    if not _source_spec.strip():
        st.error("Enter a source.")
    else:
        try:
            reference = pd.read_csv(_ref) if _ref.strip() else None
            with st.spinner(f"Fetching and validating: {_source_spec}"):
                rec = run_batch(
                    parse_source(_source_spec.strip()),
                    reference=reference,
                    sink_dir=(_sink.strip() or None),
                    store=_store,
                )
            if rec.status == "completed":
                _verdict_style = {"accept": st.success, "review": st.warning,
                                  "quarantine": st.error}.get(rec.verdict or "review", st.info)
                _verdict_style(
                    f"Batch {rec.batch_id} — {(rec.verdict or '').upper()} "
                    f"({rec.severity}) — score {rec.overall_score}, "
                    f"{rec.issues_count} issues across {rec.rows} rows, {rec.duration_s}s."
                )
                if rec.narrative:
                    st.markdown(
                        f"<div style='background:rgba(0,229,255,0.06);border-left:3px solid #00e5ff;"
                        f"padding:10px 14px;font-family:Share Tech Mono;font-size:0.78rem;"
                        f"color:#b0ffb8;'>AGENT: {rec.narrative}</div>",
                        unsafe_allow_html=True,
                    )
                if rec.sink:
                    st.caption(f"Results written to {rec.sink}")
            else:
                st.error(f"Batch {rec.batch_id} QUARANTINED — {rec.error}")
        except Exception as exc:
            st.error(f"Could not run batch: {exc}")

# ---------------------------------------------------------------------------
# Tracked batches
# ---------------------------------------------------------------------------
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
section_header("// Tracked Batches")

_bcol1, _bcol2 = st.columns([1, 5])
with _bcol1:
    if st.button("REFRESH", key="pm_refresh", use_container_width=True):
        st.rerun()

records = _store.list(limit=200)

if not records:
    terminal_block("// NO BATCHES YET<br><span style='color:#5a9a5a;'>Run a batch above to start tracking.</span>")
else:
    total = len(records)
    completed = [r for r in records if r.status == "completed"]
    passed = [r for r in completed if r.passed]
    avg_score = (sum(r.overall_score for r in completed if r.overall_score is not None)
                 / max(len(completed), 1)) if completed else 0.0
    pass_rate = (len(passed) / len(completed) * 100) if completed else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_card(str(total), "Total batches"), unsafe_allow_html=True)
    with k2: st.markdown(kpi_card(f"{len(completed)}", "Completed"), unsafe_allow_html=True)
    with k3: st.markdown(kpi_card(f"{pass_rate:.0f}%", "Pass rate"), unsafe_allow_html=True)
    with k4: st.markdown(kpi_card(f"{avg_score:.1f}", "Avg score"), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    table = pd.DataFrame([{
        "Batch": r.batch_id,
        "Source": r.source,
        "Status": r.status,
        "Verdict": (r.verdict or "").upper(),
        "Score": "" if r.overall_score is None else round(r.overall_score, 1),
        "Issues": "" if r.issues_count is None else r.issues_count,
        "Rows": r.rows,
        "Agent narrative": r.narrative or "",
        "When": r.timestamp,
    } for r in records])

    st.dataframe(table, use_container_width=True, hide_index=True,
                 height=min(560, 60 + 34 * len(table)))

    # Failure detail
    failures = [r for r in records if r.status == "failed"]
    if failures:
        with st.expander(f"// {len(failures)} FAILED BATCHES", expanded=False):
            for r in failures[:20]:
                st.markdown(
                    f"<div style='font-family:Share Tech Mono;font-size:0.74rem;color:#ff6b6b;'>"
                    f"{r.batch_id} · {r.source}<br><span style='color:#b0b0b0;'>{r.error}</span></div>"
                    "<div style='height:6px'></div>",
                    unsafe_allow_html=True,
                )
