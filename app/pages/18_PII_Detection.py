"""
Page 18: PII Detection -- Scan dataset for Personally Identifiable Information.

Uses Microsoft Presidio (Apache 2.0) with spaCy en_core_web_lg for NER.
Detects 50+ entity types including NHS numbers, NI numbers, UK postcodes,
bank accounts, credit cards, passport numbers, names, emails, phone numbers.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import (
    apply_theme, page_header, section_header, terminal_block,
    workflow_breadcrumb, kpi_card,
    MATRIX_GREEN, MATRIX_CYAN, MATRIX_RED, MATRIX_ORANGE, MATRIX_YELLOW,
)
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("PII DETECTION", "Scan for Personally Identifiable Information using Microsoft Presidio")

workflow_breadcrumb([
    ("Load Data", st.session_state.get("df_raw") is not None),
    ("PII Scan", st.session_state.get("pii_scan_result") is not None),
    ("Anonymise", st.session_state.get("df_anonymised") is not None),
])

df_raw = st.session_state.get("df_raw")
if df_raw is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

# ---------------------------------------------------------------------------
# Presidio initialisation (cached — model load takes ~3s)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PII detection engine...")
def _build_analyzer():
    """Build a Presidio AnalyzerEngine with spaCy NLP + UK-specific recognizers."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Use spaCy en_core_web_lg for best NER accuracy
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)


@st.cache_resource(show_spinner=False)
def _build_anonymizer():
    from presidio_anonymizer import AnonymizerEngine
    return AnonymizerEngine()


try:
    analyzer = _build_analyzer()
    anonymizer = _build_anonymizer()
    _engine_ok = True
except Exception as _e:
    st.error(
        "PII detection engine could not be loaded. "
        "Make sure presidio-analyzer, presidio-anonymizer, and spacy en_core_web_lg are installed."
    )
    st.caption(f"Details: {type(_e).__name__}")
    st.stop()

# ---------------------------------------------------------------------------
# Entity type reference
# ---------------------------------------------------------------------------
_ENTITY_GROUPS = {
    "Identity": [
        "PERSON", "NRP",                              # names, nationality/religion/politics
    ],
    "Finance": [
        "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER",
    ],
    "Government IDs": [
        "US_SSN", "US_PASSPORT", "UK_NHS", "SG_NRIC_FIN",
        "AU_ABN", "AU_ACN", "AU_TFN", "IN_PAN", "IN_AADHAAR",
    ],
    "Contact": [
        "EMAIL_ADDRESS", "PHONE_NUMBER", "URL",
    ],
    "Location": [
        "LOCATION", "IP_ADDRESS",
    ],
    "Dates & IDs": [
        "DATE_TIME", "MEDICAL_LICENSE", "US_DRIVER_LICENSE",
        "US_ITIN",
    ],
    "Crypto": [
        "CRYPTO",
    ],
}

_ALL_ENTITIES = sorted({e for group in _ENTITY_GROUPS.values() for e in group})

_SEVERITY_MAP = {
    "CREDIT_CARD": "CRITICAL", "IBAN_CODE": "CRITICAL", "US_SSN": "CRITICAL",
    "US_PASSPORT": "CRITICAL", "UK_NHS": "CRITICAL",
    "PERSON": "HIGH", "PHONE_NUMBER": "HIGH", "EMAIL_ADDRESS": "HIGH",
    "US_BANK_NUMBER": "HIGH", "DATE_TIME": "MEDIUM",
    "LOCATION": "MEDIUM", "IP_ADDRESS": "MEDIUM",
    "URL": "LOW", "NRP": "MEDIUM",
}
_SEV_COLOR = {
    "CRITICAL": MATRIX_RED, "HIGH": MATRIX_ORANGE,
    "MEDIUM": MATRIX_YELLOW, "LOW": MATRIX_CYAN,
}

# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------
section_header("// Scan Configuration")

text_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
all_cols = df_raw.columns.tolist()

cfg_col1, cfg_col2 = st.columns([2, 1], gap="large")

with cfg_col1:
    selected_cols = st.multiselect(
        "Columns to scan",
        all_cols,
        default=text_cols[:10],
        key="pii_cols",
        help="Select which columns to run PII detection on. Text columns are pre-selected.",
    )

with cfg_col2:
    score_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        key="pii_threshold",
        help="Only flag detections above this confidence score.",
    )
    max_rows = st.number_input(
        "Max rows to scan",
        min_value=10, max_value=min(100_000, len(df_raw)),
        value=min(5_000, len(df_raw)),
        step=500,
        key="pii_max_rows",
    )

with st.expander("Entity types to detect", expanded=False):
    selected_entities = []
    for group_name, entities in _ENTITY_GROUPS.items():
        st.markdown(
            f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.72rem;'
            f'letter-spacing:1px;margin-bottom:4px;">{group_name}</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(4, len(entities)))
        for i, ent in enumerate(entities):
            if cols[i % len(cols)].checkbox(ent, value=True, key=f"pii_ent_{ent}"):
                selected_entities.append(ent)

if not selected_entities:
    selected_entities = _ALL_ENTITIES

# ---------------------------------------------------------------------------
# Run scan
# ---------------------------------------------------------------------------
if st.button("RUN PII SCAN", key="run_pii_scan", use_container_width=True, type="primary"):
    if not selected_cols:
        st.error("Select at least one column to scan.")
    else:
        df_scan = df_raw.head(max_rows)
        findings = []

        progress = st.progress(0, text="Scanning...")
        total = len(selected_cols) * len(df_scan)
        done = 0

        for col in selected_cols:
            for row_idx, val in df_scan[col].items():
                cell_str = str(val) if pd.notna(val) else ""
                if cell_str and cell_str.lower() not in ("nan", "none", ""):
                    try:
                        results = analyzer.analyze(
                            text=cell_str,
                            entities=selected_entities,
                            language="en",
                            score_threshold=score_threshold,
                        )
                        for r in results:
                            findings.append({
                                "row": row_idx,
                                "column": col,
                                "entity_type": r.entity_type,
                                "value": cell_str[r.start:r.end],
                                "confidence": round(r.score, 3),
                                "severity": _SEVERITY_MAP.get(r.entity_type, "MEDIUM"),
                                "start": r.start,
                                "end": r.end,
                            })
                    except Exception:
                        pass

                done += 1
                if done % max(1, total // 50) == 0:
                    progress.progress(min(done / total, 1.0), text=f"Scanning {col}...")

        progress.empty()
        st.session_state["pii_scan_result"] = pd.DataFrame(findings)
        st.session_state["pii_scanned_cols"] = selected_cols
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
pii_results = st.session_state.get("pii_scan_result")

if pii_results is not None:
    if pii_results.empty:
        st.success("No PII detected above the confidence threshold in the scanned columns.")
    else:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section_header("// Scan Results")

        # KPI row
        n_critical = len(pii_results[pii_results["severity"] == "CRITICAL"])
        n_high     = len(pii_results[pii_results["severity"] == "HIGH"])
        n_cols_hit = pii_results["column"].nunique()
        n_rows_hit = pii_results["row"].nunique()

        k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
        k1.markdown(kpi_card(str(len(pii_results)), "Total Findings", ""), unsafe_allow_html=True)
        k2.markdown(kpi_card(str(n_critical), "Critical", "danger"), unsafe_allow_html=True)
        k3.markdown(kpi_card(str(n_high), "High", "warn"), unsafe_allow_html=True)
        k4.markdown(kpi_card(str(n_cols_hit), "Columns Affected", "cyan"), unsafe_allow_html=True)
        k5.markdown(kpi_card(str(n_rows_hit), "Rows Affected", ""), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Summary by entity type
        section_header("// Breakdown by Entity Type")
        summary = (
            pii_results.groupby(["entity_type", "severity"])
            .agg(count=("row", "count"), columns=("column", lambda x: ", ".join(sorted(x.unique()))))
            .reset_index()
            .sort_values(["severity", "count"], ascending=[True, False])
        )

        for _, row in summary.iterrows():
            sev_color = _SEV_COLOR.get(row["severity"], MATRIX_CYAN)
            st.markdown(
                f'<div style="background:rgba(0,5,1,0.8);border:1px solid rgba(255,255,255,0.08);'
                f'border-radius:6px;padding:10px 14px;margin-bottom:6px;'
                f'display:flex;justify-content:space-between;align-items:center;'
                f'font-family:Share Tech Mono;">'
                f'<span style="color:{MATRIX_GREEN};font-size:0.82rem;">{row["entity_type"]}</span>'
                f'<span style="font-size:0.62rem;color:{sev_color};border:1px solid {sev_color};'
                f'padding:1px 7px;border-radius:4px;letter-spacing:1px;">{row["severity"]}</span>'
                f'<span style="color:#4a7a4f;font-size:0.72rem;">columns: {row["columns"]}</span>'
                f'<span style="color:{MATRIX_CYAN};font-size:0.82rem;font-weight:bold;">'
                f'{row["count"]} finding{"s" if row["count"] != 1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Detailed findings table
        section_header("// Detailed Findings")
        display_df = pii_results[[
            "row", "column", "entity_type", "severity", "value", "confidence"
        ]].copy()
        display_df.columns = ["Row", "Column", "Entity Type", "Severity", "Detected Value", "Confidence"]
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.0%}")

        severity_filter = st.multiselect(
            "Filter by severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"],
            key="pii_sev_filter",
        )
        filtered = display_df[display_df["Severity"].isin(severity_filter)]
        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            hide_index=True,
            height=min(500, 40 + 35 * len(filtered)),
        )
        if len(filtered) > 500:
            st.caption(f"Showing first 500 of {len(filtered):,} findings.")

        # Export findings
        csv_findings = pii_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "EXPORT PII FINDINGS [CSV]",
            data=csv_findings,
            file_name="pii_findings.csv",
            mime="text/csv",
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------------
        # Anonymisation
        # ---------------------------------------------------------------------------
        section_header("// Anonymise Data")

        st.markdown(
            '<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.75rem;">'
            'Replace detected PII with anonymised tokens. Original data is never modified — '
            'a new anonymised copy is created.</p>',
            unsafe_allow_html=True,
        )

        anon_col1, anon_col2 = st.columns(2, gap="large")
        with anon_col1:
            anon_operator = st.selectbox(
                "Anonymisation method",
                ["replace", "redact", "mask", "hash"],
                key="pii_anon_op",
                help=(
                    "replace: substitute with <ENTITY_TYPE> token. "
                    "redact: delete the value entirely. "
                    "mask: replace characters with *. "
                    "hash: SHA-256 hash (preserves referential integrity)."
                ),
            )
        with anon_col2:
            anon_cols = st.multiselect(
                "Columns to anonymise",
                st.session_state.get("pii_scanned_cols", []),
                default=pii_results[pii_results["severity"].isin(["CRITICAL", "HIGH"])]["column"].unique().tolist(),
                key="pii_anon_cols",
            )

        if st.button("ANONYMISE SELECTED COLUMNS", key="run_anon", use_container_width=True):
            from presidio_anonymizer.entities import OperatorConfig

            op_map = {
                "replace": OperatorConfig("replace"),
                "redact":  OperatorConfig("redact"),
                "mask":    OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False}),
                "hash":    OperatorConfig("hash", {"hash_type": "sha256"}),
            }

            df_anon = df_raw.copy()
            anon_count = 0
            progress2 = st.progress(0, text="Anonymising...")

            for i, col in enumerate(anon_cols):
                col_findings = pii_results[pii_results["column"] == col]
                affected_rows = col_findings["row"].unique()

                for row_idx in affected_rows:
                    val = str(df_anon.at[row_idx, col]) if pd.notna(df_anon.at[row_idx, col]) else ""
                    if not val or val.lower() in ("nan", "none"):
                        continue
                    try:
                        row_results = analyzer.analyze(
                            text=val,
                            entities=selected_entities,
                            language="en",
                            score_threshold=score_threshold,
                        )
                        if row_results:
                            from presidio_anonymizer.entities import RecognizerResult as _RR
                            anon_result = anonymizer.anonymize(
                                text=val,
                                analyzer_results=row_results,
                                operators={e: op_map[anon_operator] for e in selected_entities},
                            )
                            df_anon.at[row_idx, col] = anon_result.text
                            anon_count += 1
                    except Exception:
                        pass

                progress2.progress((i + 1) / len(anon_cols), text=f"Anonymising {col}...")

            progress2.empty()
            st.session_state["df_anonymised"] = df_anon
            st.success(f"Anonymised {anon_count:,} cells across {len(anon_cols)} column(s). Original data unchanged.")

        # Download anonymised
        df_anon_out = st.session_state.get("df_anonymised")
        if df_anon_out is not None:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            csv_anon = df_anon_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "DOWNLOAD ANONYMISED DATA [CSV]",
                data=csv_anon,
                file_name="anonymised_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.dataframe(df_anon_out.head(50), use_container_width=True, hide_index=False, height=300)

elif pii_results is None:
    terminal_block(
        "// RUN A SCAN TO SEE PII FINDINGS<br>"
        "<span style='color:#4a7a4f;'>Select columns above and click RUN PII SCAN.</span>"
    )
