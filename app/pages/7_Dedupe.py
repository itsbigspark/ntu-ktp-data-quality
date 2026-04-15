"""
Page 7: Deduplicate -- Find and merge duplicate records.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header, terminal_block, kpi_card
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("DEDUPLICATION", "Find and merge duplicate records")

# Use best available data
df = next(
    (st.session_state.get(k) for k in ("df_corrected", "df_cleaned", "df_raw")
     if st.session_state.get(k) is not None),
    None,
)

if df is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

st.markdown(
    f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
    f'Working with: {len(df)} rows, {len(df.columns)} columns</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------
section_header("// Exact Duplicate Detection")

exact_dups = df.duplicated().sum()
kpi_cols = st.columns(3, gap="medium")
kpi_cols[0].markdown(kpi_card(str(len(df)), "Total Rows", "cyan"), unsafe_allow_html=True)
kpi_cols[1].markdown(kpi_card(str(exact_dups), "Exact Duplicates", "warn" if exact_dups > 0 else ""), unsafe_allow_html=True)
kpi_cols[2].markdown(kpi_card(f"{(1 - exact_dups/len(df))*100:.1f}%", "Uniqueness", ""), unsafe_allow_html=True)

if exact_dups > 0:
    if st.button("REMOVE EXACT DUPLICATES", key="remove_exact", use_container_width=True):
        df_deduped = df.drop_duplicates()
        st.session_state["df_deduplicated"] = df_deduped
        st.success(f"Removed {exact_dups} exact duplicates. {len(df_deduped)} rows remaining.")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fuzzy duplicate detection
# ---------------------------------------------------------------------------
section_header("// Fuzzy Duplicate Detection")

st.markdown(
    '<p style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;">'
    'Find near-duplicate records using text similarity matching.</p>',
    unsafe_allow_html=True,
)

text_cols = df.select_dtypes(include=["object"]).columns.tolist()
match_cols = st.multiselect("Columns to match on", text_cols, default=text_cols[:3], key="fuzzy_cols")
threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.85, 0.05, key="fuzzy_threshold")

if match_cols and st.button("FIND FUZZY DUPLICATES", key="find_fuzzy", use_container_width=True):
    with st.spinner("Computing similarity scores..."):
        try:
            from core.vectorizers import build_vectorizers
            from core.matcher import compute_pair_scores, label_pairs

            vectorizers = build_vectorizers(df, match_cols)
            pairs = compute_pair_scores(df, vectorizers, match_cols)
            labelled = label_pairs(pairs, threshold=threshold)

            dups = labelled[labelled["label"] == "duplicate"]
            st.session_state["dedup_pairs"] = dups

            st.success(f"Found {len(dups)} potential duplicate pairs (threshold: {threshold})")

            if not dups.empty:
                st.dataframe(dups.head(50), use_container_width=True, hide_index=True, height=300)

                csv = dups.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "EXPORT DUPLICATE PAIRS [CSV]",
                    data=csv,
                    file_name="duplicate_pairs.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Fuzzy matching failed: {e}")

# ---------------------------------------------------------------------------
# Download deduplicated data
# ---------------------------------------------------------------------------
deduped = st.session_state.get("df_deduplicated")
if deduped is not None:
    section_header("// Export Deduplicated Data")
    csv = deduped.to_csv(index=False).encode("utf-8")
    st.download_button("DOWNLOAD DEDUPLICATED DATA [CSV]", data=csv, file_name="deduplicated_data.csv", mime="text/csv", use_container_width=True)
