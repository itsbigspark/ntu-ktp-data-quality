"""
Page 7: Deduplicate
Find and merge duplicate records using exact matching, KNN, or blocking + KNN.
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

page_header("DEDUPLICATION", "Exact and fuzzy duplicate detection with blocking")

# ---------------------------------------------------------------------------
# Data guard
# ---------------------------------------------------------------------------
df = next(
    (st.session_state.get(k) for k in ("df_corrected", "df_cleaned", "df_raw")
     if st.session_state.get(k) is not None),
    None,
)
if df is None:
    terminal_block("// NO DATA LOADED<br><span style='color:#4a7a4f;'>Go to Load Data first.</span>")
    st.stop()

FUZZY_ROW_CAP = 50_000
_large = len(df) > FUZZY_ROW_CAP

st.markdown(
    f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
    f'Working with: {len(df):,} rows &nbsp;|&nbsp; {len(df.columns)} columns'
    + (' &nbsp;<span style="color:#ff9100;">[LARGE DATASET]</span>' if _large else '')
    + '</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Section 1: Exact Duplicates
# ---------------------------------------------------------------------------
section_header("// 01. Exact Duplicate Detection")

exact_dups = int(df.duplicated().sum())
kpi_cols = st.columns(3, gap="medium")
kpi_cols[0].markdown(kpi_card(f"{len(df):,}", "Total Rows", "cyan"), unsafe_allow_html=True)
kpi_cols[1].markdown(kpi_card(str(exact_dups), "Exact Duplicates", "warn" if exact_dups > 0 else ""), unsafe_allow_html=True)
kpi_cols[2].markdown(kpi_card(f"{(1 - exact_dups/max(len(df),1))*100:.1f}%", "Uniqueness", ""), unsafe_allow_html=True)

if exact_dups > 0:
    if st.button("REMOVE EXACT DUPLICATES", key="remove_exact", use_container_width=True):
        df_deduped = df.drop_duplicates()
        st.session_state["df_deduplicated"] = df_deduped
        st.session_state["df_corrected"] = df_deduped
        st.success(f"Removed {exact_dups:,} exact duplicates. {len(df_deduped):,} rows remaining.")
        st.rerun()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Fuzzy Duplicate Detection
# ---------------------------------------------------------------------------
section_header("// 02. Fuzzy Duplicate Detection")

text_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols  = df.select_dtypes(include=["number"]).columns.tolist()
all_cols  = df.columns.tolist()

if not text_cols:
    st.warning("No text columns found — fuzzy matching requires at least one string column.")
    st.stop()

# ── Core column config ──────────────────────────────────────────────────────
c1, c2 = st.columns([2, 1])
with c1:
    match_cols = st.multiselect(
        "Columns to match on",
        text_cols,
        default=text_cols[:min(3, len(text_cols))],
        key="fuzzy_cols",
        help="Text columns whose similarity is computed. Pick the most identifying ones (name, address, email…).",
    )
with c2:
    threshold = st.slider(
        "Duplicate threshold",
        min_value=0.5, max_value=1.0, value=0.85, step=0.05,
        key="fuzzy_threshold",
        help="Pairs scoring at or above this are labelled 'duplicate'.",
    )

# ── Run-mode selector ───────────────────────────────────────────────────────
st.markdown(
    '<p style="color:#b0ffb8;font-family:Share Tech Mono;font-size:0.78rem;margin:8px 0 4px 0;">'
    'MATCHING STRATEGY</p>',
    unsafe_allow_html=True,
)

_default_mode = "Blocking + KNN" if _large else "Fast KNN"
run_mode = st.radio(
    "Run mode",
    ["Fast KNN", "Blocking + KNN", "NxN (full)"],
    index=["Fast KNN", "Blocking + KNN", "NxN (full)"].index(_default_mode),
    horizontal=True,
    key="run_mode",
    label_visibility="collapsed",
    help=(
        "Fast KNN: approx nearest-neighbour search — best for <50k rows. "
        "Blocking + KNN: partition by key column first, then KNN within each block — handles millions of rows. "
        "NxN (full): exact cosine matrix — only feasible for <5k rows."
    ),
)

if run_mode == "NxN (full)" and len(df) > 5_000:
    st.error(
        f"NxN mode computes {len(df):,} x {len(df):,} = {len(df)**2:,} pairs. "
        "This will run out of memory. Use Fast KNN or Blocking + KNN instead."
    )

# ── Blocking config ─────────────────────────────────────────────────────────
block_cols = []
if run_mode in ("Blocking + KNN", "Blocking"):
    st.markdown(
        '<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.75rem;margin:10px 0 2px 0;">'
        'BLOCK KEY COLUMNS — records are only compared within the same bucket '
        '(e.g. same country, same product category). Pick low-cardinality columns.</p>',
        unsafe_allow_html=True,
    )
    block_cols = st.multiselect(
        "Block on columns",
        all_cols,
        default=[],
        key="block_cols",
        help="Rows with the same value in ALL block columns are grouped together. Only pairs within the same group are compared.",
    )
    if not block_cols:
        st.warning("No block columns selected — falling back to global Fast KNN.")

# ── Advanced settings ───────────────────────────────────────────────────────
with st.expander("Advanced Settings", expanded=False):
    adv1, adv2 = st.columns(2)
    with adv1:
        comparison_mode = st.selectbox(
            "Comparison mode",
            ["Row-concatenated", "Column-wise (weighted)", "Column-wise (FS-style probabilistic)"],
            index=0,
            key="comparison_mode",
            help=(
                "Row-concatenated: all columns merged into one TF-IDF vector. "
                "Column-wise (weighted): per-column similarity with configurable weights. "
                "Column-wise (FS-style probabilistic): Fellegi-Sunter log-likelihood ratio."
            ),
        )
        topk = st.slider(
            "KNN top-k neighbours",
            min_value=5, max_value=100, value=20, step=5,
            key="topk",
            help="Each row is compared only against its top-k nearest neighbours in the vector space.",
        )
    with adv2:
        similar_thr = st.slider(
            "Similar threshold",
            min_value=0.3, max_value=float(threshold), value=min(0.6, float(threshold) - 0.05),
            step=0.05,
            key="similar_thr",
            help="Pairs scoring between this and the duplicate threshold are labelled 'similar' (not merged).",
        )
        use_jaro = st.checkbox(
            "Blend Jaro-Winkler on key columns",
            value=False,
            key="use_jaro",
            help="Adds a character-level similarity boost for short-string columns (names, IDs).",
        )

    key_cols = []
    key_alpha = 0.4
    if use_jaro:
        key_cols = st.multiselect(
            "Jaro-Winkler key columns",
            text_cols,
            default=text_cols[:1],
            key="jaro_cols",
            help="Jaro-Winkler score on these columns is blended with the vector score.",
        )
        key_alpha = st.slider(
            "Jaro-Winkler weight (alpha)",
            0.0, 1.0, 0.4, 0.05,
            key="jaro_alpha",
            help="0 = pure vector similarity, 1 = pure Jaro-Winkler.",
        )

    st.markdown("**Vectorizer**")
    vc1, vc2 = st.columns(2)
    with vc1:
        use_char = st.checkbox("Char n-grams", value=True, key="use_char")
        char_min, char_max = st.select_slider(
            "Char n-gram range",
            options=[2, 3, 4, 5, 6],
            value=(3, 5),
            key="char_range",
        )
    with vc2:
        use_word = st.checkbox("Word n-grams", value=False, key="use_word")
        word_min, word_max = st.select_slider(
            "Word n-gram range",
            options=[1, 2, 3],
            value=(1, 2),
            key="word_range",
        )
    svd_k = st.slider("SVD components (0 = off)", 0, 128, 0, 16, key="svd_k",
                      help="Latent Semantic Analysis dimensionality reduction before similarity scoring.")

# ── Size estimate ───────────────────────────────────────────────────────────
_working_rows = min(len(df), FUZZY_ROW_CAP)
if run_mode == "Fast KNN":
    _est = f"~{_working_rows * topk:,} candidate pairs (KNN top-{topk})"
elif run_mode in ("Blocking + KNN",) and block_cols:
    _est = f"KNN within each {' × '.join(block_cols)} bucket — scales to millions of rows"
elif run_mode == "NxN (full)":
    _est = f"{_working_rows * (_working_rows - 1) // 2:,} pairs (full matrix)"
else:
    _est = f"~{_working_rows * topk:,} candidate pairs"

st.markdown(
    f'<p style="color:#4a7a4f;font-family:Share Tech Mono;font-size:0.72rem;margin:6px 0;">'
    f'Candidate pairs: {_est}</p>',
    unsafe_allow_html=True,
)

# ── Run button ───────────────────────────────────────────────────────────────
if not match_cols:
    st.info("Select at least one column to match on.")
elif st.button("FIND FUZZY DUPLICATES", key="find_fuzzy", use_container_width=True):
    with st.spinner("Building vectors and computing similarity scores..."):
        try:
            from core.vectorizers import build_vectorizers
            from core.matcher import compute_pair_scores, label_pairs

            # Enforce row cap
            df_fuzzy = df
            if len(df) > FUZZY_ROW_CAP:
                df_fuzzy = df.head(FUZZY_ROW_CAP).copy()
                st.warning(
                    f"Dataset has {len(df):,} rows. Fuzzy matching runs on first "
                    f"{FUZZY_ROW_CAP:,} rows. Use Blocking + KNN to cover more rows per run."
                )

            vec_cfg = {
                "use_char": use_char,
                "use_word": use_word,
                "char_range": (char_min, char_max),
                "word_range": (word_min, word_max),
                "svd_components": svd_k,
            }
            vec_pack = build_vectorizers(df_fuzzy, match_cols, vec_cfg)

            if vec_pack.get("dim", 0) == 0:
                st.error("Vectorization produced zero features. Try different columns or n-gram settings.")
                st.stop()

            # Build weights for column-wise mode
            weights = None
            if comparison_mode == "Column-wise (weighted)":
                n = len(match_cols)
                weights = {c: 1.0 / n for c in match_cols}

            _effective_run_mode = run_mode
            if run_mode in ("Blocking + KNN",) and not block_cols:
                _effective_run_mode = "Fast KNN"

            scores_df = compute_pair_scores(
                df=df_fuzzy,
                vec_pack=vec_pack,
                comparison_mode=comparison_mode,
                weights=weights,
                run_mode=_effective_run_mode,
                topk=topk,
                block_cols=block_cols if block_cols else None,
                key_cols=key_cols if use_jaro else None,
                key_alpha=key_alpha if use_jaro else 0.4,
            )

            labelled = label_pairs(scores_df, similar_thr=similar_thr, duplicate_thr=threshold)
            dups    = labelled[labelled["label"] == "duplicate"].copy()
            similar = labelled[labelled["label"] == "similar"].copy()

            st.session_state["dedup_pairs"]   = dups
            st.session_state["dedup_similar"] = similar
            st.session_state["dedup_labelled"] = labelled

            # KPI row
            k1, k2, k3, k4 = st.columns(4, gap="medium")
            k1.markdown(kpi_card(f"{len(labelled):,}", "Candidate Pairs", "cyan"), unsafe_allow_html=True)
            k2.markdown(kpi_card(str(len(dups)), "Duplicates", "danger" if len(dups) > 0 else ""), unsafe_allow_html=True)
            k3.markdown(kpi_card(str(len(similar)), "Similar", "warn"), unsafe_allow_html=True)
            k4.markdown(kpi_card(f"{_effective_run_mode.split()[0]}", "Mode Used", ""), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Matching failed — check your column selection and try again. ({type(e).__name__})")

# ---------------------------------------------------------------------------
# Results: duplicates + similar
# ---------------------------------------------------------------------------
dups    = st.session_state.get("dedup_pairs")
similar = st.session_state.get("dedup_similar")

if dups is not None and not dups.empty:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_header("// 03. Results — Duplicate Pairs")

    tab_dup, tab_sim = st.tabs([f"Duplicates ({len(dups)})", f"Similar ({len(similar) if similar is not None else 0})"])

    with tab_dup:
        st.dataframe(dups.head(200), use_container_width=True, hide_index=True, height=320)
        csv_d = dups.to_csv(index=False).encode("utf-8")
        st.download_button(
            "EXPORT DUPLICATE PAIRS [CSV]",
            data=csv_d, file_name="duplicate_pairs.csv", mime="text/csv",
        )

    with tab_sim:
        if similar is not None and not similar.empty:
            st.dataframe(similar.head(200), use_container_width=True, hide_index=True, height=320)
            csv_s = similar.to_csv(index=False).encode("utf-8")
            st.download_button(
                "EXPORT SIMILAR PAIRS [CSV]",
                data=csv_s, file_name="similar_pairs.csv", mime="text/csv",
            )
        else:
            st.info("No similar-but-not-duplicate pairs found.")

    # ── Merge / Apply deduplication ──────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_header("// 04. Apply Deduplication")

    merge_strategy = st.selectbox(
        "Merge strategy",
        ["fill_nulls", "keep_first", "keep_last", "concatenate"],
        index=0,
        key="merge_strategy",
        help=(
            "fill_nulls: keep master record, fill its blanks from duplicates. "
            "keep_first: keep the first occurrence. "
            "keep_last: keep the last occurrence. "
            "concatenate: join non-null values with pipe separator."
        ),
    )

    if st.button("APPLY DEDUPLICATION & MERGE", key="apply_dedup", use_container_width=True):
        with st.spinner("Clustering and merging duplicate groups..."):
            try:
                from core.dedup_clustering import cluster_duplicates, apply_deduplication

                clusters = cluster_duplicates(dups, id_col_a="row_i", id_col_b="row_j")
                master_selections = {cid: rows[0] for cid, rows in clusters.items()}

                df_source = st.session_state.get("dedup_df_used", df)
                df_merged = apply_deduplication(
                    df_source, clusters, master_selections,
                    merge_strategy=merge_strategy,
                )

                removed = len(df_source) - len(df_merged)
                st.session_state["df_deduplicated"] = df_merged
                st.session_state["df_corrected"]    = df_merged

                k1, k2, k3 = st.columns(3, gap="medium")
                k1.markdown(kpi_card(f"{len(clusters):,}", "Clusters Found", "cyan"), unsafe_allow_html=True)
                k2.markdown(kpi_card(str(removed), "Rows Removed", "warn"), unsafe_allow_html=True)
                k3.markdown(kpi_card(f"{len(df_merged):,}", "Rows Remaining", ""), unsafe_allow_html=True)
                st.success(f"Deduplication applied. {removed:,} rows merged. Dataset updated.")

            except Exception as e:
                st.error(f"Merge failed — try a different strategy or re-run matching. ({type(e).__name__})")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
deduped = st.session_state.get("df_deduplicated")
if deduped is not None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_header("// 05. Export Deduplicated Data")
    st.markdown(
        f'<p style="color:#00e5ff;font-family:Share Tech Mono;font-size:0.78rem;">'
        f'{len(deduped):,} rows &nbsp;|&nbsp; {len(deduped.columns)} columns</p>',
        unsafe_allow_html=True,
    )
    csv = deduped.to_csv(index=False).encode("utf-8")
    st.download_button(
        "DOWNLOAD DEDUPLICATED DATA [CSV]",
        data=csv, file_name="deduplicated_data.csv", mime="text/csv",
        use_container_width=True,
    )
