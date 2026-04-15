"""
Page 12: Entity Resolution -- Find matching records across multiple datasets.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header
from shared.state import init_state
from core.io_utils import load_any

init_state()
apply_theme()
require_auth()

page_header("ENTITY RESOLUTION", "Find matching records across multiple datasets")

st.markdown(
    '<div class="glass-card">'
    '<p style="color:#b0ffb8;font-size:0.78rem;">'
    'Advanced fuzzy matching across datasets. Supports blocking, column-specific weights, '
    'phonetic matching for names, semantic matching with sentence transformers, '
    'and ultra-fast ChromaDB vector search.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
files = st.file_uploader(
    "Upload 2+ datasets to compare",
    type=["csv", "xlsx", "parquet"],
    accept_multiple_files=True,
    key="er_file_uploader",
)

if not files or len(files) < 2:
    st.info("Upload at least 2 datasets to find matches.")
    st.stop()

dfs = {f.name: load_any(f) for f in files}
st.success(f"Loaded {len(dfs)} datasets: {', '.join(dfs.keys())}")

with st.expander("Dataset Information"):
    for name, df in dfs.items():
        st.markdown(f"**{name}:** {len(df)} rows x {len(df.columns)} columns -- Columns: {', '.join(df.columns.tolist())}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
section_header("// Matching Configuration")

config_col1, config_col2 = st.columns(2)

with config_col1:
    threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.80, 0.05, help="Minimum similarity to consider a match")

    use_blocking = st.checkbox("Use Blocking (faster for large datasets)", value=False)
    blocking_column = None
    if use_blocking:
        common_cols = set(dfs[list(dfs.keys())[0]].columns)
        for df in dfs.values():
            common_cols = common_cols.intersection(set(df.columns))
        if common_cols:
            blocking_column = st.selectbox("Blocking Column", ["None"] + list(common_cols))
            if blocking_column == "None":
                blocking_column = None
        else:
            st.warning("No common columns found across all datasets")
            use_blocking = False

with config_col2:
    use_column_weights = st.checkbox("Use Column-Specific Weights", value=False)
    use_phonetic = st.checkbox("Use Phonetic Matching for Names", value=False)

    phonetic_columns = []
    if use_phonetic:
        common_text = set(dfs[list(dfs.keys())[0]].select_dtypes(include="object").columns)
        for df in dfs.values():
            common_text = common_text.intersection(set(df.select_dtypes(include="object").columns))
        if common_text:
            phonetic_columns = st.multiselect("Columns for Phonetic Matching", list(common_text))
        else:
            st.warning("No common text columns found")
            use_phonetic = False

    use_semantic = st.checkbox("Use Semantic Matching (AI-powered)", value=False)
    semantic_model_er = None
    if use_semantic:
        try:
            from core.semantic_matcher import get_available_models
            models_info = get_available_models()
            semantic_model_er = st.selectbox(
                "Semantic Model",
                options=list(models_info.keys()),
                format_func=lambda x: f"{models_info[x]['name']} ({models_info[x]['speed']})",
                key="er_semantic_model",
            )
        except ImportError:
            st.warning("core.semantic_matcher not available")
            use_semantic = False

    use_chromadb = st.checkbox("Use ChromaDB Vector Index (Ultra Fast)", value=False)
    chromadb_model = None
    if use_chromadb:
        try:
            from core.vector_db_manager import VectorDBManager
            if "vector_db_manager" not in st.session_state:
                st.session_state["vector_db_manager"] = VectorDBManager()
            from core.semantic_matcher import get_available_models
            models_info = get_available_models()
            chromadb_model = st.selectbox(
                "Embedding Model for ChromaDB",
                options=list(models_info.keys()),
                format_func=lambda x: f"{models_info[x]['name']} ({models_info[x]['speed']})",
                key="er_chromadb_model",
            )
        except ImportError:
            st.warning("ChromaDB or semantic_matcher not available")
            use_chromadb = False

# Column weights
column_weights = None
if use_column_weights:
    section_header("// Column Weights")
    common_cols = set(dfs[list(dfs.keys())[0]].columns)
    for df in dfs.values():
        common_cols = common_cols.intersection(set(df.columns))

    if common_cols:
        column_weights = {}
        weight_cols = st.columns(min(3, len(common_cols)))
        for idx, col in enumerate(sorted(common_cols)):
            with weight_cols[idx % len(weight_cols)]:
                weight = st.slider(f"{col}", 0.0, 2.0, 1.0, 0.1, key=f"weight_{col}")
                if weight > 0:
                    column_weights[col] = weight
        if not column_weights:
            st.warning("At least one column must have weight > 0")
            column_weights = None
    else:
        st.warning("No common columns found for weighting")

st.markdown("---")

# ---------------------------------------------------------------------------
# Run matching
# ---------------------------------------------------------------------------
if st.button("Find Matches", type="primary", use_container_width=True):
    with st.spinner("Finding matches across datasets..."):
        from core.entity_resolution import find_entity_matches, analyze_clusters

        er_pairs = find_entity_matches(
            datasets=dfs,
            threshold=threshold,
            use_blocking=use_blocking,
            blocking_column=blocking_column,
            column_weights=column_weights,
            use_phonetic=use_phonetic,
            phonetic_columns=phonetic_columns if use_phonetic else None,
            use_semantic=use_semantic,
            semantic_model=semantic_model_er if use_semantic else None,
            use_chromadb=use_chromadb,
            chromadb_model=chromadb_model if use_chromadb else None,
        )

        st.session_state["er_pairs"] = er_pairs
        st.session_state["er_datasets"] = dfs

        if len(er_pairs) > 0:
            st.session_state["er_cluster_stats"] = analyze_clusters(er_pairs)

    st.success(f"Found {len(er_pairs)} matches")
    st.rerun()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "er_pairs" in st.session_state and st.session_state["er_pairs"] is not None and len(st.session_state["er_pairs"]) > 0:
    er_pairs = st.session_state["er_pairs"]

    section_header("// Matching Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Matches", len(er_pairs))
    r2.metric("Avg Similarity", f"{er_pairs['score'].mean():.3f}")
    r3.metric("High Confidence (>=0.9)", len(er_pairs[er_pairs["score"] >= 0.9]))
    if "er_cluster_stats" in st.session_state:
        r4.metric("Clusters Found", st.session_state["er_cluster_stats"]["total_clusters"])

    # Filtering
    f1, f2 = st.columns(2)
    with f1:
        min_score_filter = st.slider(
            "Minimum Score", float(er_pairs["score"].min()), 1.0,
            float(er_pairs["score"].min()), 0.05, key="er_score_filter",
        )
    with f2:
        show_limit = st.selectbox("Rows to Display", [50, 100, 500, "All"], index=1, key="er_display_limit")

    filtered_pairs = er_pairs[er_pairs["score"] >= min_score_filter]
    display_pairs = filtered_pairs if show_limit == "All" else filtered_pairs.head(show_limit)

    st.dataframe(display_pairs, use_container_width=True, height=400)

    # Side-by-side comparison
    st.markdown("---")
    section_header("// Side-by-Side Record Comparison")

    er_datasets = st.session_state.get("er_datasets", {})
    if not display_pairs.empty and er_datasets:
        pair_labels = []
        for _, row in display_pairs.iterrows():
            pair_labels.append(
                f"{row['A']} row {row['A_row']}  --  {row['B']} row {row['B_row']}  --  {row['score']*100:.1f}%"
            )

        selected_label = st.selectbox("Choose a matched pair to inspect:", pair_labels, key="er_pair_selector")
        selected_idx = pair_labels.index(selected_label)
        selected_pair = display_pairs.iloc[selected_idx]

        df_a = er_datasets.get(selected_pair["A"])
        df_b = er_datasets.get(selected_pair["B"])

        if df_a is not None and df_b is not None:
            row_a_idx = int(selected_pair["A_row"])
            row_b_idx = int(selected_pair["B_row"])
            score = selected_pair["score"]

            if score >= 0.9:
                st.success(f"{score*100:.1f}% match (High Confidence)")
            elif score >= 0.75:
                st.warning(f"{score*100:.1f}% match (Medium Confidence)")
            else:
                st.error(f"{score*100:.1f}% match (Low Confidence)")

            rec_a = df_a.iloc[row_a_idx] if row_a_idx < len(df_a) else None
            rec_b = df_b.iloc[row_b_idx] if row_b_idx < len(df_b) else None

            if rec_a is not None and rec_b is not None:
                all_cols = list(dict.fromkeys(list(df_a.columns) + list(df_b.columns)))
                comparison_rows = []
                for col in all_cols:
                    val_a = str(rec_a[col]) if col in df_a.columns else "--"
                    val_b = str(rec_b[col]) if col in df_b.columns else "--"
                    is_diff = val_a.strip().lower() != val_b.strip().lower()
                    comparison_rows.append({
                        "Field": col,
                        selected_pair["A"]: val_a,
                        selected_pair["B"]: val_b,
                        "Match": "Same" if not is_diff else "Different",
                    })

                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df, use_container_width=True, height=min(60 + len(comparison_rows) * 35, 600))

    # Cluster analysis
    if "er_cluster_stats" in st.session_state:
        cluster_stats = st.session_state["er_cluster_stats"]
        st.markdown("---")
        section_header("// Cluster Analysis")

        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Total Clusters", cluster_stats["total_clusters"])
            st.metric("Nodes in Graph", cluster_stats["total_nodes"])
        with a2:
            st.metric("Avg Cluster Size", f"{cluster_stats['avg_cluster_size']:.1f}")
            st.metric("Max Cluster Size", cluster_stats["max_cluster_size"])
        with a3:
            st.metric("Multi-Node Clusters", cluster_stats["multi_node_clusters"])
            st.metric("Singleton Nodes", cluster_stats["singleton_clusters"])

        if cluster_stats.get("golden_candidates"):
            st.markdown("---")
            section_header("// Golden Record Candidates")
            golden_df = pd.DataFrame([
                {
                    "Cluster Size": c["cluster_size"],
                    "Golden Record": c["golden_record"],
                    "Centrality": f"{c['centrality']:.3f}",
                    "Avg Similarity": c["avg_similarity"],
                }
                for c in cluster_stats["golden_candidates"][:20]
            ])
            st.dataframe(golden_df, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Matched Pairs (CSV)",
            er_pairs.to_csv(index=False).encode("utf-8"),
            "entity_resolution_pairs.csv", "text/csv",
            key="download_er_pairs", use_container_width=True,
        )
    with d2:
        if "er_cluster_stats" in st.session_state and st.session_state["er_cluster_stats"].get("golden_candidates"):
            golden_df = pd.DataFrame(st.session_state["er_cluster_stats"]["golden_candidates"])
            st.download_button(
                "Download Golden Record Candidates (CSV)",
                golden_df.to_csv(index=False).encode("utf-8"),
                "golden_record_candidates.csv", "text/csv",
                key="download_golden_records", use_container_width=True,
            )

elif "er_pairs" in st.session_state and st.session_state["er_pairs"] is not None and len(st.session_state["er_pairs"]) == 0:
    st.info("No matches found with current settings. Try lowering the similarity threshold.")
