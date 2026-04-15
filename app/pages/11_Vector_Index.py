"""
Page 11: Vector Index Manager -- Manage ChromaDB semantic search indexes.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from shared.auth import require_auth
from shared.theme import apply_theme, page_header, section_header
from shared.state import init_state

init_state()
apply_theme()
require_auth()

page_header("VECTOR INDEX", "Manage semantic search indexes using ChromaDB")

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
try:
    from core.vector_db_manager import VectorDBManager
    from core.semantic_matcher import get_sentence_transformer_model
except ImportError:
    st.error("Could not import Vector DB Manager. Ensure core/vector_db_manager.py exists and ChromaDB is installed.")
    st.code("pip install chromadb>=0.4.0", language="bash")
    st.stop()

if "vector_db_manager" not in st.session_state:
    st.session_state["vector_db_manager"] = VectorDBManager()

vdb = st.session_state["vector_db_manager"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
index_tab, query_tab, duplicate_tab, stats_tab = st.tabs(["Collections", "Query Index", "Find Duplicates", "Statistics"])

# ======================== COLLECTIONS TAB ========================
with index_tab:
    section_header("// Manage Vector Collections")

    db_info = vdb.get_database_size()
    c1, c2, c3 = st.columns(3)
    c1.metric("Collections", db_info["num_collections"])
    c2.metric("Total Records", f"{db_info['total_records']:,}")
    c3.metric("Storage (MB)", f"{db_info['estimated_storage_mb']:.1f}")

    st.markdown("---")

    collections = vdb.list_collections()

    if not collections:
        st.info("No collections yet. Create one below.")
    else:
        for coll in collections:
            with st.expander(f"{coll['name']} ({coll['count']:,} records)", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Description:** {coll['metadata'].get('description', 'No description')}")
                    st.markdown(f"**Created:** {coll['created_at']}")
                with c2:
                    st.markdown(f"**Records:** {coll['count']:,}")

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("View Stats", key=f"stats_{coll['name']}"):
                        st.session_state["selected_collection_stats"] = coll["name"]
                        st.info("Switch to Statistics tab to view detailed stats")
                with a2:
                    if st.button("Delete", key=f"delete_coll_{coll['name']}"):
                        if vdb.delete_collection(coll["name"]):
                            st.success(f"Deleted: {coll['name']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete collection")

    st.markdown("---")
    section_header("// Create New Collection")

    if not isinstance(st.session_state.get("df_raw"), pd.DataFrame):
        st.warning("Please upload data in the Load Data page first.")
    else:
        df = st.session_state["df_raw"]

        c1, c2 = st.columns(2)
        with c1:
            collection_name = st.text_input("Collection Name", placeholder="e.g., customer_names", key="new_coll_name")
        with c2:
            collection_desc = st.text_input("Description", placeholder="Brief description", key="new_coll_desc")

        text_columns = st.multiselect(
            "Columns to Index", df.columns.tolist(),
            help="Select columns to combine and index for semantic search",
        )

        model_name = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            format_func=lambda x: {
                "all-MiniLM-L6-v2": "all-MiniLM-L6-v2 (Fast, 80MB, Recommended)",
                "all-mpnet-base-v2": "all-mpnet-base-v2 (Best Quality, 420MB)",
                "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual (50+ languages, 420MB)",
            }[x],
        )

        overwrite = st.checkbox("Overwrite if exists", value=False)

        if st.button("Create and Index Collection", type="primary", use_container_width=True):
            if not collection_name:
                st.error("Please provide a collection name")
            elif not text_columns:
                st.error("Please select at least one column to index")
            else:
                with st.spinner("Creating collection and generating embeddings..."):
                    collection = vdb.create_collection(name=collection_name, description=collection_desc, overwrite=overwrite)
                    model = get_sentence_transformer_model(model_name)

                    texts = df[text_columns].astype(str).agg(" | ".join, axis=1).tolist()
                    ids = [f"row_{i}" for i in range(len(texts))]
                    metadata = []
                    for i, row in df.iterrows():
                        meta = {col: str(row[col]) for col in text_columns}
                        meta["row_index"] = int(i)
                        metadata.append(meta)

                    progress_bar = st.progress(0)
                    batch_size = 100
                    all_embeddings = []

                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        embeddings = model.encode(batch, show_progress_bar=False)
                        all_embeddings.append(embeddings)
                        progress_bar.progress(min((i + batch_size) / len(texts), 1.0))

                    all_embeddings = np.vstack(all_embeddings)

                    result = vdb.add_to_collection(
                        collection_name=collection.name,
                        embeddings=all_embeddings,
                        texts=texts, ids=ids, metadata=metadata,
                    )

                    progress_bar.progress(1.0)
                    st.success(f"Indexed {result['count']:,} records in {result['elapsed']:.2f}s ({result['rate']:.0f} records/sec)")
                    st.rerun()

# ======================== QUERY TAB ========================
with query_tab:
    section_header("// Query Vector Index")

    collections = vdb.list_collections()
    if not collections:
        st.info("No collections available. Create one in the Collections tab.")
    else:
        collection_names = [c["name"] for c in collections]
        selected_coll = st.selectbox("Select Collection", collection_names, key="query_collection_select")

        query_text = st.text_input("Search Query", placeholder="Enter text to find similar records...", key="query_text_input")
        c1, c2 = st.columns(2)
        with c1:
            n_results = st.slider("Number of Results", 1, 50, 10)
        with c2:
            min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.5, 0.05)

        if st.button("Search", type="primary", use_container_width=True):
            if not query_text:
                st.warning("Please enter a search query")
            else:
                with st.spinner("Searching..."):
                    result = vdb.query_collection(
                        collection_name=selected_coll,
                        query_texts=[query_text],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"],
                    )

                if result["results"]["ids"]:
                    docs = result["results"]["documents"][0]
                    distances = result["results"]["distances"][0]
                    metadatas = result["results"].get("metadatas", [[]])[0]

                    similarities = [1 - (d * d / 2) for d in distances]

                    filtered = [
                        (doc, sim, meta)
                        for doc, sim, meta in zip(docs, similarities, metadatas)
                        if sim >= min_similarity
                    ]

                    if filtered:
                        st.success(f"Found {len(filtered)} results above {min_similarity:.0%} similarity")
                        for idx, (doc, sim, meta) in enumerate(filtered, 1):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**{idx}. {doc}**")
                                if meta:
                                    st.caption(f"Row: {meta.get('row_index', 'N/A')}")
                            with c2:
                                if sim >= 0.9:
                                    st.success(f"**{sim:.1%}**")
                                elif sim >= 0.7:
                                    st.info(f"**{sim:.1%}**")
                                else:
                                    st.warning(f"**{sim:.1%}**")
                    else:
                        st.info(f"No results above {min_similarity:.0%} similarity threshold")
                else:
                    st.info("No results found")

# ======================== FIND DUPLICATES TAB ========================
with duplicate_tab:
    section_header("// Find Duplicates in Collection")

    collections = vdb.list_collections()
    if not collections:
        st.info("No collections available. Create one in the Collections tab.")
    else:
        collection_names = [c["name"] for c in collections]
        selected_coll = st.selectbox("Select Collection", collection_names, key="duplicate_collection_select")

        c1, c2 = st.columns(2)
        with c1:
            threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.85, 0.05)
        with c2:
            batch_size = st.number_input("Batch Size", 10, 1000, 100, 10)

        if st.button("Find Duplicates", type="primary", use_container_width=True):
            import time
            with st.spinner("Searching for duplicates..."):
                start_time = time.time()
                duplicates = vdb.find_duplicates(collection_name=selected_coll, threshold=threshold, batch_size=batch_size)
                elapsed = time.time() - start_time

            if duplicates:
                dup_df = pd.DataFrame(duplicates)
                st.warning(f"Found {len(duplicates)} duplicate pairs ({elapsed:.2f}s)")

                c1, c2, c3 = st.columns(3)
                c1.metric("Duplicate Pairs", len(duplicates))
                c2.metric("Avg Similarity", f"{dup_df['similarity'].mean():.1%}")
                c3.metric("Max Similarity", f"{dup_df['similarity'].max():.1%}")

                display_df = dup_df.copy()
                display_df["similarity"] = display_df["similarity"].apply(lambda x: f"{x:.1%}")
                st.dataframe(display_df, use_container_width=True, height=400)

                st.download_button(
                    "Download Duplicates (CSV)",
                    dup_df.to_csv(index=False).encode("utf-8"),
                    f"{selected_coll}_duplicates.csv", "text/csv",
                    use_container_width=True,
                )
            else:
                st.success("No duplicates found above threshold")

# ======================== STATISTICS TAB ========================
with stats_tab:
    section_header("// Collection Statistics")

    collections = vdb.list_collections()
    if not collections:
        st.info("No collections available. Create one in the Collections tab.")
    else:
        collection_names = [c["name"] for c in collections]
        default_idx = 0
        if "selected_collection_stats" in st.session_state and st.session_state["selected_collection_stats"] in collection_names:
            default_idx = collection_names.index(st.session_state["selected_collection_stats"])

        selected_coll = st.selectbox("Select Collection", collection_names, index=default_idx, key="stats_collection_select")
        stats = vdb.get_collection_stats(selected_coll)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{stats['count']:,}")
        c2.metric("Dimensions", stats["embedding_dimension"])
        c3.metric("Storage (MB)", f"{stats['storage_mb']:.2f}")
        if stats["count"] > 0:
            bytes_per = (stats["storage_mb"] * 1024 * 1024) / stats["count"]
            c4.metric("Bytes/Record", f"{bytes_per:.0f}")

        st.markdown("---")
        ic1, ic2 = st.columns(2)
        with ic1:
            st.markdown(f"**Name:** {stats['name']}")
            st.markdown(f"**Created:** {stats['created_at']}")
        with ic2:
            st.markdown(f"**Description:** {stats['metadata'].get('description', 'No description')}")

        st.markdown("---")
        section_header("// Export Collection")

        if st.button("Export to DataFrame", use_container_width=True):
            with st.spinner("Exporting..."):
                export_df = vdb.export_collection(selected_coll)
            st.success(f"Exported {len(export_df):,} records")
            st.dataframe(export_df.head(10), use_container_width=True)
            st.download_button(
                "Download CSV",
                export_df.to_csv(index=False).encode("utf-8"),
                f"{selected_coll}_export.csv", "text/csv",
                use_container_width=True,
            )
