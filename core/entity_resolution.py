# core/entity_resolution.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def find_entity_matches(
    datasets: Dict[str, pd.DataFrame],
    threshold: float = 0.80,
    use_blocking: bool = False,
    blocking_column: Optional[str] = None,
    column_weights: Optional[Dict[str, float]] = None,
    use_phonetic: bool = False,
    phonetic_columns: Optional[List[str]] = None,
    use_semantic: bool = False,
    semantic_model: Optional[str] = None,
    use_chromadb: bool = False,
    chromadb_model: Optional[str] = None
) -> pd.DataFrame:
    """
    Find matching entities across multiple datasets with advanced options.

    Args:
        datasets: Dict of {filename: dataframe}
        threshold: Similarity threshold (0.0-1.0)
        use_blocking: Whether to use blocking for speed
        blocking_column: Column to use for blocking (exact match required)
        column_weights: Dict of {column_name: weight} for weighted matching
        use_phonetic: Whether to use phonetic matching for names
        phonetic_columns: List of column names to apply phonetic matching
        use_semantic: Whether to use semantic (sentence transformer) matching
        semantic_model: Name of sentence transformer model to use
        use_chromadb: Whether to use ChromaDB vector index for ultra-fast matching
        chromadb_model: Name of sentence transformer model for ChromaDB embeddings

    Returns:
        DataFrame with columns: A, A_row, B, B_row, score, match_details
    """

    import itertools

    pairs = []

    # ChromaDB mode - semantic cross-dataset matching with embedding cache
    if use_chromadb and chromadb_model:
        from core.semantic_matcher import encode_texts
        from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
        import numpy as np

        # Process all pairs of datasets
        for (nameA, dfA), (nameB, dfB) in itertools.combinations(datasets.items(), 2):
            # Get common columns
            common_cols = list(set(dfA.columns).intersection(set(dfB.columns)))

            if not common_cols:
                continue

            # Prepare texts
            A_text = dfA[common_cols].astype(str).agg(" | ".join, axis=1).tolist()
            B_text = dfB[common_cols].astype(str).agg(" | ".join, axis=1).tolist()

            # Encode — results are cached in-process by encode_texts()
            # so a second run on the same data skips the model entirely
            A_embeddings = encode_texts(A_text, chromadb_model)
            B_embeddings = encode_texts(B_text, chromadb_model)

            # Vectorised similarity matrix (exact, no ChromaDB round-trip needed)
            sim_matrix = _cosine_sim(A_embeddings, B_embeddings)

            for i in range(len(A_text)):
                for j in range(len(B_text)):
                    sim = float(sim_matrix[i, j])
                    if sim >= threshold:
                        pairs.append({
                            'A': nameA,
                            'A_row': i,
                            'B': nameB,
                            'B_row': j,
                            'score': sim,
                            'match_details': f"Semantic vector similarity: {sim:.3f}"
                        })

    # Standard processing (existing code)
    else:
        # Process all pairs of datasets
        for (nameA, dfA), (nameB, dfB) in itertools.combinations(datasets.items(), 2):

            # Apply blocking if requested
            if use_blocking and blocking_column:
                block_pairs = _find_pairs_with_blocking(
                    dfA, dfB, nameA, nameB, blocking_column, threshold,
                    column_weights, use_phonetic, phonetic_columns,
                    use_semantic, semantic_model
                )
                pairs.extend(block_pairs)
            else:
                # Standard all-pairs comparison
                block_pairs = _find_pairs_standard(
                    dfA, dfB, nameA, nameB, threshold,
                    column_weights, use_phonetic, phonetic_columns,
                    use_semantic, semantic_model
                )
                pairs.extend(block_pairs)

    # Convert to DataFrame
    if pairs:
        er_df = pd.DataFrame(pairs)
        # Sort by score descending
        er_df = er_df.sort_values('score', ascending=False).reset_index(drop=True)
    else:
        er_df = pd.DataFrame(columns=['A', 'A_row', 'B', 'B_row', 'score', 'match_details'])

    return er_df


def _find_pairs_standard(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    nameA: str,
    nameB: str,
    threshold: float,
    column_weights: Optional[Dict[str, float]] = None,
    use_phonetic: bool = False,
    phonetic_columns: Optional[List[str]] = None,
    use_semantic: bool = False,
    semantic_model: Optional[str] = None
) -> List[Dict]:
    """Standard all-pairs comparison"""

    pairs = []

    # Semantic matching (if enabled)
    if use_semantic and semantic_model:
        from core.semantic_matcher import compute_semantic_similarity

        # Get common columns
        common_cols = list(set(dfA.columns).intersection(set(dfB.columns)))

        if common_cols:
            A_text = dfA[common_cols].astype(str).agg(" ".join, axis=1).tolist()
            B_text = dfB[common_cols].astype(str).agg(" ".join, axis=1).tolist()

            scores_matrix = compute_semantic_similarity(
                A_text,
                B_text,
                model_name=semantic_model,
                batch_size=32
            )
        else:
            scores_matrix = np.zeros((len(dfA), len(dfB)))

    # Column-weighted matching
    elif column_weights:
        scores_matrix = _compute_weighted_similarity(
            dfA, dfB, column_weights, use_phonetic, phonetic_columns
        )
    else:
        # Simple TF-IDF on all columns
        A_text = dfA.astype(str).agg(" ".join, axis=1).str.lower()
        B_text = dfB.astype(str).agg(" ".join, axis=1).str.lower()

        # Apply phonetic to specified columns if requested
        if use_phonetic and phonetic_columns:
            A_text = _apply_phonetic_preprocessing(dfA, phonetic_columns) + " " + A_text
            B_text = _apply_phonetic_preprocessing(dfB, phonetic_columns) + " " + B_text

        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=10000)
        try:
            X = vec.fit_transform(list(A_text) + list(B_text))
            L, R = X[:len(A_text)], X[len(A_text):]
            scores_matrix = cosine_similarity(L, R)
        except Exception:
            # Fallback to empty scores
            scores_matrix = np.zeros((len(dfA), len(dfB)))

    # Find matches above threshold
    for i in range(len(dfA)):
        j = int(scores_matrix[i].argmax())
        score = float(scores_matrix[i, j])

        if score >= threshold:
            # Get match details
            details = _get_match_details(dfA.iloc[i], dfB.iloc[j], score)

            pairs.append({
                "A": nameA,
                "A_row": i,
                "B": nameB,
                "B_row": j,
                "score": round(score, 3),
                "match_details": details
            })

    return pairs


def _find_pairs_with_blocking(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    nameA: str,
    nameB: str,
    blocking_column: str,
    threshold: float,
    column_weights: Optional[Dict[str, float]] = None,
    use_phonetic: bool = False,
    phonetic_columns: Optional[List[str]] = None,
    use_semantic: bool = False,
    semantic_model: Optional[str] = None
) -> List[Dict]:
    """Blocking-based comparison for speed"""

    pairs = []

    # Check if blocking column exists in both datasets
    if blocking_column not in dfA.columns or blocking_column not in dfB.columns:
        # Fall back to standard matching
        return _find_pairs_standard(dfA, dfB, nameA, nameB, threshold, column_weights, use_phonetic, phonetic_columns, use_semantic, semantic_model)

    # Create blocks based on exact match of blocking column
    blocksA = dfA.groupby(blocking_column).groups
    blocksB = dfB.groupby(blocking_column).groups

    # Find common block keys
    common_keys = set(blocksA.keys()).intersection(set(blocksB.keys()))

    # Process each block
    for block_key in common_keys:
        idxA = blocksA[block_key]
        idxB = blocksB[block_key]

        subA = dfA.loc[idxA]
        subB = dfB.loc[idxB]

        # Compare within block
        if len(subA) > 0 and len(subB) > 0:
            # Column-weighted matching
            if column_weights:
                scores_matrix = _compute_weighted_similarity(
                    subA, subB, column_weights, use_phonetic, phonetic_columns
                )
            else:
                # Simple TF-IDF
                A_text = subA.astype(str).agg(" ".join, axis=1).str.lower()
                B_text = subB.astype(str).agg(" ".join, axis=1).str.lower()

                if use_phonetic and phonetic_columns:
                    A_text = _apply_phonetic_preprocessing(subA, phonetic_columns) + " " + A_text
                    B_text = _apply_phonetic_preprocessing(subB, phonetic_columns) + " " + B_text

                vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000)
                try:
                    X = vec.fit_transform(list(A_text) + list(B_text))
                    L, R = X[:len(A_text)], X[len(A_text):]
                    scores_matrix = cosine_similarity(L, R)
                except Exception:
                    scores_matrix = np.zeros((len(subA), len(subB)))

            # Find matches in block
            for local_i, global_i in enumerate(idxA):
                local_j = int(scores_matrix[local_i].argmax())
                global_j = idxB[local_j]
                score = float(scores_matrix[local_i, local_j])

                if score >= threshold:
                    details = _get_match_details(dfA.loc[global_i], dfB.loc[global_j], score)

                    pairs.append({
                        "A": nameA,
                        "A_row": global_i,
                        "B": nameB,
                        "B_row": global_j,
                        "score": round(score, 3),
                        "match_details": details,
                        "block_key": str(block_key)
                    })

    return pairs


def _compute_weighted_similarity(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    column_weights: Dict[str, float],
    use_phonetic: bool = False,
    phonetic_columns: Optional[List[str]] = None
) -> np.ndarray:
    """Compute weighted similarity across specified columns"""

    # Get common columns
    common_cols = set(dfA.columns).intersection(set(dfB.columns))
    weighted_cols = {col: weight for col, weight in column_weights.items() if col in common_cols}

    if not weighted_cols:
        # No valid columns, return zeros
        return np.zeros((len(dfA), len(dfB)))

    # Normalize weights
    total_weight = sum(weighted_cols.values())
    normalized_weights = {col: w/total_weight for col, w in weighted_cols.items()}

    # Compute similarity for each column and combine
    combined_scores = np.zeros((len(dfA), len(dfB)))

    for col, weight in normalized_weights.items():
        # Extract column data
        A_col = dfA[col].fillna("").astype(str).str.lower()
        B_col = dfB[col].fillna("").astype(str).str.lower()

        # Apply phonetic if applicable
        if use_phonetic and phonetic_columns and col in phonetic_columns:
            A_col = A_col.apply(_soundex)
            B_col = B_col.apply(_soundex)

        # Compute similarity for this column
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=1000)
        try:
            X = vec.fit_transform(list(A_col) + list(B_col))
            L, R = X[:len(A_col)], X[len(A_col):]
            col_scores = cosine_similarity(L, R)

            # Add weighted scores
            combined_scores += col_scores * weight
        except Exception:
            # Skip this column if vectorization fails
            continue

    return combined_scores


def _apply_phonetic_preprocessing(df: pd.DataFrame, phonetic_columns: List[str]) -> pd.Series:
    """Apply phonetic encoding to specified columns"""

    phonetic_text = []

    for _, row in df.iterrows():
        phonetic_parts = []
        for col in phonetic_columns:
            if col in df.columns:
                val = str(row[col]) if pd.notna(row[col]) else ""
                if val:
                    phonetic_parts.append(_soundex(val))
        phonetic_text.append(" ".join(phonetic_parts))

    return pd.Series(phonetic_text)


def _soundex(name: str) -> str:
    """
    Soundex phonetic algorithm for name matching.
    Converts names to phonetic code so similar-sounding names match.

    Example: "Smith" and "Smyth" both become "S530"
    """

    if not name:
        return ""

    name = name.upper().strip()
    if not name:
        return ""

    # Soundex mapping
    soundex_mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Keep first letter
    first_letter = name[0]

    # Convert rest to codes
    code = []
    prev_code = soundex_mapping.get(first_letter, '0')

    for char in name[1:]:
        if char in soundex_mapping:
            current_code = soundex_mapping[char]
            # Only add if different from previous
            if current_code != prev_code:
                code.append(current_code)
                prev_code = current_code
        else:
            # Non-mapped characters reset the previous code
            prev_code = '0'

    # Pad or truncate to 3 digits
    code_str = "".join(code)[:3].ljust(3, '0')

    return first_letter + code_str


def _get_match_details(rowA: pd.Series, rowB: pd.Series, score: float) -> str:
    """Generate match details string showing what matched"""

    details = []

    # Find columns that exist in both
    common_cols = set(rowA.index).intersection(set(rowB.index))

    # Check exact matches
    exact_matches = []
    fuzzy_matches = []

    for col in common_cols:
        valA = str(rowA[col]) if pd.notna(rowA[col]) else ""
        valB = str(rowB[col]) if pd.notna(rowB[col]) else ""

        if valA and valB:
            if valA.lower() == valB.lower():
                exact_matches.append(col)
            elif valA.lower() in valB.lower() or valB.lower() in valA.lower():
                fuzzy_matches.append(col)

    if exact_matches:
        details.append(f"Exact: {', '.join(exact_matches[:3])}")
    if fuzzy_matches:
        details.append(f"Fuzzy: {', '.join(fuzzy_matches[:3])}")

    details.append(f"Score: {score:.3f}")

    return " | ".join(details)


def analyze_clusters(er_pairs: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze entity resolution clusters to find connected components.

    Returns statistics about clusters, golden record candidates, etc.
    """

    import networkx as nx

    # Build graph from pairs
    G = nx.Graph()

    for _, row in er_pairs.iterrows():
        nodeA = f"{row['A']}:{row['A_row']}"
        nodeB = f"{row['B']}:{row['B_row']}"
        G.add_edge(nodeA, nodeB, weight=row['score'])

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Analyze clusters
    cluster_stats = {
        'total_clusters': len(clusters),
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'cluster_sizes': [len(c) for c in clusters],
        'avg_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
        'max_cluster_size': max([len(c) for c in clusters]) if clusters else 0,
        'singleton_clusters': sum(1 for c in clusters if len(c) == 1),
        'multi_node_clusters': sum(1 for c in clusters if len(c) > 1),
    }

    # Find potential golden records (highest centrality in each cluster)
    golden_candidates = []

    for cluster in clusters:
        if len(cluster) > 1:
            # Get subgraph
            subgraph = G.subgraph(cluster)

            # Calculate degree centrality (most connections)
            centrality = nx.degree_centrality(subgraph)

            # Get node with highest centrality
            best_node = max(centrality, key=centrality.get)
            avg_score = np.mean([d['weight'] for u, v, d in subgraph.edges(data=True) if u == best_node or v == best_node])

            golden_candidates.append({
                'cluster_size': len(cluster),
                'golden_record': best_node,
                'centrality': centrality[best_node],
                'avg_similarity': round(avg_score, 3),
                'cluster_nodes': list(cluster)
            })

    cluster_stats['golden_candidates'] = golden_candidates

    # Cluster size distribution
    size_dist = {}
    for size in cluster_stats['cluster_sizes']:
        size_dist[size] = size_dist.get(size, 0) + 1
    cluster_stats['size_distribution'] = dict(sorted(size_dist.items()))

    return cluster_stats


def get_record_preview(datasets: Dict[str, pd.DataFrame], node_id: str) -> Dict[str, Any]:
    """
    Get full record details for a node ID (dataset:row format)

    Args:
        datasets: Dict of {filename: dataframe}
        node_id: String like "customers.csv:15"

    Returns:
        Dict with record data
    """

    try:
        dataset_name, row_idx = node_id.rsplit(":", 1)
        row_idx = int(row_idx)

        if dataset_name in datasets:
            df = datasets[dataset_name]
            if row_idx < len(df):
                record = df.iloc[row_idx].to_dict()
                return {
                    'dataset': dataset_name,
                    'row': row_idx,
                    'data': record
                }
    except Exception:
        pass

    return {'error': 'Record not found'}
