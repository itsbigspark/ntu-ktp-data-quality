# core/dedup_clustering.py
"""
Duplicate clustering utilities for grouping related duplicate pairs.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Dict, Set, Any
import networkx as nx


def cluster_duplicates(pairs_df: pd.DataFrame, id_col_a: str = 'id_A', id_col_b: str = 'id_B') -> Dict[int, List[int]]:
    """
    Group duplicate pairs into clusters using graph connected components.

    Args:
        pairs_df: DataFrame with duplicate pairs (id_A, id_B columns)
        id_col_a: Name of first ID column
        id_col_b: Name of second ID column

    Returns:
        Dict mapping cluster_id to list of row IDs in that cluster
        {
            0: [5, 12, 34],  # Cluster 0 has rows 5, 12, 34
            1: [67, 89],     # Cluster 1 has rows 67, 89
            ...
        }
    """
    # Build a graph where nodes are row IDs and edges are duplicate relationships
    G = nx.Graph()

    for _, row in pairs_df.iterrows():
        id_a = int(row[id_col_a])
        id_b = int(row[id_col_b])
        G.add_edge(id_a, id_b)

    # Find connected components (clusters)
    clusters = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        clusters[cluster_id] = sorted(list(component))

    return clusters


def create_cluster_summary(
    clusters: Dict[int, List[int]],
    df_original: pd.DataFrame,
    pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a summary DataFrame for each cluster.

    Args:
        clusters: Dict from cluster_duplicates()
        df_original: Original dataset
        pairs_df: Pairs DataFrame with similarity scores

    Returns:
        DataFrame with columns:
        - cluster_id
        - size (number of records in cluster)
        - row_ids (list of row IDs)
        - avg_similarity (average similarity within cluster)
        - preview (preview of first record)
    """
    summaries = []

    for cluster_id, row_ids in clusters.items():
        # Calculate average similarity for this cluster
        # Note: pairs_df has 'row_i', 'row_j', 'score' columns from matcher.py
        cluster_pairs = pairs_df[
            (pairs_df['row_i'].isin(row_ids)) &
            (pairs_df['row_j'].isin(row_ids))
        ]

        avg_sim = cluster_pairs['score'].mean() if len(cluster_pairs) > 0 else 0.0

        # Get preview of first record
        first_row_id = row_ids[0]
        if first_row_id in df_original.index:
            preview_row = df_original.loc[first_row_id]
            # Create a short preview (first 3 non-null columns)
            preview_parts = []
            for col in df_original.columns[:5]:
                val = preview_row[col]
                if pd.notna(val):
                    preview_parts.append(f"{col}: {str(val)[:30]}")
                if len(preview_parts) >= 3:
                    break
            preview = " | ".join(preview_parts)
        else:
            preview = "N/A"

        summaries.append({
            'cluster_id': cluster_id,
            'size': len(row_ids),
            'row_ids': row_ids,
            'avg_similarity': avg_sim,
            'preview': preview
        })

    # Create DataFrame, handle empty case
    if summaries:
        summary_df = pd.DataFrame(summaries)
        # Sort by cluster size (largest first)
        summary_df = summary_df.sort_values('size', ascending=False).reset_index(drop=True)
    else:
        # Return empty DataFrame with correct columns
        summary_df = pd.DataFrame(columns=['cluster_id', 'size', 'row_ids', 'avg_similarity', 'preview'])

    return summary_df


def get_cluster_details(
    cluster_id: int,
    clusters: Dict[int, List[int]],
    df_original: pd.DataFrame,
    pairs_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Get detailed information for a specific cluster.

    Returns:
        {
            'cluster_id': int,
            'row_ids': List[int],
            'records': DataFrame (subset of original data),
            'pairs': DataFrame (all pairs within this cluster),
            'suggested_master': int (row_id of suggested master record)
        }
    """
    row_ids = clusters.get(cluster_id, [])

    # Get all records in this cluster
    records = df_original.loc[df_original.index.isin(row_ids)].copy()
    records['row_id'] = records.index

    # Get all pairs within this cluster
    # Note: pairs_df has 'row_i', 'row_j', 'score' columns from matcher.py
    cluster_pairs = pairs_df[
        (pairs_df['row_i'].isin(row_ids)) &
        (pairs_df['row_j'].isin(row_ids))
    ].copy()

    # Rename columns for display consistency
    cluster_pairs = cluster_pairs.rename(columns={
        'row_i': 'id_A',
        'row_j': 'id_B',
        'score': 'sim_avg'
    })

    # Suggest master record (the one with fewest nulls)
    if len(records) > 0:
        records['null_count'] = records.isnull().sum(axis=1)
        suggested_master = records.nsmallest(1, 'null_count').index[0]
    else:
        suggested_master = row_ids[0] if row_ids else None

    return {
        'cluster_id': cluster_id,
        'row_ids': row_ids,
        'records': records,
        'pairs': cluster_pairs,
        'suggested_master': suggested_master
    }


def merge_cluster_records(
    cluster_records: pd.DataFrame,
    master_row_id: int,
    merge_strategy: str = 'keep_master'
) -> pd.Series:
    """
    Merge multiple records in a cluster into a single master record.

    Args:
        cluster_records: DataFrame with all records in cluster
        master_row_id: Row ID of the master record
        merge_strategy:
            - 'keep_master': Keep all values from master
            - 'fill_nulls': Fill master's nulls from other records
            - 'most_common': Use most common value for each field

    Returns:
        Series representing the merged master record
    """
    if master_row_id not in cluster_records.index:
        # Fallback to first record
        master_row_id = cluster_records.index[0]

    master_record = cluster_records.loc[master_row_id].copy()

    if merge_strategy == 'keep_master':
        # Simply return the master as-is
        return master_record

    elif merge_strategy == 'fill_nulls':
        # Fill master's null values from other records
        for col in cluster_records.columns:
            if pd.isna(master_record[col]):
                # Try to find non-null value from other records
                for idx, row in cluster_records.iterrows():
                    if idx != master_row_id and pd.notna(row[col]):
                        master_record[col] = row[col]
                        break
        return master_record

    elif merge_strategy == 'most_common':
        # For each column, use most common value
        merged = master_record.copy()
        for col in cluster_records.columns:
            if col == 'row_id' or col == 'null_count':
                continue
            # Get most common non-null value
            values = cluster_records[col].dropna()
            if len(values) > 0:
                most_common = values.mode()
                if len(most_common) > 0:
                    merged[col] = most_common.iloc[0]
        return merged

    return master_record


def apply_deduplication(
    df_original: pd.DataFrame,
    clusters: Dict[int, List[int]],
    master_selections: Dict[int, int],
    merge_strategy: str = 'keep_master'
) -> pd.DataFrame:
    """
    Create deduplicated dataset by merging clusters.

    Args:
        df_original: Original dataset
        clusters: Cluster mapping
        master_selections: Dict mapping cluster_id to selected master row_id
        merge_strategy: How to merge records

    Returns:
        Deduplicated DataFrame
    """
    # Start with all rows
    deduplicated_indices = set(df_original.index)
    merged_records = []

    for cluster_id, row_ids in clusters.items():
        master_id = master_selections.get(cluster_id)

        if master_id is None:
            # No master selected, keep all records
            continue

        # Get cluster records
        cluster_records = df_original.loc[df_original.index.isin(row_ids)]

        # Merge into master
        merged_record = merge_cluster_records(cluster_records, master_id, merge_strategy)

        # Remove all cluster records from deduplicated set
        for rid in row_ids:
            deduplicated_indices.discard(rid)

        # Add back only the merged master
        merged_records.append(merged_record)

    # Build final dataset
    kept_records = df_original.loc[list(deduplicated_indices)]

    if merged_records:
        merged_df = pd.DataFrame(merged_records)
        final_df = pd.concat([kept_records, merged_df], ignore_index=True)
    else:
        final_df = kept_records.reset_index(drop=True)

    return final_df
