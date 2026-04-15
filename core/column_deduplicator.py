"""
Column-Based Duplicate Detection
Identifies duplicate, similar, and derived columns in a dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def detect_column_duplicates(
    df: pd.DataFrame,
    similarity_threshold: float = 0.85,
    check_exact: bool = True,
    check_fuzzy: bool = True,
    check_subset: bool = True,
    check_derived: bool = True
) -> pd.DataFrame:
    """
    Detect duplicate and similar columns

    Args:
        df: Input dataframe
        similarity_threshold: Threshold for considering columns similar (0.0-1.0)
        check_exact: Check for exact duplicate columns
        check_fuzzy: Check for fuzzy similar columns
        check_subset: Check if one column is subset of another
        check_derived: Check for derived columns (transformations)

    Returns:
        DataFrame with column pairs and their relationship
        Columns: columnA, columnB, relationship, similarity, recommendation
    """
    results = []
    columns = df.columns.tolist()

    for i, col_a in enumerate(columns):
        for j, col_b in enumerate(columns[i + 1:], start=i + 1):
            # Skip if both columns are completely empty
            if df[col_a].isna().all() and df[col_b].isna().all():
                continue

            # Check exact duplicates
            if check_exact:
                if _are_columns_exact_duplicates(df, col_a, col_b):
                    results.append({
                        'columnA': col_a,
                        'columnB': col_b,
                        'relationship': 'exact_duplicate',
                        'similarity': 1.00,
                        'description': 'Columns are 100% identical',
                        'recommendation': f'Keep {col_a}, remove {col_b}'
                    })
                    continue

            # Check derived columns (transformations)
            if check_derived:
                derived_rel = _check_derived_relationship(df, col_a, col_b)
                if derived_rel:
                    results.append(derived_rel)
                    continue

            # Check subset relationship
            if check_subset:
                subset_rel = _check_subset_relationship(df, col_a, col_b, similarity_threshold)
                if subset_rel:
                    results.append(subset_rel)
                    continue

            # Check fuzzy similarity
            if check_fuzzy:
                similarity = _compute_column_similarity(df, col_a, col_b)
                if similarity >= similarity_threshold:
                    results.append({
                        'columnA': col_a,
                        'columnB': col_b,
                        'relationship': 'high_similarity',
                        'similarity': round(similarity, 3),
                        'description': f'{int(similarity*100)}% similar values',
                        'recommendation': f'Consider merging or keeping one column'
                    })

    return pd.DataFrame(results)


def _are_columns_exact_duplicates(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    """Check if two columns are exactly identical"""
    return df[col_a].equals(df[col_b])


def _check_derived_relationship(
    df: pd.DataFrame,
    col_a: str,
    col_b: str
) -> Optional[Dict[str, Any]]:
    """
    Check if one column is a transformation of another

    Checks for:
    - Uppercase/lowercase transformation
    - Whitespace trimming
    - String concatenation/splitting
    """
    # Convert to strings for comparison
    series_a = df[col_a].astype(str).str.strip()
    series_b = df[col_b].astype(str).str.strip()

    # Check case transformation
    if series_a.str.lower().equals(series_b.str.lower()):
        if not series_a.equals(series_b):
            return {
                'columnA': col_a,
                'columnB': col_b,
                'relationship': 'case_transformed',
                'similarity': 0.99,
                'description': 'One column is case-transformed version of the other',
                'recommendation': f'Keep {col_a}, remove {col_b}'
            }

    # Check whitespace normalization
    a_normalized = series_a.str.replace(r'\s+', ' ', regex=True)
    b_normalized = series_b.str.replace(r'\s+', ' ', regex=True)

    if a_normalized.equals(b_normalized) and not series_a.equals(series_b):
        return {
            'columnA': col_a,
            'columnB': col_b,
            'relationship': 'whitespace_normalized',
            'similarity': 0.98,
            'description': 'Columns differ only in whitespace',
            'recommendation': f'Keep {col_a}, remove {col_b}'
        }

    # Check if one column contains the other (string containment)
    # This might indicate concatenation or extraction
    non_null_mask = series_a.notna() & series_b.notna()
    if non_null_mask.sum() > 0:
        a_contains_b = series_a[non_null_mask].str.contains(
            series_b[non_null_mask].str.replace(r'[^\w\s]', '', regex=True),
            regex=False,
            na=False
        ).mean()

        b_contains_a = series_b[non_null_mask].str.contains(
            series_a[non_null_mask].str.replace(r'[^\w\s]', '', regex=True),
            regex=False,
            na=False
        ).mean()

        if a_contains_b > 0.9:
            return {
                'columnA': col_a,
                'columnB': col_b,
                'relationship': 'contains',
                'similarity': round(a_contains_b, 3),
                'description': f'{col_a} contains values from {col_b}',
                'recommendation': f'{col_b} might be extracted from {col_a}'
            }

        if b_contains_a > 0.9:
            return {
                'columnA': col_a,
                'columnB': col_b,
                'relationship': 'contains',
                'similarity': round(b_contains_a, 3),
                'description': f'{col_b} contains values from {col_a}',
                'recommendation': f'{col_a} might be extracted from {col_b}'
            }

    return None


def _check_subset_relationship(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    threshold: float = 0.90
) -> Optional[Dict[str, Any]]:
    """
    Check if one column's unique values are a subset of another

    Useful for detecting:
    - Filtered/sampled columns
    - Lookup tables
    """
    # Get unique non-null values
    unique_a = set(df[col_a].dropna().unique())
    unique_b = set(df[col_b].dropna().unique())

    if len(unique_a) == 0 or len(unique_b) == 0:
        return None

    # Check if A is subset of B
    if len(unique_a) <= len(unique_b):
        overlap = len(unique_a.intersection(unique_b))
        ratio = overlap / len(unique_a)

        if ratio >= threshold:
            return {
                'columnA': col_a,
                'columnB': col_b,
                'relationship': 'subset',
                'similarity': round(ratio, 3),
                'description': f'{int(ratio*100)}% of {col_a} values exist in {col_b}',
                'recommendation': f'{col_a} might be subset of {col_b}'
            }

    # Check if B is subset of A
    if len(unique_b) < len(unique_a):
        overlap = len(unique_b.intersection(unique_a))
        ratio = overlap / len(unique_b)

        if ratio >= threshold:
            return {
                'columnA': col_a,
                'columnB': col_b,
                'relationship': 'subset',
                'similarity': round(ratio, 3),
                'description': f'{int(ratio*100)}% of {col_b} values exist in {col_a}',
                'recommendation': f'{col_b} might be subset of {col_a}'
            }

    return None


def _compute_column_similarity(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """
    Compute fuzzy similarity between two columns

    Uses Jaccard similarity for categorical and TF-IDF for text
    """
    # Remove null values
    mask = df[col_a].notna() & df[col_b].notna()
    if mask.sum() == 0:
        return 0.0

    series_a = df.loc[mask, col_a]
    series_b = df.loc[mask, col_b]

    # For numeric columns, compute correlation
    if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b):
        try:
            corr = abs(series_a.corr(series_b))
            return corr if not pd.isna(corr) else 0.0
        except:
            pass

    # For categorical/text columns, use Jaccard similarity
    unique_a = set(series_a.unique())
    unique_b = set(series_b.unique())

    intersection = len(unique_a.intersection(unique_b))
    union = len(unique_a.union(unique_b))

    if union == 0:
        return 0.0

    jaccard = intersection / union

    # Also check row-by-row equality
    row_equality = (series_a.astype(str) == series_b.astype(str)).mean()

    # Return weighted average
    return 0.5 * jaccard + 0.5 * row_equality


def get_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed statistics for each column

    Returns:
        DataFrame with column statistics
    """
    stats = []

    for col in df.columns:
        series = df[col]

        stat = {
            'column': col,
            'dtype': str(series.dtype),
            'non_null_count': series.notna().sum(),
            'null_count': series.isna().sum(),
            'null_percentage': round(100 * series.isna().sum() / len(df), 2),
            'unique_count': series.nunique(),
            'uniqueness_ratio': round(series.nunique() / series.notna().sum(), 3) if series.notna().sum() > 0 else 0,
        }

        # Add type-specific stats
        if pd.api.types.is_numeric_dtype(series):
            stat['min'] = series.min()
            stat['max'] = series.max()
            stat['mean'] = round(series.mean(), 2) if pd.notna(series.mean()) else None
            stat['median'] = series.median()

        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                stat['min_length'] = non_null.str.len().min()
                stat['max_length'] = non_null.str.len().max()
                stat['avg_length'] = round(non_null.str.len().mean(), 1)
                stat['mode'] = series.mode()[0] if len(series.mode()) > 0 else None

        stats.append(stat)

    return pd.DataFrame(stats)


def recommend_column_removals(
    duplicate_results: pd.DataFrame,
    column_stats: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Generate recommendations for which columns to remove

    Args:
        duplicate_results: Output from detect_column_duplicates()
        column_stats: Output from get_column_statistics()

    Returns:
        List of recommendations with reasoning
    """
    recommendations = []

    # Process exact duplicates
    exact_dupes = duplicate_results[duplicate_results['relationship'] == 'exact_duplicate']

    for _, row in exact_dupes.iterrows():
        col_a = row['columnA']
        col_b = row['columnB']

        # Prefer column with better name (shorter, more descriptive)
        if len(col_a) <= len(col_b):
            keep, remove = col_a, col_b
        else:
            keep, remove = col_b, col_a

        recommendations.append({
            'action': 'remove',
            'column': remove,
            'reason': f'Exact duplicate of {keep}',
            'priority': 'high',
            'relationship': 'exact_duplicate'
        })

    # Process derived columns
    derived = duplicate_results[duplicate_results['relationship'].isin([
        'case_transformed', 'whitespace_normalized'
    ])]

    for _, row in derived.iterrows():
        col_b = row['columnB']
        col_a = row['columnA']

        recommendations.append({
            'action': 'remove',
            'column': col_b,
            'reason': f'Transformation of {col_a}',
            'priority': 'high',
            'relationship': row['relationship']
        })

    # Process high similarity columns
    similar = duplicate_results[
        (duplicate_results['relationship'] == 'high_similarity') &
        (duplicate_results['similarity'] >= 0.95)
    ]

    for _, row in similar.iterrows():
        col_a = row['columnA']
        col_b = row['columnB']

        # Check which column has more complete data
        stats_a = column_stats[column_stats['column'] == col_a].iloc[0]
        stats_b = column_stats[column_stats['column'] == col_b].iloc[0]

        if stats_a['non_null_count'] >= stats_b['non_null_count']:
            keep, remove = col_a, col_b
        else:
            keep, remove = col_b, col_a

        recommendations.append({
            'action': 'consider_removing',
            'column': remove,
            'reason': f'{int(row["similarity"]*100)}% similar to {keep}, has less data',
            'priority': 'medium',
            'relationship': 'high_similarity'
        })

    return recommendations
