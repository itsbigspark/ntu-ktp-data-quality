"""
Typo detection using edit distance and corpus-based matching.

This module detects subtle typos by comparing values against a corpus
of known-good values using Levenshtein distance.
"""

from typing import List, Tuple, Optional, Set
from collections import Counter
import pandas as pd


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    This is the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into the other.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_closest_match(value: str, corpus: List[str], max_distance: int = 2) -> Tuple[Optional[str], int]:
    """
    Find the closest match in corpus using edit distance.

    Args:
        value: Value to check
        corpus: List of known-good values
        max_distance: Maximum edit distance to consider

    Returns:
        (closest_match, distance) or (None, -1) if no close match
    """
    if not value or not corpus:
        return None, -1

    value_lower = str(value).lower().strip()
    best_match = None
    best_distance = float('inf')

    for candidate in corpus:
        candidate_lower = str(candidate).lower().strip()

        # Exact match - no typo
        if value_lower == candidate_lower:
            return None, 0

        distance = levenshtein_distance(value_lower, candidate_lower)

        if distance < best_distance:
            best_distance = distance
            best_match = candidate

    if best_distance <= max_distance:
        return best_match, int(best_distance)

    return None, -1


def build_corpus(df: pd.DataFrame, column: str, min_frequency: int = 2) -> List[str]:
    """
    Build a corpus of likely-correct values from a column.

    Uses frequency analysis - values that appear multiple times are
    assumed to be correct, while rare values might be typos.

    Args:
        df: DataFrame
        column: Column name
        min_frequency: Minimum frequency to include in corpus

    Returns:
        List of likely-correct values
    """
    if column not in df.columns:
        return []

    # Remove nulls and convert to string
    values = df[column].dropna().astype(str).str.strip()

    # Filter out obvious placeholders
    values = values[~values.str.lower().isin(['', 'null', 'none', 'na', 'n/a', 'nan'])]

    if len(values) == 0:
        return []

    # Count frequencies
    freq = Counter(values)

    # Include values that appear at least min_frequency times
    corpus = [val for val, count in freq.items() if count >= min_frequency]

    return corpus


def detect_typos(
    df: pd.DataFrame,
    column: str,
    corpus: Optional[List[str]] = None,
    max_distance: int = 2,
    min_corpus_frequency: int = 3
) -> pd.DataFrame:
    """
    Detect likely typos in a column.

    Args:
        df: DataFrame to check
        column: Column name
        corpus: Known-good values (if None, build from data)
        max_distance: Maximum edit distance to flag as typo
        min_corpus_frequency: Min frequency for corpus building

    Returns:
        DataFrame with columns: row_id, column, value, issue, detail, severity,
                                expected (closest match), rule
    """
    if column not in df.columns:
        return pd.DataFrame(columns=["row_id", "column", "value", "issue", "detail", "severity", "expected", "rule"])

    # Build corpus if not provided
    if corpus is None:
        corpus = build_corpus(df, column, min_frequency=min_corpus_frequency)

    if not corpus:
        # Not enough data to build corpus
        return pd.DataFrame(columns=["row_id", "column", "value", "issue", "detail", "severity", "expected", "rule"])

    # Convert corpus to set for quick lookup
    corpus_lower = {str(v).lower().strip() for v in corpus}

    # Check each value
    typos = []
    for i, value in enumerate(df[column]):
        # Skip nulls
        if pd.isna(value):
            continue

        value_str = str(value).strip()
        value_lower = value_str.lower()

        # Skip if exact match in corpus (case-insensitive)
        if value_lower in corpus_lower:
            continue

        # Check if it's a typo
        closest, distance = find_closest_match(value_str, corpus, max_distance)

        if closest is not None and distance > 0:
            typos.append({
                'row_id': i,
                'column': column,
                'value': value_str,
                'issue': 'likely_typo',
                'detail': f"possible typo (edit distance {distance} from '{closest}')",
                'severity': 'medium' if distance == 1 else 'low',
                'expected': {'closest_match': closest, 'edit_distance': distance},
                'rule': 'typo_detection'
            })

    if not typos:
        return pd.DataFrame(columns=["row_id", "column", "value", "issue", "detail", "severity", "expected", "rule"])

    return pd.DataFrame(typos)


def detect_typos_all_columns(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
    max_distance: int = 2,
    min_corpus_frequency: int = 3
) -> pd.DataFrame:
    """
    Detect typos across multiple text columns.

    Args:
        df: DataFrame to check
        text_columns: Columns to check (if None, auto-detect text columns)
        max_distance: Maximum edit distance for typos
        min_corpus_frequency: Min frequency for corpus building

    Returns:
        DataFrame with all detected typos
    """
    if text_columns is None:
        # Auto-detect text columns (string/object dtype, not too many unique values)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                unique_ratio = df[col].nunique() / len(df)
                # Only check columns with 2-50% unique values (categorical-ish)
                if 0.02 <= unique_ratio <= 0.50:
                    text_columns.append(col)

    all_typos = []
    for col in text_columns:
        typos = detect_typos(df, col, max_distance=max_distance, min_corpus_frequency=min_corpus_frequency)
        if not typos.empty:
            all_typos.append(typos)

    if not all_typos:
        return pd.DataFrame(columns=["row_id", "column", "value", "issue", "detail", "severity", "expected", "rule"])

    return pd.concat(all_typos, ignore_index=True)
