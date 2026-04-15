# core/corpus_validation.py
"""
Comprehensive corpus-based validation and standardization.
Integrates with CorpusManager to validate and fix data using loaded corpora.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def validate_with_corpus(
    df: pd.DataFrame,
    column: str,
    corpus_name: str,
    corpus_manager,
    corpus_type: str,
    confidence_threshold: float = 0.80
) -> pd.DataFrame:
    """
    Validate a single column against a corpus.

    Args:
        df: Dataframe to validate
        column: Column name to validate
        corpus_name: Name of corpus in Redis
        corpus_manager: CorpusManager instance
        corpus_type: 'alias', 'lookup', or 'validation'
        confidence_threshold: Minimum confidence for suggestions

    Returns:
        DataFrame with columns: row_id, column, issue_type, value, suggested_fix, confidence, source
    """
    issues = []

    if column not in df.columns:
        return pd.DataFrame()

    for idx, value in df[column].items():
        # Skip null values
        if pd.isna(value):
            continue

        value_str = str(value).strip()
        if not value_str:
            continue

        issue = None

        if corpus_type == 'alias':
            # Check for alias → canonical mapping
            canonical = corpus_manager.query_alias(
                corpus_name=corpus_name,
                key=value_str,
                lookup_key=None,
                normalize=True
            )

            if canonical and canonical.lower() != value_str.lower():
                # Found an alias that should be standardized
                issues.append({
                    'row_id': idx,
                    'column': column,
                    'issue_type': 'non_canonical_value',
                    'value': value_str,
                    'suggested_fix': canonical,
                    'confidence': 0.95,  # High confidence for exact alias match
                    'source': 'corpus',
                    'description': f'Value has canonical form in {corpus_name} corpus'
                })

        elif corpus_type == 'validation':
            # Check if value is in valid set
            is_valid = corpus_manager.is_valid(
                corpus_name=corpus_name,
                value=value_str,
                normalize=True
            )

            if not is_valid:
                # Value not in valid set
                issues.append({
                    'row_id': idx,
                    'column': column,
                    'issue_type': 'invalid_value',
                    'value': value_str,
                    'suggested_fix': None,  # Can't suggest a fix for invalid value
                    'confidence': 0.90,
                    'source': 'corpus',
                    'description': f'Value not found in {corpus_name} validation corpus'
                })

        elif corpus_type == 'lookup':
            # Lookup exists, but we don't validate - just note it's available for enrichment
            # Enrichment happens in Clean & Trim tab, not validation
            pass

    return pd.DataFrame(issues)


def validate_all_corpus_mappings(
    df: pd.DataFrame,
    corpus_mappings: Dict[str, Dict[str, Any]],
    corpus_manager
) -> pd.DataFrame:
    """
    Validate multiple columns using their mapped corpora.

    Args:
        df: Dataframe to validate
        corpus_mappings: Dict mapping column names to corpus config
            {
                "email": {"corpus": "email_domains", "type": "alias"},
                "postcode": {"corpus": "uk_postcodes", "type": "validation"},
                ...
            }
        corpus_manager: CorpusManager instance

    Returns:
        Unified DataFrame with all corpus validation issues
    """
    all_issues = []

    for column, config in corpus_mappings.items():
        if column not in df.columns:
            continue

        corpus_name = config.get('corpus')
        corpus_type = config.get('type', 'alias')

        if not corpus_name:
            continue

        # Validate this column
        column_issues = validate_with_corpus(
            df=df,
            column=column,
            corpus_name=corpus_name,
            corpus_manager=corpus_manager,
            corpus_type=corpus_type
        )

        if len(column_issues) > 0:
            all_issues.append(column_issues)

    if not all_issues:
        return pd.DataFrame(columns=[
            'row_id', 'column', 'issue_type', 'value', 'suggested_fix',
            'confidence', 'source', 'description'
        ])

    return pd.concat(all_issues, ignore_index=True)


def apply_corpus_standardization(
    df: pd.DataFrame,
    corpus_mappings: Dict[str, Dict[str, Any]],
    corpus_manager
) -> pd.DataFrame:
    """
    Apply corpus-based standardization to create a cleaned dataframe.
    Only applies alias-type corpora (canonical forms).

    Args:
        df: Original dataframe
        corpus_mappings: Column to corpus mapping
        corpus_manager: CorpusManager instance

    Returns:
        Copy of df with standardized values
    """
    df_standardized = df.copy()

    for column, config in corpus_mappings.items():
        if column not in df_standardized.columns:
            continue

        corpus_name = config.get('corpus')
        corpus_type = config.get('type', 'alias')

        # Only auto-apply alias type (standardization)
        if corpus_type != 'alias' or not corpus_name:
            continue

        # Apply standardization
        for idx, value in df_standardized[column].items():
            if pd.isna(value):
                continue

            value_str = str(value).strip()
            if not value_str:
                continue

            # Query for canonical form
            canonical = corpus_manager.query_alias(
                corpus_name=corpus_name,
                key=value_str,
                lookup_key=None,
                normalize=True
            )

            if canonical and canonical != value_str:
                df_standardized.at[idx, column] = canonical

    return df_standardized


def enrich_from_corpus(
    df: pd.DataFrame,
    key_column: str,
    corpus_name: str,
    fill_columns: List[str],
    corpus_manager
) -> Tuple[pd.DataFrame, int]:
    """
    Enrich missing data using lookup corpus.

    Args:
        df: Dataframe to enrich
        key_column: Column to use as lookup key (e.g., 'postcode')
        corpus_name: Name of lookup corpus
        fill_columns: Columns to fill if empty (e.g., ['address', 'city'])
        corpus_manager: CorpusManager instance

    Returns:
        Tuple of (enriched_df, count_of_enriched_rows)
    """
    df_enriched = df.copy()
    enriched_count = 0

    for idx, row in df_enriched.iterrows():
        key_value = row.get(key_column)

        if pd.isna(key_value):
            continue

        key_str = str(key_value).strip()
        if not key_str:
            continue

        # Lookup in corpus
        lookup_data = corpus_manager.query_lookup(
            corpus_name=corpus_name,
            key=key_str,
            normalize=True
        )

        if not lookup_data or not isinstance(lookup_data, dict):
            continue

        # Fill empty columns
        row_enriched = False
        for fill_col in fill_columns:
            if fill_col not in df_enriched.columns:
                continue

            # Only fill if current value is empty
            if pd.isna(row.get(fill_col)) or str(row.get(fill_col, '')).strip() == '':
                if fill_col in lookup_data:
                    df_enriched.at[idx, fill_col] = lookup_data[fill_col]
                    row_enriched = True

        if row_enriched:
            enriched_count += 1

    return df_enriched, enriched_count


def normalize_column_for_validation(
    series: pd.Series,
    normalization_type: str
) -> pd.Series:
    """
    Normalize a column for better validation matching.

    Args:
        series: Pandas Series to normalize
        normalization_type: 'email', 'postcode', 'text', 'phone', etc.

    Returns:
        Normalized series
    """
    if normalization_type == 'email':
        return series.astype(str).str.lower().str.strip()

    elif normalization_type == 'postcode':
        # Remove spaces, uppercase
        return series.astype(str).str.replace(' ', '').str.upper().str.strip()

    elif normalization_type == 'phone':
        # Keep only digits
        return series.astype(str).str.replace(r'\D', '', regex=True)

    elif normalization_type == 'text':
        # Lowercase, trim, collapse spaces
        return (series.astype(str)
                .str.lower()
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True))

    else:
        # Default: just trim
        return series.astype(str).str.strip()


def create_normalized_dataframe_for_validation(
    df: pd.DataFrame,
    normalization_config: Dict[str, str]
) -> pd.DataFrame:
    """
    Create a normalized copy of dataframe for better validation.
    Does NOT modify original df.

    Args:
        df: Original dataframe
        normalization_config: Dict mapping column names to normalization types
            {
                "email": "email",
                "postcode": "postcode",
                "company": "text"
            }

    Returns:
        Normalized copy of df
    """
    df_norm = df.copy()

    for column, norm_type in normalization_config.items():
        if column not in df_norm.columns:
            continue

        df_norm[column] = normalize_column_for_validation(df_norm[column], norm_type)

    return df_norm
