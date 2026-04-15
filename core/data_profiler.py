# core/data_profiler.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from collections import Counter

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data profiling with statistics and insights.

    Returns a dictionary with:
    - dataset_summary: overall statistics
    - column_profiles: detailed stats per column
    - correlations: correlation matrix for numeric columns
    - patterns: detected patterns in string columns
    - outliers: outlier detection results
    """

    profile = {
        "dataset_summary": _get_dataset_summary(df),
        "column_profiles": _profile_columns(df),
        "correlations": _compute_correlations(df),
        "patterns": _detect_patterns(df),
        "outliers": _detect_outliers(df)
    }

    return profile


def _get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get overall dataset statistics"""

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": df.size,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_cells": df.isna().sum().sum(),
        "missing_percentage": round(100 * df.isna().sum().sum() / df.size, 2),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percentage": round(100 * df.duplicated().sum() / len(df), 2) if len(df) > 0 else 0,
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": len(df.select_dtypes(include=['object']).columns),
        "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
    }


def _profile_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Profile each column with detailed statistics"""

    profiles = {}

    for col in df.columns:
        col_data = df[col]
        col_profile = {
            "dtype": str(col_data.dtype),
            "count": col_data.count(),
            "missing": col_data.isna().sum(),
            "missing_pct": round(100 * col_data.isna().sum() / len(df), 2),
            "unique": col_data.nunique(),
            "unique_pct": round(100 * col_data.nunique() / len(df), 2) if len(df) > 0 else 0,
        }

        # Numeric column statistics
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update(_profile_numeric_column(col_data))

        # String/categorical column statistics
        elif pd.api.types.is_object_dtype(col_data):
            col_profile.update(_profile_string_column(col_data))

        # Datetime column statistics
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_profile.update(_profile_datetime_column(col_data))

        profiles[col] = col_profile

    return profiles


def _profile_numeric_column(col_data: pd.Series) -> Dict[str, Any]:
    """Profile numeric column with statistics"""

    clean_data = col_data.dropna()

    if len(clean_data) == 0:
        return {
            "min": None, "max": None, "mean": None, "median": None,
            "std": None, "q25": None, "q75": None, "iqr": None,
            "skewness": None, "kurtosis": None
        }

    return {
        "min": float(clean_data.min()),
        "max": float(clean_data.max()),
        "mean": float(clean_data.mean()),
        "median": float(clean_data.median()),
        "std": float(clean_data.std()),
        "q25": float(clean_data.quantile(0.25)),
        "q75": float(clean_data.quantile(0.75)),
        "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
        "skewness": float(clean_data.skew()) if len(clean_data) > 1 else None,
        "kurtosis": float(clean_data.kurtosis()) if len(clean_data) > 1 else None,
        "zeros": int((clean_data == 0).sum()),
        "negatives": int((clean_data < 0).sum()),
        "positives": int((clean_data > 0).sum()),
    }


def _profile_string_column(col_data: pd.Series) -> Dict[str, Any]:
    """Profile string/categorical column"""

    clean_data = col_data.dropna().astype(str)

    if len(clean_data) == 0:
        return {
            "min_length": None, "max_length": None, "avg_length": None,
            "mode": None, "mode_frequency": None, "top_values": []
        }

    lengths = clean_data.str.len()
    value_counts = clean_data.value_counts()

    profile = {
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "avg_length": round(lengths.mean(), 2),
        "mode": value_counts.index[0] if len(value_counts) > 0 else None,
        "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        "top_values": [
            {"value": str(val), "count": int(count), "percentage": round(100 * count / len(clean_data), 2)}
            for val, count in value_counts.head(10).items()
        ]
    }

    # Detect common patterns
    patterns = _detect_column_patterns(clean_data)
    profile["detected_patterns"] = patterns

    return profile


def _profile_datetime_column(col_data: pd.Series) -> Dict[str, Any]:
    """Profile datetime column"""

    clean_data = col_data.dropna()

    if len(clean_data) == 0:
        return {
            "min_date": None, "max_date": None, "date_range_days": None
        }

    min_date = clean_data.min()
    max_date = clean_data.max()
    date_range = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None

    return {
        "min_date": str(min_date),
        "max_date": str(max_date),
        "date_range_days": date_range,
        "year_range": f"{min_date.year} - {max_date.year}" if pd.notna(min_date) and pd.notna(max_date) else None,
    }


def _detect_column_patterns(col_data: pd.Series) -> Dict[str, int]:
    """Detect common patterns in string column"""

    patterns = {
        "email": 0,
        "phone": 0,
        "url": 0,
        "postcode_uk": 0,
        "postcode_us": 0,
        "date_like": 0,
        "numeric_string": 0,
        "alphanumeric": 0,
    }

    # Sample up to 1000 values for performance
    sample = col_data.sample(min(1000, len(col_data)))

    for val in sample:
        val_str = str(val)

        # Email pattern
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', val_str):
            patterns["email"] += 1

        # Phone pattern
        elif re.match(r'^[\d\s\-\(\)\+]{7,}$', val_str):
            patterns["phone"] += 1

        # URL pattern
        elif re.match(r'^https?://', val_str):
            patterns["url"] += 1

        # UK postcode
        elif re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$', val_str.upper()):
            patterns["postcode_uk"] += 1

        # US ZIP
        elif re.match(r'^\d{5}(-\d{4})?$', val_str):
            patterns["postcode_us"] += 1

        # Date-like
        elif re.match(r'^\d{4}-\d{2}-\d{2}', val_str) or re.match(r'^\d{2}/\d{2}/\d{4}', val_str):
            patterns["date_like"] += 1

        # Numeric string
        elif re.match(r'^\d+$', val_str):
            patterns["numeric_string"] += 1

        # Alphanumeric
        elif re.match(r'^[a-zA-Z0-9]+$', val_str):
            patterns["alphanumeric"] += 1

    # Scale back to full column
    scale_factor = len(col_data) / len(sample)
    patterns = {k: int(v * scale_factor) for k, v in patterns.items()}

    # Only return patterns with significant presence (>10%)
    threshold = len(col_data) * 0.1
    significant_patterns = {k: v for k, v in patterns.items() if v > threshold}

    return significant_patterns


def _detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect patterns across all columns in the dataset"""

    all_patterns = {}

    # Check each string column for patterns
    for col in df.select_dtypes(include=['object']).columns:
        col_patterns = _detect_column_patterns(df[col])
        if col_patterns:
            all_patterns[col] = col_patterns

    return {
        "column_patterns": all_patterns,
        "total_columns_with_patterns": len(all_patterns)
    }


def _compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlation matrix for numeric columns"""

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return {
            "has_correlations": False,
            "message": "Need at least 2 numeric columns for correlation analysis"
        }

    corr_matrix = numeric_df.corr()

    # Find strong correlations (>0.7 or <-0.7)
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) > 0.7:
                strong_correlations.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": round(corr_val, 3),
                    "strength": "strong positive" if corr_val > 0.7 else "strong negative"
                })

    return {
        "has_correlations": True,
        "correlation_matrix": corr_matrix.round(3).to_dict(),
        "strong_correlations": strong_correlations,
        "num_numeric_columns": len(numeric_df.columns)
    }


def _detect_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect outliers in numeric columns using IQR method"""

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) == 0:
        return {
            "has_outliers": False,
            "message": "No numeric columns for outlier detection"
        }

    outlier_summary = {}

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()

        if len(col_data) == 0:
            continue

        # IQR method
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

        if len(outliers) > 0:
            outlier_summary[col] = {
                "count": len(outliers),
                "percentage": round(100 * len(outliers) / len(col_data), 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": outliers.head(10).tolist()
            }

    return {
        "has_outliers": len(outlier_summary) > 0,
        "outlier_columns": outlier_summary,
        "total_outlier_columns": len(outlier_summary)
    }


def get_value_distributions(df: pd.DataFrame, column: str, bins: int = 20) -> Dict[str, Any]:
    """Get value distribution for a specific column (for plotting)"""

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    col_data = df[column].dropna()

    if len(col_data) == 0:
        return {"error": "No data in column"}

    result = {"column": column, "dtype": str(df[column].dtype)}

    # Numeric distribution
    if pd.api.types.is_numeric_dtype(df[column]):
        hist, bin_edges = np.histogram(col_data, bins=bins)
        result["distribution_type"] = "numeric"
        result["bins"] = bin_edges.tolist()
        result["frequencies"] = hist.tolist()
        result["stats"] = {
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
        }

    # Categorical distribution
    else:
        value_counts = col_data.value_counts().head(20)
        result["distribution_type"] = "categorical"
        result["categories"] = value_counts.index.tolist()
        result["frequencies"] = value_counts.values.tolist()

    return result
