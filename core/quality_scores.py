# core/quality_scores.py

import pandas as pd
import numpy as np
import re

def compute_quality_scores(df: pd.DataFrame) -> dict:
    """
    Comprehensive data quality scoring across 6 dimensions:
    - Completeness: % of non-missing values
    - Uniqueness: % of non-duplicate records
    - Consistency: % of values matching expected formats/patterns
    - Validity: % of values within valid ranges/domains
    - Accuracy: % of values appearing correct (no obvious errors)
    - Timeliness: % of date fields with recent/valid dates
    """

    score = {}

    if df.empty:
        return {
            "completeness": 0.0,
            "uniqueness": 0.0,
            "consistency": 0.0,
            "validity": 0.0,
            "accuracy": 0.0,
            "timeliness": 0.0,
            "missing_by_column": {}
        }

    # 1. COMPLETENESS SCORE
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    score["completeness"] = round(100 * (1 - missing_cells / total_cells), 2) if total_cells > 0 else 0.0

    # 2. UNIQUENESS SCORE
    dup = df.duplicated().sum()
    score["uniqueness"] = round(100 * (1 - dup / len(df)), 2) if len(df) > 0 else 0.0

    # 3. CONSISTENCY SCORE (format consistency within each column)
    consistency_scores = []
    for col in df.select_dtypes(include='object').columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        # Check format consistency (e.g., all emails, all phones, all dates have similar formats)
        formats = col_data.apply(lambda x: _detect_format(str(x)))
        most_common_format = formats.mode()[0] if not formats.mode().empty else None

        if most_common_format:
            format_match_rate = (formats == most_common_format).sum() / len(formats)
            consistency_scores.append(format_match_rate * 100)

    score["consistency"] = round(np.mean(consistency_scores), 2) if consistency_scores else 75.0

    # 4. VALIDITY SCORE (% of values passing basic validation checks)
    validity_checks = []

    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        valid_count = 0
        total_count = len(col_data)

        # Check based on inferred column type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric: check for outliers (values within 3 std devs)
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                valid_count = ((col_data >= mean_val - 3*std_val) & (col_data <= mean_val + 3*std_val)).sum()
            else:
                valid_count = total_count
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Date: check for valid date range (not too far in past/future)
            current_year = pd.Timestamp.now().year
            valid_count = ((col_data.dt.year >= 1900) & (col_data.dt.year <= current_year + 10)).sum()
        else:
            # String: check for reasonable length and no control characters
            col_str = col_data.astype(str)
            valid_count = col_str.apply(lambda x: len(x) > 0 and len(x) < 500 and not bool(re.search(r'[\x00-\x1f]', x))).sum()

        if total_count > 0:
            validity_checks.append((valid_count / total_count) * 100)

    score["validity"] = round(np.mean(validity_checks), 2) if validity_checks else 85.0

    # 5. ACCURACY SCORE (% of values without obvious errors/typos)
    accuracy_checks = []

    for col in df.select_dtypes(include='object').columns:
        col_data = df[col].dropna().astype(str)
        if len(col_data) == 0:
            continue

        # Check for common error patterns
        error_count = 0

        for val in col_data:
            # Check for repeated characters (e.g., "aaaaa", "1111")
            if len(val) > 3 and re.search(r'(.)\1{4,}', val):
                error_count += 1
            # Check for placeholder values
            elif val.lower() in ['null', 'none', 'n/a', 'na', 'unknown', 'xxx', 'test', 'example']:
                error_count += 1
            # Check for excessive whitespace
            elif val != val.strip() or '  ' in val:
                error_count += 1

        accuracy_rate = ((len(col_data) - error_count) / len(col_data)) * 100
        accuracy_checks.append(accuracy_rate)

    score["accuracy"] = round(np.mean(accuracy_checks), 2) if accuracy_checks else 90.0

    # 6. TIMELINESS SCORE (for date columns, % of recent/valid dates)
    timeliness_checks = []
    current_date = pd.Timestamp.now()

    for col in df.columns:
        # Try to detect date columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Check if dates are within last 10 years or future 1 year
                recent_count = ((col_data >= current_date - pd.Timedelta(days=3650)) &
                               (col_data <= current_date + pd.Timedelta(days=365))).sum()
                timeliness_checks.append((recent_count / len(col_data)) * 100)
        elif df[col].dtype == 'object':
            # Try to parse as date
            try:
                col_dates = pd.to_datetime(df[col].dropna(), errors='coerce')
                valid_dates = col_dates.dropna()
                if len(valid_dates) > len(df[col].dropna()) * 0.5:  # If >50% parsed as dates
                    recent_count = ((valid_dates >= current_date - pd.Timedelta(days=3650)) &
                                   (valid_dates <= current_date + pd.Timedelta(days=365))).sum()
                    timeliness_checks.append((recent_count / len(valid_dates)) * 100)
            except:
                pass

    score["timeliness"] = round(np.mean(timeliness_checks), 2) if timeliness_checks else 80.0

    # Column-level missing %
    score["missing_by_column"] = (
        df.isna().mean().round(3).sort_values(ascending=False).to_dict()
    )

    return score


def _detect_format(value: str) -> str:
    """Helper function to detect the format/pattern of a string value"""
    if not value or len(value) == 0:
        return "empty"

    # Email pattern
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
        return "email"

    # Phone pattern (various formats)
    if re.match(r'^[\d\s\-\(\)\+]{7,}$', value):
        return "phone"

    # Date patterns
    if re.match(r'^\d{4}-\d{2}-\d{2}', value):
        return "date_iso"
    if re.match(r'^\d{2}/\d{2}/\d{4}', value):
        return "date_slash"

    # Postcode/ZIP patterns
    if re.match(r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$', value.upper()):
        return "postcode_uk"
    if re.match(r'^\d{5}(-\d{4})?$', value):
        return "postcode_us"

    # Number patterns
    if re.match(r'^[\d\.,\-\+]+$', value):
        return "numeric"

    # Mixed alphanumeric
    if re.match(r'^[a-zA-Z0-9]+$', value):
        return "alphanumeric"

    # Default: text
    return "text"