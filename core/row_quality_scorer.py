"""
Row-Level Quality Scoring
Assigns quality scores to individual rows based on multiple dimensions.

Performance: all four dimensions are computed with column-wise vectorised
operations (numpy / pandas) rather than per-row Python loops, giving
~10-50x speed-up on large datasets.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def score_all_rows(
    df: pd.DataFrame,
    rules: Optional[List[Dict]] = None,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Score all rows in a dataframe.

    Returns
    -------
    DataFrame with columns:
        row_id, overall_score, completeness, validity, consistency,
        conformity, issue_count, issues, recommendation
    """
    if weights is None:
        weights = {
            'completeness': 0.30,
            'validity':     0.40,
            'consistency':  0.20,
            'conformity':   0.10,
        }

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=[
            'row_id', 'overall_score', 'completeness', 'validity',
            'consistency', 'conformity', 'issue_count', 'issues', 'recommendation'
        ])

    # ── Completeness (pure numpy, one line) ──────────────────────────────────
    completeness = (df.notna().sum(axis=1) / max(len(df.columns), 1) * 100).clip(0, 100)

    # ── Validity (iterate over rules, not rows) ───────────────────────────────
    validity_arr, validity_issues = _score_validity_vec(df, rules, n)

    # ── Consistency (iterate over columns, IQR pre-computed once) ─────────────
    consistency_arr, consistency_issues = _score_consistency_vec(df, n)

    # ── Conformity (iterate over columns by name pattern) ────────────────────
    conformity_arr, conformity_issues = _score_conformity_vec(df, n)

    # ── Overall ───────────────────────────────────────────────────────────────
    overall = (
        completeness.values * weights['completeness'] +
        validity_arr         * weights['validity'] +
        consistency_arr      * weights['consistency'] +
        conformity_arr       * weights['conformity']
    )

    # ── Merge per-row issue strings ───────────────────────────────────────────
    all_issues = []
    for i in range(n):
        parts = [p for p in [validity_issues[i], consistency_issues[i], conformity_issues[i]] if p]
        all_issues.append("; ".join(parts) if parts else "No issues")

    issue_counts = [
        len(s.split("; ")) if s != "No issues" else 0
        for s in all_issues
    ]

    return pd.DataFrame({
        'row_id':           df.index,
        'overall_score':    np.round(overall, 1),
        'completeness':     np.round(completeness.values, 1),
        'validity':         np.round(validity_arr, 1),
        'consistency':      np.round(consistency_arr, 1),
        'conformity':       np.round(conformity_arr, 1),
        'issue_count':      issue_counts,
        'issues':           all_issues,
        'recommendation':   [_generate_recommendation(s) for s in overall],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised dimension helpers
# ──────────────────────────────────────────────────────────────────────────────

def _score_validity_vec(
    df: pd.DataFrame,
    rules: Optional[List[Dict]],
    n: int
) -> tuple:
    """
    Validity: % of rule checks passed per row.
    Iterates over rules (typically 5–20), not over rows.
    """
    passed       = np.zeros(n, dtype=np.float64)
    total_checks = np.zeros(n, dtype=np.float64)
    issues       = [""] * n

    if not rules or not isinstance(rules, (list, tuple)):
        return np.full(n, 100.0), issues

    def _append_issue(mask, msg):
        for pos in np.where(mask)[0]:
            issues[pos] = (issues[pos] + "; " + msg) if issues[pos] else msg

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        column = rule.get('column')
        if column not in df.columns:
            continue

        rule_type = rule.get('rule_type', rule.get('type', ''))
        col_vals  = df[column]
        not_null  = col_vals.notna()

        if not_null.sum() == 0:
            continue

        total_checks += not_null.values.astype(np.float64)

        if rule_type == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            numeric = pd.to_numeric(col_vals, errors='coerce')
            valid_num = not_null & numeric.notna()
            not_num   = not_null & numeric.isna()
            _append_issue(not_num.values, f"{column}: not numeric")
            total_checks -= not_num.values.astype(np.float64)  # don't double-count

            in_range = valid_num.copy()
            if min_val is not None:
                too_low = valid_num & (numeric < min_val)
                _append_issue(too_low.values, f"{column}: value < minimum {min_val}")
                in_range &= (numeric >= min_val)
            if max_val is not None:
                too_high = valid_num & (numeric > max_val)
                _append_issue(too_high.values, f"{column}: value > maximum {max_val}")
                in_range &= (numeric <= max_val)
            passed += in_range.values.astype(np.float64)

        elif rule_type in ('regex', 'pattern'):
            pattern = rule.get('pattern', '')
            if pattern:
                str_vals  = col_vals.astype(str)
                match     = not_null & str_vals.str.match(pattern, na=False)
                no_match  = not_null & ~match
                _append_issue(no_match.values, f"{column}: doesn't match pattern")
                passed += match.values.astype(np.float64)

        elif rule_type == 'allowed_values':
            allowed    = rule.get('allowed_values', [])
            in_allowed = not_null & col_vals.isin(allowed)
            not_in     = not_null & ~col_vals.isin(allowed)
            _append_issue(not_in.values, f"{column}: value not in allowed list")
            passed += in_allowed.values.astype(np.float64)

        elif rule_type == 'length':
            min_len = rule.get('min_length', 0)
            max_len = rule.get('max_length', float('inf'))
            lengths = col_vals.astype(str).str.len()
            ok      = not_null & lengths.between(min_len, max_len)
            bad     = not_null & ~ok
            _append_issue(bad.values, f"{column}: length out of [{min_len}, {max_len}]")
            passed += ok.values.astype(np.float64)

        else:
            # Unknown rule type — don't count it
            total_checks -= not_null.values.astype(np.float64)

    scores = np.where(total_checks == 0, 100.0, passed / total_checks * 100)
    return scores, issues


def _score_consistency_vec(df: pd.DataFrame, n: int) -> tuple:
    """
    Consistency: outliers, negative values in positive-only columns,
    start/end date ordering. IQR bounds computed once per column.
    """
    checks_passed = np.zeros(n, dtype=np.float64)
    total_checks  = np.zeros(n, dtype=np.float64)
    issues        = [""] * n

    def _append_issue(mask, msg):
        for pos in np.where(mask)[0]:
            issues[pos] = (issues[pos] + "; " + msg) if issues[pos] else msg

    # Numeric outlier check (IQR computed once per column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) <= 10:
            continue
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 3 * iqr, q3 + 3 * iqr

        not_null = df[col].notna()
        total_checks += not_null.values.astype(np.float64)
        in_range  = not_null & df[col].between(lb, ub)
        outlier   = not_null & ~df[col].between(lb, ub)
        checks_passed += in_range.values.astype(np.float64)
        _append_issue(outlier.values, f"{col}: extreme outlier")

    # Negative values in positive-only columns
    pos_pattern = re.compile(r'age|price|quantity|amount|count|total', re.IGNORECASE)
    sus_cols = [c for c in numeric_cols if pos_pattern.search(c)]
    for col in sus_cols:
        not_null = df[col].notna()
        total_checks  += not_null.values.astype(np.float64)
        non_neg        = not_null & (df[col] >= 0)
        neg_mask       = not_null & (df[col] < 0)
        checks_passed += non_neg.values.astype(np.float64)
        _append_issue(neg_mask.values, f"{col}: negative value")

    # Date consistency: start_* < end_* column pairs
    date_cols = [c for c in df.columns if re.search(r'date|time', c, re.IGNORECASE)]
    if len(date_cols) >= 2:
        for k, col1 in enumerate(date_cols):
            for col2 in date_cols[k + 1:]:
                if 'start' in col1.lower() and 'end' in col2.lower():
                    d1 = pd.to_datetime(df[col1], errors='coerce')
                    d2 = pd.to_datetime(df[col2], errors='coerce')
                    both_valid = d1.notna() & d2.notna()
                    total_checks  += both_valid.values.astype(np.float64)
                    ok             = both_valid & (d1 <= d2)
                    bad            = both_valid & (d1 > d2)
                    checks_passed += ok.values.astype(np.float64)
                    _append_issue(bad.values, f"{col1} after {col2}")

    scores = np.where(total_checks == 0, 100.0, checks_passed / total_checks * 100)
    return scores, issues


def _score_conformity_vec(df: pd.DataFrame, n: int) -> tuple:
    """
    Conformity: email, phone, postcode, URL, name pattern checks.
    Iterates over columns (typically 5–20), not over rows.
    """
    checks_passed = np.zeros(n, dtype=np.float64)
    total_checks  = np.zeros(n, dtype=np.float64)
    issues        = [""] * n

    EMAIL_RE    = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    UK_POST_RE  = r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'
    US_ZIP_RE   = r'^\d{5}(-\d{4})?$'
    URL_RE      = r'^https?://'
    DIGIT_RE    = r'\d'
    PHONE_STRIP = re.compile(r'[\s\-\(\)\+]')

    def _append_issue(mask, msg):
        for pos in np.where(mask)[0]:
            issues[pos] = (issues[pos] + "; " + msg) if issues[pos] else msg

    for col in df.columns:
        col_lower = col.lower()
        col_vals  = df[col]
        not_null  = col_vals.notna()
        str_vals  = col_vals.astype(str).str.strip()
        non_empty = not_null & (str_vals != '') & (str_vals != 'nan')

        if 'email' in col_lower or 'mail' in col_lower:
            total_checks  += non_empty.values.astype(np.float64)
            match          = non_empty & str_vals.str.match(EMAIL_RE, na=False)
            checks_passed += match.values.astype(np.float64)
            _append_issue((non_empty & ~match).values, f"{col}: invalid email format")

        elif any(k in col_lower for k in ('phone', 'tel', 'mobile')):
            total_checks  += non_empty.values.astype(np.float64)
            cleaned        = str_vals.str.replace(PHONE_STRIP, '', regex=True)
            valid          = non_empty & (cleaned.str.len() >= 7) & cleaned.str.isdigit()
            checks_passed += valid.values.astype(np.float64)
            _append_issue((non_empty & ~valid).values, f"{col}: invalid phone format")

        elif any(k in col_lower for k in ('postcode', 'postal', 'zip')):
            total_checks  += non_empty.values.astype(np.float64)
            upper          = str_vals.str.upper()
            valid          = non_empty & (
                upper.str.match(UK_POST_RE, na=False) |
                str_vals.str.match(US_ZIP_RE, na=False)
            )
            checks_passed += valid.values.astype(np.float64)
            _append_issue((non_empty & ~valid).values, f"{col}: invalid postcode/ZIP format")

        elif any(k in col_lower for k in ('url', 'website', 'link')):
            total_checks  += non_empty.values.astype(np.float64)
            valid          = non_empty & str_vals.str.match(URL_RE, na=False)
            checks_passed += valid.values.astype(np.float64)
            _append_issue((non_empty & ~valid).values, f"{col}: invalid URL format")

        elif 'name' in col_lower and 'user' not in col_lower:
            total_checks  += non_empty.values.astype(np.float64)
            valid          = non_empty & ~str_vals.str.contains(DIGIT_RE, na=False)
            checks_passed += valid.values.astype(np.float64)
            _append_issue((non_empty & ~valid).values, f"{col}: name contains numbers")

    scores = np.where(total_checks == 0, 100.0, checks_passed / total_checks * 100)
    return scores, issues


# ──────────────────────────────────────────────────────────────────────────────
# Unchanged helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_recommendation(score: float, issues: List[str] = None) -> str:
    if score >= 90:
        return "Good quality - no action needed"
    elif score >= 70:
        return "Moderate quality - review issues"
    elif score >= 50:
        return "Low quality - fix critical issues"
    elif score >= 30:
        return "Poor quality - consider manual review"
    else:
        return "Very poor quality - consider deletion"


def get_quality_summary(row_scores_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'total_rows':         len(row_scores_df),
        'avg_score':          round(row_scores_df['overall_score'].mean(), 1),
        'median_score':       round(row_scores_df['overall_score'].median(), 1),
        'std_score':          round(row_scores_df['overall_score'].std(), 1),
        'min_score':          round(row_scores_df['overall_score'].min(), 1),
        'max_score':          round(row_scores_df['overall_score'].max(), 1),
        'rows_excellent':     len(row_scores_df[row_scores_df['overall_score'] >= 90]),
        'rows_good':          len(row_scores_df[(row_scores_df['overall_score'] >= 70) & (row_scores_df['overall_score'] < 90)]),
        'rows_moderate':      len(row_scores_df[(row_scores_df['overall_score'] >= 50) & (row_scores_df['overall_score'] < 70)]),
        'rows_poor':          len(row_scores_df[(row_scores_df['overall_score'] >= 30) & (row_scores_df['overall_score'] < 50)]),
        'rows_very_poor':     len(row_scores_df[row_scores_df['overall_score'] < 30]),
        'total_issues':       int(row_scores_df['issue_count'].sum()),
        'avg_issues_per_row': round(row_scores_df['issue_count'].mean(), 1),
    }


def get_worst_rows(row_scores_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return row_scores_df.nsmallest(n, 'overall_score')


def get_rows_by_quality(row_scores_df: pd.DataFrame, quality_level: str) -> pd.DataFrame:
    if quality_level == 'excellent':
        return row_scores_df[row_scores_df['overall_score'] >= 90]
    elif quality_level == 'good':
        return row_scores_df[(row_scores_df['overall_score'] >= 70) & (row_scores_df['overall_score'] < 90)]
    elif quality_level == 'moderate':
        return row_scores_df[(row_scores_df['overall_score'] >= 50) & (row_scores_df['overall_score'] < 70)]
    elif quality_level == 'poor':
        return row_scores_df[(row_scores_df['overall_score'] >= 30) & (row_scores_df['overall_score'] < 50)]
    elif quality_level == 'very_poor':
        return row_scores_df[row_scores_df['overall_score'] < 30]
    else:
        return row_scores_df
