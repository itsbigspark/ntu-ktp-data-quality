# ==========================================================
# core/type_detector.py
# ==========================================================
from __future__ import annotations
import re
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Common regex patterns (extendable)
# ----------------------------------------------------------
PATTERNS = {
    "email": re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"),
    "phone": re.compile(r"^\+?\d[\d\s\-()]{6,}$"),
    "postcode": re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$", re.IGNORECASE),
    "id_code": re.compile(r"^[A-Z0-9\-]{6,}$", re.IGNORECASE),
}


# ----------------------------------------------------------
# Regex coverage scoring and dominant pattern detection
# ----------------------------------------------------------
def regex_pattern_coverage(series: pd.Series) -> dict[str, float]:
    """
    Measure what % of sampled values match each regex pattern.
    Returns a dict {pattern_name: coverage_ratio}.
    """
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return {}
    sample = s.sample(min(200, len(s)), random_state=0).tolist()
    ratios = {}
    for name, pat in PATTERNS.items():
        match_ratio = sum(bool(pat.match(v)) for v in sample) / len(sample)
        ratios[name] = round(match_ratio, 3)
    return ratios


def dominant_regex_pattern(series: pd.Series, threshold: float = 0.6) -> str | None:
    """
    Return the regex pattern name with highest coverage above threshold.
    Falls back to None if no strong pattern found.
    """
    ratios = regex_pattern_coverage(series)
    if not ratios:
        return None
    top_pat, top_val = max(ratios.items(), key=lambda kv: kv[1])
    return top_pat if top_val >= threshold else None


# ----------------------------------------------------------
# Infer semantic type of a column (optionally using regex hints)
# ----------------------------------------------------------
def infer_column_type(series: pd.Series, regex_hint: str | None = None) -> str:
    """
    Infer semantic type of a pandas Series using:
      - Regex coverage and dominant pattern
      - Optional regex hint (from validation rules)
      - Regex-based and statistical heuristics
    """
    s = series.dropna().astype(str).str.strip()
    n = len(s)
    if n == 0:
        return "unknown"

    # --- 1️⃣ Use regex hint if provided ---
    if regex_hint:
        hint = regex_hint.lower()
        if "@" in hint or "email" in hint:
            return "email"
        if "phone" in hint or r"\d{10,}" in hint or r"\+" in hint:
            return "phone"
        if "postcode" in hint or re.search(r"[A-Z]{1,2}\d", hint):
            return "postcode"
        if "date" in hint or re.search(r"\d{4}[-/]", hint):
            return "date"
        if "id" in hint or "code" in hint or re.search(r"[A-Z0-9\-]{4,}", hint):
            return "id_code"

    # --- 2️⃣ Numeric or date detection ---
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return "date"
    except Exception:
        pass

    # --- 3️⃣ Regex-based detection ---
    dominant = dominant_regex_pattern(series)
    if dominant:
        return dominant

    # --- 4️⃣ Heuristics (textual, categorical, etc.) ---
    uniq_ratio = s.nunique(dropna=True) / n
    avg_len = np.mean(s.str.len())

    if uniq_ratio < 0.2:
        return "categorical"
    elif avg_len > 25:
        return "text"
    else:
        return "string"


# ----------------------------------------------------------
# Compare types for compatibility
# ----------------------------------------------------------
def are_types_compatible(type_a: str, type_b: str) -> bool:
    """
    Return True if two inferred types are semantically comparable.
    """
    if type_a == type_b:
        return True

    compatible = {
        "string": {"categorical", "id_code"},
        "categorical": {"string"},
        "id_code": {"string", "categorical"},
        "postcode": {"string"},
        "phone": {"string"},
        "email": {"string"},
        "date": {"string"},
        "numeric": {"string", "categorical"},
    }
    return (
        type_b in compatible.get(type_a, set())
        or type_a in compatible.get(type_b, set())
    )


# ----------------------------------------------------------
# Summarize inferred column types
# ----------------------------------------------------------
def summarize_types(df: pd.DataFrame, regex_hints: dict | None = None) -> pd.DataFrame:
    """
    Produce a summary table of inferred semantic types per column.

    Args:
        df: Input DataFrame
        regex_hints: Optional dict of {column_name: regex_pattern}
                     (from validation rules or regex inference)
    """
    rows = []
    regex_hints = regex_hints or {}
    for col in df.columns:
        try:
            col_type = infer_column_type(df[col], regex_hint=regex_hints.get(col))
        except Exception:
            col_type = "unknown"
        rows.append({"column": col, "inferred_type": col_type})
    return pd.DataFrame(rows)


# ----------------------------------------------------------
# Helper: find compatible columns across datasets
# ----------------------------------------------------------
def find_compatible_columns(
    df_anchor: pd.DataFrame,
    df_other: pd.DataFrame,
    regex_hints_anchor: dict | None = None,
    regex_hints_other: dict | None = None,
    exact_only: bool = False,
) -> list[tuple[str, str]]:
    """
    Return a list of (anchor_col, other_col) pairs that are type-compatible.
    If exact_only=True, only return exact column name matches (case-insensitive).
    """
    regex_hints_anchor = regex_hints_anchor or {}
    regex_hints_other = regex_hints_other or {}

    types_anchor = {
        c: infer_column_type(df_anchor[c], regex_hint=regex_hints_anchor.get(c))
        for c in df_anchor.columns
    }
    types_other = {
        c: infer_column_type(df_other[c], regex_hint=regex_hints_other.get(c))
        for c in df_other.columns
    }

    compatible_pairs = []
    for a, ta in types_anchor.items():
        for b, tb in types_other.items():
            if exact_only:
                if a.strip().lower() == b.strip().lower():
                    compatible_pairs.append((a, b))
            else:
                if are_types_compatible(ta, tb):
                    compatible_pairs.append((a, b))

    return compatible_pairs