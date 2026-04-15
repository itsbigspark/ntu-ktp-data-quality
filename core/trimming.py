# core/trimming.py
from __future__ import annotations
import numpy as np
import pandas as pd

# 1) Always exclude helper/meta columns from similarity/trimming
EXCLUDE_ALWAYS = {"__row_concat_norm", "__row_concat_raw", "__has_issue"}

def uniqueness_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return s.nunique(dropna=False) / len(s)

# 2) Low-information: drop constants; drop near-unique ONLY if numeric
def low_information_columns(df: pd.DataFrame, low_thresh: float = 0.01, high_thresh: float = 0.99) -> list[str]:
    drops = []
    for c in df.columns:
        if c in EXCLUDE_ALWAYS:
            drops.append(c)
            continue
        ur = uniqueness_ratio(df[c].astype("object"))
        # almost constant => drop
        if ur < low_thresh:
            drops.append(c)
            continue
        # near-unique => drop only if NUMERIC (keep text IDs / names by default)
        if ur > high_thresh and pd.api.types.is_numeric_dtype(df[c]):
            drops.append(c)
    return drops

def high_missing_columns(df: pd.DataFrame, missing_thresh: float = 0.80) -> list[str]:
    return [c for c in df.columns if df[c].isna().mean() > missing_thresh]

# 3) Correlation: restrict to numeric-only to avoid spurious drops
def correlated_columns(df: pd.DataFrame, corr_thresh: float = 0.90) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return []
    work = df[num_cols].copy()
    for c in num_cols:
        s = work[c]
        work[c] = s.astype(float).fillna(s.mean())
    corr = work.corr().abs()
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i, j] >= corr_thresh:
                to_drop.add(cols[j])
    return list(to_drop)

def compute_kept_columns(df: pd.DataFrame,
                         low_thresh: float = 0.01,
                         high_thresh: float = 0.99,
                         missing_thresh: float = 0.80,
                         corr_thresh: float = 0.90,
                         protect: list[str] | None = None) -> list[str]:
    protect = set(protect or [])
    candidates = [c for c in df.columns if c not in EXCLUDE_ALWAYS]
    df = df[candidates].copy()

    drops = set()
    drops.update(low_information_columns(df, low_thresh, high_thresh))
    drops.update(high_missing_columns(df, missing_thresh))
    drops.update(correlated_columns(df, corr_thresh))

    kept = [c for c in candidates if (c not in drops) or (c in protect)]
    return kept

def apply_trimming(df: pd.DataFrame, **kwargs):
    kept = compute_kept_columns(df, **kwargs)
    return df[kept].copy(), kept