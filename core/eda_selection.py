from __future__ import annotations
import re
import pandas as pd

LOW_INFO_VALUES = {"", "missing", "null", "none", "na", "n/a", "-", "--"}
LOW_INFO_THRESHOLD = 0.90

def _low_info_frac(series: pd.Series) -> float:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return float((s.isin(LOW_INFO_VALUES)).mean())

def run_eda_selection(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in df.columns if _low_info_frac(df[c]) < LOW_INFO_THRESHOLD]
    trimmed = df[keep].copy()

    # Drop datetime-like columns
    dt_cols = [c for c in trimmed.columns if str(trimmed[c].dtype).startswith('datetime')]
    if dt_cols:
        trimmed = trimmed.drop(columns=dt_cols)

    def norm_cell(x):
        t = str(x).lower()
        t = t.replace("&", " and ")
        t = re.sub(r"[^a-z0-9]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    norm = trimmed.applymap(norm_cell).astype(str)
    trimmed['__row_concat_norm'] = norm.apply(lambda r: "_".join(r.values), axis=1)
    trimmed['__row_concat_raw']  = trimmed.apply(lambda r: "_".join(map(str, r.values)), axis=1)
    return trimmed
