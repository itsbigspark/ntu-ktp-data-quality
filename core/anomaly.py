# core/validator/anomaly.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Optional: sklearn for stronger numeric + ML anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ======================================================================
# Utilities
# ======================================================================

_MISSING_TOKENS = {
    "", " ", "none", "null", "nan", "na", "n/a", "-", "--", "unknown", "missing", "not available",
    "nil", "0", "00"
}

def _is_missing_scalar(x: Any) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in _MISSING_TOKENS

def _missing_mask(s: pd.Series) -> pd.Series:
    # Mark pandas-NA or placeholder strings as missing
    m = s.isna()
    # Also treat common placeholders as missing
    try:
        m = m | s.astype(str).str.strip().str.lower().isin(_MISSING_TOKENS)
    except Exception:
        pass
    return m.fillna(False)


# ======================================================================
# Heuristic anomaly detectors
# ======================================================================

def _numeric_outliers(s: pd.Series) -> pd.Series:
    """Return boolean mask for numeric outliers (IsolationForest if available, else IQR)."""
    x = pd.to_numeric(s, errors="coerce")
    m = x.notna()
    if m.sum() < 8:
        return pd.Series(False, index=s.index)
    if _HAVE_SK:
        mdl = IsolationForest(random_state=42, contamination="auto")
        try:
            lbl = mdl.fit_predict(x[m].to_numpy().reshape(-1, 1))
            iso = (lbl == -1)
            out = pd.Series(False, index=s.index)
            out.loc[m] = iso
            return out
        except Exception:
            pass
    # Fallback: IQR rule
    q1, q3 = x[m].quantile(0.25), x[m].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (x < lo) | (x > hi)
    return out.fillna(False)


def _rare_categories(s: pd.Series, min_ratio: float = 0.01, min_count: int = 3) -> pd.Series:
    """
    Boolean mask: rare categories by frequency.
    Excludes missing/placeholder tokens from frequency counting.
    """
    t = s.astype(str).str.strip()
    # Exclude missing tokens when computing frequencies
    non_missing = ~t.str.lower().isin(_MISSING_TOKENS)
    t_nm = t[non_missing]
    if t_nm.empty:
        return pd.Series(False, index=s.index)
    vc = t_nm.value_counts(dropna=False)
    n = len(t_nm)
    thr = max(min_count, int(np.ceil(min_ratio * n)))
    rare_vals = set(v for v, c in vc.items() if c < thr)
    mask = t.isin(rare_vals)
    # Ensure we don't mark missing values as "rare_category" — they’re "missing"
    mask = mask & ~_missing_mask(s)
    return mask.reindex(s.index, fill_value=False)


def _text_length_outliers(s: pd.Series) -> pd.Series:
    """Use IQR on string length to catch extremes (typos, concatenations, truncations)."""
    t = s.astype(str)
    lens = t.str.len()
    if lens.count() < 8:
        return pd.Series(False, index=s.index)
    q1, q3 = lens.quantile(0.25), lens.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (lens < lo) | (lens > hi)
    # Exclude missing from length logic
    return (mask & ~_missing_mask(s)).fillna(False)


def detect_anomalies(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    """
    Return heuristic anomalies as a DataFrame with columns:
      row_id, column, issue, detail, severity, value, score, expected, rule

    Issues:
      - missing                  (high)
      - numeric_outlier          (high)
      - rare_category            (medium)
      - text_length_outlier      (low)
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])

    # max records per issue type per column (to keep outputs manageable)
    MAX_PER_COL = 20000

    for c in df.columns:
        s = df[c]

        # 1) Missing
        miss = _missing_mask(s)
        miss_idx = np.where(miss.values)[0][:MAX_PER_COL]
        for i in miss_idx:
            rows.append(dict(
                row_id=int(i), column=c, issue="missing", severity="high",
                value=None if pd.isna(s.iloc[i]) else s.iloc[i],
                detail="missing / placeholder detected",
                score=None, expected=None, rule="heuristic_missing"
            ))

        # 2) Numeric outliers
        if pd.api.types.is_numeric_dtype(s):
            mask = _numeric_outliers(s)
            idx = np.where(mask.values)[0][:MAX_PER_COL]
            for i in idx:
                rows.append(dict(
                    row_id=int(i), column=c, issue="numeric_outlier", severity="high",
                    value=s.iloc[i], detail="IsolationForest/IQR flagged",
                    score=None, expected=None, rule="heuristic_numeric"
                ))
        else:
            # 3) Rare categories
            rare = _rare_categories(s)
            idx = np.where(rare.values)[0][:MAX_PER_COL]
            for i in idx:
                rows.append(dict(
                    row_id=int(i), column=c, issue="rare_category", severity="medium",
                    value=str(s.iloc[i]), detail="Uncommon value vs column frequency",
                    score=None, expected=None, rule="heuristic_freq"
                ))

            # 4) Text length extremes
            tl = _text_length_outliers(s)
            idx = np.where(tl.values)[0][:MAX_PER_COL]
            for i in idx:
                rows.append(dict(
                    row_id=int(i), column=c, issue="text_length_outlier", severity="low",
                    value=str(s.iloc[i]), detail="Too short/long vs peers (IQR)",
                    score=None, expected=None, rule="heuristic_len"
                ))

    if not rows:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])
    rep = pd.DataFrame(rows)
    # Ordering: high → medium → low, then column, then row
    rep["severity"] = rep["severity"].astype(str).str.lower()
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    rep["__sev"] = rep["severity"].map(sev_rank).fillna(3)
    rep = rep.sort_values(["__sev","column","row_id"]).drop(columns="__sev").reset_index(drop=True)
    return rep


# ======================================================================
# ML anomaly detection (optional, for entire rows)
# ======================================================================

def _row_concat(df: pd.DataFrame) -> pd.Series:
    """Concatenate row values into a string (robust to None/NaN)."""
    return df.astype(str).fillna("").apply(lambda r: " | ".join(r.values), axis=1)


def _fit_and_score_model(model_name: str, X_train, X_score) -> np.ndarray:
    """Train a novelty model and return normalized anomaly scores (0=normal, 1=anomalous)."""
    if model_name == "IsolationForest":
        mdl = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
        mdl.fit(X_train)
        s = -mdl.decision_function(X_score)  # higher = more anomalous
    elif model_name == "OneClassSVM":
        mdl = OneClassSVM(gamma="scale", nu=0.05)
        mdl.fit(X_train)
        s = -mdl.decision_function(X_score)
    elif model_name == "LOF":
        # novelty=True allows calling score_samples on new data
        mdl = LocalOutlierFactor(n_neighbors=35, novelty=True)
        mdl.fit(X_train)
        s = -mdl.score_samples(X_score)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Normalize 0..1
    s = np.asarray(s, dtype=np.float64)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def ml_anomaly_report(
    df_unclean: pd.DataFrame,
    df_ref: Optional[pd.DataFrame],
    use_reference: bool,
    models: List[str],
    ensemble: bool = True,
) -> pd.DataFrame:
    """
    Train one or more novelty models on reference (if provided) or on unclean data itself,
    using TF-IDF over row-concatenated text. Return top-15% most anomalous rows.

    Output columns: row_id, column="__row__", issue="ml_anomaly",
                    detail, severity, value=None, score, expected=None, rule
    """
    if not _HAVE_SK:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])
    if not isinstance(df_unclean, pd.DataFrame) or df_unclean.empty:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])

    txt_unclean = _row_concat(df_unclean)
    X_train_text = txt_unclean
    if use_reference and isinstance(df_ref, pd.DataFrame) and not df_ref.empty:
        txt_ref = _row_concat(df_ref)
        X_train_text = txt_ref

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    vec.fit(X_train_text)
    X_score = vec.transform(txt_unclean)
    X_train = vec.transform(X_train_text)

    scores = []
    used = []
    for name in models:
        try:
            s = _fit_and_score_model(name, X_train, X_score)
            scores.append(s)
            used.append(name)
        except Exception:
            # Skip models that fail (e.g., numerical issues on tiny datasets)
            continue

    if not scores:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])

    if ensemble and len(scores) > 1:
        score = np.mean(scores, axis=0)
        rule_name = "ml_ensemble"
        detail_model = f"ensemble({','.join(used)})"
    else:
        score = scores[0]
        rule_name = f"ml_{used[0].lower()}"
        detail_model = used[0]

    # Top 15% anomalies
    thr = np.quantile(score, 0.85)
    idx = np.where(score >= thr)[0]
    if idx.size == 0:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])

    hi_thr = np.quantile(score, 0.95)
    def sev(x: float) -> str:
        return "high" if x >= hi_thr else "medium"

    rows = []
    for i in idx:
        rows.append(dict(
            row_id=int(i),
            column="__row__",
            value=None,
            issue="ml_anomaly",
            detail=f"{detail_model} score={float(score[i]):.3f}",
            severity=sev(float(score[i])),
            score=float(score[i]),
            expected=None,
            rule=rule_name
        ))
    return pd.DataFrame(rows)