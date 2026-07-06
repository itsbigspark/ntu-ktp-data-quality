# core/validator/anomaly.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional: sklearn for stronger numeric + ML anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import DBSCAN as _DBSCAN
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


def _principled_threshold(score: np.ndarray) -> tuple:
    """
    Find a principled anomaly threshold using a 2-component GMM.

    Strategy (from diagnostic results):
      1. Fit GaussianMixture(n_components=2) to the score distribution.
      2. If the two component means are separated by > 0.1, find the
         valley (antimode) between them — that is the natural decision
         boundary between the "normal" and "anomalous" clusters.
      3. If GMM fails or the gap is too small (unimodal), fall back to
         the 90th percentile (more conservative than the old 85th).

    Returns: (threshold_float, method_description_str)
    """
    if not _HAVE_SK or len(score) < 20:
        thr = float(np.quantile(score, 0.90))
        return thr, "fallback_quantile_90 (too few rows for GMM)"

    try:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300,
                              n_init=5)
        gmm.fit(score.reshape(-1, 1))
        means = gmm.means_.flatten()
        mu_low, mu_high = float(means.min()), float(means.max())

        if mu_high - mu_low < 0.15:
            # Components too close — no meaningful separation
            raise ValueError(
                f"GMM gap too small ({mu_high - mu_low:.3f}); "
                "distribution is effectively unimodal"
            )

        # Scan the valley between the two peaks to find the antimode
        xs         = np.linspace(mu_low, mu_high, 500).reshape(-1, 1)
        log_probs  = gmm.score_samples(xs)
        antimode   = float(xs[np.argmin(log_probs)])

        # Safety: antimode must lie strictly between the means
        antimode = float(np.clip(antimode, mu_low + 0.01, mu_high - 0.01))

        # Extra guard: never flag more than 20% or fewer than 1% of rows.
        # 20% cap: above this the precision lift degrades below useful levels
        # (empirically verified on TEST2/TEST3 diagnostic).
        pct_flagged = float((score >= antimode).mean())
        if pct_flagged > 0.20:
            antimode = float(np.quantile(score, 0.80))
            method = (
                f"GMM_antimode_capped (would flag {pct_flagged*100:.1f}%; "
                f"capped at 80th pct={antimode:.3f})"
            )
        elif pct_flagged < 0.01:
            antimode = float(np.quantile(score, 0.90))
            method = (
                f"GMM_antimode_raised (would flag {pct_flagged*100:.2f}%; "
                f"raised to 90th pct={antimode:.3f})"
            )
        else:
            method = (
                f"GMM_antimode μ_low={mu_low:.3f} μ_high={mu_high:.3f} "
                f"flags={pct_flagged*100:.1f}%"
            )

        return antimode, method

    except Exception as exc:
        thr = float(np.quantile(score, 0.90))
        return thr, f"fallback_quantile_90 (GMM error: {exc})"


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1]. Higher = more anomalous."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _score_component_tfidf_row(txt_train, txt_score) -> np.ndarray:
    """Component 1: row-level TF-IDF char n-grams → IsolationForest."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          max_features=600, sublinear_tf=True)
    vec.fit(txt_train)
    X_train = vec.transform(txt_train).toarray()
    X_score = vec.transform(txt_score).toarray()
    iso = IsolationForest(n_estimators=150, contamination="auto",
                          random_state=42, n_jobs=-1)
    iso.fit(X_train)
    # IsolationForest: lower decision_function = more anomalous → invert
    return _norm(-iso.decision_function(X_score))


def _score_component_tfidf_col(df: pd.DataFrame) -> np.ndarray:
    """Component 2: per-column TF-IDF → IsolationForest on concatenated features."""
    col_vecs = []
    for col in df.columns:
        vals = df[col].fillna("").astype(str).tolist()
        if len(set(vals)) < 3:
            continue
        try:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                  max_features=60, sublinear_tf=True)
            v = vec.fit_transform(vals).toarray()
            col_vecs.append(v)
        except Exception:
            continue
    if not col_vecs:
        return None
    X = np.hstack(col_vecs)
    iso = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=42, n_jobs=-1)
    iso.fit(X)
    return _norm(-iso.decision_function(X))


def _score_component_numeric(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Component 3: numeric columns → IsolationForest."""
    from sklearn.preprocessing import StandardScaler
    num_cols = []
    for col in df.columns:
        try:
            num = pd.to_numeric(
                df[col].astype(str).str.replace("£", "", regex=False)
                                   .str.replace(",", "", regex=False),
                errors="coerce",
            )
            if num.notna().sum() > max(10, len(df) * 0.05):
                num_cols.append(num.fillna(num.median()))
        except Exception:
            continue
    if not num_cols:
        return None
    X = StandardScaler().fit_transform(np.column_stack(num_cols))
    iso = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=42, n_jobs=-1)
    iso.fit(X)
    return _norm(-iso.decision_function(X))


def _score_component_lof_row(txt_train, txt_score) -> Optional[np.ndarray]:
    """LOF component: row-level TF-IDF char n-grams -> LocalOutlierFactor (novelty).

    Fits on the reference text and scores the unclean rows by local density
    deviation. Higher score = more anomalous.
    """
    try:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                              max_features=600, sublinear_tf=True)
        vec.fit(txt_train)
        X_train = vec.transform(txt_train).toarray()
        X_score = vec.transform(txt_score).toarray()
        n_neighbors = int(min(35, max(5, len(X_train) - 1)))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)
        # score_samples: higher = more normal -> invert so higher = anomalous
        return _norm(-lof.score_samples(X_score))
    except Exception as exc:
        logger.warning("lof_row component failed: %s", exc)
        return None


def _score_component_dbscan_row(txt_train, txt_score) -> Optional[np.ndarray]:
    """DBSCAN component: row-level TF-IDF char n-grams -> DBSCAN noise scoring.

    Fits DBSCAN on the reference text to find dense clusters. Scores each
    unclean row by its distance to the nearest core sample — noise points
    (label=-1) and points far from any core get high anomaly scores.
    """
    try:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                              max_features=600, sublinear_tf=True)
        vec.fit(txt_train)
        X_train = vec.transform(txt_train).toarray()
        X_score = vec.transform(txt_score).toarray()

        db = _DBSCAN(eps=0.3, min_samples=3, metric="cosine")
        db.fit(X_train)

        core_mask = np.zeros(len(X_train), dtype=bool)
        if hasattr(db, 'core_sample_indices_') and len(db.core_sample_indices_) > 0:
            core_mask[db.core_sample_indices_] = True
        else:
            logger.warning("dbscan_row: no core samples found")
            return None

        X_core = X_train[core_mask]
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(X_core)
        dists, _ = nn.kneighbors(X_score)
        scores = dists.flatten()
        return _norm(scores)
    except Exception as exc:
        logger.warning("dbscan_row component failed: %s", exc)
        return None


def ml_anomaly_report(
    df_unclean: pd.DataFrame,
    df_ref: Optional[pd.DataFrame],
    use_reference: bool,
    models: List[str],
    ensemble: bool = True,
) -> pd.DataFrame:
    """
    Three-component ensemble anomaly scorer with GMM-fitted threshold.

    Components:
      1. Row-level TF-IDF char n-gram → IsolationForest
      2. Per-column TF-IDF → IsolationForest on stacked features
      3. Numeric columns → StandardScaler → IsolationForest

    Threshold: GMM antimode (principled boundary between normal/anomalous
    clusters). Falls back to 90th-percentile if GMM finds no clear gap.

    Output columns: row_id, column="__row__", issue="ml_anomaly",
                    detail, severity, value=None, score, expected=None, rule
    """
    _EMPTY = pd.DataFrame(columns=[
        "row_id", "column", "issue", "detail",
        "severity", "value", "score", "expected", "rule",
    ])
    if not _HAVE_SK:
        return _EMPTY
    if not isinstance(df_unclean, pd.DataFrame) or df_unclean.empty:
        return _EMPTY

    # ── Row cap ───────────────────────────────────────────────────────────────
    ML_ROW_CAP = 20_000
    if len(df_unclean) > ML_ROW_CAP:
        df_unclean = df_unclean.sample(n=ML_ROW_CAP, random_state=42).reset_index(drop=True)
        logger.info("ml_anomaly_report: dataset sampled to %d rows", ML_ROW_CAP)

    # ── Reference text (for Component 1 training) ────────────────────────────
    txt_unclean = _row_concat(df_unclean)
    if use_reference and isinstance(df_ref, pd.DataFrame) and not df_ref.empty:
        if len(df_ref) > ML_ROW_CAP:
            df_ref = df_ref.sample(n=ML_ROW_CAP, random_state=42)
        txt_train = _row_concat(df_ref)
    else:
        txt_train = txt_unclean

    # ── Which algorithm families to include ──────────────────────────────────
    # `models` selects the ensemble members. Recognised names:
    #   "IsolationForest" -> 3 Isolation Forest components (row TF-IDF, column
    #                        TF-IDF, numeric)
    #   "LOF"             -> Local Outlier Factor on row TF-IDF
    #   "DBSCAN"          -> DBSCAN noise scoring on row TF-IDF
    # An empty/None list defaults to the full ensemble for backward compatibility.
    if isinstance(models, list) and len(models) > 0:
        selected = {str(m) for m in models}
    else:
        selected = {"IsolationForest", "LOF", "DBSCAN"}
    use_iforest = "IsolationForest" in selected
    use_lof     = "LOF" in selected
    use_dbscan  = "DBSCAN" in selected

    # ── Build component scores ────────────────────────────────────────────────
    component_scores = []
    component_names  = []

    if use_iforest:
        try:
            s1 = _score_component_tfidf_row(txt_train, txt_unclean)
            component_scores.append(s1)
            component_names.append("iforest_tfidf_row")
        except Exception as exc:
            logger.warning("ml_anomaly_report: iforest_tfidf_row failed: %s", exc)

        try:
            s2 = _score_component_tfidf_col(df_unclean)
            if s2 is not None:
                component_scores.append(s2)
                component_names.append("iforest_tfidf_col")
        except Exception as exc:
            logger.warning("ml_anomaly_report: iforest_tfidf_col failed: %s", exc)

        try:
            s3 = _score_component_numeric(df_unclean)
            if s3 is not None:
                component_scores.append(s3)
                component_names.append("iforest_numeric")
        except Exception as exc:
            logger.warning("ml_anomaly_report: iforest_numeric failed: %s", exc)

    if use_lof:
        s4 = _score_component_lof_row(txt_train, txt_unclean)
        if s4 is not None:
            component_scores.append(s4)
            component_names.append("lof_row")

    if use_dbscan:
        s5 = _score_component_dbscan_row(txt_train, txt_unclean)
        if s5 is not None:
            component_scores.append(s5)
            component_names.append("dbscan_row")

    if not component_scores:
        logger.warning("ml_anomaly_report: all components failed or none selected")
        return _EMPTY

    score       = np.mean(component_scores, axis=0)
    rule_name   = f"ml_ensemble_{len(component_scores)}c"
    detail_pfx  = f"ensemble({'+'.join(component_names)})"

    # ── Principled threshold (GMM antimode, Strategy 1) ──────────────────────
    thr, thr_method = _principled_threshold(score)
    logger.info("ml_anomaly_report: thr=%.4f method=%s n_components=%d",
                thr, thr_method, len(component_scores))

    idx = np.where(score >= thr)[0]
    if idx.size == 0:
        return _EMPTY

    # Confidence tiers: how far above threshold relative to max
    score_max = score.max()
    def sev(x: float) -> str:
        span = score_max - thr + 1e-9
        rel  = (x - thr) / span
        if rel >= 0.6:
            return "high"
        if rel >= 0.25:
            return "medium"
        return "low"

    rows = []
    for i in idx:
        s_val = float(score[i])
        rows.append(dict(
            row_id=int(i),
            column="__row__",
            value=None,
            issue="ml_anomaly",
            detail=f"{detail_pfx} score={s_val:.3f} thr={thr:.3f} [{thr_method}]",
            severity=sev(s_val),
            score=s_val,
            expected=None,
            rule=rule_name,
        ))
    return pd.DataFrame(rows)