# core/reporting.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def _pct(x: float) -> float:
    return float(max(0.0, min(1.0, x))) * 100.0


def _count_issues(rep: pd.DataFrame, issues: List[str]) -> int:
    if rep is None or rep.empty or "issue" not in rep.columns:
        return 0
    return int(rep["issue"].astype(str).isin(issues).sum())


def _rows_impacted(rep: pd.DataFrame) -> int:
    if rep is None or rep.empty or "row_id" not in rep.columns:
        return 0
    return len(set(pd.to_numeric(rep["row_id"], errors="coerce").dropna().astype(int).tolist()))


def compute_quality_scores(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    validation_report: Optional[pd.DataFrame],
    pairs_scored: Optional[pd.DataFrame],
    date_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute 6 dimension (0–100) scores for 'before' and 'after' snapshots.
    Heuristics (simple, explainable). You can refine as needed.

    Dimensions:
      - Accuracy: 1 - (regex/minmax/allowed/date violations) / (total cells)
      - Completeness: 1 - missing_fraction
      - Consistency: 1 - type/date parse errors / (total cells)
      - Validity: 1 - invalid set membership / (total cells)
      - Uniqueness: 1 - (rows participating in duplicate pairs) / N
      - Timeliness: share of parseable dates within last 365 days (or non-future), averaged across date cols
    """

    def _score_snapshot(df: pd.DataFrame, rep: Optional[pd.DataFrame]) -> Dict[str, float]:
        if df is None or len(df) == 0:
            total_cells = 1
            missing_frac = 1.0
        else:
            total_cells = max(1, int(df.shape[0] * df.shape[1]))
            missing_frac = float(pd.isna(df).sum().sum()) / float(total_cells)

        # Issues (from validation_report)
        acc_issues = _count_issues(rep, ["regex mismatch", "<min", ">max", "invalid date"])
        val_issues = _count_issues(rep, ["not in allowed set"])
        type_issues = _count_issues(rep, ["not number", "invalid date"])

        accuracy = _pct(1.0 - (acc_issues / total_cells))
        completeness = _pct(1.0 - missing_frac)
        consistency = _pct(1.0 - (type_issues / total_cells))
        validity = _pct(1.0 - (val_issues / total_cells))

        # Timeliness
        timeliness_values: List[float] = []
        if date_columns:
            now = pd.Timestamp(datetime.utcnow().date())
            for c in date_columns:
                if c not in df.columns:
                    continue
                s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
                ok = s.notna()
                if ok.any():
                    within = ((now - s) <= pd.Timedelta(days=365)) & (s <= now)
                    timeliness_values.append(float(within.fillna(False).mean()))
        timeliness = _pct(np.mean(timeliness_values)) if timeliness_values else 100.0  # if no dates, assume timely

        return dict(accuracy=accuracy, completeness=completeness, consistency=consistency, validity=validity, timeliness=timeliness)

    # Uniqueness from dedupe pairs (approximate)
    def _uniqueness(df: pd.DataFrame, pairs: Optional[pd.DataFrame]) -> float:
        if df is None or len(df) == 0:
            return 0.0
        if pairs is None or pairs.empty or "label" not in pairs.columns:
            return 1.0  # no duplicates found
        dup_pairs = pairs[pairs["label"] == "duplicate"]
        if dup_pairs.empty:
            return 1.0
        rows_flagged = set()
        for _, r in dup_pairs.iterrows():
            try:
                rows_flagged.add(int(r["row_i"]))
                rows_flagged.add(int(r["row_j"]))
            except Exception:
                continue
        dup_rows = len(rows_flagged)
        return max(0.0, 1.0 - (dup_rows / max(1, len(df))))

    before = _score_snapshot(df_before, validation_report)
    after = _score_snapshot(df_after, validation_report)  # using same report; you can recompute if you want
    before["uniqueness"] = _pct(_uniqueness(df_before, pairs_scored))
    after["uniqueness"]  = _pct(_uniqueness(df_after, pairs_scored))

    return {"before": before, "after": after}