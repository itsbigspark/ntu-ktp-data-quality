# core/auto_pipeline.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Core imports from your project ---
from core.validator.discover import (
    infer_rules_from_unclean,
    rules_from_reference,
    merge_rules,
    RuleQualityConfig,
    prune_rules_by_quality,
)
from core.validator.rules_schema import normalize_rules
from core.validator.validate import validate_df  # (Expert Mode does fixes; Auto just annotates)
from core.preprocess import basic_clean
from core.trimming import apply_trimming, EXCLUDE_ALWAYS
from core.vectorizers import build_vectorizers
from core.matcher import compute_pair_scores, label_pairs


@dataclass
class AutoConfig:
    # Cleaning
    lowercase: bool = True
    strip_ws: bool = True
    normalize_punct: bool = True

    # Trimming
    protect_cols: Optional[List[str]] = None  # columns never dropped
    write_kept_path: str = "eda_selection_output/csv/final_kept_columns.csv"

    # Dedupe controls (user-configurable)
    comparison_mode: str = "Column-wise (weighted)"  # or "Row-concatenated"
    dedupe_cols: Optional[List[str]] = None          # user-chosen columns subset
    col_weights: Optional[Dict[str, float]] = None   # user-specified weights for column-wise
    use_char: bool = True
    use_word: bool = False
    char_range: Tuple[int, int] = (3, 5)
    word_range: Tuple[int, int] = (1, 2)
    svd_components: int = 0  # 0 disables SVD
    run_mode: str = "NxN (full)"  # "NxN (full)" | "Fast KNN" | "Blocking" | "Blocking + KNN"
    topk: int = 20
    block_cols: Optional[List[str]] = None
    key_cols: Optional[List[str]] = None
    key_alpha: float = 0.4
    similar_thr: float = 0.60
    duplicate_thr: float = 0.80
    max_rows_for_dedupe: int = 5000  # head(N) for performance

    # Side-by-side artifact
    side_by_side_limit: int = 200  # first N pairs expanded to columns

    # Artifacts
    out_dir: str = "auto_artifacts"


# ------------------------
# Helpers
# ------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path):
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def _save_json(obj: Dict[str, Any], path: Path):
    _ensure_dir(path.parent)
    import json
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _annotate_has_issue(df: pd.DataFrame, validation_report: pd.DataFrame) -> pd.DataFrame:
    """Create a __has_issue flag per row based on validation_report row_id."""
    out = df.copy()
    if validation_report is None or validation_report.empty or "row_id" not in validation_report.columns:
        out["__has_issue"] = False
        return out
    bad = set(pd.to_numeric(validation_report["row_id"], errors="coerce").dropna().astype(int).tolist())
    out["__has_issue"] = out.index.to_series().isin(bad)
    return out


def _safe_get_row(df_: pd.DataFrame, idx: Any):
    """Try label access, then positional access; return a Series or None."""
    try:
        if idx in df_.index:
            return df_.loc[idx]
    except Exception:
        pass
    try:
        ii = int(idx)
        if 0 <= ii < len(df_):
            return df_.iloc[ii]
    except Exception:
        pass
    return None


def _build_side_by_side(df_src: pd.DataFrame, pairs_df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Expand the top N pairs into side-by-side columns for quick review."""
    rows = []
    use_cols = [c for c in df_src.columns if not str(c).startswith("__")]
    for _, r in pairs_df.head(limit).iterrows():
        i_raw, j_raw = r.get("row_i"), r.get("row_j")
        row_i = _safe_get_row(df_src, i_raw)
        row_j = _safe_get_row(df_src, j_raw)
        if row_i is None or row_j is None:
            continue
        rec = {"row_i": i_raw, "row_j": j_raw, "score": r.get("score"), "label": r.get("label")}
        for c in use_cols:
            rec[f"{c}__i"] = row_i.get(c, None)
            rec[f"{c}__j"] = row_j.get(c, None)
        rows.append(rec)
    return pd.DataFrame(rows)


def _normalize_weights(cols: List[str], weights: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Keep only weights for present cols, set missing to 0, renormalize to sum=1.
    Returns None if no valid positive weights found.
    """
    if not cols:
        return None
    if not isinstance(weights, dict) or len(weights) == 0:
        return None
    w = {c: float(weights.get(c, 0.0)) for c in cols}
    total = sum(v for v in w.values() if v > 0)
    if total <= 0:
        return None
    return {c: (w[c] / total if w[c] > 0 else 0.0) for c in cols}


# ------------------------
# Main pipeline
# ------------------------
def run_auto_pipeline(
    df_raw: pd.DataFrame,
    df_ref: Optional[pd.DataFrame] = None,
    rules_json: Optional[Dict[str, Any]] = None,
    config: Optional[AutoConfig] = None
) -> Dict[str, Any]:
    """
    Run the full pipeline with safe defaults:
      1) Infer + merge rules
      2) Validate (rule-only)
      3) Clean text
      4) EDA trim (kept columns saved)
      5) Dedupe scoring + labels (+ side-by-side expansion)
      6) Package artifacts
    """
    cfg = config or AutoConfig()

    # ------------------------
    # 1) Rules (infer + merge + prune)
    # ------------------------
    inferred = infer_rules_from_unclean(df_raw)
    ref_rules = rules_from_reference(df_ref) if isinstance(df_ref, pd.DataFrame) else {"columns": {}}
    merged = merge_rules(rules_json, ref_rules, inferred)
    merged = normalize_rules(merged)

    # Prune weak/noisy inferences to keep Auto Mode robust
    quality_cfg = RuleQualityConfig(
        min_regex_coverage=0.20,       # drop tiny regex shards
        max_regex_patterns=8,          # cap patterns
        min_regex_overall_hit=0.65,    # union coverage must be decent
        min_allowed_coverage=0.70,     # allowed_values must cover most values
        min_date_parse_rate=0.70,      # only keep date type if parses reliably
        numeric_bounds_quantiles=(0.01, 0.99),  # robust bounds
        max_dup_rate_for_unique=0.05,  # ≤5% dupes to mark unique
    )
    try:
        merged = prune_rules_by_quality(df_raw, merged, quality_cfg)
        merged = normalize_rules(merged)
    except Exception:
        # If pruning helpers are not present, proceed with merged rules as-is
        pass

    # Save merged rules
    out_dir = Path(cfg.out_dir)
    _save_json(merged, out_dir / "rules_merged.json")

    # ------------------------
    # 2) Validate (rule-only) + annotate
    # ------------------------
    validation_report = validate_df(df_raw, merged)
    _save_csv(validation_report, out_dir / "validation_report.csv")

    df_validated = _annotate_has_issue(df_raw, validation_report)
    _save_csv(df_validated, out_dir / "validated_dataset.csv")

    # ------------------------
    # 3) Clean text
    # ------------------------
    df_clean = basic_clean(
        df_validated,
        lowercase=cfg.lowercase,
        strip_ws=cfg.strip_ws,
        normalize_punct=cfg.normalize_punct,
    )
    _save_csv(df_clean, out_dir / "cleaned_dataset.csv")

    # ------------------------
    # 4) Trim (kept columns)
    # ------------------------
    trimmed_df, kept_cols = apply_trimming(df_clean, protect=(cfg.protect_cols or []))
    _ensure_dir(Path("eda_selection_output/csv"))
    pd.Series(kept_cols).to_csv(Path(cfg.write_kept_path), index=False, header=False)
    _save_csv(trimmed_df, out_dir / "trimmed_dataset.csv")

    # ------------------------
    # 5) Dedupe (on trimmed by default)
    # ------------------------
    base = trimmed_df.copy()

    # choose columns for dedupe (exclude helpers and any __-prefixed)
    available_cols = [c for c in base.columns if c not in EXCLUDE_ALWAYS and not str(c).startswith("__")]

    # apply user-specified dedupe column subset if given (intersect with available)
    if cfg.dedupe_cols:
        cols = [c for c in cfg.dedupe_cols if c in available_cols]
        if not cols:
            cols = available_cols  # fallback if user passed none that exist
    else:
        cols = available_cols

    base = base[cols].copy()

    # row cap
    base = base.head(cfg.max_rows_for_dedupe).reset_index(drop=True).copy()

    # vectorizers
    vec_cfg = {
        "use_char": cfg.use_char,
        "use_word": cfg.use_word,
        "char_range": cfg.char_range,
        "word_range": cfg.word_range,
        "svd_components": cfg.svd_components,
    }
    vec_pack = build_vectorizers(base, cols, vec_cfg)

    # If no features, short-circuit
    if vec_pack.get("dim", 0) == 0:
        pairs_scored = pd.DataFrame(columns=["row_i", "row_j", "score", "label"])
        side_df = pd.DataFrame()
    else:
        # weights logic (only for column-wise)
        weights = None
        if cfg.comparison_mode.startswith("Column-wise"):
            # Try user weights first; if not valid, fall back to equal weights
            weights = _normalize_weights(cols, cfg.col_weights)
            if weights is None and len(cols) > 0:
                weights = {c: 1.0 / len(cols) for c in cols}

        scores_df = compute_pair_scores(
            df=base,
            vec_pack=vec_pack,
            comparison_mode=cfg.comparison_mode,
            weights=weights,            # None for row-concatenated; normalized/equal for column-wise
            run_mode=cfg.run_mode,
            topk=cfg.topk,
            block_cols=(cfg.block_cols or []),
            key_cols=(cfg.key_cols or []),
            key_alpha=cfg.key_alpha,
        )
        pairs_scored = label_pairs(scores_df, similar_thr=cfg.similar_thr, duplicate_thr=cfg.duplicate_thr)
        side_df = _build_side_by_side(base, pairs_scored, limit=cfg.side_by_side_limit)

    # Save dedupe outputs
    _save_csv(pairs_scored, out_dir / "pairs_scored.csv")
    _save_csv(side_df,      out_dir / "pairs_side_by_side.csv")

    # ------------------------
    # 6) Package & return
    # ------------------------
    return {
        "config": asdict(cfg),
        "rules_merged": merged,
        "validation_report": validation_report,
        "df_validated": df_validated,
        "df_clean": df_clean,
        "kept_columns": kept_cols,
        "df_trimmed": trimmed_df,
        "pairs_scored": pairs_scored,
        "pairs_side_by_side": side_df,
        "artifacts": {
            "rules_merged": str(out_dir / "rules_merged.json"),
            "validation_report": str(out_dir / "validation_report.csv"),
            "validated_dataset": str(out_dir / "validated_dataset.csv"),
            "cleaned_dataset": str(out_dir / "cleaned_dataset.csv"),
            "trimmed_dataset": str(out_dir / "trimmed_dataset.csv"),
            "kept_columns_file": str(Path(cfg.write_kept_path)),
            "pairs_scored": str(out_dir / "pairs_scored.csv"),
            "pairs_side_by_side": str(out_dir / "pairs_side_by_side.csv"),
        },
    }