# core/validator/discover.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import re
import math
import pandas as pd
from collections import Counter
from datetime import datetime

# ----------------- Tunables (lightweight defaults) -----------------
MAJORITY_COVERAGE = 0.80
MIN_SAMPLES_FOR_REGEX = 1
LOW_CARDINALITY_MAX_ABS = 25
UNIQUE_RATIO_THR = 0.99
MIN_REGEX_COVERAGE = 0.05
MAX_REGEX_PATTERNS = 100

# Presence inference: a column populated in at least this fraction of rows is
# treated as "required" (missing values are errors); otherwise "nullable".
PRESENCE_REQUIRED_THRESHOLD = 0.85
# Categorical inference: values appearing fewer than this many times are treated
# as likely errors and excluded from the inferred allowed-value set (weak
# supervision heuristic — see tests/ENGINE_FIXES_AND_TESTS.md).
MIN_ALLOWED_VALUE_COUNT = 2

# Placeholder tokens that count as "missing" for presence inference (kept in
# sync with core/validator/validate.py MISSING_TOKENS).
_MISSING_TOKENS = {
    "", "missing", "null", "none", "na", "n/a", "n.a.", "-", "--",
    "nan", "nil", "unknown",
}

def _present_rate(s_full: pd.Series) -> float:
    """Fraction of rows that hold a real (non-missing, non-placeholder) value."""
    if s_full is None or len(s_full) == 0:
        return 0.0
    def _present(v) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
        return str(v).strip().lower() not in _MISSING_TOKENS
    return float(s_full.map(_present).mean())

# Common date formats to try (expand as needed)
COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%m-%Y", "%m-%d-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y", "%Y.%m.%d",
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
]

# ----------------- Helpers -----------------
def _non_null_str_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=object)
    return s.dropna().astype(str).map(lambda x: x.strip())

def _is_email_like_series(s: pd.Series) -> bool:
    s = _non_null_str_series(s).head(2000)
    if s.empty:
        return False
    pat = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    return (s.map(lambda x: bool(pat.fullmatch(x))).mean() > 0.6)

def _is_uk_postcode_like_series(s: pd.Series) -> bool:
    s = _non_null_str_series(s).str.upper().head(2000)
    if s.empty:
        return False
    pat = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$")
    return (s.map(lambda x: bool(pat.fullmatch(x.replace(' ', '')))).mean() > 0.6)

def _looks_numeric_series(s: pd.Series) -> bool:
    s = _non_null_str_series(s).head(1000)
    if s.empty: return False
    ok = 0
    for v in s:
        try:
            float(str(v).replace(",", ""))
            ok += 1
        except Exception:
            pass
    # Strict: only treat a column as numeric when almost every value parses.
    # Alphanumeric identifiers (e.g. 'SC123456') must not become numeric.
    return ok / max(len(s), 1) > 0.99

def _try_numeric_bounds(s: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    s = _non_null_str_series(s)
    vals = []
    for v in s:
        try:
            vals.append(float(str(v).replace(",", "")))
        except Exception:
            continue
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))

def _is_date_series(s: pd.Series) -> bool:
    s = _non_null_str_series(s).head(1000)
    if s.empty: return False
    hits = 0
    for v in s:
        if _try_parse_date_any(v) is not None:
            hits += 1
    return hits / max(len(s), 1) > 0.6

def _try_parse_date_any(v: str) -> Optional[datetime]:
    for fmt in COMMON_DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return dt.to_pydatetime()
    except Exception:
        pass
    return None

def _date_bounds_iso(s: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    s = _non_null_str_series(s)
    dts = []
    for v in s:
        dt = _try_parse_date_any(v)
        if dt: dts.append(dt)
    if not dts:
        return None, None
    return min(dts).strftime("%Y-%m-%d"), max(dts).strftime("%Y-%m-%d")

def _infer_date_formats(s: pd.Series, max_formats: int = 4) -> List[str]:
    s = _non_null_str_series(s).head(2000)
    counts: Counter = Counter()
    for v in s:
        for fmt in COMMON_DATE_FORMATS:
            try:
                datetime.strptime(v, fmt)
                counts[fmt] += 1
                break
            except Exception:
                continue
    fmts = [fmt for fmt, _ in counts.most_common(max_formats)]
    return fmts

# --- Regex compression helpers ---
def _char_class(ch: str) -> str:
    if ch.isdigit():
        return r"\d"
    if ch.isalpha():
        return r"[A-Za-z]"
    if ch.isspace():
        return r"\s"
    return r"[^A-Za-z0-9\s]"

def _compress_to_regex(s: str) -> str:
    if not s:
        return r"^$"
    classes: List[str] = []
    for ch in s:
        classes.append(_char_class(ch))

    pieces: List[str] = []
    i = 0
    while i < len(classes):
        c = classes[i]
        j = i
        while j + 1 < len(classes) and classes[j + 1] == c:
            j += 1
        run_len = j - i + 1
        if run_len == 1:
            pieces.append(c)
        else:
            pieces.append(f"{c}{{{run_len}}}")
        i = j + 1

    pat = "".join(pieces)
    return f"^{pat}$"

def _infer_regexes_from_series(s: pd.Series,
                               min_coverage: float = MIN_REGEX_COVERAGE,
                               max_patterns: int = MAX_REGEX_PATTERNS) -> Tuple[List[str], List[Tuple[str, float]]]:
    s = _non_null_str_series(s)
    if len(s) < MIN_SAMPLES_FOR_REGEX:
        return [], []
    pats = [_compress_to_regex(v) for v in s]
    cnt = Counter(pats)
    total = max(len(s), 1)
    covered = [(p, c / total) for p, c in cnt.most_common()]
    kept = [(p, cov) for p, cov in covered if cov >= min_coverage][:max_patterns]
    return [p for p, _ in kept], kept

# ----------------- Public API: rule discovery -----------------
def infer_rules_from_unclean(df: pd.DataFrame, prune: bool = True) -> Dict[str, Any]:
    rules: Dict[str, Any] = {"columns": {}, "uniqueness": {}, "cross_field": []}
    for c in df.columns:
        s_full = df[c]
        s = _non_null_str_series(s_full)
        col_rule: Dict[str, Any] = {}

        # Presence inference (P2): mark every column required or nullable so the
        # validator checks missing values consistently, not only on columns that
        # happened to receive another rule.
        if _present_rate(s_full) >= PRESENCE_REQUIRED_THRESHOLD:
            col_rule["required"] = True
        else:
            col_rule["nullable"] = True

        if s.empty:
            rules["columns"][c] = col_rule
            continue

        # Present (non-missing) values only, so placeholder tokens do not pollute
        # type/format/value inference (e.g. an empty-string '^$' regex pattern).
        s_present = s[~s.str.lower().isin(_MISSING_TOKENS)]
        if s_present.empty:
            rules["columns"][c] = col_rule
            continue

        # Numeric
        if _looks_numeric_series(s_present):
            mn, mx = _try_numeric_bounds(s_present)
            if mn is not None and mx is not None:
                col_rule["type"] = "number"
                col_rule["min"], col_rule["max"] = mn, mx
                rules["columns"][c] = col_rule
                continue

        # Date
        if _is_date_series(s_present):
            col_rule["type"] = "date"
            dmin, dmax = _date_bounds_iso(s_present)
            if dmin: col_rule["min_date"] = dmin
            if dmax: col_rule["max_date"] = dmax
            fmts = _infer_date_formats(s_present)
            if fmts:
                col_rule["date_format"] = fmts
            rules["columns"][c] = col_rule
            continue

        # Uniqueness — only for text columns that are almost entirely distinct
        # (identifier-like). Numeric/date columns returned above, so merely
        # mostly-distinct dates are never treated as unique.
        uniq_ratio = s_present.nunique(dropna=True) / max(len(s_present), 1)
        if uniq_ratio >= UNIQUE_RATIO_THR:
            rules["uniqueness"][c] = True

        # Low-cardinality allowed_values (P3): infer the allowed set only from
        # values that appear at least MIN_ALLOWED_VALUE_COUNT times, so rare
        # erroneous values (e.g. a stray 'gbp' among 'GBP') are not mistaken for
        # a valid category and therefore remain detectable.
        nunique_non_null = s_present.nunique(dropna=True)
        if nunique_non_null <= LOW_CARDINALITY_MAX_ABS:
            vc = s_present.value_counts()
            vals = [v for v in vc.index.tolist() if int(vc[v]) >= MIN_ALLOWED_VALUE_COUNT]
            if vals:
                col_rule["allowed_values"] = vals[:200]
                total = max(int(vc.sum()), 1)
                col_rule["_allowed_values_meta"] = [
                    {"value": str(v), "count": int(vc[v]), "coverage": float(vc[v]/total)}
                    for v in vals[:200]
                ]

        # Regex patterns — only when the column is NOT already captured as a
        # categorical. Redundant regex on categoricals only adds false-positive
        # risk; over-fit regex on free-text is pruned below.
        if "allowed_values" not in col_rule:
            patterns, meta = _infer_regexes_from_series(s_present)
            if patterns:
                col_rule["regex"] = patterns
                col_rule["_regex_meta"] = [{"pattern": p, "coverage": cov} for p, cov in meta]

        rules["columns"][c] = col_rule

    # Quality pruning (P1): drop over-fit regex on free-text columns (union
    # coverage below threshold) so names/addresses/emails are not mass-flagged.
    # Numeric-bound tightening is disabled so inferred min/max (which catch
    # out-of-range values) are preserved.
    if prune:
        rules = prune_rules_by_quality(
            df, rules,
            RuleQualityConfig(
                numeric_bounds_quantiles=(0.0, 1.0),  # keep inferred min/max
                min_regex_overall_hit=0.90,           # drop regex on free-text
                min_regex_coverage=0.0,               # once kept, retain all valid
                                                      # formats (don't flag the rare
                                                      # but legitimate ID/postcode
                                                      # patterns)
            ),
        )

    return rules

def rules_from_reference(df_ref: pd.DataFrame) -> Dict[str, Any]:
    rules: Dict[str, Any] = {"columns": {}, "uniqueness": {}, "cross_field": []}
    for c in df_ref.columns:
        s_full = df_ref[c]
        s = _non_null_str_series(s_full)
        col_rule: Dict[str, Any] = {}
        if s.empty:
            rules["columns"][c] = col_rule
            continue

        nunq = s_full.nunique(dropna=True)
        uniq_ratio = nunq / max(len(s_full), 1)
        if uniq_ratio >= UNIQUE_RATIO_THR:
            rules["uniqueness"][c] = True

        if _looks_numeric_series(s):
            mn, mx = _try_numeric_bounds(s)
            if mn is not None and mx is not None:
                col_rule["type"] = "number"
                col_rule["min"], col_rule["max"] = mn, mx

        if _is_date_series(s_full):
            col_rule["type"] = "date"
            dmin, dmax = _date_bounds_iso(s_full)
            if dmin: col_rule["min_date"] = dmin
            if dmax: col_rule["max_date"] = dmax
            fmts = _infer_date_formats(s_full)
            if fmts:
                col_rule["date_format"] = fmts

        nunique_non_null = s.nunique(dropna=True)
        if nunique_non_null <= LOW_CARDINALITY_MAX_ABS:
            vc = s.value_counts()
            vals = vc.index.tolist()
            col_rule["allowed_values"] = vals[:200]
            total = max(int(vc.sum()), 1)
            col_rule["_allowed_values_meta"] = [
                {"value": str(v), "count": int(vc[v]), "coverage": float(vc[v]/total)}
                for v in vals[:200]
            ]

        patterns, meta = _infer_regexes_from_series(s)
        if patterns:
            col_rule["regex"] = patterns
            col_rule["_regex_meta"] = [{"pattern": p, "coverage": cov} for p, cov in meta]

        rules["columns"][c] = col_rule
    return rules

def merge_rules(json_rules: Optional[dict],
                ref_rules: Optional[dict],
                inferred_rules: Optional[dict]) -> dict:
    merged: Dict[str, Any] = {"columns": {}, "uniqueness": {}, "cross_field": []}
    sources = [r for r in [inferred_rules, ref_rules, json_rules] if r]

    for src in sources:
        for c, rule in (src.get("columns") or {}).items():
            base = merged["columns"].get(c, {})
            base.update(rule or {})
            merged["columns"][c] = base

        for c, need_u in (src.get("uniqueness") or {}).items():
            if need_u:
                merged["uniqueness"][c] = True

        for rule in (src.get("cross_field") or []):
            merged.setdefault("cross_field", [])
            key = str(rule)
            if key not in map(str, merged["cross_field"]):
                merged["cross_field"].append(rule)

    return merged

# ----------------- Quality pruning (to avoid noisy rules in Auto Mode) -----------------
@dataclass
class RuleQualityConfig:
    min_regex_coverage: float = 0.20
    max_regex_patterns: int = 8
    min_regex_overall_hit: float = 0.65
    min_allowed_coverage: float = 0.70
    min_date_parse_rate: float = 0.70
    numeric_bounds_quantiles: Tuple[float, float] = (0.01, 0.99)
    max_dup_rate_for_unique: float = 0.05

def prune_rules_by_quality(df: pd.DataFrame, rules: Dict[str, Any], cfg: RuleQualityConfig) -> Dict[str, Any]:
    """
    Return a copy of rules with weak/low-coverage inferences removed or tightened.
    """
    out = {"columns": {}, "uniqueness": {}, "cross_field": rules.get("cross_field", [])}
    n = len(df)

    for c, r in (rules.get("columns") or {}).items():
        s = df[c] if c in df.columns else pd.Series([], dtype=object)
        s_non = _non_null_str_series(s)
        total_non = max(1, len(s_non))
        new_r = dict(r)

        # --- Uniqueness ---
        if c in (rules.get("uniqueness") or {}):
            nunq = s.nunique(dropna=True)
            dup_rate = 1.0 - nunq / max(len(s), 1) if len(s) else 1.0
            if dup_rate <= cfg.max_dup_rate_for_unique:
                out["uniqueness"][c] = True  # keep
            # else: drop uniqueness

        # --- Date quality ---
        if new_r.get("type") == "date":
            parse_hits = 0
            for v in s_non.head(2000):
                if _try_parse_date_any(v) is not None:
                    parse_hits += 1
            parse_rate = parse_hits / max(1, min(2000, total_non))
            if parse_rate < cfg.min_date_parse_rate:
                # Drop 'type: date' if we can't parse reliably
                new_r.pop("type", None)
                new_r.pop("date_format", None)
                new_r.pop("min_date", None)
                new_r.pop("max_date", None)

        # --- Numeric bounds: tighten to robust quantiles if present ---
        if new_r.get("type") == "number" and total_non > 0:
            try:
                vals = pd.to_numeric(s_non.str.replace(",", "", regex=False), errors="coerce").dropna()
                if not vals.empty:
                    lo_q, hi_q = cfg.numeric_bounds_quantiles
                    ql = float(vals.quantile(lo_q))
                    qh = float(vals.quantile(hi_q))
                    new_r["min"] = ql
                    new_r["max"] = qh
            except Exception:
                pass

        # --- Allowed values: ensure meaningful coverage ---
        if isinstance(new_r.get("allowed_values"), list):
            vals = new_r["allowed_values"]
            vc = s_non.value_counts()
            covered = float(sum(vc.get(v, 0) for v in vals)) / max(1, int(vc.sum()))
            if covered < cfg.min_allowed_coverage:
                new_r.pop("allowed_values", None)
                new_r.pop("_allowed_values_meta", None)

        # --- Regex patterns: drop tiny/too many; ensure union hit-rate ---
        if isinstance(new_r.get("regex"), list) and new_r["regex"]:
            pats = new_r["regex"][: cfg.max_regex_patterns]
            # union coverage
            def _hit_any(v: str) -> bool:
                for p in pats:
                    try:
                        if re.fullmatch(p.strip("^$"), v) or re.fullmatch(p, v):
                            return True
                    except re.error:
                        continue
                return False
            hits = int(s_non.map(_hit_any).sum()) if total_non > 0 else 0
            union_rate = hits / total_non
            if union_rate < cfg.min_regex_overall_hit:
                new_r.pop("regex", None)
                new_r.pop("_regex_meta", None)
            else:
                # also drop patterns whose individual coverage < min_regex_coverage
                if "_regex_meta" in new_r:
                    meta = [m for m in new_r["_regex_meta"] if m.get("coverage", 0) >= cfg.min_regex_coverage]
                    pats_keep = [m["pattern"] for m in meta][: cfg.max_regex_patterns]
                    if pats_keep:
                        new_r["regex"] = pats_keep
                        new_r["_regex_meta"] = meta[: cfg.max_regex_patterns]
                    else:
                        new_r.pop("regex", None)
                        new_r.pop("_regex_meta", None)

        out["columns"][c] = new_r

    return out