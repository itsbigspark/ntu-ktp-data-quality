# core/validator/validate.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

# -----------------------------
# Format validators import (generic / data-agnostic only)
# -----------------------------
try:
    from .format_validators import (
        validate_email,
        validate_date,
        validate_url
    )
    FORMAT_VALIDATORS_AVAILABLE = True
except Exception:
    FORMAT_VALIDATORS_AVAILABLE = False

# -----------------------------
# Typo detector import
# -----------------------------
try:
    from .typo_detector import detect_typos_all_columns
    TYPO_DETECTOR_AVAILABLE = True
except Exception:
    TYPO_DETECTOR_AVAILABLE = False

# -----------------------------
# Optional anomaly imports
# -----------------------------
try:
    from .anomaly import detect_anomalies, ml_anomaly_report  # ml_anomaly_report added
except Exception:
    def detect_anomalies(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","expected","rule"])
    def ml_anomaly_report(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","score","expected","rule"])


# Tokens considered "missing"
MISSING_TOKENS = {
    "", "missing", "null", "none", "na", "n/a", "n.a.", "-", "--", "nan", "nil", "unknown"
}


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (small, dependency-free)."""
    a, b = str(a), str(b)
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]

# Column type detection patterns (data-agnostic only)
# Only generic formats that apply to ANY dataset regardless of domain.
# Domain-specific formats (UK postcode, UK company number, phone) are NOT
# auto-detected — they must be explicitly specified via JSON rules.
COLUMN_TYPE_PATTERNS = {
    'email': ['email', 'e-mail', 'e_mail'],
    'date': ['date', 'created', 'updated', 'modified', 'birth', 'dob', 'time'],
    'url': ['url', 'website', 'link', 'uri', 'web', 'homepage'],
}

# Human-readable descriptions for every issue type the pipeline can produce.
# These are deterministic, require no AI model, and are always consistent.
ISSUE_DESCRIPTIONS = {
    # ── Rule-based checks (validate_df / _check_value) ──
    "missing":
        "Value is empty, null, or a placeholder (e.g. 'N/A', 'none', '-'). "
        "This field requires a real value.",
    "format_error":
        "Value does not match the expected format for this column type. "
        "For example, an email column expects 'user@domain.com', "
        "a date column expects a parseable date string.",
    "not number":
        "This column expects numeric values, but this cell contains text or "
        "a non-numeric string that cannot be converted to a number.",
    "<min":
        "Numeric value is below the minimum allowed threshold defined in the "
        "validation rules. The value may be a data entry error or an outlier.",
    ">max":
        "Numeric value exceeds the maximum allowed threshold defined in the "
        "validation rules. The value may be a data entry error or an outlier.",
    "invalid date":
        "Value is expected to be a date but cannot be parsed as one. "
        "Common causes: wrong format, impossible date (e.g. month 13), "
        "or non-date text in a date field.",
    "not in allowed set":
        "Value is not in the list of allowed/expected values for this column. "
        "This may indicate a typo, an outdated category, or a data entry error.",
    "regex mismatch":
        "Value does not match the expected pattern (regular expression) for "
        "this column. For example, an ID column expecting 'C001' format "
        "received a value that does not follow that structure.",
    "duplicate_value":
        "This value appears more than once in a column that is expected to "
        "contain unique values. The first occurrence is kept; subsequent "
        "duplicates are flagged.",

    # ── Heuristic anomaly detection (detect_anomalies) ──
    "numeric_outlier":
        "This numeric value is statistically unusual compared to the rest of "
        "the column. Detected using Isolation Forest (ML) or the IQR method. "
        "It may be a genuine extreme value or a data entry error.",
    "rare_category":
        "This text value appears very infrequently in the column compared to "
        "other values. It may be a misspelling, a one-off entry, or a "
        "legitimate but uncommon value.",
    "text_length_outlier":
        "The length of this text value (number of characters) is unusually "
        "short or long compared to other values in the same column. "
        "This may indicate truncation, concatenation errors, or extra whitespace.",

    # ── Typo detection (typo_detector) ──
    "likely_typo":
        "This value is within 1-2 character edits (insertions, deletions, or "
        "substitutions) of a more common value in the same column. "
        "It is likely a spelling mistake or keyboard error.",

    # ── ML row-level anomaly (ml_anomaly_report) ──
    "ml_anomaly":
        "This entire row was flagged as anomalous by one or more machine "
        "learning models (Isolation Forest, One-Class SVM, or Local Outlier "
        "Factor). The combination of values across all columns in this row "
        "is unusual compared to the rest of the dataset.",
}


def attach_issue_descriptions(report: pd.DataFrame) -> pd.DataFrame:
    """Add a 'description' column with human-readable explanations.

    Uses the issue column (either 'issue' or 'issue_type') to look up
    descriptions from ISSUE_DESCRIPTIONS.  Unknown issue types get a
    generic fallback message.
    """
    if report.empty:
        report["description"] = None
        return report

    _issue_col = "issue_type" if "issue_type" in report.columns else "issue"
    if _issue_col not in report.columns:
        report["description"] = None
        return report

    report["description"] = report[_issue_col].map(
        lambda x: ISSUE_DESCRIPTIONS.get(str(x).strip(), f"Issue of type '{x}' was detected in this cell.")
    )
    return report


# -----------------------------
# Helpers for rule-based checks
# -----------------------------
def _detect_column_type(column_name: str, column_data: Optional[pd.Series] = None) -> Optional[str]:
    """
    Detect format type using column name as a hint, then CONFIRM by
    sampling actual data.  Returns a type only when both the name matches
    AND >= 50% of a non-null sample pass the corresponding validator.

    Returns:
        Format type ('email', 'date', 'url') or None
    """
    if not FORMAT_VALIDATORS_AVAILABLE:
        return None

    # Step 1: column-name hint
    col_lower = column_name.lower().strip()
    candidate = None
    for format_type, patterns in COLUMN_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in col_lower:
                candidate = format_type
                break
        if candidate:
            break

    if candidate is None:
        return None

    # Step 2: confirm with data sample (if data provided)
    if column_data is None:
        return None  # no data to confirm — do not assume

    sample = column_data.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if len(sample) == 0:
        return None

    sample = sample.head(100)  # check up to 100 values

    hits = 0
    for val in sample:
        try:
            if candidate == "email":
                ok, _ = validate_email(val)
            elif candidate == "date":
                ok, _ = validate_date(val)
            elif candidate == "url":
                ok, _ = validate_url(val)
            else:
                ok = False
            if ok:
                hits += 1
        except Exception:
            continue

    # Confirm only if >= 50% of sample matches the format
    if hits / len(sample) >= 0.50:
        return candidate

    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s in MISSING_TOKENS


def _check_value(v: Any, rule: Dict[str, Any], column_name: str = "", confirmed_type: Optional[str] = None) -> Tuple[bool, str, str, str, Dict[str, Any], str]:
    """
    Return:
      ok: bool
      issue: short code (e.g., 'missing', 'not number', 'regex mismatch', '<min', '>max', 'invalid date', 'not in allowed set')
      detail: human-friendly detail
      severity: 'high' | 'medium' | 'low'
      expected: dict of constraint hints
      rule_name: which rule triggered

    confirmed_type: pre-validated format type ('email', 'date', 'url') or None.
                    Only set when column name matched AND data sample confirmed.
    """
    # Presence / Missing check (FIRST) — honor nullable/required from rule
    if _is_missing(v):
        if rule.get("nullable") is True or rule.get("required") is False:
            return True, "", "", "low", {}, "presence"
        return False, "missing", "value is missing / placeholder", "high", {"not_missing": True}, "presence"

    # Basic pre-normalisation
    s_raw = "" if v is None else str(v)
    s = s_raw.strip()
    expected: Dict[str, Any] = {}

    # Format validation (BEFORE regex) — runs if a confirmed type was passed in
    # OR if the rule explicitly declares type: "email".
    _eff_type = confirmed_type or (rule.get("type") if rule.get("type") in ("email", "url") else None)
    if FORMAT_VALIDATORS_AVAILABLE and _eff_type:
        detected_type = _eff_type
        is_valid, error_reason = False, None

        try:
            if detected_type == 'email':
                is_valid, error_reason = validate_email(s)
                expected["format"] = "email"
                # Secondary: domain-quality check for structurally valid emails.
                # Common provider domains — misspellings like gmial.com, yahooo.com
                # pass RFC structure check but are clearly wrong.
                if is_valid and "@" in s:
                    _domain = s.rsplit("@", 1)[1].lower().strip()
                    _KNOWN_PROVIDERS = {
                        "gmail.com", "googlemail.com",
                        "yahoo.com", "yahoo.co.uk", "yahoo.fr", "yahoo.de",
                        "hotmail.com", "hotmail.co.uk", "hotmail.fr",
                        "outlook.com", "outlook.co.uk",
                        "icloud.com", "me.com", "mac.com",
                        "live.com", "live.co.uk",
                        "btinternet.com", "bt.com",
                        "virginmedia.com", "sky.com", "talktalk.net",
                        "aol.com", "protonmail.com", "proton.me",
                    }
                    # Flag ONLY genuine near-typos of a known provider. Requires the
                    # domain to be a real provider-length string (>=6 chars), not
                    # already a known provider, and within a 1-2 character edit of one
                    # of similar length. This deliberately never fires on ordinary
                    # company domains (e.g. 'x.com', 'acme.io') — see P1 in
                    # tests/ENGINE_FIXES_AND_TESTS.md.
                    if _domain not in _KNOWN_PROVIDERS and len(_domain) >= 6:
                        _best, _bd = None, 99
                        for _p in _KNOWN_PROVIDERS:
                            if abs(len(_p) - len(_domain)) > 1:
                                continue
                            _d = _edit_distance(_domain, _p)
                            if _d < _bd:
                                _best, _bd = _p, _d
                        if _best is not None and 1 <= _bd <= 2:
                            is_valid = False
                            error_reason = f"domain '{_domain}' looks like a misspelling of '{_best}'"
            elif detected_type == 'date':
                is_valid, error_reason = validate_date(s)
                expected["format"] = "date"
            elif detected_type == 'url':
                is_valid, error_reason = validate_url(s)
                expected["format"] = "URL"
        except Exception:
            # If format validator fails, continue to other checks
            is_valid = True  # don't falsely flag on validator crash

        if not is_valid and error_reason:
            return False, "format_error", f"invalid {detected_type}: {error_reason}", "high", expected, "format"

    # Number type + bounds  ("number", "numeric", "integer", "int" share bounds logic)
    if rule.get("type") in ("number", "numeric", "integer", "int"):
        expected["type"] = "numeric"
        # First attempt: parse directly (no stripping).
        # If that fails, try stripping currency symbols — if it THEN parses,
        # that means the value has a currency prefix which is itself a format error.
        _CURRENCY_RE = re.compile(r"[£$€¥]")
        try:
            x = float(s)
        except (ValueError, TypeError):
            s_stripped = _CURRENCY_RE.sub("", s).replace(",", "").strip()
            try:
                float(s_stripped)  # parses after stripping → currency symbol is the issue
                return False, "format_error", f"value '{s}' contains currency symbol; expected plain number", "high", expected, "type"
            except (ValueError, TypeError):
                return False, "not number", f"value '{s}' is not numeric", "high", expected, "type"
        # Bounds check on successfully parsed numeric value
        if "min" in rule:
            expected["min"] = rule["min"]
            if x < float(rule["min"]):
                return False, "<min", f"value {x} < min {rule['min']}", "high", expected, "bounds"
        if "max" in rule:
            expected["max"] = rule["max"]
            if x > float(rule["max"]):
                return False, ">max", f"value {x} > max {rule['max']}", "high", expected, "bounds"

    # Date type — checks format, parseability AND optional min/max bounds
    if rule.get("type") == "date":
        expected["type"] = "date"
        # If rule specifies a strict format, enforce it before attempting parse.
        _date_format = rule.get("format", "")
        if _date_format.upper() in ("YYYY-MM-DD", "ISO"):
            # Must match YYYY-MM-DD exactly (no other representations allowed)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                expected["format"] = "YYYY-MM-DD"
                return False, "format_error", f"date '{s}' is not in ISO YYYY-MM-DD format", "high", expected, "format"
        try:
            parsed_dt = pd.to_datetime(s, errors="raise")
        except Exception:
            return False, "invalid date", "cannot parse date/time", "high", expected, "type"
        # Bounds check. Accept min/max OR min_date/max_date (date-specific keys).
        _min_bound = rule.get("min", rule.get("min_date"))
        _max_bound = rule.get("max", rule.get("max_date"))
        if _min_bound is not None or _max_bound is not None:
            try:
                from dateutil.parser import parse as _parse_date
                _today = pd.Timestamp.today().normalize()
                _dayfirst = "DD/MM" in str(rule.get("format", "")).upper() or "DD-MM" in str(rule.get("format", "")).upper()
                def _resolve(v) -> pd.Timestamp:
                    if str(v).lower() in ("today", "now"):
                        return _today
                    return pd.Timestamp(_parse_date(str(v), dayfirst=_dayfirst))
                # Re-parse the value with dayfirst awareness for non-ISO formats.
                _val_dt = pd.Timestamp(_parse_date(s, dayfirst=_dayfirst)) if _dayfirst else parsed_dt
                if _min_bound is not None:
                    expected["min"] = _min_bound
                    if _val_dt < _resolve(_min_bound):
                        return False, "<min", f"date {s} is before minimum {_min_bound}", "high", expected, "bounds"
                if _max_bound is not None:
                    expected["max"] = _max_bound
                    if _val_dt > _resolve(_max_bound):
                        return False, ">max", f"date {s} is after maximum {_max_bound}", "high", expected, "bounds"
            except Exception:
                pass  # If bounds parsing fails, skip bounds check silently

    # Allowed values (categorical constraints)
    if "allowed_values" in rule and isinstance(rule["allowed_values"], (list, tuple)):
        allowed = set(map(str, rule["allowed_values"]))
        expected["allowed_values"] = list(rule["allowed_values"])
        if s not in allowed:
            return False, "not in allowed set", "value not in allowed set", "medium", expected, "allowed_values"

    # Regex / pattern (single string or list).
    # Rules may use "regex" or "pattern" interchangeably — try both keys.
    _regex_source = "regex" if "regex" in rule else ("pattern" if "pattern" in rule else None)
    if _regex_source:
        pats = rule[_regex_source]
        pats = pats if isinstance(pats, list) else [pats]
        ok_any = False
        for pat in pats:
            try:
                if re.fullmatch(pat, s or ""):
                    ok_any = True
                    break
            except re.error:
                # bad pattern -> ignore that pattern
                continue
        if not ok_any and len(pats) > 0:
            expected["regex"] = pats[0]  # show one for explainability
            return False, "regex mismatch", "value does not match pattern", "high", expected, _regex_source

    # All checks passed / or no rule parts
    return True, "", "", "low", expected, ""


def _validate_uniqueness(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    """Return rows violating uniqueness constraints for columns flagged True in rules['uniqueness']."""
    uniq = (rules.get("uniqueness") or {})
    rows: List[Dict[str, Any]] = []
    for c, need_unique in uniq.items():
        if not need_unique or c not in df.columns:
            continue
        # consider only non-missing values for uniqueness
        s = df[c].astype(str)
        miss_mask = s.str.strip().str.lower().isin(MISSING_TOKENS)
        dup_mask = s[~miss_mask].duplicated(keep="first")
        if dup_mask.any():
            bad_idx = dup_mask[dup_mask].index
            for i in bad_idx:
                rows.append(dict(
                    row_id=int(i),
                    column=c,
                    value=df.iloc[i][c],
                    issue="duplicate_value",
                    detail="violates uniqueness",
                    severity="high",
                    expected={"unique": True},
                    rule="uniqueness",
                ))
    if not rows:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","expected","rule"])
    return pd.DataFrame(rows)


# ------------------------------------------------
# Public: rule-only validation
# ------------------------------------------------
def validate_df(df: pd.DataFrame, rules: Dict[str, Any], enable_typo_detection: bool = False) -> pd.DataFrame:
    """
    Rule-based validation only (no anomaly detection).

    Args:
        df: DataFrame to validate
        rules: Validation rules
        enable_typo_detection: If True, add fuzzy matching for typo detection

    Returns:
        DataFrame with columns: row_id, column, issue, detail, severity, value, expected, rule
    """
    rows: List[Dict[str, Any]] = []

    # Pre-compute confirmed column types by sampling actual data.
    # This avoids false positives from column names alone.
    _confirmed_types: Dict[str, Optional[str]] = {}
    if FORMAT_VALIDATORS_AVAILABLE:
        for c in df.columns:
            _confirmed_types[c] = _detect_column_type(c, column_data=df[c])

    # Per-cell checks
    col_rules = (rules.get("columns") or {})
    for c in df.columns:
        rule = col_rules.get(c, {})
        ct = _confirmed_types.get(c)
        # Even if no rule exists, check format validators if type confirmed
        if not rule and ct:
            rule = {}
        elif not rule:
            continue

        col_vals = df[c]
        for i, val in enumerate(col_vals):
            ok, issue, detail, severity, expected, rule_name = _check_value(val, rule, column_name=c, confirmed_type=ct)
            if not ok:
                rows.append(dict(
                    row_id=int(i),
                    column=c,
                    value=val,
                    issue=issue,
                    detail=detail,
                    severity=severity,
                    expected=expected,
                    rule=rule_name or "value_rule"
                ))

    # Uniqueness checks
    uniq_df = _validate_uniqueness(df, rules)
    if not uniq_df.empty:
        rows.extend(uniq_df.to_dict("records"))

    # Typo detection (optional)
    if enable_typo_detection and TYPO_DETECTOR_AVAILABLE:
        typo_df = detect_typos_all_columns(df, max_distance=2, min_corpus_frequency=3)
        if not typo_df.empty:
            rows.extend(typo_df.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","expected","rule"])
    rep = pd.DataFrame(rows)
    rep = rep.sort_values(["severity","column","row_id"], ascending=[True, True, True]).reset_index(drop=True)
    return rep


# ------------------------------------------------------
# Public: rule + anomaly combined (with optional ML)
# ------------------------------------------------------
def validate_and_anomaly_report(
    df: pd.DataFrame,
    rules: Dict[str, Any],
    use_ml: bool = False,
    ml_models: Optional[List[str]] = None,
    ensemble: bool = True,
    use_reference: bool = True,
    df_ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Combine rule-based validation (validate_df) with heuristic and optional ML anomalies.

    Output columns: row_id, column, issue, detail, severity, value, expected, rule
    - If called with only (df, rules), behaves like rule + heuristics.
    - If use_ml=True, adds ML row-level anomalies (column='__row__') using TF-IDF over row-concat.
    """
    # Rule violations
    try:
        rule_rep = validate_df(df, rules, enable_typo_detection=True)
    except Exception:
        rule_rep = pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","expected","rule"])

    # Heuristic anomalies (numeric/rare/length/missing)
    anom = detect_anomalies(df, rules)

    # ML anomalies (optional)
    ml_rep = pd.DataFrame(columns=["row_id","column","issue","detail","severity","value","expected","rule"])
    if use_ml and isinstance(ml_models, list) and len(ml_models) > 0:
        try:
            ml_raw = ml_anomaly_report(
                df_unclean=df,
                df_ref=df_ref,
                use_reference=use_reference,
                models=ml_models,
                ensemble=ensemble,
            )
            if not ml_raw.empty:
                ml_rep = ml_raw[["row_id","column","issue","detail","severity","value","expected","rule"]].copy()
        except Exception:
            pass

    # Harmonize columns
    for k in ["expected","rule","detail","severity"]:
        if k not in rule_rep.columns: rule_rep[k] = None
        if k not in anom.columns:     anom[k]     = None
        if k not in ml_rep.columns:   ml_rep[k]   = None

    rep = pd.concat([
        rule_rep[["row_id","column","issue","detail","severity","value","expected","rule"]],
        anom[["row_id","column","issue","detail","severity","value","expected","rule"]],
        ml_rep[["row_id","column","issue","detail","severity","value","expected","rule"]],
    ], ignore_index=True)

    if not rep.empty:
        rep = rep.sort_values(["severity","column","row_id"], ascending=[True, True, True]).reset_index(drop=True)
    return rep


# ------------------------------------------------------
# Reporting helper: summarize ML performance (optional)
# ------------------------------------------------------
def summarize_ml_performance(
    ml_report_unclean: pd.DataFrame,
    rule_report_unclean: pd.DataFrame,
    df_ref: Optional[pd.DataFrame] = None,
    ml_report_ref: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build a small dict of metrics for UI cards:
      - contamination_rate on unclean (rows flagged / total)
      - overlap_with_rules (% of ml anomalies whose row_id/columns also in rule violations)
      - ref_false_positive_rate (if df_ref provided and ml_report_ref available)
    """
    out: Dict[str, Any] = {
        "contamination_rate": None,
        "overlap_with_rules_pct": None,
        "ref_false_positive_rate": None,
        "counts": {}
    }

    # Contamination on unclean
    if isinstance(ml_report_unclean, pd.DataFrame) and not ml_report_unclean.empty:
        flagged = ml_report_unclean["row_id"].nunique()
        # Try to infer total rows from max row_id
        try:
            total = int(ml_report_unclean["row_id"].max()) + 1
        except Exception:
            total = None
        if total and total > 0:
            out["contamination_rate"] = flagged / total
        out["counts"]["ml_flagged_rows"] = flagged

    # Overlap with rules
    if isinstance(ml_report_unclean, pd.DataFrame) and not ml_report_unclean.empty \
       and isinstance(rule_report_unclean, pd.DataFrame) and not rule_report_unclean.empty:
        ml_rows = set(ml_report_unclean["row_id"].tolist())
        rule_rows = set(rule_report_unclean["row_id"].tolist())
        if ml_rows:
            out["overlap_with_rules_pct"] = len(ml_rows & rule_rows) / len(ml_rows)

    # False positive on reference (if provided)
    if isinstance(df_ref, pd.DataFrame) and df_ref is not None \
       and isinstance(ml_report_ref, pd.DataFrame) and not ml_report_ref.empty:
        # assume reference is "clean"; anything flagged by ML is counted as FP
        fp_rows = ml_report_ref["row_id"].nunique()
        out["ref_false_positive_rate"] = fp_rows / len(df_ref)

    return out


# -------------------------------------------
# Public: attach human-friendly suggestions
# -------------------------------------------
def _detect_dominant_format(series: pd.Series) -> Optional[str]:
    """Detect the dominant format pattern in a column.

    Returns a strftime-compatible format string for dates, or a regex pattern
    string for structured text. Returns None if no clear dominant format.

    Priority: detect date formats first, then structured patterns.
    """
    import re as _re
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if len(vals) < 5:
        return None

    sample = vals.head(200)

    # ── Date format detection ──
    _DATE_FORMATS = [
        ("%Y-%m-%d",       r"^\d{4}-\d{2}-\d{2}$"),
        ("%d/%m/%Y",       r"^\d{2}/\d{2}/\d{4}$"),
        ("%m/%d/%Y",       r"^\d{2}/\d{2}/\d{4}$"),
        ("%d-%m-%Y",       r"^\d{2}-\d{2}-\d{4}$"),
        ("%Y/%m/%d",       r"^\d{4}/\d{2}/\d{2}$"),
        ("%d %b %Y",       r"^\d{2}\s[A-Za-z]{3}\s\d{4}$"),
        ("%b %d, %Y",      r"^[A-Za-z]{3}\s\d{1,2},\s\d{4}$"),
        ("%d %B %Y",       r"^\d{2}\s[A-Za-z]+\s\d{4}$"),
    ]
    best_fmt = None
    best_count = 0
    for fmt, pat in _DATE_FORMATS:
        matches = sample.str.fullmatch(pat).sum()
        if matches > best_count:
            best_count = matches
            best_fmt = fmt
    if best_fmt and best_count / len(sample) >= 0.60:
        return f"date:{best_fmt}"

    # ── Structured text pattern detection ──
    # Detect dominant casing/structure patterns
    _PATTERNS = [
        ("upper",    lambda s: s.str.match(r'^[A-Z\s]+$')),
        ("lower",    lambda s: s.str.match(r'^[a-z\s]+$')),
        ("title",    lambda s: s.apply(lambda x: x == x.title() if isinstance(x, str) else False)),
    ]
    for name, check_fn in _PATTERNS:
        try:
            matches = check_fn(sample).sum()
            if matches / len(sample) >= 0.80:
                return f"case:{name}"
        except Exception:
            continue

    return None


def _reformat_value(value: str, dominant_format: str) -> Optional[str]:
    """Reformat a value to match the dominant format detected for its column."""
    if not value or not dominant_format:
        return None

    if dominant_format.startswith("date:"):
        target_fmt = dominant_format[5:]
        try:
            parsed = pd.to_datetime(value, errors="raise", dayfirst=True)
            reformatted = parsed.strftime(target_fmt)
            if reformatted != value:
                return reformatted
        except Exception:
            return None

    elif dominant_format == "case:upper":
        fixed = value.upper()
        return fixed if fixed != value else None
    elif dominant_format == "case:lower":
        fixed = value.lower()
        return fixed if fixed != value else None
    elif dominant_format == "case:title":
        fixed = value.title()
        return fixed if fixed != value else None

    return None


def _load_corpus_csvs(corpus_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load corpus CSV files from disk.  Returns a dict keyed by column name:

        {
            "bank_name": {
                "aliases": {"hsbc": "HSBC UK", "barclays bank": "Barclays", ...},
                "canonicals": ["HSBC UK", "Barclays", "NatWest", ...],
            },
            "transaction_type": {
                "aliases": {},
                "canonicals": ["Debit", "Credit", "Transfer", ...],
            },
            ...
        }

    Auto-discovers the corpus directory relative to this file if not given.
    """
    import os
    if corpus_dir is None:
        # Walk up from core/validator/ to project root, then into TEST2_DATA/corpus
        _this = os.path.dirname(os.path.abspath(__file__))
        corpus_dir = os.path.join(_this, os.pardir, os.pardir, "TEST2_DATA", "corpus")
        corpus_dir = os.path.normpath(corpus_dir)

    if not os.path.isdir(corpus_dir):
        return {}

    # Mapping: filename → (column_name, type)
    # "type" is "alias" (has alias+canonical cols) or "validation" (single value col)
    _KNOWN_CORPORA = {
        "corpus_bank_names.csv":        ("bank_name",        "alias"),
        "corpus_company_names.csv":     ("merchant_name",    "alias"),
        "corpus_country_codes.csv":     ("country",          "alias"),
        "corpus_email_domains.csv":     ("email",            "email_domain"),
        "corpus_transaction_types.csv": ("transaction_type", "validation"),
        "corpus_uk_postcodes.csv":      ("postcode",         "validation_postcode"),
    }

    result: Dict[str, Dict[str, Any]] = {}

    for fname, (col_name, ctype) in _KNOWN_CORPORA.items():
        fpath = os.path.join(corpus_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            cdf = pd.read_csv(fpath)
        except Exception:
            continue

        if ctype == "alias" and "alias" in cdf.columns and "canonical" in cdf.columns:
            aliases = {}
            for _, row in cdf.iterrows():
                a = str(row["alias"]).strip()
                c = str(row["canonical"]).strip()
                aliases[a.lower()] = c
            canonicals = list(set(aliases.values()))
            result[col_name] = {"aliases": aliases, "canonicals": canonicals}

        elif ctype == "email_domain" and "alias" in cdf.columns and "canonical" in cdf.columns:
            # Special: maps typo domains to correct domains
            aliases = {}
            for _, row in cdf.iterrows():
                a = str(row["alias"]).strip().lower()
                c = str(row["canonical"]).strip().lower()
                aliases[a] = c
            canonicals = list(set(aliases.values()))
            result[col_name] = {"aliases": aliases, "canonicals": canonicals,
                                "type": "email_domain"}

        elif ctype == "validation" and "value" in cdf.columns:
            vals = cdf["value"].dropna().astype(str).str.strip().tolist()
            result[col_name] = {"aliases": {}, "canonicals": vals}

        elif ctype == "validation_postcode" and "postcode" in cdf.columns:
            vals = cdf["postcode"].dropna().astype(str).str.strip().tolist()
            result[col_name] = {"aliases": {}, "canonicals": vals}

    return result


def _corpus_lookup(value: str, corpus_entry: Dict[str, Any]) -> Optional[str]:
    """Try to find a canonical value using the corpus entry.

    1.  Exact alias lookup  (case-insensitive).
    2.  For email_domain type: extract domain, lookup, reconstruct.
    3.  Jaro-Winkler against canonical list (>= 0.85 threshold — lower
        than the generic 0.90 because corpus canonicals are trusted).
    """
    if not value or not corpus_entry:
        return None

    aliases = corpus_entry.get("aliases", {})
    canonicals = corpus_entry.get("canonicals", [])
    ctype = corpus_entry.get("type", "")

    # ── Email domain special handling ──
    if ctype == "email_domain" and "@" in value:
        local, domain = value.rsplit("@", 1)
        domain_lower = domain.lower().strip()
        if domain_lower in aliases:
            fixed_domain = aliases[domain_lower]
            return f"{local}@{fixed_domain}"
        # Jaro-Winkler on the domain part only
        try:
            import jellyfish
            best, best_sim = None, 0.0
            for canon in canonicals:
                sim = jellyfish.jaro_winkler_similarity(domain_lower, canon.lower())
                if sim > best_sim:
                    best_sim = sim
                    best = canon
            if best and best_sim >= 0.85:
                return f"{local}@{best}"
        except Exception:
            pass
        return None

    # ── Standard alias lookup (case-insensitive) ──
    val_lower = value.strip().lower()
    if val_lower in aliases:
        canonical = aliases[val_lower]
        if canonical != value:
            return canonical

    # ── Jaro-Winkler against canonical values ──
    if canonicals:
        try:
            import jellyfish
            best, best_sim = None, 0.0
            for canon in canonicals:
                sim = jellyfish.jaro_winkler_similarity(val_lower, canon.lower())
                if sim > best_sim:
                    best_sim = sim
                    best = canon
            if best and best_sim >= 0.85 and best != value:
                return best
        except Exception:
            # Fallback: simple set-overlap similarity
            sa = set(val_lower.split())
            best, best_sim = None, 0.0
            for canon in canonicals:
                ca = set(canon.lower().split())
                sim = len(sa & ca) / max(1, len(sa | ca))
                if sim > best_sim:
                    best_sim = sim
                    best = canon
            if best and best_sim >= 0.85 and best != value:
                return best

    return None


def attach_suggestions(report: pd.DataFrame,
                       df: pd.DataFrame,
                       df_ref: Optional[pd.DataFrame],
                       rules: Optional[Dict[str, Any]] = None,
                       corpus_map: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Adds a 'suggested_fix' column using 3-priority format detection:
      1. JSON rules (explicit format if defined)
      2. Reference data (dominant format from clean dataset)
      3. Inferred (dominant format from the column itself)

    Additionally uses corpus CSV files (alias → canonical mappings) for:
      - bank_name, merchant_name, country, email domain, transaction_type, postcode

    For each issue:
      - format_error / invalid date → reformat to dominant format
      - numeric issues → clip to bounds
      - typos / categorical → corpus alias lookup first, then Jaro-Winkler
      - missing → corpus suggestion if available
    """
    try:
        from .corpus_correction import suggest_fix
    except Exception:
        def suggest_fix(value: str, column: str, corpus: Optional[List[str]] = None) -> Optional[str]:
            return None

    if report.empty:
        out = report.copy()
        out["suggested_fix"] = None
        return out

    if rules is None:
        rules = {}

    # ── Load corpus CSVs (auto-discovered or passed in) ──
    if corpus_map is None:
        corpus_map = _load_corpus_csvs()

    # ── Pre-compute dominant formats per column (cached) ──
    _col_formats: Dict[str, Optional[str]] = {}
    for col in df.columns:
        # Priority 1: JSON rules
        col_rules = (rules.get("columns") or {}).get(col, {})
        if isinstance(col_rules, dict) and "date_format" in col_rules:
            _col_formats[col] = f"date:{col_rules['date_format']}"
            continue
        if isinstance(col_rules, dict) and "case" in col_rules:
            _col_formats[col] = f"case:{col_rules['case']}"
            continue

        # Priority 2: Reference data
        if df_ref is not None and col in df_ref.columns:
            fmt = _detect_dominant_format(df_ref[col])
            if fmt:
                _col_formats[col] = fmt
                continue

        # Priority 3: Inferred from unclean data
        fmt = _detect_dominant_format(df[col])
        if fmt:
            _col_formats[col] = fmt

    # ── Build corpus per column (cached) ──
    # Priority: corpus CSV canonicals > reference data > high-frequency from data
    _col_corpus: Dict[str, List[str]] = {}
    from collections import Counter
    for col in df.columns:
        # Start with corpus CSV canonicals if available
        if col in corpus_map and corpus_map[col].get("canonicals"):
            _col_corpus[col] = corpus_map[col]["canonicals"]
        elif df_ref is not None and col in df_ref.columns:
            _col_corpus[col] = df_ref[col].dropna().astype(str).tolist()
        else:
            freq = Counter(df[col].dropna().astype(str).str.strip().tolist())
            corpus = [v for v, c in freq.items() if c >= 3]
            if not corpus:
                corpus = df[col].dropna().astype(str).tolist()
            _col_corpus[col] = corpus

    # ── Generate suggestions ──
    # Determine the issue column name (may have been renamed to issue_type)
    _issue_attr = "issue_type" if "issue_type" in report.columns else "issue"

    out = report.copy()
    fixes: List[Any] = []
    for r in out.itertuples(index=False):
        suggestion = None
        col = getattr(r, "column", None)
        val = getattr(r, "value", None)
        issue = getattr(r, _issue_attr, "")

        if col is None or col not in df.columns:
            fixes.append(None)
            continue

        # ── Corpus alias lookup (FIRST — highest confidence) ──
        # If a corpus CSV exists for this column, try direct alias mapping.
        if col in corpus_map and val is not None:
            suggestion = _corpus_lookup(str(val), corpus_map[col])

        # ── If corpus lookup gave an answer, use it ──
        if suggestion is not None:
            fixes.append(suggestion)
            continue

        # ── Numeric issues → clip to bounds ──
        if issue in ("numeric_outlier", "<min", ">max", "not number"):
            x = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - 1e-12 if (q3 - q1) == 0 else (q3 - q1)
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            exp = getattr(r, "expected", None)
            if isinstance(exp, dict):
                if "min" in exp: lo = exp["min"]
                if "max" in exp: hi = exp["max"]

            try:
                suggestion = float(np.clip(float(val), float(lo), float(hi)))
            except Exception:
                suggestion = None

        # ── Format errors → try reformatting to dominant format first ──
        elif issue in ("format_error", "invalid date", "regex mismatch"):
            fmt = _col_formats.get(col)
            if fmt and val is not None:
                suggestion = _reformat_value(str(val), fmt)
            # Email domain typo correction (e.g. outlok.co -> outlook.com)
            if suggestion is None and val is not None and "@" in str(val):
                try:
                    from .corpus_correction import suggest_email_fix
                    suggestion = suggest_email_fix(str(val))
                except Exception:
                    pass
            # If still nothing, fall back to corpus matching
            if suggestion is None:
                corpus = _col_corpus.get(col, [])
                suggestion = suggest_fix("" if val is None else str(val), col, corpus=corpus)

        # ── Typos, rare categories, other issues → corpus matching ──
        else:
            corpus = _col_corpus.get(col, [])
            suggestion = suggest_fix("" if val is None else str(val), col, corpus=corpus)

        fixes.append(suggestion)

    out["suggested_fix"] = fixes
    return out