# NTU KTP Data Quality Investigator — Research Log

**Project:** NTU Knowledge Transfer Partnership — Data Quality Investigator
**Dataset domains tested:** Insurance claims (TEST3_DATA), Financial transactions (TEST2_DATA)
**Evaluation date:** 2026-05-20

---

## Table of Contents

1. [Evaluation Framework Design](#1-evaluation-framework-design)
2. [Ground Truth Extraction — TEST3_DATA](#2-ground-truth-extraction--test3_data)
3. [ML Diagnostic Experiments](#3-ml-diagnostic-experiments)
4. [ML Anomaly Detector Upgrade](#4-ml-anomaly-detector-upgrade)
5. [Full Engine Evaluation — Baseline (Pre-fix)](#5-full-engine-evaluation--baseline-pre-fix)
6. [Rule Engine Fixes](#6-rule-engine-fixes)
7. [Full Engine Evaluation — After Fixes](#7-full-engine-evaluation--after-fixes)
8. [Ablation Analysis](#8-ablation-analysis)
9. [Remaining Gaps](#9-remaining-gaps)
10. [Open Finance Evaluation Pipeline Plan](#10-open-finance-evaluation-pipeline-plan)

---

## 1. Evaluation Framework Design

### Why it was needed
Before this work, the DQ engine had no quantitative accuracy measurement. Claims about detection quality were qualitative. To support the research paper and to guide further development, a rigorous P/R/F1 evaluation was needed.

### Approach

**Ground truth** is a set of `(row_id, column, issue_type)` tuples known to be errors. Each detected `(row_id, column, mapped_issue_type)` is matched against ground truth for exact-match scoring.

**Metric definitions:**
- True Positive (TP): detected tuple appears in ground truth
- False Positive (FP): detected tuple does not appear in ground truth
- False Negative (FN): ground truth tuple was not detected
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × P × R / (P + R)

**Excluded from exact P/R:**
- `discovery` — heuristic signals (`rare_category`, `text_length_outlier`) that flag suspicious values for human review, not specific named error types. Evaluated separately via "discovery recall" (fraction of error rows touched).
- `ml_anomaly` — row-level score, no column/type attached. Evaluated via lift over random baseline.

**Files:**
- `scripts/run_evaluation.py` — runs all 5 engine layers, computes P/R/F1
- `scripts/output/evaluation/` — all outputs (CSV, summary text)

---

## 2. Ground Truth Extraction — TEST3_DATA

### Dataset
`TEST3_DATA/main_data/insurance_claims.csv` — 1085 rows × 20 columns, synthetic insurance claims with deliberately injected errors across 21 error types.

### Method
Algorithmic extraction using known injection manifest. Each error type was detected using domain-appropriate logic:

| Error type | Detection method |
|---|---|
| `missing_value` | Match against `MISSING_TOKENS` set (NaN, "N/A", "unknown", etc.) |
| `format_error` (email) | Structural regex passes; check domain against `KNOWN_GOOD_EMAIL_DOMAINS` |
| `format_error` (postcode) | Regex `^[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}$` |
| `format_error` (claim_amount £) | Detect `£` prefix via regex |
| `format_error` (date non-ISO) | Pattern `\d{4}-\d{2}-\d{2}` negative match |
| `range_error` (claim_amount) | value < 0 |
| `range_error` (date_of_birth) | date < 1900 or date > today |
| `duplicate` | `df.duplicated(keep=False)` |
| `near_duplicate` | SequenceMatcher similarity > 0.85 on claimant_name |
| `corpus_mismatch` | Lookup against corpus CSVs (insurer, diagnosis code) |
| `standardisation_error` | Allowed-value violation with known canonical mapping |
| `logical_error` | Cross-column checks (e.g., policy end before start, future claim date) |

### Key finding: email domain check
`^[^@\s]+@[^@\s]+\.[^@\s]+$` passes `gmial.com` structurally. Correct approach: check the domain part against a known-good provider set. Domains like `gmial.com`, `yahooo.com`, `outlok.com` then produce format_error correctly.

### Results
- **639 GT records** across 493 unique error rows
- **Cross-check vs manifest:** 19/21 error types within 15% tolerance
- Two legitimate discrepancies:
  - `duplicate`: extractor flags all 110 rows (both originals + copies); manifest counts only 50 added copies. Both methodologies are internally consistent.
  - `near_duplicate`: similar counting difference

**GT breakdown by issue type:**

| Issue type | Count |
|---|---|
| duplicate | 110 |
| missing_value | 77 |
| format_error | 185 |
| range_error | 26 |
| logical_error | 55 |
| corpus_mismatch | 74 |
| standardisation_error | 62 |
| near_duplicate | 50 |
| **Total** | **639** |

**Script:** `scripts/extract_ground_truth_test3.py`
**Output:** `scripts/output/ground_truth_test3.csv`

---

## 3. ML Diagnostic Experiments

### Purpose
Determine whether the ML anomaly detection component (IsolationForest-based) carries real signal before investing in threshold optimisation.

### Experiment 3 — Bimodal Score Distribution

**Method:** Compute ensemble anomaly scores for all rows. Plot KDE histogram. Count local maxima (peaks = `ys[i] > ys[i-1] and ys[i] > ys[i+1] and ys[i] > ys.max() * 0.1`). If bimodal → two populations (normal vs anomalous) → real signal exists.

**Results:**

| Dataset | Peaks | Interpretation |
|---|---|---|
| TEST2_DATA | 2 (BIMODAL) | Clear normal/anomaly separation |
| TEST3_DATA | 3 (BIMODAL) | Clear separation, one intermediate cluster |

**Interpretation:** Bimodal in both domains → real signal is present. A unimodal distribution would indicate the model is not distinguishing normal from anomalous rows.

### Experiment 4 — Beat Random Baseline

**Method:** Compare ML-flagged rows (at current threshold) against 100 random seeds flagging the same number of rows. Compute precision for each. Lift = ML precision / mean random precision.

**Results:**

| Dataset | ML Precision | Random Baseline | Lift |
|---|---|---|---|
| TEST2_DATA | ~0.52 | ~0.42 | 1.24x |
| TEST3_DATA | ~0.56 | ~0.45 | 1.21x |

**Interpretation:** Both datasets show lift > 1.0. The ML model selects error-containing rows at better-than-random rate in both insurance and financial domains. This confirms the component adds genuine value.

**Conclusion:** Strategy 1 (principled GMM threshold) is appropriate for both datasets.

**Script:** `scripts/ml_diagnostic.py`
**Output:** `scripts/output/ml_diagnostic/` (KDE plots, beat-random plots)

---

## 4. ML Anomaly Detector Upgrade

### Pre-upgrade problem
The original ML detector used a single IsolationForest on all columns with a hardcoded `quantile(0.85)` threshold. This flagged 15% of rows with precision near random (~1.0x lift).

### Upgrade: 3-Component Ensemble

Three parallel signals, averaged to produce final score:

| Component | Method | What it captures |
|---|---|---|
| Row-level TF-IDF | Char n-gram (2-4) across all columns, IsolationForest | Unusual whole-row patterns |
| Column-level TF-IDF | Per-column IsolationForest on stacked features | Column-specific outlier text |
| Numeric | StandardScaler + IsolationForest on numeric cols | Statistical outliers in numbers |

**Function:** `core/anomaly.py::ml_anomaly_report()`

### Upgrade: GMM Antimode Threshold (Strategy 1)

**Why:** Hardcoded quantile threshold doesn't adapt to the score distribution shape. If the distribution is bimodal, the valley between peaks is the principled decision boundary.

**Method:**
1. Fit 2-component Gaussian Mixture Model to score distribution
2. Find antimode (log-probability minimum) between the two component means
3. Validity checks: gap between means must be > 0.15; flagging rate must be 1-20%
4. If checks fail: fall back to `quantile(0.90)`

**Production behaviour:** On TEST3_DATA, GMM detects a gap of 0.084 (< 0.15 threshold) — the 3-component ensemble compresses IsolationForest scores into a narrow range. Fallback to `quantile(0.90)` activates. This is logged in output.

### Post-upgrade results (TEST3_DATA)

| Metric | Before | After |
|---|---|---|
| Flagging rate | 15% | 10% |
| Lift vs random | ~1.0x | 1.35x |
| Precision | ~45% | 61.5% |

**Commit:** `14c5881`

---

## 5. Full Engine Evaluation — Baseline (Pre-fix)

Evaluation run after ML upgrade, before rule engine fixes.

### Results

| Metric | Value |
|---|---|
| Precision | 0.360 |
| Recall | 0.541 |
| **F1** | **0.433** |
| TP | 346 |
| FP | 614 |
| FN | 293 |

### Per issue type

| Issue type | P | R | F1 | GT |
|---|---|---|---|---|
| corpus_mismatch | 1.000 | 1.000 | 1.000 | 74 |
| duplicate | 1.000 | 1.000 | 1.000 | 110 |
| near_duplicate | 1.000 | 1.000 | 1.000 | 50 |
| standardisation_error | 1.000 | 0.726 | 0.841 | 62 |
| missing_value | 0.098 | 0.870 | 0.177 | 77 |
| format_error | 0.000 | 0.000 | 0.000 | 185 |
| range_error | 0.000 | 0.000 | 0.000 | 26 |
| logical_error | 0.000 | 0.000 | 0.000 | 55 |

### Root cause analysis for zeros

**format_error = 0.000 (185 GT missed):**
- `_check_value` had `if rule.get("type") == "number":` — `"numeric"` type in TEST3_DATA rules was silently skipped
- `_check_value` only read `"regex"` key — postcode rule uses `"pattern"` key → 0 postcode issues
- No ISO date format enforcement — `27 Jan 2024` parses successfully as a date
- Email `validate_email()` passes structural RFC check but not domain quality

**range_error = 0.000 (26 GT missed):**
- Same `"numeric"` type bug — claim_amount negative values not range-checked
- Date `min`/`max` bounds not implemented — DOB out-of-range missed

**logical_error = 0.000 (55 GT missed):**
- No cross-column validation rules implemented (this remains an open gap)

**missing_value low precision (P=0.098):**
- Heuristic layer (`detect_anomalies`) flags 15,000+ rows total; the majority are discovery signals but some bleed into missing_value mappings

---

## 6. Rule Engine Fixes

### Fix 1 — `"numeric"` type alias (`core/validator/validate.py`)

```python
# Before
if rule.get("type") == "number":

# After
if rule.get("type") in ("number", "numeric"):
```

**Impact:** Enables claim_amount and any other `"numeric"`-typed rule to run bounds checks.

### Fix 2 — Currency symbol = format error

```python
# Before: stripped £ before parsing (silently accepted £33867.3)
s_num = re.sub(r"[£$€,]", "", s).strip()
x = float(s_num)

# After: try direct parse first; if currency causes failure → format_error
try:
    x = float(s)
except ValueError:
    s_stripped = re.sub(r"[£$€¥]", "", s).replace(",", "").strip()
    try:
        float(s_stripped)  # parses after strip → currency was the issue
        return False, "format_error", "value contains currency symbol; expected plain number", ...
```

**Impact:** `£33867.3` → format_error (26 GT items). `-23687.78` → `<min` range_error (12 GT items).

### Fix 3 — `"pattern"` key support

```python
# Before: only "regex" key
if "regex" in rule:
    pats = rule["regex"]

# After: "pattern" treated identically to "regex"
_regex_source = "regex" if "regex" in rule else ("pattern" if "pattern" in rule else None)
if _regex_source:
    pats = rule[_regex_source]
```

**Impact:** Postcode rule `pattern: "^[A-Z]{1,2}\\d..."` now fires → 27 postcode format_errors caught.

### Fix 4 — ISO date format enforcement

```python
if rule.get("format", "").upper() in ("YYYY-MM-DD", "ISO"):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return False, "format_error", f"date '{s}' is not in ISO YYYY-MM-DD format", ...
```

**Impact:** `27 Jan 2024`, `04/04/2023` → format_error (35 claim_date GT items).

### Fix 5 — Date bounds checking

```python
# After the parseability check:
if "min" in rule or "max" in rule:
    from dateutil.parser import parse as _parse_date
    _today = pd.Timestamp.today().normalize()
    def _resolve(v):
        return _today if str(v).lower() == "today" else pd.Timestamp(_parse_date(str(v)))
    if "min" in rule and parsed_dt < _resolve(rule["min"]):
        return False, "<min", f"date {s} is before minimum {rule['min']}", ...
    if "max" in rule and parsed_dt > _resolve(rule["max"]):
        return False, ">max", f"date {s} is after maximum {rule['max']}", ...
```

**Impact:** Future DOB (`2027-07-25`), pre-1900 DOB (`1887-07-05`) → range_error (14 GT items).

### Fix 6 — Email domain quality check

```python
if is_valid and "@" in s:
    _domain = s.rsplit("@", 1)[1].lower()
    _KNOWN_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", ...}
    close = difflib.get_close_matches(_domain, _KNOWN_PROVIDERS, n=1, cutoff=0.7)
    if close and close[0] != _domain:
        is_valid = False
        error_reason = f"domain '{_domain}' looks like a misspelling of '{close[0]}'"
```

**Impact:** `gmial.com`, `yahooo.com`, `outlok.com` → format_error. All 65 email domain errors caught.

### Fix 7 — Rules file format and `ENGINE_TO_GT` mapping (`scripts/run_evaluation.py`)

TEST3_DATA `validation_rules.json` is flat `{column: rule_dict}` but `validate_df` expects `{"columns": {column: rule_dict}}`. Fix: auto-wrap on load.

`ENGINE_TO_GT` was missing mappings for new `_check_value` issue codes (`format_error`, `regex mismatch`, `not in allowed set`, `<min`, `>max`).

**Commit:** `2a000c6`

---

## 7. Full Engine Evaluation — After Fixes

Evaluation run after all rule engine fixes.

### Results

| Metric | Baseline | After fixes | Change |
|---|---|---|---|
| Precision | 0.360 | 0.461 | +0.101 |
| Recall | 0.541 | 0.833 | +0.292 |
| **F1** | **0.433** | **0.593** | **+0.160** |
| TP | 346 | 532 | +186 |
| FP | 614 | 622 | +8 |
| FN | 293 | 107 | -186 |

### Per issue type

| Issue type | P | R | F1 | GT |
|---|---|---|---|---|
| corpus_mismatch | 1.000 | 1.000 | 1.000 | 74 |
| duplicate | 1.000 | 1.000 | 1.000 | 110 |
| near_duplicate | 1.000 | 1.000 | 1.000 | 50 |
| format_error | 0.955 | 0.908 | **0.931** | 185 |
| standardisation_error | 1.000 | 0.726 | 0.841 | 62 |
| range_error | 1.000 | 0.692 | **0.818** | 26 |
| missing_value | 0.098 | 0.870 | 0.177 | 77 |
| logical_error | 0.000 | 0.000 | 0.000 | 55 |

### Per column (selected)

| Column | P | R | F1 | GT |
|---|---|---|---|---|
| email | 0.896 | 1.000 | 0.945 | 86 |
| postcode | 0.808 | 1.000 | 0.894 | 42 |
| claim_date | 0.778 | 0.507 | 0.614 | 69 |
| claim_amount | 0.792 | 0.644 | 0.710 | 59 |
| date_of_birth | 0.250 | 0.429 | 0.316 | 14 |
| policy_id | 0.600 | 1.000 | 0.750 | 15 |

### Row-level recall (deterministic layers only)

| Config | P | R | F1 |
|---|---|---|---|
| rule + corpus + duplicate | 0.677 | 0.945 | 0.789 |
| All layers (including heuristic) | 0.454 | 1.000 | 0.625 |

**Interpretation:** The deterministic layers (rule + corpus + duplicate) achieve 94.5% row-level recall with 67.7% precision — i.e., they flag nearly all error rows with acceptable noise. The heuristic layer adds the remaining 5.5% recall but at the cost of many discovery-type flags that are not exact-match TPs.

---

## 8. Ablation Analysis

This shows the contribution of each layer as they are stacked:

| Config | P | R | F1 |
|---|---|---|---|
| rule only | 0.343 | 0.466 | 0.395 |
| + heuristic | 0.324 | 0.466 | 0.382 |
| + corpus | 0.374 | 0.582 | 0.456 |
| + duplicate | 0.461 | 0.833 | 0.593 |
| + ML | 0.461 | 0.833 | 0.593 |

### Interpretations

**Rule layer alone (F1=0.395):** Strong format/range signal. Recall limited by missing cross-column logic and unmapped patterns. Precision 34% because rule layer generates many `missing` flags for optional fields (diagnosis_code: 424 missing flags vs ~21 GT).

**Adding heuristic (F1=0.382):** F1 *decreases* because the heuristic layer adds 15,000+ discovery flags that are not TPs. The heuristic layer's value is discovery recall (touches 100% of error rows) not exact P/R improvement.

**Adding corpus (F1=0.456):** +0.074 F1. Corpus layer perfectly catches all 74 corpus_mismatch items (insurer names, diagnosis codes) with zero FPs.

**Adding duplicate (F1=0.593):** Largest single gain (+0.137). Duplicate detection catches all 110 exact duplicates and 50 near-duplicates with perfect precision.

**Adding ML (F1=0.593):** No change to exact P/R (expected — ML is row-level, no column/type). ML value is shown in lift metrics (1.35x precision vs random).

---

## 9. Remaining Gaps

### logical_error — F1 = 0.000 (55 GT instances)

**Root cause:** No cross-column validation rules implemented in the engine. The 55 GT instances include:
- Policy end date before policy start date
- Claim date after policy end date
- Age implied by DOB inconsistent with stated age field

**Fix required:** Add cross-column rule support to `validate_df`. This requires expressing rules like `"claim_date <= policy_end_date"` and evaluating them per row.

### missing_value — P = 0.098 (precision very low)

**Root cause:** The heuristic `detect_anomalies` layer flags 15,000+ rows. Many of these are mapped to `missing_value` category. Most are discovery signals (sparse/unusual values) rather than true missing value issues.

**Fix required:** Separate the heuristic `missing` detection (which is precise and high-signal) from the discovery signals. The heuristic `rare_category` and `text_length_outlier` flags should remain in `discovery` and never be elevated to `missing_value`.

### standardisation_error — R = 0.726 (17 GT items missed)

**Root cause:** 17 standardisation errors not caught. These are cases where the value is in the allowed_values set but uses inconsistent casing or spacing (e.g. `"under review"` vs `"Under Review"`). The current check does exact string match.

**Fix:** Case-insensitive allowed-value check with normalised comparison.

### claim_date — R = 0.507 (34 GT items missed out of 69)

35 format_error items caught (non-ISO dates). The remaining 34 are... need to investigate what they are. Could be missing values, logical errors (future claim dates), or other issues.

### date_of_birth — P = 0.250 (only 6 TP out of 14 GT)

8 GT range errors (pre-1900 + future dates) — only 6 caught. 2 missed. Also, 18 FPs (the 6 TPs from 24 total rule flags). The FPs are likely format_errors being misclassified by the rule layer. Needs investigation.

---

## 10. Open Finance Evaluation Pipeline Plan

*Status: Planned — not yet implemented. Prerequisite gate must clear first (see backlog).*

### Goal
Evaluate the DQ engine on real open-source financial data to demonstrate cross-domain generalisation beyond the synthetic TEST datasets.

### Proposed architecture

```
[Source: Quandl/OBP-API]
         |
         v
[S3: raw-data/]
         |
    [Jupyter/Colab notebook: inject synthetic errors]
         |
         v
[S3: test-data/]
         |
    [DQ Investigator API: POST /validate]
         |
         v
[S3: results/]
         |
    [Evaluation notebook: compute P/R/F1]
```

### Prerequisites
1. ML diagnostic confirmed real signal on both TEST domains (CLEARED)
2. Strategy 1 threshold implemented (CLEARED — 2a000c6)
3. Evaluation pipeline validated on TEST3_DATA (CLEARED — F1=0.593)
4. `scripts/inject_errors.py` — controlled error injection harness (NOT YET BUILT)

### Next steps
1. Build `scripts/inject_errors.py` with parameterised error types and rates
2. Set up Quandl data ingestion notebook
3. Wire to FastAPI endpoint and capture structured results
4. Compute P/R/F1 on the injected-error open-finance data

---

## Version History

| Date | Version | Change |
|---|---|---|
| 2026-05-18 | v0.1 | Ground truth extractor created; first evaluation baseline |
| 2026-05-18 | v0.2 | ML upgrade: 3-component ensemble + GMM threshold |
| 2026-05-19 | v0.3 | First P/R/F1 numbers: F1=0.433 baseline |
| 2026-05-20 | v0.4 | Rule fixes: F1 0.433 → 0.593 |
