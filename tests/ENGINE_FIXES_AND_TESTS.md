# Engine Precision Fixes (P1–P3) and Test Harness

This document records the correctness fixes made to the core validation engine,
the datasets used to test them, the test cases that guard them, and the known
limitations that remain. It is the companion to `tests/test_engine_quality.py`.

Run the harness:

```bash
./.venv/bin/python -m pytest tests/test_engine_quality.py -v
```

---

## 1. What was wrong

Live testing showed the engine was **high-recall but low-precision** — it found
real errors but flagged large numbers of valid values, especially in free-text
columns. On a *clean* Companies House sample it flagged ~22% of all cells.

Four independent root causes were identified and fixed.

| ID | Symptom | Root cause | Fix |
|----|---------|-----------|-----|
| **P1a** | 49 of 50 valid emails flagged as "misspelling of a known provider" | `validate.py` email domain heuristic used a loose 0.7 similarity ratio; `x.com` scored ~0.73 vs `bt.com` | Only flag genuine near-typos: domain ≥6 chars, not already a known provider, within a 1–2 character edit of a similar-length provider (`_edit_distance`) |
| **P1b** | Free-text columns (names, addresses) mass-flagged by `regex mismatch` | Inferred char-class regex was applied to every column and never pruned in the inference path | Apply `prune_rules_by_quality` inside `infer_rules_from_unclean` (default on); drop a column's regex when the top-8 patterns cover < 90% of values (free-text) |
| **P2** | Missing `N/A` in `company_name` not flagged, but flagged in other columns | `validate_df` skipped columns with no inferred rule, and presence was never inferred | Infer `required` / `nullable` for **every** column (present in ≥85% of rows → required) so missing is checked consistently |
| **P3** | A stray `gbp`/`EURO` among `GBP` not flagged | Allowed-value set was inferred from **all** distinct values, so rare errors became "allowed" | Infer `allowed_values` only from values seen ≥ 2 times (`MIN_ALLOWED_VALUE_COUNT`) |

Two further precision issues surfaced while testing and were also fixed:

- **Alphanumeric IDs mis-typed as numeric** (`company_number` like `SC123456` →
  17 "not number" false positives). Fixed by tightening numeric inference to
  require > 99% of values to parse (`_looks_numeric_series`).
- **Near-unique dates treated as unique** (`date_of_creation` → 13 false
  `duplicate_value`). Fixed by raising `UNIQUE_RATIO_THR` to 0.99 and only
  inferring uniqueness for text columns (numeric/date columns return earlier).

Additional hardening: inference now runs on **present values only** (placeholder
tokens like `""`/`N/A` are excluded), so an empty-string `^$` pattern can no
longer pollute a column's regex; and regex is skipped entirely on columns already
captured as categoricals (redundant, adds only false-positive risk).

Files changed: `core/validator/discover.py`, `core/validator/validate.py`.

---

## 2. Result

Clean Companies House sample (300 rows × 11 columns = 3,300 cells):

| Stage | False-positive cells | Rate |
|-------|----------------------|------|
| Before any fix | ~745 | ~22% |
| After P1a + P1b + P2 + P3 | 88 | 2.7% |
| After ID/uniqueness/pattern-retention fixes | **23** | **0.70%** |

Of the final 23, two are genuinely-missing localities (correct), two are rare
categories (borderline), and the rest are residual postcode patterns (see
Limitations). Detection of injected errors (typos, invalid categories,
out-of-range dates, missing values) is preserved — see the positive-detection
tests.

The fixes are **not Companies-House-specific**: on the insurance domain
(`TEST3_DATA`) the flag rate on a 250-row base is 1.72% and injected errors are
caught.

---

## 3. Datasets used for testing

All are already in the repository. No new bulk datasets were created; unit-test
fixtures are generated inline and documented in each test.

| Dataset | Location | Role in tests |
|---------|----------|---------------|
| Companies House (clean) | `companies_house_clean.csv` (4,662 rows, 11 cols) | Real-data false-positive rate + injected-error recall |
| Insurance claims (TEST3) | `TEST3_DATA/main_data/insurance_claims.csv` (1,085 rows, 20 cols) | Second-domain generalisation |
| Insurance corpora (TEST3) | `TEST3_DATA/corpus/*.csv` | Reference categorical value lists (policy/claim/insurer/diagnosis) |
| Banking (TEST2) | `TEST2_DATA/` | Available for future cross-domain runs (not yet in the harness) |
| Inline fixtures | generated in `test_engine_quality.py` | Deterministic P1–P3 and edge-case cases |

Inline fixtures are used for the precision tests because they need **known
ground truth** (which cells are valid vs corrupted) with no ambiguity. Each
fixture is described in the docstring of its test.

---

## 4. Test cases (`tests/test_engine_quality.py`)

| Test | Guards |
|------|--------|
| `test_p1a_valid_emails_not_flagged` | P1a — the 50-email regression: only the malformed one is flagged |
| `test_p1a_provider_typo_still_flagged` | P1a — `gmial.com` is still caught |
| `test_p1a_ordinary_company_domains_not_flagged` | P1a — `x.com`, `acme.io`, `foo.co`, `bigspark.ai` never flagged |
| `test_p1b_freetext_names_not_massflagged` | P1b — free-text names produce 0 flags; regex is pruned |
| `test_p2_missing_in_required_column_flagged` | P2 — `N/A`/`""`/`null`/`-`/`none` in a required column flagged |
| `test_p2_missing_in_nullable_column_not_flagged` | P2 — blanks in a mostly-empty (nullable) column not flagged |
| `test_p3_rare_invalid_category_flagged` | P3 — stray `gbp`/`EURO` flagged; allowed set = `['GBP']` |
| `test_p3_valid_categories_not_flagged` | P3 — valid balanced categories all pass |
| `test_positive_invalid_category_flagged` | Recall — typo'd category `Actvie` caught |
| `test_positive_out_of_range_date_flagged` | Recall — impossible date `3019` caught |
| `test_positive_numeric_out_of_range_flagged` | Recall — numeric outlier caught against a clean reference |
| `test_real_data_low_false_positive_rate` | Real CH false-positive rate < 2% |
| `test_real_data_injected_errors_caught` | Real CH — injected errors recovered |
| `test_insurance_domain_not_massflagged_and_catches_injected` | Second-domain generalisation |
| `test_edge_empty_dataframe` … `test_edge_unicode_and_symbols` | Edge cases: empty / 1-row / single-column / all-missing / all-numeric / unicode — no crashes |

---

## 5. Known limitations (documented, not bugs)

- **UK postcodes via inferred regex.** Postcodes have more distinct structural
  forms (`A9 9AA`, `A99 9AA`, `AA9A 9AA`, …) than the 8-pattern cap used to
  distinguish structured from free-text columns. Rare forms in a sample can be
  flagged. The correct fix for postcode-heavy data is to supply a schema rule
  with the proper UK postcode pattern, or use the postcode corpus — both are
  already supported. Raising the pattern cap was rejected because it re-breaks
  free-text pruning.
- **Numeric out-of-range needs a clean reference.** Bounds inferred from data
  that already contains an outlier absorb that outlier. Detection of numeric
  out-of-range therefore requires a clean reference (or provided bounds), which
  is how the batch pipeline is intended to run.
- **Rare-but-valid categories.** Excluding values seen only once from the allowed
  set (P3) can flag a legitimate one-off category. This is an intentional
  weak-supervision trade-off favouring recall of invalid categories.
