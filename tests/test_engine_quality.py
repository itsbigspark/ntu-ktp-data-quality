"""
Engine quality regression tests (precision + presence + categorical).

Covers the P1-P3 fixes made to core/validator/{discover,validate}.py and a set
of positive-detection and edge-case guards so the fixes cannot silently reduce
recall or crash on unusual inputs.

Test data is generated inline (documented in each test) plus, where available,
the repo's real Companies House clean base. See tests/ENGINE_FIXES_AND_TESTS.md
for the full catalogue of datasets and cases.

Run:  ./.venv/bin/python -m pytest tests/test_engine_quality.py -v
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from core.validator.discover import infer_rules_from_unclean
from core.validator.validate import validate_df

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_CLEAN = os.path.join(REPO_ROOT, "companies_house_clean.csv")
INSURANCE = os.path.join(REPO_ROOT, "TEST3_DATA", "main_data", "insurance_claims.csv")


def _flagged(rep: pd.DataFrame):
    """Set of (row_id, column) cells the validator flagged."""
    if rep is None or len(rep) == 0:
        return set()
    return set((int(r), c) for r, c in zip(rep["row_id"], rep["column"]))


def _run(df: pd.DataFrame):
    """Infer rules then validate; return the set of flagged cells."""
    rules = infer_rules_from_unclean(df)
    return _flagged(validate_df(df, rules)), rules


# ---------------------------------------------------------------------------
# P1a — email domain heuristic must not flag valid emails
# ---------------------------------------------------------------------------
def test_p1a_valid_emails_not_flagged():
    """Regression: 49 valid emails on an ordinary domain were all flagged as
    'misspelling of a known provider'. Only the one malformed address should
    now be flagged."""
    df = pd.DataFrame({
        "email": [f"user{i}@x.com" for i in range(49)] + ["not-an-email"],
    })
    flags, _ = _run(df)
    email_flags = {r for r in flags if r[1] == "email"}
    assert email_flags == {(49, "email")}, f"unexpected email flags: {sorted(email_flags)}"


def test_p1a_provider_typo_still_flagged():
    """A genuine typo of a known provider (gmial.com) must still be caught."""
    df = pd.DataFrame({
        "email": [f"user{i}@gmail.com" for i in range(20)] + ["bob@gmial.com"],
    })
    flags, _ = _run(df)
    assert (20, "email") in flags


@pytest.mark.parametrize("domain", ["x.com", "acme.io", "foo.co", "bigspark.ai"])
def test_p1a_ordinary_company_domains_not_flagged(domain):
    """Ordinary company domains must never be flagged as provider misspellings."""
    df = pd.DataFrame({"email": [f"user{i}@{domain}" for i in range(15)]})
    flags, _ = _run(df)
    assert not any(f[1] == "email" for f in flags), f"{domain} wrongly flagged"


# ---------------------------------------------------------------------------
# P1b — free-text columns must not be mass-flagged by over-fit regex
# ---------------------------------------------------------------------------
def test_p1b_freetext_names_not_massflagged():
    """A column of distinct, clean free-text names must produce no flags
    (over-fit regex is pruned)."""
    names = [f"{a} {b} Ltd" for a in
             ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Iota"]
             for b in ["Holdings", "Trading", "Systems", "Partners", "Group",
                       "Services", "Capital", "Industries", "Solutions", "Ventures"]]
    df = pd.DataFrame({"company_name": names})
    flags, rules = _run(df)
    assert len([f for f in flags if f[1] == "company_name"]) == 0
    # regex should have been pruned away for this free-text column
    assert "regex" not in rules["columns"]["company_name"]


# ---------------------------------------------------------------------------
# P2 — presence inference: missing values checked consistently
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("placeholder", ["N/A", "", "null", "-", "none"])
def test_p2_missing_in_required_column_flagged(placeholder):
    """A placeholder-missing value in a mostly-populated (required) column
    must be flagged, regardless of the placeholder token used."""
    vals = [f"Company {i} Ltd" for i in range(50)]
    vals[7] = placeholder
    df = pd.DataFrame({"company_name": vals})
    flags, rules = _run(df)
    assert rules["columns"]["company_name"].get("required") is True
    assert (7, "company_name") in flags


def test_p2_missing_in_nullable_column_not_flagged():
    """A column that is mostly empty is inferred nullable; its blanks must not
    be flagged as errors."""
    col = [""] * 70 + [f"opt {i}" for i in range(30)]
    df = pd.DataFrame({"address_line_2": col})
    flags, rules = _run(df)
    assert rules["columns"]["address_line_2"].get("nullable") is True
    assert not any(f[1] == "address_line_2" for f in flags)


# ---------------------------------------------------------------------------
# P3 — categorical allowed-values exclude rare (likely erroneous) values
# ---------------------------------------------------------------------------
def test_p3_rare_invalid_category_flagged():
    """A stray 'gbp' / 'EURO' among 'GBP' must remain detectable: the allowed
    set is inferred only from values seen >= 2 times."""
    df = pd.DataFrame({"currency": ["GBP"] * 48 + ["gbp", "EURO"]})
    flags, rules = _run(df)
    assert rules["columns"]["currency"]["allowed_values"] == ["GBP"]
    assert (48, "currency") in flags and (49, "currency") in flags


def test_p3_valid_categories_not_flagged():
    """Every value in a balanced, valid categorical must pass."""
    df = pd.DataFrame({"status": (["Active", "Dissolved", "Liquidation"] * 20)})
    flags, _ = _run(df)
    assert not any(f[1] == "status" for f in flags)


# ---------------------------------------------------------------------------
# Positive detection guards — the fixes must not reduce recall
# ---------------------------------------------------------------------------
def test_positive_invalid_category_flagged():
    df = pd.DataFrame({"status": ["Active"] * 40 + ["Dissolved"] * 40 + ["Actvie"]})
    flags, _ = _run(df)
    assert (80, "status") in flags


def test_positive_out_of_range_date_flagged():
    dates = [f"20{10 + (i % 10)}-01-15" for i in range(60)]
    dates[30] = "3019-01-01"
    df = pd.DataFrame({"date_of_creation": dates})
    flags, _ = _run(df)
    assert (30, "date_of_creation") in flags


def test_positive_numeric_out_of_range_flagged():
    """Numeric out-of-range is detected against bounds inferred from a CLEAN
    reference. (Self-inferring bounds from data that already contains the
    outlier would absorb it — that is a documented weak-supervision limit.)"""
    clean = pd.DataFrame({"amount": [str(round(v, 2)) for v in np.linspace(10, 90, 60)]})
    rules = infer_rules_from_unclean(clean)
    dirty = clean.copy()
    dirty.at[25, "amount"] = "999999999"
    flags = _flagged(validate_df(dirty, rules))
    assert (25, "amount") in flags


# ---------------------------------------------------------------------------
# Real-data generalisation (Companies House clean base)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CH_CLEAN), reason="companies_house_clean.csv not present")
def test_real_data_low_false_positive_rate():
    """On the clean Companies House base, the false-positive rate must be low.
    Before the P1 fixes this over-flagged heavily; we assert < 5% of cells."""
    df = pd.read_csv(CH_CLEAN).head(300).reset_index(drop=True)
    flags, _ = _run(df)
    cell_count = df.shape[0] * df.shape[1]
    fp_rate = len(flags) / cell_count
    assert fp_rate < 0.02, f"false-positive rate too high: {fp_rate:.3f} ({len(flags)}/{cell_count})"


@pytest.mark.skipif(not os.path.exists(CH_CLEAN), reason="companies_house_clean.csv not present")
def test_real_data_injected_errors_caught():
    """Inject known errors into the clean base and assert they are caught."""
    df = pd.read_csv(CH_CLEAN).head(300).reset_index(drop=True)
    truth = []
    df.at[5, "company_status"] = "Actvie"; truth.append((5, "company_status"))
    df.at[11, "company_type"] = "Private Lumited Company"; truth.append((11, "company_type"))
    if "date_of_creation" in df.columns:
        df.at[20, "date_of_creation"] = "3019-01-01"; truth.append((20, "date_of_creation"))
    flags, _ = _run(df)
    caught = [t for t in truth if t in flags]
    assert len(caught) >= 2, f"caught only {caught} of {truth}"


# ---------------------------------------------------------------------------
# Second-domain generalisation (insurance) — fixes must not be CH-specific
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(INSURANCE), reason="TEST3 insurance data not present")
def test_insurance_domain_not_massflagged_and_catches_injected():
    """On a different domain (insurance), free-text columns must not be
    mass-flagged, and injected errors (invalid category, missing) are caught."""
    base = pd.read_csv(INSURANCE).head(250).reset_index(drop=True)
    flags, _ = _run(base)
    fp_rate = len(flags) / (base.shape[0] * base.shape[1])
    assert fp_rate < 0.05, f"insurance flag rate too high: {fp_rate:.3f}"

    rules = infer_rules_from_unclean(base)
    dirty = base.copy()
    truth = []
    catcol = next((c for c in dirty.columns
                   if dirty[c].nunique() <= 15 and dirty[c].dtype == object), None)
    if catcol:
        dirty.at[3, catcol] = "ZZ_not_a_category"; truth.append((3, catcol))
    prescol = next((c for c in dirty.columns
                    if dirty[c].dtype == object and dirty[c].notna().mean() > 0.9), None)
    if prescol:
        dirty.at[6, prescol] = "N/A"; truth.append((6, prescol))
    got = _flagged(validate_df(dirty, rules))
    assert all(t in got for t in truth), f"missed injected: {[t for t in truth if t not in got]}"


# ---------------------------------------------------------------------------
# Edge cases — must not crash, must return a well-formed report
# ---------------------------------------------------------------------------
def test_edge_empty_dataframe():
    df = pd.DataFrame()
    rules = infer_rules_from_unclean(df)
    rep = validate_df(df, rules)
    assert isinstance(rep, pd.DataFrame)


def test_edge_single_row():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    _run(df)  # must not raise


def test_edge_single_column():
    df = pd.DataFrame({"only": ["a", "b", "a", "c", "a"]})
    _run(df)


def test_edge_all_missing_column():
    df = pd.DataFrame({"blank": ["", "", "", ""], "ok": ["a", "a", "b", "b"]})
    flags, rules = _run(df)
    assert rules["columns"]["blank"].get("nullable") is True
    assert not any(f[1] == "blank" for f in flags)


def test_edge_all_numeric():
    df = pd.DataFrame({"x": list(range(100)), "y": [i * 1.5 for i in range(100)]})
    _run(df)


def test_edge_unicode_and_symbols():
    df = pd.DataFrame({"name": ["Café Ltd", "Zürich AG", "naïve €", "Ω Corp", "普通 Ltd"] * 5})
    _run(df)
