"""
extract_ground_truth_test3.py
─────────────────────────────
Produces a precise row-level ground truth file for TEST3_DATA/insurance_claims.csv
by algorithmically detecting every injected error type documented in error_manifest.json.

Output: scripts/output/ground_truth_test3.csv
Schema:  row_id | column | issue_type | original_value | notes

Run:
    python scripts/extract_ground_truth_test3.py

Requirements: pandas, numpy, python-dateutil
    pip install pandas numpy python-dateutil
"""

import re
import os
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE, "TEST3_DATA", "main_data", "insurance_claims.csv")
CORPUS_DIR     = os.path.join(BASE, "TEST3_DATA", "corpus")
MANIFEST_PATH  = os.path.join(BASE, "TEST3_DATA", "error_manifest.json")
OUTPUT_DIR     = os.path.join(BASE, "scripts", "output")
OUTPUT_PATH    = os.path.join(OUTPUT_DIR, "ground_truth_test3.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
print(f"  {len(df)} rows, {len(df.columns)} columns")

# ─────────────────────────────────────────────────────────────────────────────
# Load corpus files
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus_canonicals(filename):
    """Return set of canonical values from a corpus alias file."""
    path = os.path.join(CORPUS_DIR, filename)
    c = pd.read_csv(path, dtype=str)
    return set(c["canonical"].dropna().str.strip().unique())

def load_corpus_all_values(filename):
    """Return set of ALL values (alias + canonical) — anything in this set is recognised."""
    path = os.path.join(CORPUS_DIR, filename)
    c = pd.read_csv(path, dtype=str)
    aliases   = set(c["alias"].dropna().str.strip().unique())
    canonicals = set(c["canonical"].dropna().str.strip().unique())
    return aliases | canonicals

insurer_canonicals  = load_corpus_canonicals("corpus_insurer_names.csv")
insurer_all         = load_corpus_all_values("corpus_insurer_names.csv")
status_canonicals   = load_corpus_canonicals("corpus_claim_status.csv")
status_all          = load_corpus_all_values("corpus_claim_status.csv")

# diagnosis codes: only the code column matters
diag_path = os.path.join(CORPUS_DIR, "corpus_diagnosis_codes.csv")
diag_df   = pd.read_csv(diag_path, dtype=str)
valid_diag_codes = set(diag_df["code"].dropna().str.strip().unique())

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}$")
POLICY_RE   = re.compile(r"^POL-\d{6}$")
PHONE_RE    = re.compile(r"^\+44\s7\d{3}\s\d{6}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Known legitimate email domains for this synthetic dataset.
# Any structurally valid email whose domain is NOT in this set is a format error
# (misspelled domain such as gmial.com, outlok.com, hotmial.com, etc.)
KNOWN_GOOD_EMAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "yahoo.com",
    "outlook.com", "icloud.com", "btinternet.com",
}

TODAY = date.today()
MIN_DOB = date(1900, 1, 1)
MAX_PLAUSIBLE_DOB = date(TODAY.year - 110, TODAY.month, TODAY.day)

def is_blank(val):
    return pd.isna(val) or str(val).strip() == ""

def try_parse_date(val):
    """Return a date object or None."""
    val = str(val).strip()
    if not val:
        return None
    try:
        return dateutil_parser.parse(val, dayfirst=False).date()
    except Exception:
        return None

def strip_currency(val):
    """Strip £, $, commas and return float or None."""
    val = str(val).strip().replace("£", "").replace("$", "").replace(",", "")
    try:
        return float(val)
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Accumulate ground truth rows
# ─────────────────────────────────────────────────────────────────────────────
records = []

def flag(row_id, column, issue_type, original_value, notes=""):
    records.append({
        "row_id":         row_id,
        "column":         column,
        "issue_type":     issue_type,
        "original_value": str(original_value),
        "notes":          notes,
    })

print("Scanning for injected errors...")

for idx, row in df.iterrows():

    # ── 1. Completely empty records ──────────────────────────────────────────
    non_empty = sum(1 for v in row.values if not is_blank(v))
    if non_empty == 0:
        flag(idx, "all_columns", "missing_value", "", "Completely empty record")
        continue   # no point checking individual columns on an empty row

    # ── 2. Missing values — email ────────────────────────────────────────────
    if is_blank(row.get("email", "")):
        flag(idx, "email", "missing_value", row.get("email", ""), "Blank email")

    # ── 3. Format error — email (bad domain) ─────────────────────────────────
    else:
        email_val = str(row.get("email", "")).strip()
        if not EMAIL_RE.match(email_val):
            flag(idx, "email", "format_error", email_val, "Invalid email structure")
        else:
            domain = email_val.split("@")[-1].lower()
            if domain not in KNOWN_GOOD_EMAIL_DOMAINS:
                flag(idx, "email", "format_error", email_val,
                     f"Misspelled email domain: {domain}")

    # ── 4. Missing value — phone ─────────────────────────────────────────────
    if is_blank(row.get("phone", "")):
        flag(idx, "phone", "missing_value", row.get("phone", ""), "Blank phone")

    # ── 5. Missing value — postcode ──────────────────────────────────────────
    pc = str(row.get("postcode", "")).strip()
    if is_blank(pc):
        flag(idx, "postcode", "missing_value", pc, "Blank postcode")

    # ── 6. Format error — postcode (no space / wrong case) ───────────────────
    elif not POSTCODE_RE.match(pc.upper()):
        flag(idx, "postcode", "format_error", pc, "Postcode not in valid UK format (missing space or bad pattern)")

    # ── 7. Format error — policy_id ──────────────────────────────────────────
    pid = str(row.get("policy_id", "")).strip()
    if not is_blank(pid) and not POLICY_RE.match(pid):
        flag(idx, "policy_id", "format_error", pid, "Policy ID not in POL-XXXXXX format")

    # ── 8. Format error — claimant_name whitespace ───────────────────────────
    name = str(row.get("claimant_name", ""))
    if name != name.strip():
        flag(idx, "claimant_name", "format_error", name, "Leading/trailing whitespace in name")

    # ── 9. Standardisation error — city capitalisation ───────────────────────
    city = str(row.get("city", "")).strip()
    if city and (city != city.title()) and city == city.lower():
        flag(idx, "city", "standardisation_error", city, "City not properly capitalised")

    # ── 10. Corpus mismatch — insurer ────────────────────────────────────────
    insurer = str(row.get("insurer", "")).strip()
    if not is_blank(insurer) and insurer not in insurer_canonicals:
        flag(idx, "insurer", "corpus_mismatch", insurer, "Insurer name not in canonical form")

    # ── 11. Standardisation error — claim_status ─────────────────────────────
    status = str(row.get("claim_status", "")).strip()
    if not is_blank(status) and status not in status_canonicals:
        flag(idx, "claim_status", "standardisation_error", status, "Claim status not standardised")

    # ── 12. Format error — claim_amount contains currency symbol ─────────────
    amt_raw = str(row.get("claim_amount", "")).strip()
    if "£" in amt_raw or "$" in amt_raw:
        flag(idx, "claim_amount", "format_error", amt_raw, "Claim amount contains currency symbol")

    # ── 13. Range error — negative claim_amount ──────────────────────────────
    amt_num = strip_currency(amt_raw)
    if amt_num is not None and amt_num < 0:
        flag(idx, "claim_amount", "range_error", amt_raw, "Negative claim amount")

    # ── 14. Logical error — claim_amount exceeds coverage_limit ─────────────
    cov_raw = str(row.get("coverage_limit", "")).strip()
    cov_num = strip_currency(cov_raw)
    if amt_num is not None and cov_num is not None and amt_num > 0 and amt_num > cov_num:
        flag(idx, "claim_amount", "logical_error", amt_raw,
             f"Claim amount ({amt_num}) exceeds coverage limit ({cov_num})")

    # ── 15. Format error — date_of_birth (must be ISO) ───────────────────────
    dob_raw = str(row.get("date_of_birth", "")).strip()
    dob = None
    if not is_blank(dob_raw):
        is_iso_dob = ISO_DATE_RE.match(dob_raw)
        if not is_iso_dob:
            flag(idx, "date_of_birth", "format_error", dob_raw, "DOB not in ISO YYYY-MM-DD format")
        # Always try to parse regardless of format — catches range issues in non-ISO dates too
        dob = try_parse_date(dob_raw)

    # ── 16. Range error — DOB in future ──────────────────────────────────────
    if dob and dob > TODAY:
        flag(idx, "date_of_birth", "range_error", dob_raw, "Date of birth is in the future")

    # ── 17. Range error — DOB implies age > 110 ──────────────────────────────
    elif dob and dob < MAX_PLAUSIBLE_DOB:
        flag(idx, "date_of_birth", "range_error", dob_raw, "Date of birth implies age > 110 years")

    # ── 18. Format error — claim_date not ISO ────────────────────────────────
    cd_raw = str(row.get("claim_date", "")).strip()
    cd = None
    if not is_blank(cd_raw):
        if not ISO_DATE_RE.match(cd_raw):
            flag(idx, "claim_date", "format_error", cd_raw, "Claim date not in ISO YYYY-MM-DD format")
        else:
            cd = try_parse_date(cd_raw)

    # ── 19. Logical error — claim_date before policy_start_date ─────────────
    ps_raw = str(row.get("policy_start_date", "")).strip()
    ps = try_parse_date(ps_raw)
    if cd and ps and cd < ps:
        flag(idx, "claim_date", "logical_error", cd_raw,
             f"Claim date ({cd_raw}) is before policy start date ({ps_raw})")

    # ── 20. Corpus mismatch — diagnosis_code (if present, must be valid ICD-10)
    diag = str(row.get("diagnosis_code", "")).strip()
    if not is_blank(diag) and diag not in valid_diag_codes:
        flag(idx, "diagnosis_code", "corpus_mismatch", diag, "Diagnosis code not in valid ICD-10 list")

# ── 21. Exact duplicates ─────────────────────────────────────────────────────
print("  Checking for exact duplicates...")
dup_mask = df.duplicated(keep=False)
for idx in df[dup_mask].index:
    flag(idx, "all_columns", "duplicate", "", "Exact duplicate row")

# ── 22. Near duplicates (claimant_name) ─────────────────────────────────────
# Use a simple approach: group by (policy_id, claim_type) where the same
# policy appears more than once with slightly different claimant names.
# This catches the "same person, minor name variation" pattern from the manifest.
print("  Checking for near duplicates...")
try:
    from difflib import SequenceMatcher

    grouped = df.groupby("policy_id")
    near_dup_rows = set()
    for pol_id, grp in grouped:
        if len(grp) < 2:
            continue
        names = grp["claimant_name"].tolist()
        indices = grp.index.tolist()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ratio = SequenceMatcher(
                    None,
                    str(names[i]).strip().lower(),
                    str(names[j]).strip().lower()
                ).ratio()
                # Similar but not identical names on same policy = near duplicate
                if 0.7 < ratio < 1.0:
                    near_dup_rows.add(indices[i])
                    near_dup_rows.add(indices[j])

    for idx in near_dup_rows:
        flag(idx, "claimant_name", "near_duplicate",
             df.at[idx, "claimant_name"],
             "Near-duplicate claim: same policy_id, similar claimant name")
except Exception as e:
    print(f"  Warning: near-duplicate check failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Build output dataframe and save
# ─────────────────────────────────────────────────────────────────────────────
gt = pd.DataFrame(records, columns=["row_id", "column", "issue_type", "original_value", "notes"])
gt = gt.sort_values(["row_id", "column"]).reset_index(drop=True)
gt.to_csv(OUTPUT_PATH, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GROUND TRUTH EXTRACTION COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT_PATH}")
print(f"Total ground truth records: {len(gt)}")
print(f"Unique rows with at least one error: {gt['row_id'].nunique()}")
print()
print("Breakdown by issue_type:")
by_type = gt.groupby("issue_type").size().sort_values(ascending=False)
for itype, count in by_type.items():
    print(f"  {itype:<30} {count:>4}")
print()
print("Breakdown by column:")
by_col = gt.groupby("column").size().sort_values(ascending=False)
for col, count in by_col.items():
    print(f"  {col:<30} {count:>4}")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-check against manifest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CROSS-CHECK vs error_manifest.json")
print("=" * 60)
print(f"{'Error type':<30} {'Column':<20} {'Manifest':<10} {'Extracted':<10} {'Match?'}")
print("-" * 80)

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

for entry in manifest["errors"]:
    etype    = entry["error_type"]
    col      = entry["column"]
    expected = entry["affected_rows"]
    extracted = len(gt[(gt["issue_type"] == etype) & (gt["column"] == col)])
    # Duplicate/near-duplicate: manifest counts only added copies; we flag ALL
    # occurrences (original + copy) — expected difference. Mark as OK with note.
    if etype in ("duplicate", "near_duplicate"):
        note = "(all occurrences flagged)"
        match = "OK*"
    else:
        note = ""
        match = "OK" if abs(extracted - expected) <= max(3, expected * 0.15) else "CHECK"
    print(f"  {etype:<28} {col:<20} {expected:<10} {extracted:<10} {match}  {note}")

print()
print("Notes:")
print("  OK   = extracted within 15% of manifest count")
print("  OK*  = duplicate/near_duplicate: manifest counts added copies only;")
print("         ground truth flags ALL occurrences (original + copies).")
print("         Both are valid — depends on evaluation methodology.")
print("  CHECK = extracted differs by >15% — investigate before using for P/R.")
print()
print("Duplicate methodology chosen: keep=False (flag all occurrences).")
print("Rationale: DQ engine flags all members of a duplicate group; ground truth")
print("should match. For stricter evaluation, filter ground truth to keep=first.")
print()
print(f"Ground truth file ready: {OUTPUT_PATH}")
