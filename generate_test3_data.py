"""
generate_test3_data.py
Generates TEST3_DATA -- Insurance Claims Dataset (1000 rows)
Completely different domain from TEST2_DATA (banking transactions).
"""

import os
import json
import random
import string
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

random.seed(42)
np.random.seed(42)

OUT = "TEST3_DATA"
os.makedirs(f"{OUT}/main_data", exist_ok=True)
os.makedirs(f"{OUT}/reference", exist_ok=True)
os.makedirs(f"{OUT}/rules", exist_ok=True)
os.makedirs(f"{OUT}/corpus", exist_ok=True)
os.makedirs(f"{OUT}/entity_resolution", exist_ok=True)

# ── Lookup tables ──────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "James", "Sarah", "Mohammed", "Emily", "David", "Priya", "Robert", "Fatima",
    "John", "Charlotte", "Ali", "Jessica", "Thomas", "Amara", "William", "Hannah",
    "Michael", "Zoe", "Daniel", "Grace", "Ryan", "Olivia", "Matthew", "Sophie",
    "George", "Lucy", "Patrick", "Aisha", "Kevin", "Rebecca"
]

LAST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Evans", "Wilson",
    "Thomas", "Roberts", "Johnson", "Walker", "Wright", "Thompson", "White",
    "Hughes", "Edwards", "Green", "Hall", "Lewis", "Harris", "Clarke", "Patel",
    "Jackson", "Wood", "Turner", "Martin", "Cooper", "Hill", "Ward"
]

INSURERS = [
    "Aviva", "AXA UK", "Bupa", "Legal & General", "Zurich Insurance",
    "Allianz UK", "Direct Line", "RSA Insurance", "Liverpool Victoria", "Hastings Direct"
]

CLAIM_TYPES = [
    "Motor", "Home", "Health", "Life", "Travel", "Liability", "Pet", "Business"
]

CLAIM_STATUS = [
    "Submitted", "Under Review", "Approved", "Rejected", "Settled", "Closed"
]

DIAGNOSIS_CODES = [
    "M54.5", "J06.9", "I10", "E11.9", "F41.1", "K21.0", "M79.3",
    "G43.9", "R51", "L30.9", "J45.9", "N39.0"
]

POLICY_TYPES = [
    "Comprehensive", "Third Party", "Third Party Fire & Theft",
    "Standard", "Premium", "Basic", "Gold", "Silver"
]

CITIES = [
    "London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Liverpool",
    "Bristol", "Sheffield", "Edinburgh", "Cardiff", "Nottingham", "Leicester"
]

POSTCODES_VALID = [
    "SW1A 1AA", "M1 1AE", "B1 1BB", "LS1 1BA", "G1 1PT", "L1 0AB",
    "BS1 4DJ", "S1 2HH", "EH1 1YZ", "CF10 1EP", "NG1 5FS", "LE1 6ZZ",
    "EC1A 1BB", "WC2N 5DU", "E1 6RF", "N1 9GU", "SE1 7PB", "W1A 1AA"
]

# ── Helper functions ──────────────────────────────────────────────────────────

def rand_policy_id():
    return f"POL-{random.randint(100000, 999999)}"

def rand_claim_id():
    return f"CLM-{random.randint(10000, 99999)}"

def rand_date(start_year=2020, end_year=2024):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def rand_phone():
    return f"+44 7{random.randint(100,999)} {random.randint(100000,999999)}"

def rand_email(first, last):
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "btinternet.com"]
    sep = random.choice([".", "_", ""])
    return f"{first.lower()}{sep}{last.lower()}@{random.choice(domains)}"

def rand_amount(min_val=100, max_val=50000):
    return round(random.uniform(min_val, max_val), 2)

def rand_postcode():
    return random.choice(POSTCODES_VALID)


# ── Generate clean base records ────────────────────────────────────────────────

def generate_clean_record(idx):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    policy_start = rand_date(2018, 2022)
    policy_end = policy_start + timedelta(days=365)
    claim_date = rand_date(2023, 2024)
    dob = date(random.randint(1950, 2000), random.randint(1, 12), random.randint(1, 28))
    coverage = rand_amount(10000, 200000)
    claim_amount = rand_amount(100, min(coverage * 0.8, 50000))

    return {
        "claim_id": rand_claim_id(),
        "policy_id": rand_policy_id(),
        "claimant_name": f"{first} {last}",
        "date_of_birth": dob.isoformat(),
        "email": rand_email(first, last),
        "phone": rand_phone(),
        "address": f"{random.randint(1, 200)} {random.choice(['High Street', 'Church Lane', 'Victoria Road', 'Park Avenue', 'Station Road', 'Mill Lane'])}",
        "city": random.choice(CITIES),
        "postcode": rand_postcode(),
        "insurer": random.choice(INSURERS),
        "policy_type": random.choice(POLICY_TYPES),
        "claim_type": random.choice(CLAIM_TYPES),
        "policy_start_date": policy_start.isoformat(),
        "policy_end_date": policy_end.isoformat(),
        "claim_date": claim_date.isoformat(),
        "claim_amount": claim_amount,
        "coverage_limit": round(coverage, 2),
        "diagnosis_code": random.choice(DIAGNOSIS_CODES) if random.random() < 0.6 else "",
        "claim_status": random.choice(CLAIM_STATUS),
        "agent_id": f"AGT{random.randint(100, 999)}",
    }


records = [generate_clean_record(i) for i in range(1000)]
df = pd.DataFrame(records)

# Track errors injected
error_log = []

def log_error(row_ids, error_type, column, description, count):
    error_log.append({
        "error_type": error_type,
        "column": column,
        "description": description,
        "affected_rows": count,
        "example_rows": row_ids[:3]
    })


# ── Inject errors ─────────────────────────────────────────────────────────────

# 1. Bad email format (~60 rows)
bad_email_idx = random.sample(range(1000), 60)
email_typos = ["gmial.com", "yahooo.com", "outlok.com", "hotmial.com", "iclod.com"]
for i in bad_email_idx:
    df.at[i, "email"] = df.at[i, "email"].split("@")[0] + "@" + random.choice(email_typos)
log_error(bad_email_idx, "format_error", "email", "Misspelled email domain (gmial.com, outlok.com etc.)", 60)

# 2. Missing email (~20 rows)
missing_email_idx = random.sample([i for i in range(1000) if i not in bad_email_idx], 20)
for i in missing_email_idx:
    df.at[i, "email"] = ""
log_error(missing_email_idx, "missing_value", "email", "Blank email address", 20)

# 3. Missing phone (~30 rows)
missing_phone_idx = random.sample(range(1000), 30)
for i in missing_phone_idx:
    df.at[i, "phone"] = ""
log_error(missing_phone_idx, "missing_value", "phone", "Missing phone number", 30)

# 4. Bad postcode format -- no space (~25 rows)
bad_postcode_idx = random.sample(range(1000), 25)
for i in bad_postcode_idx:
    df.at[i, "postcode"] = df.at[i, "postcode"].replace(" ", "")
log_error(bad_postcode_idx, "format_error", "postcode", "Postcode missing space (SW1A1AA instead of SW1A 1AA)", 25)

# 5. Missing postcode (~15 rows)
missing_post_idx = random.sample([i for i in range(1000) if i not in bad_postcode_idx], 15)
for i in missing_post_idx:
    df.at[i, "postcode"] = ""
log_error(missing_post_idx, "missing_value", "postcode", "Missing postcode", 15)

# 6. Insurer name variants (~50 rows)
insurer_variants = {
    "Aviva": ["aviva", "AVIVA", "Aviva plc", "Aviva Insurance"],
    "AXA UK": ["AXA", "axa uk", "AXA Insurance UK", "Axa"],
    "Bupa": ["BUPA", "bupa", "Bupa Insurance", "BUPA Health"],
    "Legal & General": ["L&G", "Legal and General", "legal & general", "L & G"],
    "Zurich Insurance": ["Zurich", "ZURICH", "zurich insurance", "Zurich UK"],
}
insurer_variant_idx = random.sample(range(1000), 50)
for i in insurer_variant_idx:
    canonical = df.at[i, "insurer"]
    if canonical in insurer_variants:
        df.at[i, "insurer"] = random.choice(insurer_variants[canonical])
    else:
        df.at[i, "insurer"] = df.at[i, "insurer"].lower()
log_error(insurer_variant_idx, "corpus_mismatch", "insurer", "Insurer name variants (AXA vs AXA UK vs axa uk)", 50)

# 7. Claim status inconsistency (~40 rows)
status_variants = {
    "Submitted": ["submitted", "SUBMITTED", "Submited", "Submit"],
    "Under Review": ["under review", "UNDER REVIEW", "In Review", "Reviewing"],
    "Approved": ["approved", "APPROVED", "Approvd", "approve"],
    "Rejected": ["rejected", "REJECTED", "Rejectd", "Decline"],
    "Settled": ["settled", "SETTLED", "Settld", "Complete"],
    "Closed": ["closed", "CLOSED", "Closd", "Done"],
}
status_idx = random.sample(range(1000), 40)
for i in status_idx:
    s = df.at[i, "claim_status"]
    if s in status_variants:
        df.at[i, "claim_status"] = random.choice(status_variants[s])
log_error(status_idx, "standardisation_error", "claim_status", "Claim status not standardised (APPROVED, approvd, approve)", 40)

# 8. Claim amount with currency symbol (~25 rows)
currency_idx = random.sample(range(1000), 25)
for i in currency_idx:
    df.at[i, "claim_amount"] = f"£{df.at[i, 'claim_amount']}"
log_error(currency_idx, "format_error", "claim_amount", "Claim amount stored as string with £ symbol", 25)

# 9. Claim amount exceeds coverage limit (~20 rows) -- logical error
exceed_idx = random.sample(range(1000), 20)
for i in exceed_idx:
    df.at[i, "claim_amount"] = round(df.at[i, "coverage_limit"] * random.uniform(1.1, 2.0), 2)
log_error(exceed_idx, "logical_error", "claim_amount", "Claim amount exceeds coverage limit (business rule violation)", 20)

# 10. Negative claim amount (~12 rows)
neg_idx = random.sample([i for i in range(1000) if i not in exceed_idx and i not in currency_idx], 12)
for i in neg_idx:
    df.at[i, "claim_amount"] = -abs(df.at[i, "claim_amount"])
log_error(neg_idx, "range_error", "claim_amount", "Negative claim amount", 12)

# 11. Claim date before policy start date (~30 rows) -- logical error
logic_date_idx = random.sample(range(1000), 30)
for i in logic_date_idx:
    # Set claim date to before policy start
    policy_start = pd.to_datetime(df.at[i, "policy_start_date"])
    bad_claim_date = policy_start - timedelta(days=random.randint(1, 365))
    df.at[i, "claim_date"] = bad_claim_date.strftime("%Y-%m-%d")
log_error(logic_date_idx, "logical_error", "claim_date", "Claim date is before policy start date", 30)

# 12. Date of birth in future (~10 rows)
dob_future_idx = random.sample(range(1000), 10)
for i in dob_future_idx:
    future_dob = date(random.randint(2025, 2030), random.randint(1, 12), random.randint(1, 28))
    df.at[i, "date_of_birth"] = future_dob.isoformat()
log_error(dob_future_idx, "range_error", "date_of_birth", "Date of birth in the future", 10)

# 13. Date of birth impossibly old (>110 years) (~8 rows)
dob_old_idx = random.sample([i for i in range(1000) if i not in dob_future_idx], 8)
for i in dob_old_idx:
    old_dob = date(random.randint(1880, 1900), random.randint(1, 12), random.randint(1, 28))
    df.at[i, "date_of_birth"] = old_dob.isoformat()
log_error(dob_old_idx, "range_error", "date_of_birth", "Date of birth implies age > 110 years (implausible)", 8)

# 14. Date format errors on claim_date (~35 rows)
date_format_idx = random.sample([i for i in range(1000) if i not in logic_date_idx], 35)
date_formats = [
    lambda d: d.strftime("%d/%m/%Y"),
    lambda d: d.strftime("%m-%d-%Y"),
    lambda d: d.strftime("%d %b %Y"),
    lambda d: d.strftime("%B %d, %Y"),
]
for i in date_format_idx:
    try:
        d = pd.to_datetime(df.at[i, "claim_date"]).date()
        df.at[i, "claim_date"] = random.choice(date_formats)(d)
    except:
        pass
log_error(date_format_idx, "format_error", "claim_date", "Claim date not in ISO format (15/01/2024, Jan 15 2024, etc.)", 35)

# 15. Invalid diagnosis code (~20 rows)
invalid_diag_idx = random.sample(range(1000), 20)
invalid_codes = ["XYZ", "123", "UNKNOWN", "N/A", "none", "??", "TBD"]
for i in invalid_diag_idx:
    df.at[i, "diagnosis_code"] = random.choice(invalid_codes)
log_error(invalid_diag_idx, "corpus_mismatch", "diagnosis_code", "Diagnosis code not in valid ICD-10 list", 20)

# 16. Name with extra whitespace (~16 rows)
whitespace_idx = random.sample(range(1000), 16)
for i in whitespace_idx:
    df.at[i, "claimant_name"] = "  " + df.at[i, "claimant_name"] + "  "
log_error(whitespace_idx, "format_error", "claimant_name", "Leading/trailing whitespace in claimant name", 16)

# 17. City in lowercase (~16 rows)
city_lower_idx = random.sample([i for i in range(1000) if i not in whitespace_idx], 16)
for i in city_lower_idx:
    df.at[i, "city"] = df.at[i, "city"].lower()
log_error(city_lower_idx, "standardisation_error", "city", "City name not properly capitalised (london, manchester)", 16)

# 18. Bad policy_id format (~15 rows)
bad_policy_idx = random.sample(range(1000), 15)
bad_formats = ["POL{}", "P-{}", "POLICY-{}", "{}", "POL {}"]
for i in bad_policy_idx:
    num = random.randint(100000, 999999)
    df.at[i, "policy_id"] = random.choice(bad_formats).format(num)
log_error(bad_policy_idx, "format_error", "policy_id", "Policy ID not in POL-XXXXXX format", 15)

# 19. Exact duplicates (50 rows -- copy rows 0-49)
dup_source = df.iloc[:50].copy()
df = pd.concat([df, dup_source], ignore_index=True)
log_error(list(range(1000, 1050)), "duplicate", "all_columns", "Exact duplicate rows (same claim submitted twice)", 50)

# 20. Near duplicates (~25 rows -- same claimant, slightly different details)
near_dup_idx = random.sample(range(50, 200), 25)
near_dups = df.iloc[near_dup_idx].copy()
for i in range(len(near_dups)):
    # Slightly alter the name
    name = near_dups.iloc[i]["claimant_name"].strip()
    parts = name.split()
    if len(parts) >= 2:
        parts[1] = parts[1][:3] + "."   # "Smith" -> "Smi."
        near_dups.iloc[i, near_dups.columns.get_loc("claimant_name")] = " ".join(parts)
    # Slightly alter amount
    near_dups.iloc[i, near_dups.columns.get_loc("claim_amount")] = round(
        float(str(near_dups.iloc[i]["claim_amount"]).replace("£", "")) * random.uniform(0.98, 1.02), 2
    )
df = pd.concat([df, near_dups], ignore_index=True)
log_error(list(range(1050, 1075)), "near_duplicate", "claimant_name", "Near-duplicate claims (same person, minor name/amount variation)", 25)

# 21. Completely empty rows (10 rows)
empty_df = pd.DataFrame([{col: "" for col in df.columns} for _ in range(10)])
df = pd.concat([df, empty_df], ignore_index=True)
log_error(list(range(1075, 1085)), "missing_value", "all_columns", "Completely empty records", 10)

# Reset index and add row numbers
df = df.reset_index(drop=True)
total_rows = len(df)

# ── Save main dataset ──────────────────────────────────────────────────────────
df.to_csv(f"{OUT}/main_data/insurance_claims.csv", index=False)
print(f"Saved main dataset: {total_rows} rows, {len(df.columns)} columns")


# ── Reference: known fraudulent claimants ─────────────────────────────────────
fraud_data = []
for i in range(30):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    fraud_data.append({
        "claimant_name": f"{first} {last}",
        "policy_id": rand_policy_id(),
        "fraud_type": random.choice(["Duplicate Claim", "Exaggerated Loss", "Staged Accident",
                                      "Identity Fraud", "Ghost Claimant", "Provider Fraud"]),
        "reported_date": rand_date(2022, 2024).isoformat(),
        "risk_level": random.choice(["High", "High", "Medium", "Critical"]),
        "notes": random.choice([
            "Submitted identical claim at two insurers",
            "Claim amount inconsistent with policy coverage",
            "Claimant address does not match records",
            "Linked to known fraud ring",
            "Multiple claims in 6-month window",
        ])
    })
pd.DataFrame(fraud_data).to_csv(f"{OUT}/reference/known_fraudulent_claimants.csv", index=False)
print("Saved reference: known_fraudulent_claimants.csv (30 records)")


# ── Corpus files ──────────────────────────────────────────────────────────────

# Insurer name corpus
insurer_corpus = []
canonical_insurers = {
    "Aviva": ["aviva", "AVIVA", "Aviva plc", "Aviva Insurance", "Aviva UK"],
    "AXA UK": ["AXA", "axa uk", "AXA Insurance UK", "Axa", "axa", "AXA Insurance"],
    "Bupa": ["BUPA", "bupa", "Bupa Insurance", "BUPA Health", "Bupa UK"],
    "Legal & General": ["L&G", "Legal and General", "legal & general", "L & G", "LG", "Legal&General"],
    "Zurich Insurance": ["Zurich", "ZURICH", "zurich insurance", "Zurich UK", "zurich"],
    "Allianz UK": ["Allianz", "ALLIANZ", "allianz uk", "Allianz Insurance", "allianz"],
    "Direct Line": ["direct line", "DIRECT LINE", "DirectLine", "Direct-Line"],
    "RSA Insurance": ["RSA", "rsa insurance", "RSA Group", "rsa", "Royal Sun Alliance"],
    "Liverpool Victoria": ["LV=", "LV", "liverpool victoria", "Liverpool Victoria Insurance"],
    "Hastings Direct": ["Hastings", "hastings direct", "HASTINGS DIRECT", "hastings"],
}
for canonical, aliases in canonical_insurers.items():
    for alias in aliases:
        insurer_corpus.append({"alias": alias, "canonical": canonical})
pd.DataFrame(insurer_corpus).to_csv(f"{OUT}/corpus/corpus_insurer_names.csv", index=False)

# Claim status corpus
status_corpus = []
canonical_statuses = {
    "Submitted": ["submitted", "SUBMITTED", "Submited", "Submit", "New"],
    "Under Review": ["under review", "UNDER REVIEW", "In Review", "Reviewing", "In Progress"],
    "Approved": ["approved", "APPROVED", "Approvd", "approve", "Accepted"],
    "Rejected": ["rejected", "REJECTED", "Rejectd", "Decline", "Declined", "Denied"],
    "Settled": ["settled", "SETTLED", "Settld", "Complete", "Completed", "Paid"],
    "Closed": ["closed", "CLOSED", "Closd", "Done", "Finalised", "Finalized"],
}
for canonical, aliases in canonical_statuses.items():
    for alias in aliases:
        status_corpus.append({"alias": alias, "canonical": canonical})
pd.DataFrame(status_corpus).to_csv(f"{OUT}/corpus/corpus_claim_status.csv", index=False)

# Claim types corpus
claim_type_corpus = [
    {"value": "Motor"}, {"value": "Home"}, {"value": "Health"},
    {"value": "Life"}, {"value": "Travel"}, {"value": "Liability"},
    {"value": "Pet"}, {"value": "Business"}
]
pd.DataFrame(claim_type_corpus).to_csv(f"{OUT}/corpus/corpus_claim_types.csv", index=False)

# Valid diagnosis codes (ICD-10 subset)
diag_corpus = [
    {"code": "M54.5", "description": "Low back pain"},
    {"code": "J06.9", "description": "Acute upper respiratory infection"},
    {"code": "I10", "description": "Essential hypertension"},
    {"code": "E11.9", "description": "Type 2 diabetes mellitus"},
    {"code": "F41.1", "description": "Generalised anxiety disorder"},
    {"code": "K21.0", "description": "Gastro-oesophageal reflux"},
    {"code": "M79.3", "description": "Panniculitis"},
    {"code": "G43.9", "description": "Migraine"},
    {"code": "R51", "description": "Headache"},
    {"code": "L30.9", "description": "Dermatitis"},
    {"code": "J45.9", "description": "Asthma"},
    {"code": "N39.0", "description": "Urinary tract infection"},
]
pd.DataFrame(diag_corpus).to_csv(f"{OUT}/corpus/corpus_diagnosis_codes.csv", index=False)

# Policy types corpus
policy_corpus = [{"value": p} for p in POLICY_TYPES]
pd.DataFrame(policy_corpus).to_csv(f"{OUT}/corpus/corpus_policy_types.csv", index=False)

print("Saved corpus files: insurer_names, claim_status, claim_types, diagnosis_codes, policy_types")


# ── Validation rules ──────────────────────────────────────────────────────────
rules = {
    "claim_id": {
        "type": "string",
        "pattern": r"^CLM-\d{5}$",
        "description": "Claim ID must be in format CLM-XXXXX"
    },
    "policy_id": {
        "type": "string",
        "pattern": r"^POL-\d{6}$",
        "description": "Policy ID must be in format POL-XXXXXX"
    },
    "email": {
        "type": "email",
        "required": True,
        "description": "Must be a valid email address"
    },
    "phone": {
        "type": "string",
        "pattern": r"^\+44\s7\d{3}\s\d{6}$",
        "description": "Must be a valid UK mobile number (+44 7XXX XXXXXX)"
    },
    "postcode": {
        "type": "string",
        "pattern": r"^[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}$",
        "auto_fix": "uppercase_and_add_space",
        "description": "Must be a valid UK postcode with space"
    },
    "date_of_birth": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "min": "1900-01-01",
        "max": "today",
        "description": "Date of birth must be valid and not in the future"
    },
    "claim_date": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "auto_fix": "normalise_date",
        "description": "Claim date must be ISO format YYYY-MM-DD"
    },
    "claim_amount": {
        "type": "numeric",
        "min": 0,
        "max": 500000,
        "auto_fix": "strip_currency_symbol",
        "description": "Claim amount must be a positive number, no currency symbols"
    },
    "claim_status": {
        "type": "categorical",
        "allowed_values": ["Submitted", "Under Review", "Approved", "Rejected", "Settled", "Closed"],
        "auto_fix": "corpus_lookup",
        "description": "Claim status must be a standard value"
    },
    "insurer": {
        "type": "categorical",
        "auto_fix": "corpus_lookup",
        "description": "Insurer name must match known insurer list"
    },
    "claim_type": {
        "type": "categorical",
        "allowed_values": ["Motor", "Home", "Health", "Life", "Travel", "Liability", "Pet", "Business"],
        "description": "Claim type must be a valid category"
    },
    "coverage_limit": {
        "type": "numeric",
        "min": 1000,
        "max": 10000000,
        "description": "Coverage limit must be between £1,000 and £10,000,000"
    },
    "diagnosis_code": {
        "type": "categorical",
        "auto_fix": "corpus_lookup",
        "description": "Diagnosis code must be valid ICD-10 code (if provided)"
    }
}
with open(f"{OUT}/rules/validation_rules.json", "w") as f:
    json.dump(rules, f, indent=2)
print("Saved validation rules: 13 rules")


# ── Entity resolution: 4 insurer datasets ────────────────────────────────────

base_er_customers = []
for i in range(100):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    base_er_customers.append({
        "customer_id": f"CUST-{1000 + i}",
        "full_name": f"{first} {last}",
        "date_of_birth": date(random.randint(1960, 1995), random.randint(1, 12), random.randint(1, 28)).isoformat(),
        "email": rand_email(first, last),
        "phone": rand_phone(),
        "postcode": rand_postcode(),
        "active_policies": random.randint(1, 4),
    })

er_base = pd.DataFrame(base_er_customers)

# Insurer A: clean canonical
er_base.to_csv(f"{OUT}/entity_resolution/insurer_a_customers.csv", index=False)

# Insurer B: name typos, phone format variations
er_b = er_base.copy()
for i in range(len(er_b)):
    if random.random() < 0.3:
        name = er_b.at[i, "full_name"]
        parts = name.split()
        if len(parts) >= 2 and len(parts[1]) > 3:
            parts[1] = parts[1][:-1] + random.choice("aeiou")
            er_b.at[i, "full_name"] = " ".join(parts)
    if random.random() < 0.2:
        er_b.at[i, "phone"] = er_b.at[i, "phone"].replace("+44 ", "0").replace(" ", "-")
er_b.to_csv(f"{OUT}/entity_resolution/insurer_b_customers.csv", index=False)

# Insurer C: abbreviated names, lowercase postcodes
er_c = er_base.copy()
for i in range(len(er_c)):
    if random.random() < 0.4:
        name = er_c.at[i, "full_name"]
        parts = name.split()
        if len(parts) >= 2:
            parts[0] = parts[0][0] + "."
            er_c.at[i, "full_name"] = " ".join(parts)
    if random.random() < 0.3:
        er_c.at[i, "postcode"] = er_c.at[i, "postcode"].lower()
er_c.to_csv(f"{OUT}/entity_resolution/insurer_c_customers.csv", index=False)

# Insurer D: different date format, extra whitespace, missing emails
er_d = er_base.copy()
for i in range(len(er_d)):
    if random.random() < 0.4:
        try:
            d = datetime.strptime(er_d.at[i, "date_of_birth"], "%Y-%m-%d")
            er_d.at[i, "date_of_birth"] = d.strftime("%d/%m/%Y")
        except:
            pass
    if random.random() < 0.2:
        er_d.at[i, "full_name"] = "  " + er_d.at[i, "full_name"] + "  "
    if random.random() < 0.15:
        er_d.at[i, "email"] = ""
er_d.to_csv(f"{OUT}/entity_resolution/insurer_d_customers.csv", index=False)
print("Saved entity resolution files: 4 insurer datasets (100 customers each)")


# ── Generate metadata / README ────────────────────────────────────────────────

# Error summary
total_errors = sum(e["affected_rows"] for e in error_log)
error_summary_md = "\n".join([
    f"| {e['error_type']} | {e['column']} | {e['description']} | {e['affected_rows']} |"
    for e in error_log
])

readme = f"""# TEST3_DATA — Insurance Claims Demo Dataset

Completely different domain from TEST2_DATA (banking transactions).
Domain: **UK Insurance Claims Processing**

---

## Folder Structure

```
TEST3_DATA/
├── main_data/
│   └── insurance_claims.csv            ← Load this first in the app
├── reference/
│   └── known_fraudulent_claimants.csv  ← Load as Reference Data
├── rules/
│   └── validation_rules.json           ← Load as Validation Rules
├── corpus/
│   ├── corpus_insurer_names.csv        ← Insurer name aliases
│   ├── corpus_claim_status.csv         ← Claim status aliases
│   ├── corpus_claim_types.csv          ← Valid claim type values
│   ├── corpus_diagnosis_codes.csv      ← Valid ICD-10 codes
│   └── corpus_policy_types.csv         ← Valid policy type values
└── entity_resolution/
    ├── insurer_a_customers.csv         ← Clean canonical (100 customers)
    ├── insurer_b_customers.csv         ← Name typos, phone format variations
    ├── insurer_c_customers.csv         ← Abbreviated names, lowercase postcodes
    └── insurer_d_customers.csv         ← Different date format, whitespace, missing emails
```

---

## Main Dataset: insurance_claims.csv

**Total rows:** {total_rows}
**Columns:** {len(df.columns)}
**Column list:** {', '.join(df.columns.tolist())}

### Columns

| Column | Type | Description |
|--------|------|-------------|
| claim_id | string | Unique claim identifier (CLM-XXXXX) |
| policy_id | string | Policy identifier (POL-XXXXXX) |
| claimant_name | string | Full name of claimant |
| date_of_birth | date | Claimant date of birth (YYYY-MM-DD) |
| email | string | Contact email address |
| phone | string | UK mobile phone number |
| address | string | Street address |
| city | string | City |
| postcode | string | UK postcode |
| insurer | string | Insurance company name |
| policy_type | string | Type of policy (Comprehensive, Standard etc.) |
| claim_type | string | Category of claim (Motor, Health, Home etc.) |
| policy_start_date | date | When policy started |
| policy_end_date | date | When policy ends |
| claim_date | date | Date claim was submitted |
| claim_amount | numeric | Amount claimed (£) |
| coverage_limit | numeric | Maximum policy coverage (£) |
| diagnosis_code | string | ICD-10 diagnosis code (for health claims) |
| claim_status | string | Current status of claim |
| agent_id | string | Handling agent ID |

---

## Injected Errors (Total: {total_errors} across {len(error_log)} error types)

| Error Type | Column | Description | Rows Affected |
|------------|--------|-------------|---------------|
{error_summary_md}

---

## Reference: known_fraudulent_claimants.csv (30 records)

Cross-reference file for flagging high-risk or fraudulent claimants.
Columns: claimant_name, policy_id, fraud_type, reported_date, risk_level, notes

Fraud types included:
- Duplicate Claim
- Exaggerated Loss
- Staged Accident
- Identity Fraud
- Ghost Claimant
- Provider Fraud

---

## Rules: validation_rules.json (13 rules)

| Field | Rule |
|-------|------|
| claim_id | Pattern: CLM-XXXXX |
| policy_id | Pattern: POL-XXXXXX |
| email | Valid RFC email format |
| phone | UK mobile format (+44 7XXX XXXXXX) |
| postcode | UK postcode with space, uppercase |
| date_of_birth | Valid date, not in future, not before 1900 |
| claim_date | ISO format YYYY-MM-DD (auto-fix: normalise) |
| claim_amount | Positive number, no currency symbol (auto-fix: strip £) |
| claim_status | Standard values only (auto-fix: corpus lookup) |
| insurer | Known insurer list (auto-fix: corpus lookup) |
| claim_type | Valid category list |
| coverage_limit | Between £1,000 and £10,000,000 |
| diagnosis_code | Valid ICD-10 code |

---

## Corpus Files

| File | Contents |
|------|----------|
| corpus_insurer_names.csv | Aliases for 10 UK insurers |
| corpus_claim_status.csv | Aliases for 6 claim statuses |
| corpus_claim_types.csv | 8 valid claim types |
| corpus_diagnosis_codes.csv | 12 valid ICD-10 codes with descriptions |
| corpus_policy_types.csv | 8 valid policy types |

---

## Entity Resolution Files (4 insurers × 100 customers)

The SAME 100 customers represented across 4 insurers with variations:
- **insurer_a**: Clean canonical records
- **insurer_b**: Name typos (~30%), phone format differences (~20%)
- **insurer_c**: Abbreviated first names (~40%), lowercase postcodes (~30%)
- **insurer_d**: Different date format DD/MM/YYYY (~40%), whitespace (~20%), missing emails (~15%)

**Use case:** Load all 4 files into Entity Resolution tab to match the same customer across insurers.

---

## How to Load in the App

1. **Load Data page:** Upload `main_data/insurance_claims.csv`
2. **Rules page:** Upload `rules/validation_rules.json`
3. **Reference:** Upload `reference/known_fraudulent_claimants.csv`
4. **Corpus Manager page:** Upload all files from `corpus/` folder
5. **Entity Resolution page:** Upload all 4 `entity_resolution/insurer_*.csv` files

---

## Comparison with TEST2_DATA

| Feature | TEST2_DATA | TEST3_DATA |
|---------|-----------|-----------|
| Domain | Banking transactions | Insurance claims |
| Rows | 500 | {total_rows} |
| Columns | 15 | {len(df.columns)} |
| Key entities | Customers, banks, merchants | Claimants, insurers, policies |
| Unique errors | Bank name variants, merchant variants | Diagnosis codes, coverage logic, fraud flags |
| Shared errors | Email, phone, postcode, dates, duplicates | Email, phone, postcode, dates, duplicates |
"""

with open(f"{OUT}/README.md", "w") as f:
    f.write(readme)
print("Saved README.md with full metadata")

# ── Error manifest (JSON) ──────────────────────────────────────────────────────
manifest = {
    "dataset": "insurance_claims.csv",
    "total_rows": total_rows,
    "total_columns": len(df.columns),
    "columns": df.columns.tolist(),
    "total_errors_injected": total_errors,
    "error_types": len(error_log),
    "errors": error_log
}
with open(f"{OUT}/error_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Saved error_manifest.json")

print(f"\nDone. TEST3_DATA created with {total_rows} rows and {total_errors} injected errors across {len(error_log)} error types.")
