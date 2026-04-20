# TEST3_DATA — Insurance Claims Demo Dataset

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

**Total rows:** 1085
**Columns:** 20
**Column list:** claim_id, policy_id, claimant_name, date_of_birth, email, phone, address, city, postcode, insurer, policy_type, claim_type, policy_start_date, policy_end_date, claim_date, claim_amount, coverage_limit, diagnosis_code, claim_status, agent_id

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

## Injected Errors (Total: 532 across 21 error types)

| Error Type | Column | Description | Rows Affected |
|------------|--------|-------------|---------------|
| format_error | email | Misspelled email domain (gmial.com, outlok.com etc.) | 60 |
| missing_value | email | Blank email address | 20 |
| missing_value | phone | Missing phone number | 30 |
| format_error | postcode | Postcode missing space (SW1A1AA instead of SW1A 1AA) | 25 |
| missing_value | postcode | Missing postcode | 15 |
| corpus_mismatch | insurer | Insurer name variants (AXA vs AXA UK vs axa uk) | 50 |
| standardisation_error | claim_status | Claim status not standardised (APPROVED, approvd, approve) | 40 |
| format_error | claim_amount | Claim amount stored as string with £ symbol | 25 |
| logical_error | claim_amount | Claim amount exceeds coverage limit (business rule violation) | 20 |
| range_error | claim_amount | Negative claim amount | 12 |
| logical_error | claim_date | Claim date is before policy start date | 30 |
| range_error | date_of_birth | Date of birth in the future | 10 |
| range_error | date_of_birth | Date of birth implies age > 110 years (implausible) | 8 |
| format_error | claim_date | Claim date not in ISO format (15/01/2024, Jan 15 2024, etc.) | 35 |
| corpus_mismatch | diagnosis_code | Diagnosis code not in valid ICD-10 list | 20 |
| format_error | claimant_name | Leading/trailing whitespace in claimant name | 16 |
| standardisation_error | city | City name not properly capitalised (london, manchester) | 16 |
| format_error | policy_id | Policy ID not in POL-XXXXXX format | 15 |
| duplicate | all_columns | Exact duplicate rows (same claim submitted twice) | 50 |
| near_duplicate | claimant_name | Near-duplicate claims (same person, minor name/amount variation) | 25 |
| missing_value | all_columns | Completely empty records | 10 |

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
| Rows | 500 | 1085 |
| Columns | 15 | 20 |
| Key entities | Customers, banks, merchants | Claimants, insurers, policies |
| Unique errors | Bank name variants, merchant variants | Diagnosis codes, coverage logic, fraud flags |
| Shared errors | Email, phone, postcode, dates, duplicates | Email, phone, postcode, dates, duplicates |
