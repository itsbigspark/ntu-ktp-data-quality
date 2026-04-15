# TEST2_DATA — Demo Dataset Pack

Complete demo-ready dataset for the NTU-KTP Data Quality Application.

---

## Folder Structure

```
TEST2_DATA/
├── main_data/
│   └── customer_transactions.csv       ← Load this first in the app
├── reference/
│   └── known_fraudulent_merchants.csv  ← Load as Reference Data
├── rules/
│   └── validation_rules.json           ← Load as Validation Rules
├── corpus/
│   ├── corpus_company_names.csv        ← Merchant name aliases
│   ├── corpus_bank_names.csv           ← Bank name aliases
│   ├── corpus_email_domains.csv        ← Email domain typo corrections
│   ├── corpus_transaction_types.csv    ← Valid transaction type values
│   ├── corpus_uk_postcodes.csv         ← Valid UK postcodes
│   └── corpus_country_codes.csv        ← Country name aliases
└── entity_resolution/
    ├── bank1_customers.csv             ← Clean canonical (80 customers)
    ├── bank2_customers.csv             ← Same customers, name typos
    ├── bank3_customers.csv             ← Same customers, abbreviated names
    └── bank4_customers.csv             ← Same customers, format differences
```

---

## What Each File Contains

### main_data/customer_transactions.csv (500 rows)
The primary dataset. Contains ALL error types:

| Error Type         | Count     | What it looks like                          |
|--------------------|-----------|---------------------------------------------|
| Bad email          | ~60 rows  | gmial.com, hotmial.com, outlok.com          |
| Missing email      | ~20 rows  | Empty string                                |
| Missing phone      | ~30 rows  | Empty string                                |
| Missing postcode   | ~25 rows  | Empty string                                |
| Bad postcode       | ~25 rows  | SW1A1AA (no space)                          |
| Bank name variant  | ~60 rows  | hsbc, HSBC Bank, HSBC Holdings plc          |
| Merchant variant   | ~60 rows  | amazon, AMAZON, Amazon.co.uk                |
| Status error       | ~40 rows  | completed, COMPLETED, Complted              |
| Currency in amount | ~25 rows  | £1234.56 instead of 1234.56                 |
| Negative amount    | ~12 rows  | -500.00                                     |
| Date format error  | ~40 rows  | 15/01/2024 or Jan 15 2024 instead of ISO    |
| Name whitespace    | ~16 rows  | "  James Smith  "                           |
| City lowercase     | ~16 rows  | "london" instead of "London"                |
| Exact duplicates   | 50 rows   | Identical rows from rows 1-200              |
| Near duplicates    | 30 rows   | Same customer, slightly different name/email|
| All fields missing | 20 rows   | Completely empty records                    |

### reference/known_fraudulent_merchants.csv (20 merchants)
Cross-reference for flagging risky merchants in transaction data.
Columns: merchant_name, fraud_type, reported_date, risk_level

### rules/validation_rules.json (11 rules)
Validation rules for all key fields:
- customer_id: pattern C\d{3,4}
- email: RFC email regex
- phone: UK phone format
- postcode: UK postcode format (auto-fix: add space + uppercase)
- transaction_amount: -10000 to 1000000
- transaction_date: YYYY-MM-DD (auto-fix: normalize date)
- status: allowed values with case normalization
- transaction_type: allowed values list

### corpus/ files
Used by the correction suggester to map aliases to canonical values.
All files use alias → canonical format (except transaction_types which is a value list).

### entity_resolution/ (4 files × 80 customers = 320 rows)
The SAME 80 customers represented across 4 banks with variations:
- bank1: Clean canonical records
- bank2: Name typos, postcodes without spaces
- bank3: Abbreviated first names (J.), lowercase cities, street abbreviations
- bank4: Different date format (DD/MM/YYYY), whitespace in names, lowercase postcodes

**Use case for demo:** Load all 4 files into Entity Resolution tab to show the system linking the same customer across banks.

---

## How to Load in the App

1. Start app: `./RUN_APP.sh`
2. **Validate & Fix tab:** Upload `main_data/customer_transactions.csv`
3. **Rules:** Upload `rules/validation_rules.json`
4. **Reference:** Upload `reference/known_fraudulent_merchants.csv`
5. **Corpus tab:** Upload files from `corpus/` folder
6. **Entity Resolution tab:** Upload all 4 `entity_resolution/bank*.csv` files
