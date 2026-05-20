"""
aizle_metadata_to_rules.py
==========================
Reads Aizle consumer-banking metadata CSVs and generates a rules JSON
for each dataset in the format expected by the DQ Investigator engine.

Usage:
    python scripts/aizle_metadata_to_rules.py \
        --metadata_dir /path/to/aizle.../metadata \
        --output_dir   scripts/output/aizle_rules

Outputs one JSON per dataset:
    aizle_rules/transactions_rules.json
    aizle_rules/customers_rules.json
    aizle_rules/loan_accounts_rules.json
    aizle_rules/credit_card_accounts_rules.json
"""

import ast
import csv
import json
import os
import argparse
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_values(raw: str) -> Optional[list]:
    """Parse the 'Values' column which looks like: ['A', 'B', None]"""
    if not raw or raw.strip() == "":
        return None
    try:
        parsed = ast.literal_eval(raw.strip())
        if isinstance(parsed, list):
            return [v for v in parsed if v is not None]
    except Exception:
        pass
    return None


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val.strip()) if val and val.strip() else None
    except ValueError:
        return None


def read_metadata(path: str) -> list:
    """Read a metadata CSV into a list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def metadata_to_rules(rows: list, extra_rules: Dict[str, Dict] = None) -> Dict[str, Any]:
    """
    Convert metadata rows to a rules dict.

    Strategy per column type:
      string  + Values list  → allowed_values
      string  + pattern hint → pattern (regex)
      float / integer        → min / max  (from Min/Max columns)
      date                   → type=date + min/max
      bool                   → allowed_values [true, false]
      datetime               → type=date + min/max (ISO timestamps)
    """
    rules = {}

    for row in rows:
        col   = row["Name"].strip()
        ctype = row["Type"].strip().lower()
        desc  = row.get("Description", "").strip()
        mn    = row.get("Min", "").strip()
        mx    = row.get("Max", "").strip()
        values = _parse_values(row.get("Values", ""))
        nulls = row.get("# nulls", "0").strip()
        nullable = nulls and nulls != "0"

        rule: Dict[str, Any] = {}

        # ── Bool ─────────────────────────────────────────────────────────
        if ctype == "bool":
            rule["allowed_values"] = ["true", "false", "True", "False", "0", "1"]
            rule["description"] = desc
            rule["severity"] = "warning"

        # ── Integer ──────────────────────────────────────────────────────
        elif ctype == "integer":
            mn_f = _safe_float(mn)
            mx_f = _safe_float(mx)
            if mn_f is not None:
                rule["min"] = mn_f
            if mx_f is not None:
                rule["max"] = mx_f
            rule["type"] = "numeric"
            rule["description"] = desc
            rule["severity"] = "error"

        # ── Float ────────────────────────────────────────────────────────
        elif ctype == "float":
            mn_f = _safe_float(mn)
            mx_f = _safe_float(mx)
            if mn_f is not None:
                # Add 10% margin so injected outliers are clearly out-of-range
                rule["min"] = round(mn_f * 1.5 if mn_f < 0 else mn_f * 0.5, 4)
            if mx_f is not None:
                rule["max"] = round(mx_f * 1.5, 4)
            rule["type"] = "numeric"
            rule["description"] = desc
            rule["severity"] = "error"

        # ── Date ─────────────────────────────────────────────────────────
        elif ctype == "date":
            rule["type"] = "date"
            if mn:
                rule["min"] = mn[:10]   # keep YYYY-MM-DD only
            if mx:
                rule["max"] = "today"
            rule["description"] = desc
            rule["severity"] = "error"

        # ── Datetime ─────────────────────────────────────────────────────
        elif ctype == "datetime":
            rule["type"] = "date"
            if mn:
                rule["min"] = mn[:10]
            rule["max"] = "today"
            rule["description"] = desc
            rule["severity"] = "error"

        # ── String ───────────────────────────────────────────────────────
        elif ctype == "string":

            # 1. Explicit allowed values from metadata
            if values and len(values) <= 30:
                rule["allowed_values"] = values
                rule["case_sensitive"] = True
                rule["description"] = desc
                rule["severity"] = "error"

            # 2. Pattern hints from column name / description
            else:
                col_l = col.lower()

                if "sort_code" in col_l:
                    rule["pattern"] = r"^\d{2}-\d{2}-\d{2}$"
                    rule["description"] = "Sort code must be in DD-DD-DD format"
                    rule["severity"] = "error"

                elif "postcode" in col_l:
                    rule["pattern"] = r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}$"
                    rule["description"] = "Must be a valid UK postcode (e.g. SW1A 2AA)"
                    rule["severity"] = "error"
                    rule["auto_fix"] = True

                elif "mobile_number" in col_l:
                    rule["pattern"] = r"^\+447\d{9}$"
                    rule["description"] = "UK mobile must start +447 followed by 9 digits"
                    rule["severity"] = "warning"

                elif "land_line" in col_l:
                    rule["pattern"] = r"^\+44\d{10}$"
                    rule["description"] = "UK landline must start +44 followed by 10 digits"
                    rule["severity"] = "warning"

                elif col_l in ("customer_id",):
                    rule["pattern"] = r"^\d{12}$"
                    rule["description"] = "Customer ID must be 12 digits"
                    rule["severity"] = "error"

                elif "account_number" in col_l and "counterparty" not in col_l:
                    rule["min_length"] = int(_safe_float(row.get("Min Length", "8")) or 8)
                    rule["max_length"] = int(_safe_float(row.get("Max Length", "16")) or 16)
                    rule["description"] = desc
                    rule["severity"] = "warning"

                elif "iban" in col_l:
                    rule["pattern"] = r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}$"
                    rule["description"] = "Must be a valid IBAN"
                    rule["severity"] = "error"

                elif col_l in ("address_town",):
                    rule["case"] = "upper"
                    rule["description"] = "Town must be uppercase"
                    rule["severity"] = "warning"
                    rule["auto_fix"] = True

                else:
                    # Length bounds only
                    min_len = _safe_float(row.get("Min Length", ""))
                    max_len = _safe_float(row.get("Max Length", ""))
                    if min_len:
                        rule["min_length"] = int(min_len)
                    if max_len:
                        rule["max_length"] = int(max_len)
                    rule["description"] = desc
                    rule["severity"] = "warning"

        # ── Skip ID / free-text columns with no useful rule ───────────────
        if not rule:
            continue

        # Mark nullable columns so engine doesn't flag nulls as errors
        if nullable:
            rule["nullable"] = True

        rules[col] = rule

    # Apply hand-crafted overrides / additions
    if extra_rules:
        for col, override in extra_rules.items():
            if col in rules:
                rules[col].update(override)
            else:
                rules[col] = override

    return rules


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    "transactions": {
        "metadata_file": "consumer_banking-small-en_GB-v9_1_0-bank1_personal_transactions.csv",
        "output_file":   "transactions_rules.json",
        "extra_rules": {
            # transaction_type has 20 values not listed in metadata — add manually
            "transaction_type": {
                "allowed_values": [
                    "ATM", "BACS_C", "BACS_D", "BGC", "CASH_DEP", "CCR_D",
                    "CC_C", "DD_C", "DD_D", "DEP", "FP_C", "FP_D", "INTP_C",
                    "POS", "SELF_DD_C", "SELF_DD_D", "SO_C", "SO_D",
                    "TFR_C", "TFR_D",
                ],
                "case_sensitive": True,
                "description": "Must be a recognised transaction type code",
                "severity": "error",
            },
            # Loosen amount bounds — real transactions can be large
            "amount": {
                "min": -50000,
                "max": 50000,
                "type": "numeric",
                "description": "Transaction amount in GBP",
                "severity": "error",
            },
            # Drop sort_code rule for counterparty (nullable, less strict)
            "counterparty_sort_code": {
                "pattern": r"^\d{2}-\d{2}-\d{2}$",
                "nullable": True,
                "description": "Counterparty sort code in DD-DD-DD format (nullable)",
                "severity": "warning",
            },
            # Skip tid — just a UUID-like string, no meaningful rule
            "tid": {"skip": True},
            "first_party_agent_id": {"skip": True},
            "counterparty_agent_id": {"skip": True},
            "narrative": {"skip": True},
            "merchant_name": {"skip": True},
        },
    },

    "customers": {
        "metadata_file": "consumer_banking-small-en_GB-v9_1_0-bank1_personal_customers.csv",
        "output_file":   "customers_rules.json",
        "extra_rules": {
            "dob": {
                "type": "date",
                "min": "1900-01-01",
                "max": "today",
                "description": "Date of birth must be in the past and after 1900",
                "severity": "error",
            },
            # Override: customer_id is 12-digit numeric string
            "customer_id": {"skip": True},
            "agent_id":    {"skip": True},
            "second_name": {"nullable": True, "skip": True},
            "address_line1": {"skip": True},
        },
    },

    "loan_accounts": {
        "metadata_file": "consumer_banking-small-en_GB-v9_1_0-bank1_personal_loan_accounts.csv",
        "output_file":   "loan_accounts_rules.json",
        "extra_rules": {
            # Expand loan_status — metadata only shows UPTODATE (all current)
            # but valid values include ARREARS, SETTLED, DEFAULT
            "loan_status": {
                "allowed_values": ["UPTODATE", "ARREARS", "SETTLED", "DEFAULT", "WRITEOFF"],
                "case_sensitive": True,
                "description": "Loan status must be a recognised value",
                "severity": "error",
            },
            "account_id": {"skip": True},
            "start_date": {
                "type": "date",
                "min": "2000-01-01",
                "max": "today",
                "severity": "error",
            },
            "settle_date": {
                "type": "date",
                "min": "2000-01-01",
                "max": "2050-12-31",
                "severity": "error",
            },
            # Loosen bounds slightly so injected outliers are clearly wrong
            "loan_amount":      {"min": 100,  "max": 100000, "type": "numeric"},
            "interest_rate":    {"min": 0.5,  "max": 50.0,   "type": "numeric"},
            "repayment_amount": {"min": 1.0,  "max": 5000.0, "type": "numeric"},
            "total_payable":    {"min": 100,  "max": 200000, "type": "numeric"},
        },
    },

    "credit_card_accounts": {
        "metadata_file": "consumer_banking-small-en_GB-v9_1_0-bank1_personal_credit_card_accounts.csv",
        "output_file":   "credit_card_accounts_rules.json",
        "extra_rules": {
            "account_id": {"skip": True},
            "card_issue_date": {
                "type": "date",
                "min": "2000-01-01",
                "max": "today",
                "severity": "error",
            },
            "card_expiration_date": {
                "type": "date",
                "min": "today",
                "max": "2040-12-31",
                "description": "Card expiry must be in the future",
                "severity": "error",
            },
            "credit_limit":    {"min": 100,    "max": 50000,  "type": "numeric"},
            "interest_rate":   {"min": 0.1,    "max": 60.0,   "type": "numeric"},
            "current_balance": {"min": -50000, "max": 0.0,    "type": "numeric"},
        },
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Aizle metadata CSVs to DQ rules JSON")
    parser.add_argument(
        "--metadata_dir",
        default=os.path.expanduser(
            "~/Downloads/aizle-consumer_banking-small-9_1_0/metadata"
        ),
        help="Path to Aizle metadata directory",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "output", "aizle_rules"),
        help="Where to write the rules JSON files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name, cfg in DATASETS.items():
        metadata_path = os.path.join(args.metadata_dir, cfg["metadata_file"])
        output_path   = os.path.join(args.output_dir,   cfg["output_file"])

        if not os.path.exists(metadata_path):
            print(f"  SKIP {ds_name} — metadata file not found: {metadata_path}")
            continue

        rows  = read_metadata(metadata_path)
        rules = metadata_to_rules(rows, extra_rules=cfg.get("extra_rules"))

        # Remove columns marked skip
        rules = {k: v for k, v in rules.items() if not v.get("skip")}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)

        print(f"  {ds_name:25s} → {output_path}  ({len(rules)} column rules)")

    print("\nDone.")


if __name__ == "__main__":
    main()
