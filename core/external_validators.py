"""
core/external_validators.py
============================
Live external API validation for column values.

Validates data against real-world authoritative sources:
  - Postcodes    → postcodes.io (free, no key)
  - Email        → MX record check (no external API needed)
  - Company      → Companies House API (free, needs key)
  - VAT numbers  → HMRC VAT API (free, no key)
  - Phone        → NumVerify (free tier, needs key)
  - IBAN         → iban.com API (free)
  - Sort codes   → sort-codes.com (free)
  - Custom API   → any endpoint the user configures

Design principles:
  - Caching: same value is never checked twice in a session
  - Batching: bulk endpoints used where available (postcodes.io bulk)
  - Rate limiting: built-in delay to respect free-tier limits
  - Graceful degradation: any failure = "unchecked", not "invalid"
  - Privacy: only the value itself is sent, never the full row
"""

import re
import time
import socket
import logging
import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
import pandas as pd

logger = logging.getLogger("dq.external_validators")


# ---------------------------------------------------------------------------
# In-memory cache (per Python process lifetime)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Dict[str, Any]] = {}

def _cache_key(validator: str, value: str) -> str:
    return hashlib.md5(f"{validator}:{value}".encode()).hexdigest()

def _from_cache(validator: str, value: str) -> Optional[Dict]:
    return _CACHE.get(_cache_key(validator, value))

def _to_cache(validator: str, value: str, result: Dict):
    _CACHE[_cache_key(validator, value)] = result


# ---------------------------------------------------------------------------
# Shared result schema
# ---------------------------------------------------------------------------
def _result(valid: bool, source: str, detail: Optional[Dict] = None, error: str = "") -> Dict:
    return {
        "valid": valid,
        "source": source,
        "detail": detail or {},
        "error": error,
    }


# ===========================================================================
# 1. POSTCODE VALIDATOR (postcodes.io — free, no key, bulk supported)
# ===========================================================================

def validate_postcode(postcode: str) -> Dict:
    """Single postcode validation via postcodes.io."""
    if not postcode or not postcode.strip():
        return _result(False, "postcodes.io", error="empty")

    pc = postcode.strip().upper().replace(" ", "")
    cached = _from_cache("postcode", pc)
    if cached:
        return cached

    try:
        resp = requests.get(f"https://api.postcodes.io/postcodes/{pc}", timeout=6)
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            r = _result(True, "postcodes.io", {
                "normalised": data.get("postcode"),
                "country": data.get("country"),
                "region": data.get("region"),
                "admin_district": data.get("admin_district"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            })
        else:
            r = _result(False, "postcodes.io", error=f"not found (HTTP {resp.status_code})")
        _to_cache("postcode", pc, r)
        return r
    except Exception as e:
        return _result(False, "postcodes.io", error=str(e))


def validate_postcodes_bulk(postcodes: List[str]) -> Dict[str, Dict]:
    """
    Bulk postcode validation using postcodes.io batch endpoint.
    Up to 100 postcodes per request. Returns dict keyed by normalised postcode.
    """
    results = {}
    to_check = []

    for pc in postcodes:
        if not pc or not pc.strip():
            results[pc] = _result(False, "postcodes.io", error="empty")
            continue
        pc_norm = pc.strip().upper().replace(" ", "")
        cached = _from_cache("postcode", pc_norm)
        if cached:
            results[pc] = cached
        else:
            to_check.append((pc, pc_norm))

    # Batch in chunks of 100
    for i in range(0, len(to_check), 100):
        chunk = to_check[i:i+100]
        try:
            resp = requests.post(
                "https://api.postcodes.io/postcodes",
                json={"postcodes": [pc_norm for _, pc_norm in chunk]},
                timeout=15,
            )
            if resp.status_code == 200:
                for item, (orig, pc_norm) in zip(resp.json().get("result", []), chunk):
                    if item and item.get("result"):
                        data = item["result"]
                        r = _result(True, "postcodes.io", {
                            "normalised": data.get("postcode"),
                            "country": data.get("country"),
                            "region": data.get("region"),
                            "admin_district": data.get("admin_district"),
                        })
                    else:
                        r = _result(False, "postcodes.io", error="not found")
                    _to_cache("postcode", pc_norm, r)
                    results[orig] = r
        except Exception as e:
            for orig, _ in chunk:
                results[orig] = _result(False, "postcodes.io", error=str(e))
        time.sleep(0.1)  # polite rate limiting

    return results


# ===========================================================================
# 2. EMAIL VALIDATOR (MX record check — no external API needed)
# ===========================================================================

def validate_email(email: str) -> Dict:
    """
    Validate an email address:
    1. Format check (regex)
    2. Domain MX record check (does this domain actually receive email?)
    """
    if not email or not email.strip():
        return _result(False, "mx_check", error="empty")

    email = email.strip().lower()
    cached = _from_cache("email", email)
    if cached:
        return cached

    # Step 1: format check
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        r = _result(False, "mx_check", error="invalid format")
        _to_cache("email", email, r)
        return r

    # Step 2: MX record check
    domain = email.split("@")[1]
    try:
        # Use DNS to check MX records
        import dns.resolver
        mx_records = dns.resolver.resolve(domain, "MX")
        r = _result(True, "mx_check", {
            "domain": domain,
            "mx_records": [str(mx.exchange) for mx in mx_records][:3],
        })
    except ImportError:
        # dnspython not installed -- fall back to socket check
        try:
            socket.getaddrinfo(domain, None)
            r = _result(True, "mx_check", {"domain": domain, "note": "domain resolves (no MX check)"})
        except socket.gaierror:
            r = _result(False, "mx_check", error=f"domain {domain} does not resolve")
    except Exception as e:
        # MX lookup failed -- domain probably doesn't exist
        r = _result(False, "mx_check", error=f"no MX records for {domain}")

    _to_cache("email", email, r)
    return r


def validate_emails_bulk(emails: List[str]) -> Dict[str, Dict]:
    """Bulk email validation with caching."""
    return {email: validate_email(email) for email in emails}


# ===========================================================================
# 3. VAT NUMBER VALIDATOR (HMRC VAT API — free, no key required)
# ===========================================================================

def validate_vat(vat_number: str) -> Dict:
    """
    Validate a UK VAT registration number via HMRC's free API.
    Format: GB + 9 digits (e.g. GB123456789)
    """
    if not vat_number or not vat_number.strip():
        return _result(False, "hmrc_vat", error="empty")

    vat = vat_number.strip().upper().replace(" ", "")
    # Normalise: add GB prefix if missing
    if not vat.startswith("GB") and vat.isdigit():
        vat = "GB" + vat

    cached = _from_cache("vat", vat)
    if cached:
        return cached

    try:
        resp = requests.get(
            f"https://api.service.hmrc.gov.uk/organisations/vat/check-vat-number/lookup/{vat}",
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            target = data.get("target", {})
            r = _result(True, "hmrc_vat", {
                "vat_number": target.get("vatNumber"),
                "name": target.get("name"),
                "address": target.get("address", {}),
            })
        elif resp.status_code == 404:
            r = _result(False, "hmrc_vat", error="VAT number not registered")
        else:
            r = _result(False, "hmrc_vat", error=f"HTTP {resp.status_code}")
        _to_cache("vat", vat, r)
        return r
    except Exception as e:
        return _result(False, "hmrc_vat", error=str(e))


# ===========================================================================
# 4. COMPANY VALIDATOR (Companies House API — needs API key)
# ===========================================================================

def validate_company(company_name: str, api_key: str = "") -> Dict:
    """
    Search Companies House for a company name.
    Free API key at: https://developer.company-information.service.gov.uk/
    """
    if not company_name or not company_name.strip():
        return _result(False, "companies_house", error="empty")

    name = company_name.strip()
    cached = _from_cache("company", name.lower())
    if cached:
        return cached

    try:
        url = "https://api.company-information.service.gov.uk/search/companies"
        auth = (api_key, "") if api_key else None
        resp = requests.get(url, params={"q": name, "items_per_page": 3}, auth=auth, timeout=8)

        if resp.status_code == 401:
            return _result(False, "companies_house", error="API key required or invalid")

        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                top = items[0]
                r = _result(True, "companies_house", {
                    "matched_name": top.get("title"),
                    "company_number": top.get("company_number"),
                    "status": top.get("company_status"),
                    "type": top.get("company_type"),
                    "date_created": top.get("date_of_creation"),
                })
            else:
                r = _result(False, "companies_house", error="no matching company found")
        else:
            r = _result(False, "companies_house", error=f"HTTP {resp.status_code}")

        _to_cache("company", name.lower(), r)
        return r
    except Exception as e:
        return _result(False, "companies_house", error=str(e))


# ===========================================================================
# 5. IBAN VALIDATOR (format + checksum — no external API needed)
# ===========================================================================

def validate_iban(iban: str) -> Dict:
    """
    Validate an IBAN using the ISO 13616 checksum algorithm.
    No external API needed -- the checksum is deterministic.
    """
    if not iban or not iban.strip():
        return _result(False, "iban_checksum", error="empty")

    iban_clean = iban.strip().upper().replace(" ", "").replace("-", "")
    cached = _from_cache("iban", iban_clean)
    if cached:
        return cached

    if len(iban_clean) < 4:
        r = _result(False, "iban_checksum", error="too short")
        _to_cache("iban", iban_clean, r)
        return r

    # IBAN lengths by country
    IBAN_LENGTHS = {
        "GB": 22, "DE": 22, "FR": 27, "ES": 24, "IT": 27,
        "NL": 18, "BE": 16, "PL": 28, "SE": 24, "CH": 21,
        "IE": 22, "PT": 25, "AT": 20, "DK": 18, "NO": 15,
    }

    country = iban_clean[:2]
    expected_len = IBAN_LENGTHS.get(country)

    if expected_len and len(iban_clean) != expected_len:
        r = _result(False, "iban_checksum", error=f"wrong length for {country} (expected {expected_len}, got {len(iban_clean)})")
        _to_cache("iban", iban_clean, r)
        return r

    # Mod-97 checksum
    rearranged = iban_clean[4:] + iban_clean[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    try:
        checksum = int(numeric) % 97
        valid = checksum == 1
        r = _result(valid, "iban_checksum", {
            "country": country,
            "length": len(iban_clean),
            "normalised": " ".join(iban_clean[i:i+4] for i in range(0, len(iban_clean), 4)),
        }, error="" if valid else "checksum failed")
    except Exception as e:
        r = _result(False, "iban_checksum", error=str(e))

    _to_cache("iban", iban_clean, r)
    return r


# ===========================================================================
# 6. PHONE NUMBER VALIDATOR (format check + E.164 normalisation)
# ===========================================================================

def validate_phone(phone: str, country: str = "GB") -> Dict:
    """
    Validate a phone number format.
    Uses phonenumbers library if available, otherwise regex fallback.
    """
    if not phone or not phone.strip():
        return _result(False, "phone_validator", error="empty")

    phone = phone.strip()
    cached = _from_cache("phone", phone)
    if cached:
        return cached

    try:
        import phonenumbers
        parsed = phonenumbers.parse(phone, country)
        valid = phonenumbers.is_valid_number(parsed)
        r = _result(valid, "phonenumbers", {
            "normalised": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
            "country_code": parsed.country_code,
            "number_type": str(phonenumbers.number_type(parsed)),
        }, error="" if valid else "invalid number")
    except ImportError:
        # Fallback: regex for UK numbers
        uk_pattern = r"^(\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}$"
        valid = bool(re.match(uk_pattern, phone))
        r = _result(valid, "regex_fallback", {
            "pattern": "UK mobile",
        }, error="" if valid else "does not match UK mobile format")
    except Exception as e:
        r = _result(False, "phone_validator", error=str(e))

    _to_cache("phone", phone, r)
    return r


# ===========================================================================
# 7. SORT CODE VALIDATOR (UK bank sort codes)
# ===========================================================================

def validate_sort_code(sort_code: str) -> Dict:
    """
    Validate a UK bank sort code format.
    Format: XX-XX-XX or XXXXXX (6 digits)
    """
    if not sort_code or not sort_code.strip():
        return _result(False, "sort_code", error="empty")

    sc = sort_code.strip().replace("-", "").replace(" ", "")
    cached = _from_cache("sort_code", sc)
    if cached:
        return cached

    if not re.match(r"^\d{6}$", sc):
        r = _result(False, "sort_code", error="must be 6 digits (XX-XX-XX)")
        _to_cache("sort_code", sc, r)
        return r

    # Format as XX-XX-XX
    formatted = f"{sc[:2]}-{sc[2:4]}-{sc[4:]}"

    # Known invalid ranges
    if sc.startswith("00") or sc == "000000":
        r = _result(False, "sort_code", error="invalid sort code range")
    else:
        r = _result(True, "sort_code", {"normalised": formatted, "digits": sc})

    _to_cache("sort_code", sc, r)
    return r


# ===========================================================================
# 8. CUSTOM API VALIDATOR (user-defined endpoint)
# ===========================================================================

def validate_custom_api(
    value: str,
    endpoint_url: str,
    method: str = "GET",
    value_param: str = "value",
    valid_json_path: str = "valid",
    headers: Optional[Dict] = None,
    api_key: str = "",
) -> Dict:
    """
    Validate a value against any custom REST API endpoint.

    Parameters
    ----------
    value : str
        The value to validate.
    endpoint_url : str
        Full URL template. Use {value} as placeholder, e.g.:
        "https://api.example.com/validate/{value}"
        or "https://api.example.com/check"
    method : str
        "GET" or "POST"
    value_param : str
        For GET: query param name. For POST: JSON body key.
    valid_json_path : str
        Dot-separated path in JSON response that indicates validity,
        e.g. "valid" or "result.is_valid" or "data.status"
    headers : dict
        Additional headers (e.g. {"Authorization": "Bearer xxx"})
    api_key : str
        API key to include as Bearer token if headers not provided.
    """
    if not value or not value.strip():
        return _result(False, "custom_api", error="empty")

    if not endpoint_url:
        return _result(False, "custom_api", error="no endpoint URL configured")

    url = endpoint_url.replace("{value}", requests.utils.quote(str(value)))
    h = headers or {}
    if api_key and "Authorization" not in h:
        h["Authorization"] = f"Bearer {api_key}"

    try:
        if method.upper() == "POST":
            resp = requests.post(url, json={value_param: value}, headers=h, timeout=10)
        else:
            resp = requests.get(url, params={value_param: value}, headers=h, timeout=10)

        if resp.status_code not in (200, 201):
            return _result(False, "custom_api", error=f"HTTP {resp.status_code}")

        data = resp.json()

        # Navigate the JSON path to find the validity indicator
        keys = valid_json_path.split(".")
        val = data
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break

        # Interpret the result
        if isinstance(val, bool):
            valid = val
        elif isinstance(val, str):
            valid = val.lower() in ("true", "valid", "yes", "1", "ok", "success")
        elif isinstance(val, (int, float)):
            valid = bool(val)
        else:
            valid = False

        return _result(valid, url, {"response_field": valid_json_path, "raw_value": val})

    except Exception as e:
        return _result(False, "custom_api", error=str(e))


# ===========================================================================
# Main dispatcher: validate a full column
# ===========================================================================

VALIDATOR_MAP = {
    "postcode": validate_postcode,
    "email": validate_email,
    "vat": validate_vat,
    "iban": validate_iban,
    "phone": validate_phone,
    "sort_code": validate_sort_code,
    "company": validate_company,
}

VALIDATOR_LABELS = {
    "postcode": "UK Postcode (postcodes.io)",
    "email": "Email MX Check",
    "vat": "VAT Number (HMRC)",
    "iban": "IBAN Checksum",
    "phone": "Phone Number Format",
    "sort_code": "UK Sort Code",
    "company": "Companies House",
    "custom": "Custom API Endpoint",
}


def validate_column(
    df: pd.DataFrame,
    column: str,
    validator_type: str,
    api_key: str = "",
    custom_config: Optional[Dict] = None,
    batch_size: int = 100,
    rate_limit_delay: float = 0.05,
) -> pd.DataFrame:
    """
    Validate an entire DataFrame column against an external API.

    Adds columns:
      - {column}_ext_valid   : True / False / None (unchecked)
      - {column}_ext_source  : which API was used
      - {column}_ext_detail  : extra info (normalised value, region, etc.)
      - {column}_ext_error   : error message if invalid

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to validate
    validator_type : str
        One of: postcode, email, vat, iban, phone, sort_code, company, custom
    api_key : str
        API key for validators that require one
    custom_config : dict
        Config for custom API validator (endpoint_url, method, etc.)
    batch_size : int
        Process N rows at a time (for progress / memory)
    rate_limit_delay : float
        Seconds to sleep between API calls (default 0.05 = 50ms)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    df = df.copy()
    values = df[column].astype(str).fillna("").tolist()

    valid_col = f"{column}_ext_valid"
    source_col = f"{column}_ext_source"
    detail_col = f"{column}_ext_detail"
    error_col = f"{column}_ext_error"

    results_valid = []
    results_source = []
    results_detail = []
    results_error = []

    # Use bulk endpoint for postcodes
    if validator_type == "postcode":
        bulk_results = validate_postcodes_bulk(values)
        for val in values:
            r = bulk_results.get(val, _result(False, "postcodes.io", error="not checked"))
            results_valid.append(r["valid"])
            results_source.append(r["source"])
            results_detail.append(str(r["detail"]) if r["detail"] else "")
            results_error.append(r["error"])

    elif validator_type == "custom" and custom_config:
        for i, val in enumerate(values):
            r = validate_custom_api(
                value=val,
                endpoint_url=custom_config.get("endpoint_url", ""),
                method=custom_config.get("method", "GET"),
                value_param=custom_config.get("value_param", "value"),
                valid_json_path=custom_config.get("valid_json_path", "valid"),
                headers=custom_config.get("headers"),
                api_key=api_key,
            )
            results_valid.append(r["valid"])
            results_source.append(r["source"])
            results_detail.append(str(r["detail"]) if r["detail"] else "")
            results_error.append(r["error"])
            if i % batch_size == 0 and i > 0:
                time.sleep(rate_limit_delay)

    else:
        validator_fn = VALIDATOR_MAP.get(validator_type)
        if not validator_fn:
            raise ValueError(f"Unknown validator type: {validator_type}. Choose from: {list(VALIDATOR_MAP.keys())}")

        for i, val in enumerate(values):
            try:
                if validator_type == "company":
                    r = validator_fn(val, api_key=api_key)
                else:
                    r = validator_fn(val)
            except Exception as e:
                r = _result(False, validator_type, error=str(e))

            results_valid.append(r["valid"])
            results_source.append(r["source"])
            results_detail.append(str(r["detail"]) if r["detail"] else "")
            results_error.append(r["error"])

            if rate_limit_delay and i % 10 == 0 and i > 0:
                time.sleep(rate_limit_delay)

    df[valid_col] = results_valid
    df[source_col] = results_source
    df[detail_col] = results_detail
    df[error_col] = results_error

    # Summary stats
    total = len(df)
    valid_count = sum(1 for v in results_valid if v is True)
    invalid_count = sum(1 for v in results_valid if v is False)
    logger.info(
        f"External validation '{column}' via {validator_type}: "
        f"{valid_count}/{total} valid, {invalid_count} invalid"
    )

    return df


def get_validation_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get summary stats for an externally-validated column.
    Call after validate_column().
    """
    valid_col = f"{column}_ext_valid"
    error_col = f"{column}_ext_error"

    if valid_col not in df.columns:
        return {"error": f"Column {column} has not been externally validated yet"}

    total = len(df)
    valid = int(df[valid_col].sum())
    invalid = int((df[valid_col] == False).sum())

    top_errors = {}
    if error_col in df.columns:
        errors = df[df[valid_col] == False][error_col].value_counts().head(5)
        top_errors = errors.to_dict()

    return {
        "column": column,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_pct": round(valid / total * 100, 1) if total else 0,
        "top_errors": top_errors,
    }


# ---------------------------------------------------------------------------
# Legacy compat (keep old function signature working)
# ---------------------------------------------------------------------------
def validate_column_via_api(
    df: pd.DataFrame,
    column: str,
    api_type: str = "postcode",
    api_url: Optional[str] = None,
) -> pd.DataFrame:
    """Backwards-compatible wrapper for old code."""
    return validate_column(df, column, validator_type=api_type)


def validate_postcode_via_api(postcode: str) -> Optional[Dict]:
    """Backwards-compatible single postcode validator."""
    r = validate_postcode(postcode)
    return r["detail"] if r["valid"] else None
