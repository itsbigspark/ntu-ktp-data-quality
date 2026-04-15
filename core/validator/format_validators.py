# core/validator/format_validators.py
"""
Built-in format validators for common data types.

These validators provide robust pattern matching for:
- Email addresses
- UK postcodes
- Phone numbers
- Dates
- Company numbers
- URLs

Unlike inferred regex patterns, these are based on standards and domain knowledge.
"""

import re
from typing import Tuple, Optional
from datetime import datetime
import pandas as pd


# =============================================================================
# EMAIL VALIDATION
# =============================================================================

EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

def validate_email(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address (RFC 5322 simplified).

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(value, str):
        return False, "not a string"

    value = value.strip()

    if not value:
        return False, "empty email"

    if '@' not in value:
        return False, "missing @ symbol"

    if not re.match(EMAIL_REGEX, value):
        # More specific error messages
        if value.count('@') > 1:
            return False, "multiple @ symbols"
        if not re.search(r'\.[a-zA-Z]{2,}$', value):
            return False, "missing or invalid domain extension"
        if value.startswith('@') or value.endswith('@'):
            return False, "@ symbol at start or end"
        return False, "invalid email format"

    return True, None


# =============================================================================
# UK POSTCODE VALIDATION
# =============================================================================

# UK Postcode format: https://en.wikipedia.org/wiki/Postcodes_in_the_United_Kingdom
UK_POSTCODE_REGEX = r'^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$'

def validate_uk_postcode(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate UK postcode.

    Valid formats:
    - A9 9AA (e.g., M1 1AE)
    - A99 9AA (e.g., M60 1NW)
    - AA9 9AA (e.g., DN55 1PT)
    - AA99 9AA (e.g., EC1A 1BB)
    - A9A 9AA (e.g., W1A 1HQ)
    - AA9A 9AA (e.g., SW1A 1AA)

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(value, str):
        return False, "not a string"

    value = value.strip().upper()

    if not value:
        return False, "empty postcode"

    # Check basic format
    if not re.match(UK_POSTCODE_REGEX, value):
        if len(value) < 5:
            return False, "too short for valid postcode"
        if len(value) > 8:
            return False, "too long for valid postcode"
        if not any(c.isdigit() for c in value):
            return False, "missing digits"
        if not any(c.isalpha() for c in value):
            return False, "missing letters"
        return False, "invalid UK postcode format"

    return True, None


# =============================================================================
# PHONE NUMBER VALIDATION
# =============================================================================

# UK phone formats
UK_PHONE_REGEX = r'^\+?44\s?\d{10}$|^0\d{10}$|^\d{11}$'
# International E.164
INTL_PHONE_REGEX = r'^\+?[1-9]\d{1,14}$'

def validate_phone(value: str, strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate phone number.

    Args:
        value: Phone number string
        strict: If True, requires E.164 format (+44...)

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(value, str):
        return False, "not a string"

    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)]+', '', value)

    if not cleaned:
        return False, "empty phone number"

    if strict:
        # E.164 format: +[country code][number]
        if not re.match(INTL_PHONE_REGEX, cleaned):
            if not cleaned.startswith('+'):
                return False, "missing country code (should start with +)"
            return False, "invalid E.164 format"
    else:
        # Accept UK or international
        if not (re.match(UK_PHONE_REGEX, cleaned) or re.match(INTL_PHONE_REGEX, cleaned)):
            if len(cleaned) < 10:
                return False, "too short for valid phone number"
            if len(cleaned) > 15:
                return False, "too long for valid phone number"
            return False, "invalid phone number format"

    return True, None


# =============================================================================
# DATE VALIDATION
# =============================================================================

def validate_date(value: str, allow_future: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate date string.

    Accepts common formats:
    - ISO 8601: YYYY-MM-DD
    - UK format: DD/MM/YYYY
    - US format: MM/DD/YYYY

    Args:
        value: Date string
        allow_future: If False, rejects future dates

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(value, str):
        # Try pandas timestamp
        if isinstance(value, (pd.Timestamp, datetime)):
            if not allow_future and value > datetime.now():
                return False, "future date not allowed"
            return True, None
        return False, "not a string or datetime"

    value = value.strip()

    if not value:
        return False, "empty date"

    # Try multiple date formats
    formats = [
        '%Y-%m-%d',      # ISO 8601
        '%d/%m/%Y',      # UK
        '%m/%d/%Y',      # US
        '%Y/%m/%d',      # Alternative
        '%d-%m-%Y',      # Alternative UK
        '%m-%d-%Y',      # Alternative US
    ]

    parsed_date = None
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue

    if parsed_date is None:
        # Check for common errors
        if len(value.replace('-', '').replace('/', '')) < 8:
            return False, "incomplete date"
        if not any(c.isdigit() for c in value):
            return False, "no digits in date"
        return False, "unrecognized date format"

    # Validate date components
    if parsed_date.month > 12:
        return False, f"invalid month ({parsed_date.month})"
    if parsed_date.day > 31:
        return False, f"invalid day ({parsed_date.day})"
    if parsed_date.year < 1900:
        return False, f"year too old ({parsed_date.year})"
    if parsed_date.year > 2100:
        return False, f"year too far in future ({parsed_date.year})"

    # Check future date if not allowed
    if not allow_future and parsed_date > datetime.now():
        return False, "future date not allowed"

    return True, None


# =============================================================================
# UK COMPANY NUMBER VALIDATION
# =============================================================================

def validate_uk_company_number(value) -> Tuple[bool, Optional[str]]:
    """
    Validate UK Companies House registration number.

    Formats:
    - 8 digits: 12345678 or 04335942 (with leading zeros)
    - 7 digits: 1234567 (acceptable - will be zero-padded)
    - 6 digits: 123456 (acceptable - will be zero-padded)
    - 2 letters + 6 digits: OC123456 (LLP)
    - 2 letters + 6 digits: SC123456 (Scottish)

    Args:
        value: Company number (can be int or str)

    Returns:
        (is_valid, reason_if_invalid)
    """
    # Handle integers (from CSV files where leading zeros are dropped)
    if isinstance(value, (int, float)):
        if value != value:  # NaN check
            return False, "empty company number"
        # Convert to string with zero-padding to 8 digits
        value = str(int(value)).zfill(8)
    elif not isinstance(value, str):
        return False, "not a string or number"
    else:
        value = value.strip().upper()

    if not value:
        return False, "empty company number"

    # Standard format: 6-8 digits (with optional leading zeros)
    if re.match(r'^\d{6,8}$', value):
        return True, None

    # LLP format: OC + 6 digits
    if re.match(r'^OC\d{6}$', value):
        return True, None

    # Scottish format: SC + 6 digits
    if re.match(r'^SC\d{6}$', value):
        return True, None

    # Other prefixes: NI (Northern Ireland), etc.
    if re.match(r'^[A-Z]{2}\d{6}$', value):
        return True, None  # Accept other prefixes

    # Error messages
    if len(value) < 6:
        return False, "too short for company number"
    if len(value) > 8:
        return False, "too long for company number"

    return False, "invalid UK company number format"


# =============================================================================
# URL VALIDATION
# =============================================================================

URL_REGEX = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'

def validate_url(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL (http/https).

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not isinstance(value, str):
        return False, "not a string"

    value = value.strip()

    if not value:
        return False, "empty URL"

    if not re.match(URL_REGEX, value):
        if not value.startswith(('http://', 'https://')):
            return False, "missing http:// or https://"
        if '.' not in value:
            return False, "missing domain extension"
        return False, "invalid URL format"

    return True, None


# =============================================================================
# TYPO DETECTION (Edit Distance)
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def is_likely_typo(value: str, reference: str, threshold: int = 2) -> bool:
    """
    Check if value is likely a typo of reference.

    Args:
        value: Value to check
        reference: Reference value
        threshold: Maximum edit distance to consider a typo

    Returns:
        True if likely typo (edit distance <= threshold)
    """
    if not isinstance(value, str) or not isinstance(reference, str):
        return False

    distance = levenshtein_distance(value.lower(), reference.lower())
    return distance <= threshold


# =============================================================================
# VALIDATOR REGISTRY
# =============================================================================

VALIDATORS = {
    'email': validate_email,
    'uk_postcode': validate_uk_postcode,
    'phone': validate_phone,
    'date': validate_date,
    'uk_company_number': validate_uk_company_number,
    'url': validate_url,
}


def get_validator(field_type: str):
    """Get validator function by type."""
    return VALIDATORS.get(field_type)


def validate_value(value, field_type: str, **kwargs) -> Tuple[bool, Optional[str]]:
    """
    Validate a value using the appropriate validator.

    Args:
        value: Value to validate
        field_type: Type of validation ('email', 'uk_postcode', etc.)
        **kwargs: Additional arguments for validator

    Returns:
        (is_valid, reason_if_invalid)
    """
    validator = get_validator(field_type)
    if validator is None:
        raise ValueError(f"Unknown field type: {field_type}")

    return validator(value, **kwargs)
