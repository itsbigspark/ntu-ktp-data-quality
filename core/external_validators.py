# ==========================================================
# core/external_validators.py
# ==========================================================
import pandas as pd
import requests
from typing import Literal, Optional, Dict, Any


def validate_column_via_api(
    df: pd.DataFrame,
    column: str,
    api_type: Literal["postcode", "email", "company"] = "postcode",
    api_url: str | None = None,
) -> pd.DataFrame:
    """
    Validate column values via external APIs (Postcodes.io, Rapid Email Verifier, UK Companies API).
    Adds a new column `<column>_verified` with ✅ or ❌.

    Supported types:
      - postcode → https://api.postcodes.io/postcodes/
      - email    → https://rapid-email-verifier.fly.dev/api/validate
      - company  → https://api.company-information.service.gov.uk/search/companies?q=
    """
    df = df.copy()

    # Default API endpoints
    base_urls = {
        "postcode": "https://api.postcodes.io/postcodes/",
        "email": "https://rapid-email-verifier.fly.dev/api/validate",
        "company": "https://api.company-information.service.gov.uk/search/companies?q=",
    }

    base_url = api_url or base_urls.get(api_type, "")
    verified_col = f"{column}_verified"
    results, meta = [], []

    # ==========================================================
    # POSTCODE VALIDATION
    # ==========================================================
    if api_type == "postcode":
        for val in df[column].astype(str).fillna("").tolist():
            if not val.strip():
                results.append("❌")
                meta.append(None)
                continue

            try:
                # normalise by removing spaces for lookup, API is forgiving
                url = f"{base_url}{val.replace(' ', '')}"
                resp = requests.get(url, timeout=8)
                data = resp.json()
                ok = (resp.status_code == 200) and (data.get("status") == 200)
                results.append("✅" if ok else "❌")
                meta.append(data.get("result") if ok else None)
            except Exception:
                results.append("❌")
                meta.append(None)

        df[verified_col] = results
        df[f"{column}_api_meta"] = meta
        return df

    # ==========================================================
    # EMAIL VALIDATION (Rapid Email Verifier – batch domain check)
    # ==========================================================
    elif api_type == "email":
        verified, statuses, domains = [], [], []

        for val in df[column].astype(str).fillna("").tolist():
            email = val.strip()

            if not email:
                verified.append("❌")
                statuses.append("empty")
                domains.append(None)
                continue

            try:
                resp = requests.post(
                    base_url,
                    json={"email": email},
                    timeout=15
                )
                data = resp.json()

                status = data.get("status", "").upper()
                domain = email.split("@")[-1] if "@" in email else None

                verified.append("✅" if status == "VALID" else "❌")
                statuses.append(status)
                domains.append(domain)

            except Exception:
                verified.append("❌")
                statuses.append("error")
                domains.append(None)

        df[f"{column}_domain"] = domains
        df[f"{column}_api_status"] = statuses
        df[verified_col] = verified
        return df
    
    # ==========================================================
    # COMPANY VALIDATION (Companies House API)
    # ==========================================================
    elif api_type == "company":
        for val in df[column].astype(str).fillna("").tolist():
            if not val.strip():
                results.append("❌")
                meta.append(None)
                continue

            try:
                url = f"{base_url}{val}"
                resp = requests.get(url, timeout=8)
                ok = resp.status_code == 200 and bool(resp.json().get("items"))
                results.append("✅" if ok else "❌")
                meta.append(None)
            except Exception:
                results.append("❌")
                meta.append(None)

        df[verified_col] = results
        df[f"{column}_api_meta"] = meta
        return df

    # ==========================================================
    # UNKNOWN / UNSUPPORTED TYPE
    # ==========================================================
    else:
        df[verified_col] = ["❌"] * len(df)
        df[f"{column}_api_meta"] = [None] * len(df)
        return df


# ==========================================================
# Row-level helper used by postcode_corpus.py
# ==========================================================
def validate_postcode_via_api(postcode: str) -> Optional[Dict[str, Any]]:
    """
    Lightweight single-postcode validator for use inside postcode_corpus.py.

    Returns
    -------
    dict | None
        Example:
            {
                "valid": True,
                "source": "postcodes.io",
                "country": "...",
                "admin_district": "...",
                ...
            }

        Returns None if invalid OR if any error occurs.

    Notes
    -----
    - This is intentionally defensive: any failure results in `None`,
      so upstream code can just treat it as "no API validation".
    """
    if not postcode:
        return None

    try:
        base_url = "https://api.postcodes.io/postcodes/"
        # postcodes.io handles with/without spaces, but we normalise anyway
        url = f"{base_url}{postcode.replace(' ', '')}"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if data.get("status") != 200:
            return None

        result = data.get("result") or {}
        return {
            "valid": True,
            "source": "postcodes.io",
            "country": result.get("country"),
            "nhs_ha": result.get("nhs_ha"),
            "admin_district": result.get("admin_district"),
            "parliamentary_constituency": result.get("parliamentary_constituency"),
        }
    except Exception:
        return None