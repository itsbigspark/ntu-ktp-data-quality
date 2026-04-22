"""
api/auth.py -- API Key Authentication
======================================
Simple API key auth for the DQ Investigator REST API.

Keys are stored as SHA-256 hashes in the api_keys table.
Clients pass their raw key in the X-API-Key header.

Usage (in endpoints):
    from api.auth import require_api_key
    @app.post("/api/v1/validate")
    async def validate(api_key: str = Security(require_api_key)):
        ...

Generating a new key (run locally):
    python -c "from api.auth import generate_key; generate_key('ClientName')"
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("dq_api.auth")

# ---------------------------------------------------------------------------
# Header scheme
# ---------------------------------------------------------------------------
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---------------------------------------------------------------------------
# Master bypass key (for internal use / health checks that need auth)
# Set MASTER_API_KEY env var in ECS task definition to enable.
# ---------------------------------------------------------------------------
_MASTER_KEY = os.environ.get("MASTER_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of a raw API key (what we store in DB)."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _get_db_engine():
    from core.storage.database import get_engine
    return get_engine()


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------
def _validate_key_in_db(raw_key: str) -> Optional[str]:
    """
    Look up a raw key in the database.
    Returns the client_name if valid and active, else None.
    Updates last_used_at on successful lookup.
    """
    try:
        from sqlalchemy import text
        engine = _get_db_engine()
        key_hash = _hash_key(raw_key)

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, client_name FROM api_keys "
                    "WHERE key_hash = :kh AND is_active = TRUE"
                ),
                {"kh": key_hash},
            ).fetchone()

            if row is None:
                return None

            # Update last used timestamp
            conn.execute(
                text(
                    "UPDATE api_keys SET last_used_at = :ts WHERE id = :id"
                ),
                {"ts": datetime.now(timezone.utc).isoformat(), "id": row[0]},
            )
            conn.commit()
            return row[1]  # client_name

    except Exception as e:
        logger.warning(f"DB error during key validation: {e}")
        # If DB is unreachable, fall through to master key check only
        return None


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def require_api_key(raw_key: str = Security(_API_KEY_HEADER)) -> str:
    """
    FastAPI Security dependency.
    Raises 401 if key is missing or invalid.
    Returns the client name on success.
    """
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Master key bypass (internal use)
    if _MASTER_KEY and raw_key == _MASTER_KEY:
        return "internal"

    client = _validate_key_in_db(raw_key)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return client


# ---------------------------------------------------------------------------
# Schema creation (called on startup)
# ---------------------------------------------------------------------------
def ensure_api_keys_table() -> None:
    """
    Create the api_keys table if it doesn't exist.
    If BOOTSTRAP_API_KEY env var is set and the table is empty,
    insert it automatically so the API is usable immediately after deploy.
    """
    try:
        from sqlalchemy import text
        engine = _get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id          SERIAL PRIMARY KEY,
                    key_hash    VARCHAR(64)  NOT NULL UNIQUE,
                    client_name VARCHAR(255) NOT NULL,
                    created_at  TIMESTAMP    DEFAULT NOW(),
                    is_active   BOOLEAN      DEFAULT TRUE,
                    last_used_at TIMESTAMP
                )
            """))
            conn.commit()
            logger.info("api_keys table ready")

            # Bootstrap: insert first key if table is empty and env var is set
            bootstrap_key = os.environ.get("BOOTSTRAP_API_KEY", "")
            if bootstrap_key:
                count = conn.execute(text("SELECT COUNT(*) FROM api_keys")).scalar()
                if count == 0:
                    key_hash = _hash_key(bootstrap_key)
                    conn.execute(
                        text("INSERT INTO api_keys (key_hash, client_name) VALUES (:kh, :cn)"),
                        {"kh": key_hash, "cn": "NTU Internal (bootstrap)"},
                    )
                    conn.commit()
                    logger.info("Bootstrap API key inserted into api_keys table")

    except Exception as e:
        logger.warning(f"Could not create api_keys table: {e}")


# ---------------------------------------------------------------------------
# Key generation utility (run locally to create keys)
# ---------------------------------------------------------------------------
def generate_key(client_name: str) -> str:
    """
    Generate a new API key for a client and insert into the database.

    Run from project root:
        python -c "from api.auth import generate_key; generate_key('NTU Internal')"

    Prints the raw key — store it securely, it cannot be recovered.
    """
    raw_key = "dq-" + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)

    try:
        from sqlalchemy import text
        engine = _get_db_engine()
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO api_keys (key_hash, client_name) "
                    "VALUES (:kh, :cn)"
                ),
                {"kh": key_hash, "cn": client_name},
            )
            conn.commit()
        print(f"\nAPI key generated for '{client_name}':")
        print(f"  {raw_key}")
        print(f"\nShare this key with the client. It cannot be recovered.")
        print(f"To revoke: UPDATE api_keys SET is_active=FALSE WHERE client_name='{client_name}';\n")
    except Exception as e:
        print(f"DB error: {e}")
        print(f"Key hash to insert manually: {key_hash}")
        print(f"Raw key: {raw_key}")

    return raw_key


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Test Client"
    generate_key(name)
