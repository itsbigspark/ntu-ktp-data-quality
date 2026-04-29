# ==========================================================
# core/corpus_manager.py
# Dynamic Corpus Management System
# Supports uploading and managing custom corpus data in Redis
# ==========================================================

from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
import pandas as pd
import redis
import json
from pathlib import Path
import re


CorpusType = Literal["alias", "lookup", "validation"]


class CorpusManager:
    """
    Manages dynamic corpus uploads and Redis storage.

    Supports corpus types:
    - alias: Maps variations to canonical forms (e.g., email domains, company names)
    - lookup: Key-value lookups (e.g., postcode -> address data)
    - validation: Valid values list (e.g., allowed domains, country codes)
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize with optional Redis client."""
        self.redis = redis_client or self._get_redis()

    def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        try:
            client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            client.ping()
            return client
        except redis.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Redis at localhost:6379. "
                "Please ensure Redis is running."
            )

    # ==========================================================
    # CORPUS LOADING
    # ==========================================================

    def load_corpus_from_dataframe(
        self,
        df: pd.DataFrame,
        corpus_name: str,
        corpus_type: CorpusType,
        key_column: str,
        value_column: str,
        lookup_column: Optional[str] = None,
        normalize_keys: bool = True,
    ) -> Dict[str, Any]:
        """
        Load corpus data from a DataFrame into Redis.

        Args:
            df: Source DataFrame
            corpus_name: Name for this corpus (e.g., "email_domains", "postcodes")
            corpus_type: Type of corpus ("alias", "lookup", "validation")
            key_column: Column containing keys (e.g., "alias", "postcode", "domain")
            value_column: Column containing values (e.g., "canonical", "address")
            lookup_column: Optional additional lookup key (e.g., "postcode" for postcode+address alias)
            normalize_keys: Whether to normalize keys (lowercase, strip)

        Returns:
            dict with loading statistics
        """
        if df is None or df.empty:
            return {"status": "error", "message": "Empty DataFrame"}

        if key_column not in df.columns or value_column not in df.columns:
            return {
                "status": "error",
                "message": f"Required columns not found. Need: {key_column}, {value_column}"
            }

        loaded = 0
        skipped = 0
        errors = 0

        # Build the Redis key prefix
        prefix = f"corpus:{corpus_name}"

        # Store metadata
        metadata = {
            "corpus_name": corpus_name,
            "corpus_type": corpus_type,
            "key_column": key_column,
            "value_column": value_column,
            "lookup_column": lookup_column or "",
            "total_rows": len(df),
            "loaded_at": pd.Timestamp.now().isoformat(),
        }
        self.redis.hset(f"{prefix}:meta", mapping=metadata)

        # Load based on corpus type
        if corpus_type == "alias":
            loaded, skipped, errors = self._load_alias_corpus(
                df, prefix, key_column, value_column, lookup_column, normalize_keys
            )

        elif corpus_type == "lookup":
            loaded, skipped, errors = self._load_lookup_corpus(
                df, prefix, key_column, value_column, normalize_keys
            )

        elif corpus_type == "validation":
            loaded, skipped, errors = self._load_validation_corpus(
                df, prefix, key_column, normalize_keys
            )

        # Persist to RDS so corpus survives Redis restarts
        try:
            from core.storage.database import save_corpus_to_db
            save_corpus_to_db(
                corpus_name=corpus_name,
                corpus_type=corpus_type,
                key_column=key_column,
                value_column=value_column,
                df=df,
            )
        except Exception as _db_err:
            import logging
            logging.getLogger("dq_engine.corpus").warning(
                "Could not persist corpus '%s' to database: %s", corpus_name, _db_err
            )

        return {
            "status": "success",
            "corpus_name": corpus_name,
            "corpus_type": corpus_type,
            "loaded": loaded,
            "skipped": skipped,
            "errors": errors,
        }

    def _load_alias_corpus(
        self,
        df: pd.DataFrame,
        prefix: str,
        key_col: str,
        value_col: str,
        lookup_col: Optional[str],
        normalize: bool,
    ) -> tuple[int, int, int]:
        """Load alias-type corpus (variations → canonical)."""
        loaded = 0
        skipped = 0
        errors = 0

        for idx, row in df.iterrows():
            try:
                key = str(row[key_col]).strip()
                value = str(row[value_col]).strip()

                if not key or not value or pd.isna(key) or pd.isna(value):
                    skipped += 1
                    continue

                # Normalize key if requested
                if normalize:
                    key = key.lower()

                # Build Redis key
                if lookup_col and lookup_col in df.columns:
                    lookup = str(row[lookup_col]).strip().upper()
                    redis_key = f"{prefix}:alias:{lookup}:{key}"
                else:
                    redis_key = f"{prefix}:alias:{key}"

                # Store in Redis
                self.redis.set(redis_key, value)
                loaded += 1

            except Exception as e:
                errors += 1
                continue

        return loaded, skipped, errors

    def _load_lookup_corpus(
        self,
        df: pd.DataFrame,
        prefix: str,
        key_col: str,
        value_col: str,
        normalize: bool,
    ) -> tuple[int, int, int]:
        """Load lookup-type corpus (key → JSON object with multiple fields)."""
        loaded = 0
        skipped = 0
        errors = 0

        for idx, row in df.iterrows():
            try:
                key = str(row[key_col]).strip()

                if not key or pd.isna(key):
                    skipped += 1
                    continue

                # Normalize key if requested
                if normalize:
                    key = key.upper().replace(" ", "")

                # Build Redis key
                redis_key = f"{prefix}:lookup:{key}"

                # Store entire row as hash
                row_dict = row.to_dict()
                # Convert all values to strings, handle NaN
                row_dict = {k: str(v) if not pd.isna(v) else "" for k, v in row_dict.items()}

                self.redis.hset(redis_key, mapping=row_dict)
                loaded += 1

            except Exception as e:
                errors += 1
                continue

        return loaded, skipped, errors

    def _load_validation_corpus(
        self,
        df: pd.DataFrame,
        prefix: str,
        key_col: str,
        normalize: bool,
    ) -> tuple[int, int, int]:
        """Load validation-type corpus (set of valid values)."""
        loaded = 0
        skipped = 0
        errors = 0

        valid_values = []

        for idx, row in df.iterrows():
            try:
                value = str(row[key_col]).strip()

                if not value or pd.isna(value):
                    skipped += 1
                    continue

                # Normalize if requested
                if normalize:
                    value = value.lower()

                valid_values.append(value)

            except Exception:
                errors += 1
                continue

        # Store as a Redis set
        if valid_values:
            self.redis.sadd(f"{prefix}:valid", *valid_values)
            loaded = len(valid_values)

        return loaded, skipped, errors

    # ==========================================================
    # CORPUS QUERYING
    # ==========================================================

    def query_alias(
        self,
        corpus_name: str,
        key: str,
        lookup_key: Optional[str] = None,
        normalize: bool = True,
    ) -> Optional[str]:
        """
        Query an alias corpus to get canonical value.

        Args:
            corpus_name: Name of the corpus
            key: The alias/variation to look up
            lookup_key: Optional lookup key (e.g., postcode for address aliases)
            normalize: Whether to normalize the key

        Returns:
            Canonical value or None if not found
        """
        if normalize:
            key = key.lower().strip()

        prefix = f"corpus:{corpus_name}"

        if lookup_key:
            lookup_key = lookup_key.upper().strip()
            redis_key = f"{prefix}:alias:{lookup_key}:{key}"
        else:
            redis_key = f"{prefix}:alias:{key}"

        return self.redis.get(redis_key)

    def query_lookup(
        self,
        corpus_name: str,
        key: str,
        normalize: bool = True,
    ) -> Optional[Dict[str, str]]:
        """
        Query a lookup corpus to get full record.

        Args:
            corpus_name: Name of the corpus
            key: The lookup key
            normalize: Whether to normalize the key

        Returns:
            Dictionary with all fields or None if not found
        """
        if normalize:
            key = key.upper().replace(" ", "").strip()

        redis_key = f"corpus:{corpus_name}:lookup:{key}"
        result = self.redis.hgetall(redis_key)

        return result if result else None

    def is_valid(
        self,
        corpus_name: str,
        value: str,
        normalize: bool = True,
    ) -> bool:
        """
        Check if a value is in the validation corpus.

        Args:
            corpus_name: Name of the corpus
            value: Value to validate
            normalize: Whether to normalize the value

        Returns:
            True if value is valid, False otherwise
        """
        if normalize:
            value = value.lower().strip()

        redis_key = f"corpus:{corpus_name}:valid"
        return self.redis.sismember(redis_key, value)

    # ==========================================================
    # CORPUS MANAGEMENT
    # ==========================================================

    def list_corpora(self) -> List[Dict[str, Any]]:
        """List all loaded corpora with metadata."""
        meta_keys = self.redis.keys("corpus:*:meta")
        corpora = []

        for key in meta_keys:
            meta = self.redis.hgetall(key)
            if meta:
                corpora.append(meta)

        return corpora

    def get_corpus_stats(self, corpus_name: str) -> Dict[str, Any]:
        """Get statistics for a specific corpus."""
        prefix = f"corpus:{corpus_name}"
        meta = self.redis.hgetall(f"{prefix}:meta")

        if not meta:
            return {"status": "error", "message": "Corpus not found"}

        # Count keys based on corpus type
        corpus_type = meta.get("corpus_type")

        if corpus_type == "alias":
            count = len(self.redis.keys(f"{prefix}:alias:*"))
        elif corpus_type == "lookup":
            count = len(self.redis.keys(f"{prefix}:lookup:*"))
        elif corpus_type == "validation":
            count = self.redis.scard(f"{prefix}:valid")
        else:
            count = 0

        return {
            "status": "success",
            "corpus_name": corpus_name,
            "corpus_type": corpus_type,
            "key_count": count,
            "metadata": meta,
        }

    def delete_corpus(self, corpus_name: str) -> Dict[str, Any]:
        """Delete a corpus and all its data."""
        prefix = f"corpus:{corpus_name}"

        # Get all keys for this corpus
        keys = self.redis.keys(f"{prefix}:*")

        if not keys:
            return {"status": "error", "message": "Corpus not found"}

        # Delete all keys
        deleted = self.redis.delete(*keys)

        return {
            "status": "success",
            "corpus_name": corpus_name,
            "keys_deleted": deleted,
        }

    # ==========================================================
    # DB RESTORE
    # ==========================================================

    def restore_from_db(self) -> int:
        """
        Restore all corpora that were previously persisted to RDS back into Redis.

        Returns:
            Number of corpora successfully restored.
        """
        from core.storage.database import load_all_corpus_from_db

        records = load_all_corpus_from_db()
        restored = 0

        for record in records:
            try:
                result = self.load_corpus_from_dataframe(
                    df=record["df"],
                    corpus_name=record["corpus_name"],
                    corpus_type=record["corpus_type"],
                    key_column=record["key_column"],
                    value_column=record["value_column"],
                )
                if result.get("status") == "success":
                    restored += 1
                else:
                    import logging
                    logging.getLogger("dq_engine.corpus").warning(
                        "Restore skipped for '%s': %s",
                        record["corpus_name"],
                        result.get("message", "unknown error"),
                    )
            except Exception as e:
                import logging
                logging.getLogger("dq_engine.corpus").warning(
                    "Restore failed for '%s': %s", record["corpus_name"], e
                )

        return restored

    # ==========================================================
    # FILE PARSING
    # ==========================================================

    @staticmethod
    def parse_corpus_file(file) -> pd.DataFrame:
        """
        Parse corpus file (Excel, CSV, JSON, TXT) into DataFrame.

        Args:
            file: Uploaded file object from Streamlit

        Returns:
            DataFrame with parsed data
        """
        filename = file.name.lower()

        try:
            # Excel files
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)

            # CSV files
            elif filename.endswith('.csv'):
                df = pd.read_csv(file)

            # TSV files
            elif filename.endswith('.tsv'):
                df = pd.read_csv(file, sep='\t')

            # JSON files
            elif filename.endswith('.json'):
                content = file.read()
                data = json.loads(content)

                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # If dict of lists, use as-is
                    if all(isinstance(v, list) for v in data.values()):
                        df = pd.DataFrame(data)
                    # If dict of dicts, transpose
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("Unsupported JSON structure")

            # Text files (assume key-value pairs or single column)
            elif filename.endswith('.txt'):
                lines = file.read().decode('utf-8').strip().split('\n')

                # Check if it's key-value pairs (separated by tab, comma, or :)
                if '\t' in lines[0]:
                    df = pd.read_csv(pd.io.common.StringIO('\n'.join(lines)), sep='\t', header=None)
                    df.columns = ['key', 'value'] if df.shape[1] == 2 else df.columns
                elif ',' in lines[0]:
                    df = pd.read_csv(pd.io.common.StringIO('\n'.join(lines)))
                elif ':' in lines[0]:
                    pairs = [line.split(':', 1) for line in lines if ':' in line]
                    df = pd.DataFrame(pairs, columns=['key', 'value'])
                else:
                    # Single column of values
                    df = pd.DataFrame({'value': lines})

            else:
                raise ValueError(f"Unsupported file type: {filename}")

            return df

        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}")
