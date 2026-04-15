# core/corpus_client.py
import redis
from difflib import SequenceMatcher

_redis_client = None

def get_redis():
    """Lazy-init Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return _redis_client


def resolve_alias_exact(postcode: str, raw_addr: str):
    """
    Exact alias lookup:
    alias:POSTCODE:alias_lower -> canonical_address
    """
    if not postcode or not raw_addr:
        return None
    r = get_redis()
    key = f"alias:{postcode.strip().upper()}:{raw_addr.strip().lower()}"
    return r.get(key)


def resolve_alias_fuzzy(postcode: str, raw_addr: str, threshold: float = 0.90):
    """
    OPTIONAL fallback: fuzzy match within aliases for a postcode.
    This is more expensive, so only use if you really need it.
    """
    if not postcode or not raw_addr:
        return None

    r = get_redis()
    pattern = f"alias:{postcode.strip().upper()}:*"
    candidates = r.keys(pattern)
    if not candidates:
        return None

    raw_norm = raw_addr.strip().lower()
    best_score = 0.0
    best_value = None

    for k in candidates:
        # k looks like "alias:PC:alias_text"
        _, _, alias_text = k.split(":", 2)
        score = SequenceMatcher(None, raw_norm, alias_text).ratio()
        if score > best_score:
            best_score = score
            best_value = r.get(k)

    if best_score >= threshold:
        return best_value
    return None


def standardise_address_with_corpus(df, postcode_col: str, address_col: str,
                                    use_fuzzy: bool = False):
    """
    For each row, if (postcode, address) exists as alias → replace with canonical_address.
    Returns a new DataFrame.
    """
    from pandas import DataFrame
    if not isinstance(df, DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    df_out = df.copy()
    if postcode_col not in df_out.columns or address_col not in df_out.columns:
        return df_out  # silently do nothing if columns missing

    mask = df_out[postcode_col].notna() & df_out[address_col].notna()
    rows = df_out[mask].copy()

    for idx, row in rows.iterrows():
        pc = str(row[postcode_col]).strip().upper()
        addr_raw = str(row[address_col]).strip()

        canonical = resolve_alias_exact(pc, addr_raw)
        if not canonical and use_fuzzy:
            canonical = resolve_alias_fuzzy(pc, addr_raw, threshold=0.90)

        if canonical:
            df_out.at[idx, address_col] = canonical

    return df_out