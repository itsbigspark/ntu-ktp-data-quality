# core/corpus_client.py
# ---------------------------------------------------------------------
# Small wrapper that calls postcode_corpus and (optionally) applies
# corpus-based fixes directly to a DataFrame.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from .postcode_corpus import (
    postcode_address_findings_from_corpus,
    get_redis,
)


AutoMode = Literal["none", "high_conf_only", "always"]


def standardise_address_with_corpus(
    df: pd.DataFrame,
    postcode_col: str = "postcode",
    address_col: str = "address",
    use_fuzzy: bool = False,
    auto_mode: AutoMode = "always",
    min_confidence: float = 0.80,
    redis_client=None,
) -> pd.DataFrame:
    """
    High-level helper used by the Streamlit app.

    1. Calls postcode_address_findings_from_corpus(...) to get a
       findings DataFrame with columns:
         - row_id
         - column
         - issue_type
         - value
         - suggested_fix
         - source
         - confidence

    2. Depending on `auto_mode`, applies fixes directly to a copy of df.

       auto_mode:
         - "none"           → do NOT modify df, just return df as-is
         - "high_conf_only" → apply fixes where confidence >= min_confidence
         - "always"         → apply ALL non-null suggested_fix

    3. Returns the modified DataFrame (or original copy if no fixes).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if postcode_col not in df.columns or address_col not in df.columns:
        # Nothing to do if columns aren't there
        return df

    # Make a working copy
    out = df.copy()

    # Ensure we have a redis client
    redis_client = redis_client or get_redis()

    # ---- 1) Get findings from corpus layer ----
    findings = postcode_address_findings_from_corpus(
        out,
        postcode_col=postcode_col,
        address_col=address_col,
        redis_client=redis_client,
        use_fuzzy_alias=use_fuzzy,
    )

    # If postcode_corpus returns nothing, just return df unchanged
    if findings is None or len(findings) == 0:
        return out

    # Optional: stash findings on the DataFrame as an attribute
    # (not strictly used by Streamlit right now, but handy)
    out._dq_postcode_findings = findings  # type: ignore[attr-defined]

    # ---- 2) Decide whether to apply fixes or not ----
    mode = auto_mode or "always"
    mode = mode.lower()

    if mode == "none":
        # Just return the DataFrame; no auto-application
        return out

    # We only auto-apply for address_not_canonical_for_postcode
    # and where suggested_fix is present
    apply_mask = findings["suggested_fix"].notna()

    if mode == "high_conf_only" and "confidence" in findings.columns:
        apply_mask &= findings["confidence"] >= float(min_confidence)

    to_apply = findings[apply_mask]

    # ---- 3) Apply fixes row by row ----
    for row in to_apply.itertuples(index=False):
        # row has fields: row_id, column, issue_type, value,
        #                 suggested_fix, source, confidence
        row_id = getattr(row, "row_id", None)
        col = getattr(row, "column", None)
        fix = getattr(row, "suggested_fix", None)

        if row_id is None or col is None or fix is None:
            continue

        try:
            out.at[int(row_id), col] = fix
        except Exception:
            # If some index/column mismatch happens, skip silently
            continue

    return out