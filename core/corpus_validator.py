# ==========================================================
# core/corpus_validator.py
# ==========================================================
from __future__ import annotations
import pandas as pd
from rapidfuzz import fuzz, process
from pathlib import Path
from chardet import detect


# ==========================================================
# Helper: Safe corpus loader with CSV, TXT, XLSX support
# ==========================================================
def _read_corpus_file(corpus_path: str) -> list[str]:
    """
    Load a corpus file (CSV, TXT, or XLSX) with automatic encoding detection
    and resilient fallbacks for malformed or messy data.
    Returns a deduplicated list of values.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    ext = path.suffix.lower()
    corpus_values = []

    # --- Excel Handler ---
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path)
            # pick first non-empty column
            first_col = next((c for c in df.columns if df[c].notna().any()), df.columns[0])
            corpus_values = (
                df[first_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            return corpus_values
        except Exception as e:
            raise ValueError(f"Failed to read Excel corpus: {e}")

    # --- Encoding detection for text-based formats ---
    with open(path, "rb") as f:
        raw = f.read(4096)
    enc = detect(raw).get("encoding") or "utf-8"

    try:
        # --- CSV Handler ---
        if ext == ".csv":
            corpus_df = pd.read_csv(
                path,
                encoding=enc,
                on_bad_lines="skip",
                engine="python",
                dtype=str,
            )
        # --- TXT Handler ---
        else:
            corpus_df = pd.read_table(
                path,
                encoding=enc,
                header=None,
                on_bad_lines="skip",
                engine="python",
                dtype=str,
            )

        corpus_values = (
            corpus_df.iloc[:, 0]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    except Exception:
        # --- Final Fallback: Line-by-line safe read ---
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                corpus_values = [line.strip() for line in f if line.strip()]
        except Exception:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                corpus_values = [line.strip() for line in f if line.strip()]

    # --- Deduplicate & return ---
    return list(dict.fromkeys(corpus_values))


# ==========================================================
# Main: Corpus Validation with keep / replace / remove logic
# ==========================================================
def validate_column_via_corpus(
    df: pd.DataFrame,
    column: str,
    corpus_path: str,
    fuzzy: bool = False,
    threshold: float = 90.0,
    action: str = "keep",  # "keep", "replace", "remove"
) -> pd.DataFrame:
    """
    Validate column values against a local corpus (CSV, TXT, or XLSX).

    Args:
        df: Input DataFrame
        column: Column to validate
        corpus_path: Path to corpus file
        fuzzy: Whether to use fuzzy string matching
        threshold: Fuzzy match threshold (0–100)
        action: How to handle invalid rows:
                - "keep": retain all rows (default)
                - "replace": replace invalid values with best corpus match
                - "remove": drop invalid rows entirely

    Returns:
        DataFrame with extra columns:
          <column>_in_corpus
          <column>_closest_match
    """
    df = df.copy()
    corpus_values = _read_corpus_file(corpus_path)
    if not corpus_values:
        raise ValueError(f"No valid entries found in corpus: {corpus_path}")

    verified, matches = [], []

    for val in df[column].astype(str).fillna("").tolist():
        v = val.strip()
        if not v:
            verified.append("❌")
            matches.append(None)
            continue

        # --- Exact matching ---
        if not fuzzy:
            ok = v.lower() in [x.lower() for x in corpus_values]
            verified.append("✅" if ok else "❌")
            matches.append(v if ok else None)
        else:
            # --- Fuzzy matching (RapidFuzz) ---
            try:
                match, score, _ = process.extractOne(v, corpus_values, scorer=fuzz.WRatio)
                ok = score >= threshold
                verified.append("✅" if ok else "❌")
                matches.append(match if ok else f"{match} ({score:.1f}%)")
            except Exception:
                verified.append("❌")
                matches.append(None)

    # Add validation results
    df[f"{column}_in_corpus"] = verified
    df[f"{column}_closest_match"] = matches

    # ======================================================
    # Post-processing actions
    # ======================================================
    if action == "replace":
        # Replace invalid values with closest corpus match
        df[column] = df.apply(
            lambda r: (
                r[f"{column}_closest_match"]
                if r[f"{column}_in_corpus"] == "✅" and pd.notna(r[f"{column}_closest_match"])
                else r[column]
            ),
            axis=1,
        )

    elif action == "remove":
        # Drop rows not found in corpus
        df = df[df[f"{column}_in_corpus"] == "✅"].reset_index(drop=True)

    return df