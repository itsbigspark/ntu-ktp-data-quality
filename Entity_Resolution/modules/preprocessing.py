# modules/preprocessing.py
import pandas as pd
import re
from .utils import (
    clean_text,
    remove_punctuation,
    strip_currency_symbols,
    expand_abbreviations,
)

def infer_majority_format(series):
    series = series.dropna().astype(str)
    lowercase_ratio = sum(s.islower() for s in series) / len(series)
    punct_ratio = sum(bool(re.search(r"[^\w\s]", s)) for s in series) / len(series)
    return {
        "lowercase": lowercase_ratio > 0.6,
        "remove_punctuation": punct_ratio < 0.4,
        "strip_currency": False
    }

def standardize_datasets(
    dfs,
    common_columns,
    mode,
    standard_dataset=None,
    manual_rules=None,
    abbr_dict=None,
    column_sources=None
):
    cleaned_dfs = {}
    preview_samples = {}

    for name, df in dfs.items():
        df_cleaned = df.copy()

        for col in common_columns:
            if col not in df_cleaned.columns:
                continue

            original_values = df_cleaned[col].copy()

            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue

            df_cleaned[col] = df_cleaned[col].astype(str).fillna("")
            rules = {}

            if mode == "Select a dataset as standard":
                reference_values = dfs[standard_dataset][col].dropna().astype(str)
                rules = {
                    "lowercase": all(s.islower() for s in reference_values[:20]),
                    "remove_punctuation": all(not re.search(r"[^\w\s]", s) for s in reference_values[:20]),
                    "strip_currency": False,
                }

            elif mode == "Manually define standardization rules":
                rules = manual_rules.get(col, {})

            elif mode == "Select per-column source":
                chosen_dataset = column_sources.get(col)
                reference_values = dfs[chosen_dataset][col].dropna().astype(str)
                rules = {
                    "lowercase": all(s.islower() for s in reference_values[:20]),
                    "remove_punctuation": all(not re.search(r"[^\w\s]", s) for s in reference_values[:20]),
                    "strip_currency": False,
                }

            elif mode == "Use majority voting per column":
                rules = infer_majority_format(df_cleaned[col])

            if rules.get("lowercase"):
                df_cleaned[col] = df_cleaned[col].str.lower()

            if rules.get("remove_punctuation"):
                df_cleaned[col] = df_cleaned[col].apply(remove_punctuation)

            if rules.get("strip_currency"):
                df_cleaned[col] = df_cleaned[col].apply(strip_currency_symbols)

            if abbr_dict:
                df_cleaned[col] = df_cleaned[col].apply(lambda x: expand_abbreviations(x, abbr_dict))

            preview_samples[col] = {
                "original": list(original_values[:5]),
                "standardized": list(df_cleaned[col][:5])
            }

        cleaned_dfs[name] = df_cleaned

    return cleaned_dfs, preview_samples
