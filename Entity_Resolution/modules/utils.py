# modules/utils.py
import re
import pandas as pd
from unidecode import unidecode

def clean_text(text):
    if pd.isna(text):
        return ""
    text = unidecode(str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

def remove_punctuation(text):
    return re.sub(r"[^\w\s]", "", text) if isinstance(text, str) else text

def strip_currency_symbols(value):
    return re.sub(r"[^\d.]", "", str(value)) if pd.notna(value) else None

def expand_abbreviations(text, abbr_dict):
    if not isinstance(text, str):
        return text
    for short, full in abbr_dict.items():
        pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
        text = pattern.sub(full, text)
    return text
