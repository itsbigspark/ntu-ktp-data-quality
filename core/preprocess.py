from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
import yaml

def basic_clean(df: pd.DataFrame, lowercase=True, strip_ws=True, normalize_punct=True) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == 'O' or str(out[c].dtype).startswith('string'):
            s = out[c].astype(str)
            if lowercase:
                s = s.str.lower()
            if strip_ws:
                s = s.str.strip().str.replace(r"\s+", " ", regex=True)
            if normalize_punct:
                s = (s.str.replace("&", " and ")
                       .str.replace(r"[^\w\s@\.-]", " ", regex=True)
                       .str.replace(r"\s+", " ", regex=True).str.strip())
            out[c] = s
    return out



def apply_abbreviations(df: pd.DataFrame, yaml_path: Path) -> pd.DataFrame:
    out = df.copy()
    if not Path(yaml_path).exists():
        return out
    mapping = yaml.safe_load(Path(yaml_path).read_text()) or {}
    if not isinstance(mapping, dict) or not mapping:
        return out
    # token-level replace
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, mapping.keys())) + r")\b", flags=re.IGNORECASE)
    def repl(m):
        raw = m.group(0)
        low = raw.lower()
        return mapping.get(low, mapping.get(raw, raw))
    for c in out.columns:
        if out[c].dtype == 'O' or str(out[c].dtype).startswith('string'):
            out[c] = out[c].astype(str).str.replace(pattern, repl, regex=True)
    return out


