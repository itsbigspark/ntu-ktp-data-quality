from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import re
import yaml

from core.local_store import (
    get_postcode_info,
    is_valid_domain,
    get_company_record,
    get_alias_canonical,
)
from core.address_resolution import resolve_address  # already exists in your tree


# ==============================================================
# LOAD YAML RULES
# ==============================================================
def _load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


# ==============================================================
# NORMALIZATION UTILS
# ==============================================================
def _norm_value(value: str, cfg: Dict[str, Any]) -> str:
    out = str(value)
    norm = cfg.get("normalize", {})

    if norm.get("strip_spaces"):   out = out.strip()
    if norm.get("lowercase"):      out = out.lower()
    if norm.get("uppercase"):      out = out.upper()
    if norm.get("collapse_spaces"): out = re.sub(r"\s+", " ", out).strip()
    if norm.get("remove_punct"):
        out = re.sub(r"[^\w\s@.\-]", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
    if norm.get("keep_digits_only"): out = re.sub(r"\D+", "", out)

    # hyphen indexing for sort codes e.g. 123456 → 12-34-56
    if "add_hyphens" in norm:
        idxs: List[int] = norm["add_hyphens"]
        digits = re.sub(r"\D+", "", out)
        parts = []; prev = 0
        for i in idxs:
            parts.append(digits[prev:i]); prev = i
        parts.append(digits[prev:])
        out = "-".join(filter(None, parts))

    if norm.get("remove_spaces"): out = out.replace(" ", "")
    return out


# ==============================================================
# MAIN STANDARDISATION ENGINE
# ==============================================================
def standardize_dataframe(df: pd.DataFrame, rules_path: Path) -> pd.DataFrame:
    rules = _load_rules(rules_path)
    if not rules:
        return df

    df = df.copy()

    for attr, cfg in rules.items():
        columns = cfg.get("columns", [])
        pattern = cfg.get("pattern")
        regex = re.compile(pattern) if pattern else None

        for col in columns:
            if col not in df.columns:
                continue

            s = df[col].astype(str)

            # 1) Base normalisation
            s = s.map(lambda v: _norm_value(v, cfg))

            # 2) Alias → Canonical lookup (Redis set)
            if attr == "company_name" and cfg.get("use_alias_corpus"):
                s = s.map(lambda v: get_alias_canonical(v) or v)

            # 3) Canonical name enrichment
            if attr == "company_name" and cfg.get("corpus_column_key"):
                def canon(v):
                    rec = get_company_record(v)
                    return rec.get("canonical") if isinstance(rec, dict) else v
                s = s.map(canon)

            # 4) Postcode normalisation + Redis normal form
            if attr == "postcode" and cfg.get("use_corpus"):
                def fix_pc(v):
                    rec = get_postcode_info(v)
                    if isinstance(rec, dict) and "normalized" in rec:
                        return rec["normalized"]
                    return v.replace(" ", "").upper()
                s = s.map(fix_pc)

            # 5) Address enrichment using postcode-tree + fuzzy resolver
            if attr == "address" and cfg.get("use_postcode_corpus"):
                postcode_col = cfg.get("postcode_col")
                if postcode_col in df.columns:
                    df[col] = df.apply(
                        lambda r: resolve_address(r[col], r[postcode_col]) or r[col],
                        axis=1,
                    )
                    continue  # skip direct assignment below

            df[col] = s

    return df