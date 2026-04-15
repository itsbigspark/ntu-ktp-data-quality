from __future__ import annotations
from typing import Dict, Any, List, Union

def empty_rules() -> Dict[str, Any]:
    return {"columns": {}, "uniqueness": {}, "cross_field": []}

def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    return [str(x)]

def normalize_rules(rules: dict) -> dict:
    rules = rules or {}
    rules.setdefault("columns", {})
    rules.setdefault("uniqueness", {})
    rules.setdefault("cross_field", [])

    cols = rules["columns"]
    for c, r in list(cols.items()):
        r = r or {}
        # normalize regex -> list[str]
        if "regex" in r:
            r["regex"] = _as_list(r.get("regex"))

        # normalize date_format -> list[str]
        if "date_format" in r:
            r["date_format"] = _as_list(r.get("date_format"))

        # coerce numeric bounds
        if "min" in r:
            try: r["min"] = float(r["min"])
            except Exception: del r["min"]
        if "max" in r:
            try: r["max"] = float(r["max"])
            except Exception: del r["max"]

        # allowed_values -> list[str]
        if "allowed_values" in r and not isinstance(r["allowed_values"], list):
            r["allowed_values"] = _as_list(r["allowed_values"])

        cols[c] = r

    rules["columns"] = cols
    rules.setdefault("uniqueness", {})
    rules.setdefault("cross_field", [])
    return rules