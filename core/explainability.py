# core/explainability.py
from __future__ import annotations
from typing import Dict, List
import networkx as nx
import pandas as pd
from collections import defaultdict

def edge_table(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "u": u, "v": v,
            "weight": float(d.get("weight", 0.0)),
            "col_contribs": d.get("col_contribs", {})
        })
    return pd.DataFrame(rows)

def column_importance(G: nx.Graph, top_n: int = 15) -> pd.DataFrame:
    agg = defaultdict(float)
    for _, _, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        cc = d.get("col_contribs", {}) or {}
        for c, s in cc.items():
            try:
                agg[c] += float(s) * w
            except Exception:
                pass
    imp = pd.DataFrame(
        sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n],
        columns=["column","importance"]
    )
    return imp