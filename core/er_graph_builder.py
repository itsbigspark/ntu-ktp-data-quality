# core/er_graph_builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import networkx as nx
import pandas as pd

@dataclass
class RecordRef:
    dataset: str
    row_id: int

def _node_id(ds: str, rid: int) -> str:
    return f"{ds}::{rid}"

def build_er_graph(
    datasets: Dict[str, pd.DataFrame],
    pair_tables: List[pd.DataFrame],
    *,
    score_col: str = "score",
    col_contribs_col: Optional[str] = "col_contribs",  # dict per edge {col: score}
    min_score: float = 0.6
) -> nx.Graph:
    """
    pair_tables: list of DataFrames each containing:
      ['dataset_left','row_id_left','dataset_right','row_id_right', score_col, (optional) col_contribs_col]
    """
    G = nx.Graph()
    # Nodes with metadata
    for ds, df in datasets.items():
        for rid in df.index:
            nid = _node_id(ds, int(rid))
            if nid not in G:
                G.add_node(nid, dataset=ds, row_id=int(rid))

    # Edges
    for pairs in pair_tables:
        req = {"dataset_left","row_id_left","dataset_right","row_id_right", score_col}
        if not req.issubset(set(map(str, pairs.columns))):
            continue
        sub = pairs[pairs[score_col] >= min_score].copy()
        for r in sub.itertuples(index=False):
            dsL, idL = getattr(r, "dataset_left"), int(getattr(r, "row_id_left"))
            dsR, idR = getattr(r, "dataset_right"), int(getattr(r, "row_id_right"))
            s       = float(getattr(r, score_col))
            nidL, nidR = _node_id(dsL, idL), _node_id(dsR, idR)

            contribs = {}
            if col_contribs_col and hasattr(r, col_contribs_col):
                try:
                    v = getattr(r, col_contribs_col)
                    if isinstance(v, dict):
                        contribs = v
                except Exception:
                    pass

            if G.has_edge(nidL, nidR):
                # keep the best score; merge contribs
                if s > G[nidL][nidR]["weight"]:
                    G[nidL][nidR]["weight"] = s
                    G[nidL][nidR]["col_contribs"] = contribs
            else:
                G.add_edge(nidL, nidR, weight=s, col_contribs=contribs)
    return G