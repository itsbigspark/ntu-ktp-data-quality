# core/clustering.py
from __future__ import annotations
from typing import List, Dict, Any
import networkx as nx
import pandas as pd

def clusters_connected_components(G: nx.Graph, min_size: int = 1) -> List[List[str]]:
    comps = [list(c) for c in nx.connected_components(G)]
    return [c for c in comps if len(c) >= min_size]

def clusters_louvain(G: nx.Graph, min_size: int = 1) -> List[List[str]]:
    try:
        import community as community_louvain  # python-louvain
    except Exception:
        return clusters_connected_components(G, min_size=min_size)
    part = community_louvain.best_partition(G, weight="weight")
    by_comm: Dict[int, List[str]] = {}
    for n, comm in part.items():
        by_comm.setdefault(comm, []).append(n)
    return [c for c in by_comm.values() if len(c) >= min_size]

def clusters_to_df(G: nx.Graph, clusters: List[List[str]]) -> pd.DataFrame:
    rows = []
    for cid, nodes in enumerate(clusters, start=1):
        for n in nodes:
            rows.append({
                "cluster_id": cid,
                "node": n,
                "dataset": G.nodes[n].get("dataset"),
                "row_id": G.nodes[n].get("row_id")
            })
    return pd.DataFrame(rows)