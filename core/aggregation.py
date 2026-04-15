# core/aggregation.py
from __future__ import annotations
from typing import Dict, List, Callable, Tuple
import pandas as pd
import numpy as np


def _majority(series: pd.Series) -> str:
    """Return the most frequent (mode) string value."""
    vc = series.astype(str).value_counts(dropna=False)
    return vc.index[0] if not vc.empty else ""


def _numeric_reduce(series: pd.Series) -> float:
    """Return median of numeric values, ignoring NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanmedian(s)) if s.notna().any() else np.nan


def _merge_text(series: pd.Series) -> str:
    """Return merged unique text values (joined with |)."""
    vals = pd.Series(series).dropna().astype(str).unique().tolist()
    return " | ".join(sorted(vals)) if vals else ""


def build_golden_records(
    datasets: Dict[str, pd.DataFrame],
    cluster_df: pd.DataFrame,
    reduce_map: Dict[str, Callable] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build golden records from clustered entity matches.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dataset name -> DataFrame.
    cluster_df : pd.DataFrame
        Must contain columns: [cluster_id, dataset, row_id].
    reduce_map : dict[str, Callable], optional
        Optional custom reducer per column (e.g. {"age": np.mean}).

    Returns
    -------
    golden_df : pd.DataFrame
        Consolidated "golden record" per cluster.
    lineage_df : pd.DataFrame
        Mapping of cluster_id -> dataset -> row_id.
    """
    if reduce_map is None:
        reduce_map = {}

    rows = []
    lineage_rows = []

    for cid, grp in cluster_df.groupby("cluster_id"):
        parts = []

        # Collect all rows belonging to this cluster
        for _, r in grp.iterrows():
            ds, rid = r["dataset"], int(r["row_id"])
            if ds not in datasets:
                continue
            df = datasets[ds]
            if rid >= len(df):
                continue

            rec = df.iloc[rid].copy()
            rec.index = pd.MultiIndex.from_product([[ds], rec.index], names=["dataset", "column"])
            parts.append(rec)
            lineage_rows.append({"cluster_id": cid, "dataset": ds, "row_id": rid})

        if not parts:
            continue

        # Combine all records for this cluster (long-form)
        tall = pd.concat(parts)

        # ✅ Collapse duplicate (dataset, column) pairs
        tall = tall.groupby(tall.index).agg(lambda x: " | ".join(map(str, pd.Series(x).dropna().unique())))

        # ✅ Reassign proper MultiIndex level names
        if isinstance(tall.index, pd.MultiIndex):
            tall.index.set_names(["dataset", "column"], inplace=True)
        else:
            tall.index = pd.MultiIndex.from_tuples(tall.index, names=["dataset", "column"])

        # ✅ Pivot to wide format (datasets as columns)
        wide = tall.unstack(level="dataset").sort_index(axis=1)

        # ✅ Reduce across datasets per column
        out = {}
        for col in wide.index:
            col_vals = wide.loc[col].dropna()

            # Pick reducer: use user-specified or infer by dtype
            reducer = reduce_map.get(col)
            if reducer is None:
                try:
                    if pd.to_numeric(col_vals, errors="coerce").notna().any():
                        reducer = _numeric_reduce
                    else:
                        reducer = _majority
                except Exception:
                    reducer = _merge_text

            # Compute reduced value
            try:
                out[col] = reducer(col_vals)
            except Exception:
                out[col] = _merge_text(col_vals)

        out["cluster_id"] = cid
        rows.append(out)

    # ✅ Build final DataFrames
    golden = pd.DataFrame(rows).set_index("cluster_id").reset_index()
    lineage = pd.DataFrame(lineage_rows)

    return golden, lineage