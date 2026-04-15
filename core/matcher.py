from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

try:
    import jellyfish
except Exception:
    jellyfish = None


# =====================================================
# Helpers
# =====================================================
def _normalize_str(x: object) -> str:
    s = str(x).strip().lower()
    return " ".join(s.split())

def _unique_unordered_pairs(pairs: Iterable[Tuple[int,int]]) -> List[Tuple[int,int]]:
    out = set()
    for i,j in pairs:
        if i==j: continue
        a,b = (i,j) if i<j else (j,i)
        out.add((a,b))
    return list(out)

def _block_candidates(df: pd.DataFrame, cols: List[str]) -> List[Tuple[int,int]]:
    if not cols: return []
    keys = (df[cols].astype(str).applymap(_normalize_str).agg("||".join, axis=1)).tolist()
    groups = {}
    for idx,k in enumerate(keys):
        groups.setdefault(k,[]).append(idx)
    cand = []
    for idxs in groups.values():
        if len(idxs)<2: continue
        cand.extend(itertools.combinations(idxs,2))
    return _unique_unordered_pairs(cand)

def _knn_pairs(X: np.ndarray, k: int=20) -> List[Tuple[int,int]]:
    n = X.shape[0]
    if n<=1: return []
    k = max(1, min(k, n-1))
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(X)
    _, inds = nn.kneighbors(X, return_distance=True)
    pairs=[]
    for i in range(n):
        for j in inds[i,1:]:
            pairs.append((int(i), int(j)))
    return _unique_unordered_pairs(pairs)

def _pair_cosines(X: np.ndarray, pairs: List[Tuple[int,int]]) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1); norms[norms==0]=1e-9
    sims = np.empty(len(pairs), dtype=np.float32)
    for t,(i,j) in enumerate(pairs):
        sims[t] = float(np.dot(X[i],X[j])/(norms[i]*norms[j]))
    return sims

def _jaro_concat(df: pd.DataFrame, key_cols: List[str], i: int, j:int) -> float:
    if not key_cols or jellyfish is None: return 0.0
    s1 = " ".join(str(df.iloc[i][c]) for c in key_cols)
    s2 = " ".join(str(df.iloc[j][c]) for c in key_cols)
    try:
        return float(jellyfish.jaro_winkler_similarity(s1,s2))
    except Exception:
        try:
            return float(jellyfish.jaro_winkler(s1,s2))
        except Exception:
            return 0.0

def _normalize_weights(weights: Optional[Dict[str,float]], cols: List[str]) -> Dict[str,float]:
    if not cols: return {}
    if not weights: return {c:1.0/len(cols) for c in cols}
    usable = {c: float(w) for c,w in weights.items() if c in cols and float(w)>=0.0}
    total = sum(usable.values())
    if total<=0: return {c:1.0/len(cols) for c in cols}
    return {c: usable.get(c,0.0)/total for c in cols}


# =====================================================
# FS-style probabilistic weighting
# =====================================================
def _fs_probabilistic_score(
    vec_pack: dict,
    pairs: List[Tuple[int,int]],
    threshold: float = 0.85
) -> pd.DataFrame:
    """
    Compute Fellegi–Sunter style probabilistic match scores from column vectors.
    Returns DataFrame with row_i, row_j, prob_match.
    """
    col_vecs = vec_pack.get("col_vecs", {})
    if not col_vecs:
        return pd.DataFrame(columns=["row_i","row_j","score"])

    # Estimate m_i and u_i parameters from similarity distributions
    params = {}
    for col, Xc in col_vecs.items():
        norms = np.linalg.norm(Xc, axis=1); norms[norms==0]=1e-9
        sims = cosine_similarity(Xc)
        sims_flat = sims[np.triu_indices_from(sims, k=1)]
        m_i = np.mean(sims_flat >= threshold)
        u_i = np.mean(sims_flat < threshold) * 0.1 + 0.01
        m_i = np.clip(m_i, 0.01, 0.99)
        u_i = np.clip(u_i, 0.01, 0.99)
        params[col] = (m_i, u_i)

    # Compute log-likelihood ratio per pair
    rows = []
    for (i,j) in pairs:
        LLR = 0.0
        for col, Xc in col_vecs.items():
            m_i, u_i = params[col]
            norms = np.linalg.norm(Xc, axis=1); norms[norms==0]=1e-9
            sim = float(np.dot(Xc[i], Xc[j]) / (norms[i]*norms[j]))
            agree = sim >= threshold
            LLR += agree * np.log(m_i / u_i) + (1 - agree) * np.log((1 - m_i) / (1 - u_i))
        prob = 1 / (1 + np.exp(-LLR))
        rows.append({"row_i": i, "row_j": j, "score": prob})

    return pd.DataFrame(rows)


# =====================================================
# Main Scoring Function
# =====================================================
def compute_pair_scores(
    df: pd.DataFrame,
    vec_pack: dict,
    comparison_mode: str="Row-concatenated",
    weights: Optional[Dict[str,float]] = None,
    run_mode: str="Fast KNN",
    topk: int = 20,
    block_cols: Optional[List[str]] = None,
    key_cols: Optional[List[str]] = None,
    key_alpha: float = 0.4,
) -> pd.DataFrame:
    n = len(df)
    if n<=1:
        return pd.DataFrame(columns=["row_i","row_j","score"])

    # Normalise short aliases from UI radio buttons
    _aliases = {"NxN": "NxN (full)", "Blocking+KNN": "Blocking + KNN"}
    run_mode = _aliases.get(run_mode, run_mode)

    # Candidate generation
    candidates=[]
    if run_mode in ("Blocking","Blocking + KNN"):
        if block_cols:
            candidates = _block_candidates(df, block_cols)
        else:
            run_mode = "Fast KNN" if run_mode=="Blocking + KNN" else "NxN (full)"
    if run_mode=="Fast KNN":
        X0 = vec_pack["row_concat_vec"]
        candidates = _unique_unordered_pairs(candidates + _knn_pairs(X0, k=topk))
    elif run_mode=="Blocking + KNN" and block_cols:
        keys = (df[block_cols].astype(str).applymap(_normalize_str).agg("||".join, axis=1)).tolist()
        buckets={}
        for i,k in enumerate(keys): buckets.setdefault(k,[]).append(i)
        X0 = vec_pack["row_concat_vec"]
        allp=[]
        for idxs in buckets.values():
            if len(idxs)<2: continue
            Xb = X0[idxs]
            lp = _knn_pairs(Xb, k=min(topk, len(idxs)-1))
            allp.extend((idxs[i], idxs[j]) for (i,j) in lp)
        candidates = _unique_unordered_pairs(allp)
    if run_mode=="NxN (full)":
        candidates = [(i,j) for i in range(n) for j in range(i+1,n)]

    rows=[]
    if comparison_mode=="Row-concatenated":
        X = vec_pack["row_concat_vec"]
        if run_mode=="NxN (full)":
            S = cosine_similarity(X)
            use_jaro = (key_cols and jellyfish is not None and 0.0<key_alpha<=1.0)
            for (i,j) in candidates:
                base = float(S[i,j])
                if use_jaro:
                    jw = _jaro_concat(df, key_cols, i, j)
                    score = (1.0-key_alpha)*base + key_alpha*jw
                else:
                    score = base
                rows.append({"row_i":i, "row_j":j, "score":score})
        else:
            sims = _pair_cosines(X, candidates)
            use_jaro = (key_cols and jellyfish is not None and 0.0<key_alpha<=1.0)
            for t,(i,j) in enumerate(candidates):
                base = float(sims[t])
                if use_jaro:
                    jw = _jaro_concat(df, key_cols, i, j)
                    base = (1.0-key_alpha)*base + key_alpha*jw
                rows.append({"row_i":i, "row_j":j, "score":base})

    elif comparison_mode=="Column-wise (weighted)":
        cols = list(vec_pack["col_vecs"].keys())
        wts = _normalize_weights(weights, cols)
        norms = {c: np.linalg.norm(vec_pack["col_vecs"][c], axis=1) for c in cols}
        for c in cols:
            norms[c][norms[c]==0]=1e-9
        def col_cos(c,i,j):
            Xc = vec_pack["col_vecs"][c]
            return float(np.dot(Xc[i],Xc[j])/(norms[c][i]*norms[c][j]))
        for (i,j) in candidates:
            s=0.0
            for c,w in wts.items():
                if w<=0: continue
                s += w * col_cos(c,i,j)
            rows.append({"row_i":i, "row_j":j, "score":s})

    elif comparison_mode=="Column-wise (FS-style probabilistic)":
        # Call new FS-style probabilistic scoring
        out = _fs_probabilistic_score(vec_pack, candidates)
        return out.sort_values("score", ascending=False, ignore_index=True)

    else:
        raise ValueError(f"Unknown comparison_mode={comparison_mode}")

    out = pd.DataFrame(rows)
    if out.empty: 
        return pd.DataFrame(columns=["row_i","row_j","score"])
    return out.sort_values("score", ascending=False, ignore_index=True)


def label_pairs(df: pd.DataFrame, similar_thr: float=0.6, duplicate_thr: float=0.8) -> pd.DataFrame:
    if df.empty: 
        return df.assign(label=pd.Series(dtype=str))
    def lab(s):
        if s>=duplicate_thr: return "duplicate"
        if s>=similar_thr: return "similar"
        return "none"
    out = df.copy()
    out["label"] = out["score"].map(lab)
    return out