# core/vectorizers.py
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix


def _build_concat_texts(df: pd.DataFrame, cols: List[str]) -> list[str]:
    # stringify + fill; join with space
    if not cols:
        # no columns selected -> empty strings per row
        return [""] * len(df)
    return df[cols].astype(str).fillna("").agg(" ".join, axis=1).tolist()


def _safe_fit_transform(vec: TfidfVectorizer, docs: list[str]):
    """
    Fit/transform and gracefully handle 'empty vocabulary' by returning a 0-column sparse matrix.
    """
    try:
        X = vec.fit_transform(docs)
        # If vectorizer survived but produced 0 features, normalize to 0-col matrix
        if X.shape[1] == 0:
            return csr_matrix((len(docs), 0), dtype=np.float32)
        return X
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            # Return a valid 0-col matrix so callers can hstack safely
            return csr_matrix((len(docs), 0), dtype=np.float32)
        raise


def build_vectorizers(df: pd.DataFrame, cols: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "row_concat_vec": np.ndarray (dense) [n_rows x dim],  # same shape as before
        "col_vecs": Dict[str, np.ndarray],                   # per-column dense arrays
        "dim": int,                                          # final row_concat_vec width
        "skipped_columns": List[str],                        # columns that yielded no features
      }
    """
    use_char = cfg.get("use_char", True)
    use_word = cfg.get("use_word", False)
    char_range = cfg.get("char_range", (3, 5))
    word_range = cfg.get("word_range", (1, 2))
    svd_components = int(cfg.get("svd_components", 0) or 0)

    analyzers = []
    if use_char:
        analyzers.append(("char", char_range))
    if use_word:
        analyzers.append(("word", word_range))
    if not analyzers:
        analyzers = [("char", (3, 5))]

    # -------- Row-concatenated TF-IDF --------
    texts = _build_concat_texts(df, cols)
    mats = []
    for analyzer, ngram in analyzers:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=1)
        X_part = _safe_fit_transform(vec, texts)
        if X_part.shape[1] > 0:
            mats.append(X_part)

    if mats:
        X_all = hstack(mats).tocsr().astype(np.float32)
    else:
        X_all = csr_matrix((len(df), 0), dtype=np.float32)

    # Optional SVD (dense output)
    if svd_components and X_all.shape[1] > 1:
        k = min(svd_components, X_all.shape[1] - 1)
        if k > 0:
            svd = TruncatedSVD(n_components=k, random_state=42)
            row_concat_vec = svd.fit_transform(X_all).astype(np.float32)
        else:
            row_concat_vec = X_all.toarray().astype(np.float32)
    else:
        # Preserve your previous interface (dense). If memory is tight, consider keeping this sparse.
        row_concat_vec = X_all.toarray().astype(np.float32)

    # -------- Per-column char TF-IDF (as before, but safe) --------
    col_vecs: Dict[str, np.ndarray] = {}
    skipped_columns: List[str] = []
    for c in cols:
        docs_c = df[c].astype(str).fillna("").tolist()
        v = TfidfVectorizer(analyzer="char", ngram_range=char_range, min_df=1)
        Xc = _safe_fit_transform(v, docs_c)
        if Xc.shape[1] > 0:
            col_vecs[c] = Xc.toarray().astype(np.float32)
        else:
            # skip silently; track for debugging
            skipped_columns.append(c)

    dim = int(row_concat_vec.shape[1]) if row_concat_vec.ndim == 2 else 0
    return {
        "row_concat_vec": row_concat_vec,
        "col_vecs": col_vecs,
        "dim": dim,
        "skipped_columns": skipped_columns,
    }