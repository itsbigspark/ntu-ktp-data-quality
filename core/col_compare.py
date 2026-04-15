# core/col_compare.py

import math
import numpy as np
import pandas as pd
from typing import Dict, Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Jaro fallback (rapidfuzz is fast; if not installed, we just skip)
try:
    from rapidfuzz.distance import JaroWinkler
    _HAS_JARO = True
except Exception:
    _HAS_JARO = False


def _norm(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip()
    # simple normalize; leave case handling to vectorizer (lowercase=True)
    return s


def _cosine_text_safe(a: str, b: str) -> float:
    """Robust text similarity with fallbacks:
       1) char n-grams (3–5)
       2) word n-grams (1–2)
       3) Jaro-Winkler (if available)
       4) 0.0
    """
    # If both empty after normalization, treat as no signal
    if (a is None or a == "") and (b is None or b == ""):
        return 0.0

    # Try char n-grams first (doesn't suffer from stop-words)
    for analyzer, ngram_range, token_pattern in [
        ("char", (3, 5), None),                  # best default
        ("word", (1, 2), r"(?u)\b\w+\b"),        # word fallback
    ]:
        try:
            vec = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                lowercase=True,
                # don't set stop_words for char analyzer; for word analyzer keep a permissive token_pattern
                token_pattern=token_pattern,
            )
            X = vec.fit_transform([a, b])
            if X.shape[1] == 0:
                # empty vocab even after this analyzer -> try next
                continue
            return float(cosine_similarity(X[0], X[1])[0, 0])
        except ValueError:
            # empty vocab / all stop-words -> try next
            continue
        except Exception:
            # any other unexpected issue, try next fallback
            continue

    # Fallback to Jaro-Winkler if available (scaled to 0..1)
    if _HAS_JARO:
        try:
            return float(JaroWinkler.normalized_similarity(a or "", b or "")) / 100.0
        except Exception:
            pass

    return 0.0


def per_column_similarity(df: pd.DataFrame, i: int, j: int, columns: Iterable[str]) -> Dict[str, float]:
    """Return per-column similarity scores between row i and j."""
    scores: Dict[str, float] = {}
    for c in columns:
        try:
            a = _norm(df.iloc[i][c])
            b = _norm(df.iloc[j][c])

            # If both values are empty after normalization, skip or set 0.0
            if (not a) and (not b):
                scores[c] = 0.0
                continue

            scores[c] = _cosine_text_safe(a, b)
        except Exception:
            # Never crash the diagnostic; assign 0 on failure
            scores[c] = 0.0
    return scores


def aggregate(col_scores: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """Weighted mean of per-column scores (weights optional)."""
    if not col_scores:
        return 0.0
    if not weights:
        return float(np.mean(list(col_scores.values())))
    # normalize weights over the intersection
    keys = [k for k in col_scores.keys() if k in weights and weights[k] > 0]
    if not keys:
        return float(np.mean(list(col_scores.values())))
    w = np.array([weights[k] for k in keys], dtype=float)
    w = w / w.sum()
    v = np.array([col_scores[k] for k in keys], dtype=float)
    return float((w * v).sum())