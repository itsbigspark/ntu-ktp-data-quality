from __future__ import annotations
from typing import Optional, Iterable
try:
    import jellyfish
except Exception:
    jellyfish = None

def suggest_fix(value: str, column: str, corpus: Optional[Iterable[str]] = None) -> Optional[str]:
    if value is None or corpus is None:
        return None
    s = str(value)
    best = None; best_sim = -1.0
    for cand in corpus:
        cand = str(cand)
        if jellyfish is not None:
            try:
                sim = jellyfish.jaro_winkler_similarity(s, cand)
            except Exception:
                sim = jellyfish.jaro_winkler(s, cand)
        else:
            sa = set(s.lower().split()); ca = set(cand.lower().split())
            sim = len(sa & ca) / max(1, len(sa | ca))
        if sim > best_sim:
            best_sim = sim; best = cand
    if best_sim >= 0.90:
        return best
    return None
