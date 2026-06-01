from __future__ import annotations
from typing import Optional, Iterable
import re

try:
    import jellyfish
except Exception:
    jellyfish = None


def _normalize(x: str) -> str:
    """Casefold and strip non-alphanumerics so 'nat west', 'NatWest' and
    'NAT-WEST' all compare as equal. Used only for the *similarity* test;
    the original canonical string is what gets returned as the fix."""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _similarity(a: str, b: str) -> float:
    if jellyfish is not None:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            try:
                return jellyfish.jaro_winkler(a, b)
            except Exception:
                pass
    # Token-overlap fallback when jellyfish is unavailable
    sa, ca = set(a), set(b)
    return len(sa & ca) / max(1, len(sa | ca))


def suggest_fix(
    value: str,
    column: str,
    corpus: Optional[Iterable[str]] = None,
    threshold: float = 0.90,
) -> Optional[str]:
    """
    Suggest a canonical replacement for `value` by fuzzy-matching against
    `corpus`, comparing on a normalized (casefold + punctuation-stripped) form
    so that case and spacing differences do not sink an otherwise-clean match.

    Returns the best canonical candidate (in its original form) whose normalized
    similarity to `value` is >= threshold, or None.

    A candidate identical to the input value is skipped, so a typo that also
    appears in the corpus does not "suggest itself"; the loop still finds the
    correctly-cased canonical if one exists.
    """
    if value is None or corpus is None:
        return None

    s = str(value)
    s_norm = _normalize(s)
    if not s_norm:
        return None

    best = None
    best_sim = -1.0
    for cand in corpus:
        cand = str(cand)
        if cand == s:
            continue  # don't suggest the value as a fix for itself
        sim = _similarity(s_norm, _normalize(cand))
        if sim > best_sim:
            best_sim = sim
            best = cand

    if best is not None and best_sim >= threshold and best != s:
        return best
    return None


# Common email providers for domain-typo correction (extendable).
_EMAIL_DOMAINS = [
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "hotmail.co.uk",
    "live.com", "live.co.uk", "msn.com", "yahoo.com", "yahoo.co.uk", "ymail.com",
    "icloud.com", "me.com", "mac.com", "aol.com", "protonmail.com", "proton.me",
    "btinternet.com", "sky.com", "talktalk.net", "virginmedia.com", "ntlworld.com",
]


def suggest_email_fix(value: str, threshold: float = 0.82) -> Optional[str]:
    """
    Correct a likely-mistyped email domain (e.g. 'a@outlok.co' -> 'a@outlook.com',
    'b@hotmial.com' -> 'b@hotmail.com') by fuzzy-matching the domain part against
    a list of common providers. Returns the corrected address or None.

    Only the domain is altered; the local part is preserved exactly. The match
    must beat `threshold` and differ from the original domain.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s.count("@") != 1:
        return None
    local, domain = s.split("@", 1)
    if not local or not domain:
        return None

    d_norm = _normalize(domain)
    best, best_sim = None, -1.0
    for cand in _EMAIL_DOMAINS:
        sim = _similarity(d_norm, _normalize(cand))
        if sim > best_sim:
            best_sim, best = sim, cand

    if best is not None and best_sim >= threshold and best.lower() != domain.lower():
        return f"{local}@{best}"
    return None
