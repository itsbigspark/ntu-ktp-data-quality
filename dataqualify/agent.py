"""
Agent node — the per-batch brain of the autonomous pipeline.

After the engine validates a batch, the agent decides what to do with it:

  - triage:   classify the outcome (ok / warn / critical)
  - route:    accept downstream, route to human review, or quarantine
  - escalate: flag trends across recent batches (score drops, repeated failures)
  - narrate:  a plain-English summary of what was found and the recommendation

Design principle: the routing *decisions* are deterministic and auditable —
they are computed from the engine's scores and issue severities, not by an LLM.
The language model is used only to phrase the narrative (optional; a deterministic
template is the default). This keeps the "what happens to your data" decision
reproducible and defensible, while still giving you an intelligent, readable
handling of every batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AgentDecision:
    verdict: str                      # "accept" | "review" | "quarantine"
    severity: str                     # "ok" | "warn" | "critical"
    reasons: List[str] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "severity": self.severity,
            "reasons": self.reasons,
            "escalations": self.escalations,
            "recommended_actions": self.recommended_actions,
            "narrative": self.narrative,
        }


_SEVERITY_BY_VERDICT = {"quarantine": "critical", "review": "warn", "accept": "ok"}
_ACTIONS = {
    "accept": ["pass downstream"],
    "review": ["route to analyst review", "hold from downstream until reviewed"],
    "quarantine": ["quarantine batch", "block downstream", "alert the data owner"],
}


def _severity_counts(issues_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    if issues_df is None or len(issues_df) == 0 or "severity" not in getattr(issues_df, "columns", []):
        return {}
    return {str(k): int(v) for k, v in issues_df["severity"].value_counts().items()}


def _top_issue_types(issues_df: Optional[pd.DataFrame], n: int = 3) -> List[tuple]:
    if issues_df is None or len(issues_df) == 0 or "issue" not in getattr(issues_df, "columns", []):
        return []
    return [(str(k), int(v)) for k, v in issues_df["issue"].value_counts().head(n).items()]


def triage(
    rows: int,
    overall_score: Optional[float],
    passed: Optional[bool],
    issues_count: int,
    issues_df: Optional[pd.DataFrame] = None,
    history: Optional[List[Any]] = None,
    pass_threshold: float = 85.0,
    use_llm: bool = False,
) -> AgentDecision:
    """Decide the verdict, routing, escalations, and narrative for one batch.

    ``history`` is a list of prior BatchRecord-like objects (with ``.overall_score``,
    ``.passed``, ``.status``) used for trend escalation.
    """
    rows = max(int(rows or 0), 1)
    score = float(overall_score) if overall_score is not None else 0.0
    high = _severity_counts(issues_df).get("high", 0)

    # ── Routing decision (deterministic) ──────────────────────────────────
    reasons: List[str] = []
    if passed is False or score < pass_threshold:
        verdict = "quarantine"
        reasons.append(f"overall score {score:.1f} is below the pass threshold {pass_threshold:.0f}")
    elif high > 0 and (high / rows) > 0.05:
        verdict = "quarantine"
        reasons.append(f"{high} high-severity issues ({high / rows:.0%} of rows)")
    elif issues_count > 0:
        verdict = "review"
        reasons.append(f"{issues_count} issue(s) found; above the auto-accept bar")
    else:
        verdict = "accept"
        reasons.append("no issues found; meets the quality threshold")

    severity = _SEVERITY_BY_VERDICT[verdict]

    # ── Escalation (trend across recent batches) ──────────────────────────
    escalations: List[str] = []
    completed = [h for h in (history or []) if getattr(h, "status", None) == "completed"
                 and getattr(h, "overall_score", None) is not None]
    if completed:
        recent_avg = mean(h.overall_score for h in completed[:10])
        if score and score < recent_avg - 10:
            escalations.append(
                f"score dropped {recent_avg - score:.0f} points vs the recent average ({recent_avg:.0f})")
    recent_bad = [h for h in (history or [])[:3]
                  if getattr(h, "passed", None) is False or getattr(h, "status", None) == "failed"]
    if verdict == "quarantine" and len(recent_bad) >= 2:
        escalations.append("repeated failing batches from this pipeline — investigate the source")

    decision = AgentDecision(
        verdict=verdict,
        severity=severity,
        reasons=reasons,
        escalations=escalations,
        recommended_actions=list(_ACTIONS[verdict]),
    )
    decision.narrative = narrate(rows, score, issues_count, decision, issues_df, use_llm=use_llm)
    return decision


def narrate(
    rows: int,
    score: float,
    issues_count: int,
    decision: AgentDecision,
    issues_df: Optional[pd.DataFrame] = None,
    use_llm: bool = False,
) -> str:
    """Plain-English summary. Deterministic template by default; optional LLM phrasing."""
    tops = _top_issue_types(issues_df)
    top_str = ", ".join(f"{c} {t.replace('_', ' ')}" for t, c in tops) if tops else "none"
    base = (
        f"Batch of {rows} rows scored {score:.1f}. "
        f"Verdict: {decision.verdict.upper()} ({decision.severity}). "
        f"{issues_count} issue(s); top types: {top_str}. "
        f"Recommended: {'; '.join(decision.recommended_actions)}."
    )
    if decision.escalations:
        base += " Escalations: " + "; ".join(decision.escalations) + "."

    if not use_llm:
        return base

    # Optional: let a local/hosted LLM rephrase into a business narrative.
    try:  # pragma: no cover - depends on an available provider
        from core.local_llm import summarise  # type: ignore
        return summarise(base) or base
    except Exception:
        return base
