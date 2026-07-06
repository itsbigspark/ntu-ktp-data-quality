"""
Tests for the agent node (per-batch triage/routing/escalation/narration).

The routing decisions are deterministic, so they are asserted exactly. The
integration test confirms run_batch attaches a verdict + narrative to every batch.

Run:  ./.venv/bin/python -m pytest tests/test_agent.py -v
"""
from __future__ import annotations

import warnings
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from dataqualify.agent import triage, AgentDecision
from dataqualify.batch import BatchRecord, run_batch, SQLiteBatchStore
from dataqualify.sources import parse_source


def test_accept_clean_batch():
    d = triage(rows=100, overall_score=96.0, passed=True, issues_count=0)
    assert d.verdict == "accept" and d.severity == "ok"
    assert "pass downstream" in d.recommended_actions


def test_review_when_issues_present_but_passing():
    d = triage(rows=100, overall_score=90.0, passed=True, issues_count=5)
    assert d.verdict == "review" and d.severity == "warn"


def test_quarantine_when_below_threshold():
    d = triage(rows=100, overall_score=70.0, passed=False, issues_count=40)
    assert d.verdict == "quarantine" and d.severity == "critical"
    assert any("below the pass threshold" in r for r in d.reasons)


def test_quarantine_on_high_severity_rate():
    issues = pd.DataFrame({"severity": ["high"] * 10, "issue": ["missing"] * 10})
    d = triage(rows=100, overall_score=90.0, passed=True, issues_count=10, issues_df=issues)
    assert d.verdict == "quarantine"
    assert any("high-severity" in r for r in d.reasons)


def test_escalation_on_score_drop():
    history = [BatchRecord(batch_id=f"b{i}", source="s", timestamp="t",
                           status="completed", overall_score=95.0, passed=True)
               for i in range(5)]
    d = triage(rows=100, overall_score=80.0, passed=False, issues_count=30, history=history)
    assert any("dropped" in e for e in d.escalations)


def test_narrative_is_readable():
    issues = pd.DataFrame({"severity": ["low", "low"], "issue": ["missing", "likely_typo"]})
    d = triage(rows=50, overall_score=91.0, passed=True, issues_count=2, issues_df=issues)
    assert "Verdict: REVIEW" in d.narrative
    assert "Recommended:" in d.narrative


def test_run_batch_attaches_verdict_and_narrative(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"status": ["Active", "Dissolved"] * 30, "n": range(60)}).to_csv(src, index=False)
    store = SQLiteBatchStore(str(tmp_path / "b.db"))
    rec = run_batch(parse_source(str(src)), store=store)
    assert rec.verdict in {"accept", "review", "quarantine"}
    assert rec.severity in {"ok", "warn", "critical"}
    assert rec.narrative
    # persisted with the batch
    assert store.get(rec.batch_id).verdict == rec.verdict


def test_agent_can_be_disabled(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(src, index=False)
    store = SQLiteBatchStore(str(tmp_path / "b.db"))
    rec = run_batch(parse_source(str(src)), store=store, agent=False)
    assert rec.verdict is None and rec.narrative is None
