"""
Headless batch runner with tracking.

``run_batch`` is the spine of the autonomous vision: it pulls a table from any
:class:`~dataqualify.sources.SourceConnector`, runs the data-quality pipeline,
records a tracked batch (source, timestamp, score, pass/fail, timing), and
optionally writes the results to a sink directory. It has no UI dependency, so
the same function backs the CLI, the autonomous cloud pipeline, and (later) the
agent node.

    from dataqualify.sources import parse_source
    from dataqualify.batch import run_batch

    record = run_batch(parse_source("s3://bucket/incoming/data.csv"),
                       sink_dir="./out")
    print(record.overall_score, record.status)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .sources import SourceConnector


# ---------------------------------------------------------------------------
# Batch record
# ---------------------------------------------------------------------------
@dataclass
class BatchRecord:
    batch_id: str
    source: str
    timestamp: str
    status: str  # "completed" | "failed"
    rows: int = 0
    columns: int = 0
    overall_score: Optional[float] = None
    passed: Optional[bool] = None
    issues_count: Optional[int] = None
    duration_s: float = 0.0
    sink: Optional[str] = None
    error: Optional[str] = None
    # Agent-node decision (per-batch triage/routing/narration).
    verdict: Optional[str] = None       # "accept" | "review" | "quarantine"
    severity: Optional[str] = None      # "ok" | "warn" | "critical"
    narrative: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Batch store (tracking)
# ---------------------------------------------------------------------------
_DEFAULT_DB = os.environ.get(
    "DATAQUALIFY_BATCH_DB",
    os.path.join(os.path.expanduser("~"), ".dataqualify", "batches.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id      TEXT PRIMARY KEY,
    source        TEXT,
    timestamp     TEXT,
    status        TEXT,
    rows          INTEGER,
    columns       INTEGER,
    overall_score REAL,
    passed        INTEGER,
    issues_count  INTEGER,
    duration_s    REAL,
    sink          TEXT,
    error         TEXT,
    verdict       TEXT,
    severity      TEXT,
    narrative     TEXT
);
"""

# Columns added after the first release; migrated onto existing databases.
_MIGRATIONS = [("verdict", "TEXT"), ("severity", "TEXT"), ("narrative", "TEXT")]


class SQLiteBatchStore:
    """A simple, dependency-free batch tracker backed by SQLite."""

    def __init__(self, path: str = _DEFAULT_DB):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)
            existing = {r["name"] for r in c.execute("PRAGMA table_info(batches)").fetchall()}
            for col, coltype in _MIGRATIONS:
                if col not in existing:
                    c.execute(f"ALTER TABLE batches ADD COLUMN {col} {coltype}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, rec: BatchRecord) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO batches
                   (batch_id, source, timestamp, status, rows, columns,
                    overall_score, passed, issues_count, duration_s, sink, error,
                    verdict, severity, narrative)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.batch_id, rec.source, rec.timestamp, rec.status, rec.rows,
                 rec.columns, rec.overall_score,
                 None if rec.passed is None else int(rec.passed),
                 rec.issues_count, rec.duration_s, rec.sink, rec.error,
                 rec.verdict, rec.severity, rec.narrative),
            )

    def list(self, limit: int = 50) -> List[BatchRecord]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM batches ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("passed") is not None:
                d["passed"] = bool(d["passed"])
            out.append(BatchRecord(**d))
        return out

    def get(self, batch_id: str) -> Optional[BatchRecord]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        if d.get("passed") is not None:
            d["passed"] = bool(d["passed"])
        return BatchRecord(**d)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
def _new_batch_id() -> str:
    return "batch_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _write_sink(sink_dir: str, batch_id: str, result: Dict[str, Any]) -> str:
    os.makedirs(sink_dir, exist_ok=True)
    issues = result.get("issues")
    if isinstance(issues, pd.DataFrame):
        issues.to_csv(os.path.join(sink_dir, f"{batch_id}_issues.csv"), index=False)
    corrected = result.get("corrected_df")
    if isinstance(corrected, pd.DataFrame):
        corrected.to_csv(os.path.join(sink_dir, f"{batch_id}_corrected.csv"), index=False)
    report = {k: result.get(k) for k in
              ("batch_id", "timestamp", "rows_processed", "quality_scores",
               "overall_score", "pass", "issues_count")}
    with open(os.path.join(sink_dir, f"{batch_id}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return sink_dir


def run_batch(
    source: SourceConnector,
    config: Optional[Dict[str, Any]] = None,
    reference: Optional[pd.DataFrame] = None,
    rules: Optional[Dict[str, Any]] = None,
    sink_dir: Optional[str] = None,
    store: Optional[SQLiteBatchStore] = None,
    agent: bool = True,
) -> BatchRecord:
    """Fetch from ``source``, run the pipeline, triage, and record a tracked batch.

    When ``agent`` is True (default) the per-batch agent node triages the result
    (verdict / severity / narrative / escalations) after validation. Never raises
    on a data/pipeline error: failures are captured and recorded with
    ``status="failed"`` (and quarantined) so every attempt is tracked.
    """
    from core.engine import run_pipeline

    if store is None:
        store = SQLiteBatchStore()
    if config is None:
        try:
            from core.engine import load_config
            config = load_config()
        except Exception:
            config = {}

    batch_id = _new_batch_id()
    started = time.time()
    rec = BatchRecord(
        batch_id=batch_id, source=getattr(source, "name", str(source)),
        timestamp=datetime.now(timezone.utc).isoformat(), status="failed",
    )

    try:
        df = source.fetch()
        rec.rows, rec.columns = int(df.shape[0]), int(df.shape[1])

        result = run_pipeline(df, config, rules=rules, df_ref=reference, batch_id=batch_id)
        rec.overall_score = float(result.get("overall_score")) if result.get("overall_score") is not None else None
        rec.passed = bool(result.get("pass")) if result.get("pass") is not None else None
        rec.issues_count = int(result.get("issues_count", 0))
        rec.status = "completed"

        if agent:
            from .agent import triage
            decision = triage(
                rows=rec.rows, overall_score=rec.overall_score, passed=rec.passed,
                issues_count=rec.issues_count or 0, issues_df=result.get("issues"),
                history=store.list(limit=10),
            )
            rec.verdict, rec.severity, rec.narrative = (
                decision.verdict, decision.severity, decision.narrative)

        if sink_dir:
            rec.sink = _write_sink(sink_dir, batch_id, result)
    except Exception as exc:  # noqa: BLE001 - we deliberately record all failures
        rec.status = "failed"
        rec.error = f"{type(exc).__name__}: {exc}"
        rec._traceback = traceback.format_exc()  # type: ignore[attr-defined]
        if agent:
            rec.verdict, rec.severity = "quarantine", "critical"
            rec.narrative = f"Batch could not be processed and was quarantined: {rec.error}"
    finally:
        rec.duration_s = round(time.time() - started, 3)
        store.record(rec)

    return rec
