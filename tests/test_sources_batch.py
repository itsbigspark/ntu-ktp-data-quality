"""
Tests for the source-connector layer and the headless batch runner.

Covers: source spec routing, file/HTTP fetching, the SQLite batch tracker, and
run_batch end-to-end (success and recorded-failure paths).

Run:  ./.venv/bin/python -m pytest tests/test_sources_batch.py -v
"""
from __future__ import annotations

import warnings
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from dataqualify.sources import (
    parse_source, FileSource, S3Source, HttpApiSource, CompaniesHouseSource,
)
from dataqualify.batch import run_batch, SQLiteBatchStore, BatchRecord


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec,cls", [
    ("data.csv", FileSource),
    ("/abs/path/x.parquet", FileSource),
    ("s3://bucket/key.csv", S3Source),
    ("https://api.example.com/records", HttpApiSource),
    ("http://api.example.com/records", HttpApiSource),
    ("companies-house:12345678", CompaniesHouseSource),
])
def test_parse_source_routing(spec, cls):
    assert isinstance(parse_source(spec), cls)


def test_s3_source_parses_bucket_and_key():
    s = S3Source("s3://my-bucket/incoming/2026/data.csv")
    assert s.bucket == "my-bucket"
    assert s.key == "incoming/2026/data.csv"


def test_companies_house_numbers_parsed():
    s = parse_source("companies-house:12345678, SC123456")
    assert s.numbers == ["12345678", "SC123456"]


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------
def test_file_source_reads_csv(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)
    df = FileSource(str(p)).fetch()
    assert list(df.columns) == ["a", "b"] and len(df) == 2


# ---------------------------------------------------------------------------
# HTTP API source (normalisation logic, without a real network call)
# ---------------------------------------------------------------------------
def test_http_api_normalises_record_path(monkeypatch):
    src = HttpApiSource("https://x/api", record_path="data.items")
    monkeypatch.setattr(src, "_get_json",
                        lambda: {"data": {"items": [{"id": 1}, {"id": 2}]}})
    df = src.fetch()
    assert list(df["id"]) == [1, 2]


def test_http_api_single_object_becomes_one_row(monkeypatch):
    src = HttpApiSource("https://x/api")
    monkeypatch.setattr(src, "_get_json", lambda: {"company_name": "ACME", "status": "active"})
    df = src.fetch()
    assert len(df) == 1 and df.iloc[0]["company_name"] == "ACME"


# ---------------------------------------------------------------------------
# Batch store (tracking)
# ---------------------------------------------------------------------------
def test_batch_store_record_list_get(tmp_path):
    store = SQLiteBatchStore(str(tmp_path / "b.db"))
    rec = BatchRecord(batch_id="b1", source="file://x", timestamp="t",
                      status="completed", rows=10, columns=3,
                      overall_score=91.2, passed=True, issues_count=4)
    store.record(rec)
    assert store.get("b1").overall_score == 91.2
    assert store.get("b1").passed is True
    listed = store.list()
    assert len(listed) == 1 and listed[0].batch_id == "b1"


# ---------------------------------------------------------------------------
# run_batch end-to-end
# ---------------------------------------------------------------------------
def test_run_batch_completes_and_tracks(tmp_path):
    src_path = tmp_path / "src.csv"
    pd.DataFrame({
        "company_status": ["Active"] * 20 + ["Dissolved"] * 20,
        "amount": [str(round(v, 2)) for v in range(40)],
    }).to_csv(src_path, index=False)

    store = SQLiteBatchStore(str(tmp_path / "b.db"))
    rec = run_batch(parse_source(str(src_path)), sink_dir=str(tmp_path / "out"), store=store)

    assert rec.status == "completed"
    assert rec.rows == 40 and rec.columns == 2
    assert rec.overall_score is not None
    # tracked
    assert store.get(rec.batch_id) is not None
    # sink written
    assert (tmp_path / "out" / f"{rec.batch_id}_report.json").exists()
    assert (tmp_path / "out" / f"{rec.batch_id}_issues.csv").exists()


def test_run_batch_records_failure(tmp_path):
    """A source that cannot be fetched is recorded as a failed batch, not raised."""
    store = SQLiteBatchStore(str(tmp_path / "b.db"))
    rec = run_batch(FileSource(str(tmp_path / "does_not_exist.csv")), store=store)
    assert rec.status == "failed"
    assert rec.error is not None
    assert store.get(rec.batch_id).status == "failed"
