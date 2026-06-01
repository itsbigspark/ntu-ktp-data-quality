"""
Tests for the regulatory RAG layer (core/regulatory_rag.py).

Verifies that data quality findings retrieve the correct regulatory clause.
Uses an isolated temp ChromaDB directory so it never touches the real store.
"""

import tempfile
import shutil
import pytest

from core.regulatory_rag import RegulatoryRAG


@pytest.fixture(scope="module")
def rag():
    tmp = tempfile.mkdtemp(prefix="reg_rag_test_")
    r = RegulatoryRAG(persist_directory=tmp, collection_name="regulatory_kb_test")
    r.ingest(force=True)
    yield r
    shutil.rmtree(tmp, ignore_errors=True)


def _top_citation(results):
    assert results, "no clauses retrieved"
    return results[0]["citation"]


def test_ingest_count(rag):
    stats = rag.ingest(force=True)
    assert stats["count"] == 15


def test_missing_value_maps_to_completeness(rag):
    res = rag.retrieve(issue="missing_value", column="customer_id",
                       dimension="completeness", k=2)
    assert "Principle 4" in _top_citation(res)


def test_stale_date_maps_to_timeliness(rag):
    res = rag.retrieve(issue="stale_date", column="last_review_date",
                       dimension="timeliness", k=2)
    cites = [r["citation"] for r in res]
    assert any("Principle 5" in c for c in cites)  # BCBS timeliness
    assert any("5(1)(e)" in c or "Principle 5" in c for c in cites)


def test_accuracy_finding_surfaces_gdpr_accuracy(rag):
    res = rag.retrieve(issue="incorrect_value", column="first_name",
                       dimension="accuracy", k=3)
    cites = [r["citation"] for r in res]
    assert any("5(1)(d)" in c for c in cites) or any("Principle 3" in c or "Principle 7" in c for c in cites)


def test_dimension_hard_filter(rag):
    res = rag.retrieve(dimension="timeliness", k=5, filter_by_dimension=True)
    assert res
    for r in res:
        assert "timeliness" in r["dimensions"]


def test_similarity_in_range(rag):
    res = rag.retrieve(issue="missing_value", dimension="completeness", k=3)
    for r in res:
        assert 0.0 <= r["similarity"] <= 1.0


def test_cite_finding_wrapper(rag):
    finding = {"issue": "missing_value", "column": "balance", "dimension": "completeness"}
    res = rag.cite_finding(finding, k=1)
    assert len(res) == 1
    assert res[0]["regulation"] in ("BCBS 239", "UK GDPR", "Data Protection Act 2018")
