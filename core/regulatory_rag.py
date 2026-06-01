"""
Regulatory RAG Layer
====================
Retrieval-augmented regulatory citation for data quality findings.

Given a data quality finding (issue type, column, detail, and/or quality
dimension), this module retrieves the most relevant regulatory clause(s) from a
ChromaDB-backed knowledge base so that each finding can cite the exact
regulation it relates to (e.g. "BCBS 239 Principle 4 (Completeness)",
"UK GDPR Art. 5(1)(d) (Accuracy)").

Design notes
------------
- Embeddings use the SAME model as the rest of the app (all-MiniLM-L6-v2) via
  core.semantic_matcher.encode_texts, so the regulatory collection lives in the
  same embedding space as corpus / entity-resolution embeddings.
- Persistence reuses the project's ChromaDB store via core.vector_db_manager.
- The collection is created with cosine distance, and we supply precomputed
  query embeddings, so similarity = 1 - distance, in [0, 1].
- Each clause is embedded as "summary + keywords" (the requirement text plus
  data-quality keywords) for best recall against short engine findings.

The knowledge base itself lives in regulatory_kb/regulatory_clauses.json. The
summaries there are authored paraphrases for retrieval, not verbatim
reproductions of the official regulatory texts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Repo-root-relative defaults
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KB_PATH = _REPO_ROOT / "regulatory_kb" / "regulatory_clauses.json"
_DEFAULT_PERSIST_DIR = _REPO_ROOT / "chroma_db"

_COLLECTION_NAME = "regulatory_kb"
_EMBED_MODEL = "all-MiniLM-L6-v2"

# Map the engine's 6 quality dimensions to keywords that sharpen the query
# when only a dimension (not a free-text detail) is available.
_DIMENSION_HINTS = {
    "completeness": "missing value missing record incomplete null blank coverage",
    "uniqueness": "duplicate record duplicate entry single source of truth",
    "consistency": "inconsistent contradictory format mismatch cross-column reconciliation",
    "validity": "invalid value out of range format error not allowed value",
    "accuracy": "inaccurate incorrect value typo wrong data error",
    "timeliness": "stale outdated old date not up to date retention freshness",
}

# Map the engine's issue types (see ISSUE_DESCRIPTIONS in validator/validate.py)
# to the quality dimension they primarily affect. Used to sharpen retrieval and
# to filter to the most relevant clause.
ISSUE_DIMENSION = {
    # core/validator/validate.py vocabulary
    "missing": "completeness",
    "format_error": "validity",
    "not number": "validity",
    "<min": "validity",
    ">max": "validity",
    "invalid date": "timeliness",
    "not in allowed set": "accuracy",
    "regex mismatch": "validity",
    "duplicate_value": "uniqueness",
    "numeric_outlier": "accuracy",
    "rare_category": "accuracy",
    "text_length_outlier": "consistency",
    "likely_typo": "accuracy",
    "ml_anomaly": "accuracy",
    # dq_engine orchestrator vocabulary
    "missing_value": "completeness",
    "null_value": "completeness",
    "duplicate": "uniqueness",
    "near_duplicate": "uniqueness",
    "regex_mismatch": "validity",
    "type_error": "validity",
    "invalid_value": "validity",
    "out_of_range": "validity",
    "constraint_violation": "validity",
    "corpus_mismatch": "accuracy",
    "unknown_value": "accuracy",
    "referential_integrity": "accuracy",
    "inconsistency": "consistency",
    "cross_column": "consistency",
    "stale_date": "timeliness",
    "future_date": "timeliness",
    "invalid_date": "timeliness",
}


class RegulatoryRAG:
    """Retrieve regulatory clauses relevant to a data quality finding."""

    def __init__(
        self,
        kb_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = _COLLECTION_NAME,
        model_name: str = _EMBED_MODEL,
    ):
        self.kb_path = Path(kb_path) if kb_path else _DEFAULT_KB_PATH
        self.persist_directory = (
            Path(persist_directory) if persist_directory else _DEFAULT_PERSIST_DIR
        )
        self.collection_name = collection_name
        self.model_name = model_name
        self._clauses: Optional[List[Dict[str, Any]]] = None
        self._vdb = None  # lazy VectorDBManager

    # ------------------------------------------------------------------
    # Knowledge base loading
    # ------------------------------------------------------------------
    def _load_clauses(self) -> List[Dict[str, Any]]:
        if self._clauses is not None:
            return self._clauses
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Regulatory KB not found at {self.kb_path}")
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        clauses = data.get("clauses", [])
        if not clauses:
            raise ValueError(f"No clauses found in {self.kb_path}")
        self._clauses = clauses
        return clauses

    def _get_vdb(self):
        if self._vdb is None:
            from core.vector_db_manager import VectorDBManager

            self._vdb = VectorDBManager(persist_directory=str(self.persist_directory))
        return self._vdb

    def _embed_text(self, text: str) -> str:
        """Build the text that gets embedded for a clause: summary + keywords."""
        # not used directly for clauses (see ingest); kept for symmetry/testing
        return text

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, force: bool = False) -> Dict[str, Any]:
        """
        Embed all clauses and store them in the ChromaDB collection.

        Args:
            force: if True, drop and rebuild the collection.

        Returns:
            stats dict (count, collection, model).
        """
        from core.semantic_matcher import encode_texts

        clauses = self._load_clauses()
        vdb = self._get_vdb()

        # Create (cosine space) collection directly so we control the metric.
        if force:
            try:
                vdb.client.delete_collection(name=self.collection_name)
            except Exception:
                pass

        collection = vdb.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Regulatory clauses for DQ citation", "hnsw:space": "cosine"},
        )

        # Skip re-ingest if already populated and not forcing.
        if not force and collection.count() >= len(clauses):
            logger.info(
                "Regulatory KB already ingested (%d clauses); skipping. Use force=True to rebuild.",
                collection.count(),
            )
            return {
                "ingested": False,
                "count": collection.count(),
                "collection": self.collection_name,
                "model": self.model_name,
            }

        # Build embed text = summary + keywords for each clause.
        embed_texts: List[str] = []
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in clauses:
            keywords = c.get("keywords", [])
            kw_str = " ".join(keywords)
            embed_texts.append(f"{c['summary']} {kw_str}".strip())
            ids.append(c["id"])
            documents.append(c["summary"])
            dims = c.get("dimensions", [])
            meta: Dict[str, Any] = {
                "citation": c.get("citation", ""),
                "regulation": c.get("regulation", ""),
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
                "dimensions": ",".join(dims),
                "keywords": kw_str,
            }
            # Per-dimension boolean flags enable optional Chroma `where` filtering.
            for d in dims:
                meta[f"dim_{d}"] = True
            metadatas.append(meta)

        embeddings = encode_texts(embed_texts, model_name=self.model_name)

        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

        logger.info("Ingested %d regulatory clauses into '%s'", len(ids), self.collection_name)
        return {
            "ingested": True,
            "count": len(ids),
            "collection": self.collection_name,
            "model": self.model_name,
        }

    def _ensure_ingested(self) -> None:
        vdb = self._get_vdb()
        try:
            collection = vdb.client.get_collection(name=self.collection_name)
            if collection.count() > 0:
                return
        except Exception:
            pass
        self.ingest(force=False)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _build_query(
        self,
        issue: Optional[str],
        column: Optional[str],
        detail: Optional[str],
        dimension: Optional[str],
    ) -> str:
        parts: List[str] = []
        if issue:
            parts.append(str(issue).replace("_", " "))
        if column:
            parts.append(f"in field {column}")
        if detail:
            parts.append(str(detail))
        if dimension and dimension in _DIMENSION_HINTS:
            parts.append(_DIMENSION_HINTS[dimension])
        query = " ".join(parts).strip()
        return query or (dimension or "data quality issue")

    def retrieve(
        self,
        issue: Optional[str] = None,
        column: Optional[str] = None,
        detail: Optional[str] = None,
        dimension: Optional[str] = None,
        k: int = 3,
        min_similarity: float = 0.0,
        filter_by_dimension: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k regulatory clauses for a finding.

        Args:
            issue: engine issue type, e.g. "missing_value", "format_error".
            column: column name the finding relates to.
            detail: free-text description of the finding.
            dimension: one of the 6 quality dimensions; sharpens the query and,
                       if filter_by_dimension=True, restricts results to clauses
                       tagged with that dimension.
            k: number of clauses to return.
            min_similarity: drop results below this cosine similarity (0..1).
            filter_by_dimension: hard-filter to the given dimension via metadata.

        Returns:
            List of dicts: {citation, regulation, title, summary, similarity,
                            dimensions, clause_id}, best first.
        """
        from core.semantic_matcher import encode_texts

        self._ensure_ingested()
        vdb = self._get_vdb()

        query = self._build_query(issue, column, detail, dimension)
        q_emb = encode_texts([query], model_name=self.model_name)

        where = None
        if filter_by_dimension and dimension:
            where = {f"dim_{dimension}": True}

        # Over-fetch a little so min_similarity filtering still returns ~k.
        n_fetch = max(k * 2, k)
        out = vdb.query_collection(
            collection_name=self.collection_name,
            query_embeddings=q_emb,
            n_results=n_fetch,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        res = out["results"]

        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        results: List[Dict[str, Any]] = []
        for meta, dist, cid in zip(metadatas, distances, ids):
            # cosine space + normalized query: similarity = 1 - distance
            similarity = round(float(1.0 - dist), 4)
            if similarity < min_similarity:
                continue
            results.append(
                {
                    "clause_id": cid,
                    "citation": meta.get("citation", ""),
                    "regulation": meta.get("regulation", ""),
                    "title": meta.get("title", ""),
                    "summary": meta.get("summary", ""),
                    "dimensions": [d for d in meta.get("dimensions", "").split(",") if d],
                    "similarity": similarity,
                }
            )
            if len(results) >= k:
                break

        return results

    def cite_finding(self, finding: Dict[str, Any], k: int = 1) -> List[Dict[str, Any]]:
        """
        Convenience wrapper that accepts a finding dict from the engine.

        Expects keys among: issue, column, detail, dimension.
        """
        return self.retrieve(
            issue=finding.get("issue"),
            column=finding.get("column"),
            detail=finding.get("detail"),
            dimension=finding.get("dimension"),
            k=k,
        )


    def annotate_report(
        self,
        report,
        min_similarity: float = 0.15,
    ):
        """
        Attach regulatory citations to an engine issues DataFrame.

        Adds three columns:
            - regulatory_citation: e.g. "BCBS 239 Principle 4 (Completeness)"
            - regulation:          e.g. "BCBS 239"
            - citation_similarity: cosine similarity of the match (0..1)

        Retrieval is cached per distinct issue type, so cost scales with the
        number of issue *types* (a handful), not the number of rows.

        Args:
            report: pandas DataFrame with at least an 'issue' (or 'issue_type')
                    column. A 'detail' column is used if present.
            min_similarity: matches below this are left blank rather than
                            attaching a weak/irrelevant citation.

        Returns:
            The same DataFrame with the three citation columns added.
        """
        import pandas as pd  # local import to keep module import light

        if report is None or len(report) == 0:
            if report is not None:
                report["regulatory_citation"] = pd.Series(dtype="object")
                report["regulation"] = pd.Series(dtype="object")
                report["citation_similarity"] = pd.Series(dtype="float")
            return report

        # Pick the issue-type column that actually carries data. Some pipelines
        # create an empty 'issue_type' column for schema consistency while the
        # real type lives in 'issue', so prefer whichever is populated.
        issue_col = None
        for cand in ("issue_type", "issue", "rule"):
            if cand in report.columns and report[cand].notna().any():
                issue_col = cand
                break
        if issue_col is None:
            report["regulatory_citation"] = None
            report["regulation"] = None
            report["citation_similarity"] = None
            return report

        self._ensure_ingested()

        # Build a cache keyed by distinct issue type.
        cache: Dict[str, Dict[str, Any]] = {}
        for issue_type in report[issue_col].dropna().astype(str).unique():
            dimension = ISSUE_DIMENSION.get(issue_type.strip())
            hits = self.retrieve(
                issue=issue_type,
                dimension=dimension,
                k=1,
                min_similarity=min_similarity,
            )
            if hits:
                top = hits[0]
                cache[issue_type] = {
                    "regulatory_citation": top["citation"],
                    "regulation": top["regulation"],
                    "citation_similarity": top["similarity"],
                }
            else:
                cache[issue_type] = {
                    "regulatory_citation": None,
                    "regulation": None,
                    "citation_similarity": None,
                }

        report["regulatory_citation"] = report[issue_col].map(
            lambda x: cache.get(str(x).strip(), {}).get("regulatory_citation")
        )
        report["regulation"] = report[issue_col].map(
            lambda x: cache.get(str(x).strip(), {}).get("regulation")
        )
        report["citation_similarity"] = report[issue_col].map(
            lambda x: cache.get(str(x).strip(), {}).get("citation_similarity")
        )
        return report


# Module-level singleton for convenient reuse across the app.
_default_rag: Optional[RegulatoryRAG] = None


def get_regulatory_rag() -> RegulatoryRAG:
    global _default_rag
    if _default_rag is None:
        _default_rag = RegulatoryRAG()
    return _default_rag


if __name__ == "__main__":
    # Simple CLI: build the index and run a couple of sample queries.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rag = RegulatoryRAG()
    stats = rag.ingest(force=True)
    print("Ingest:", stats)

    for q in [
        {"issue": "missing_value", "column": "customer_id", "dimension": "completeness"},
        {"issue": "invalid_format", "column": "email", "dimension": "validity"},
        {"issue": "stale_date", "column": "last_review_date", "dimension": "timeliness"},
        {"issue": "typo", "column": "first_name", "dimension": "accuracy"},
    ]:
        print(f"\nQuery: {q}")
        for r in rag.retrieve(**q, k=2):
            print(f"  [{r['similarity']:.3f}] {r['citation']} - {r['title']}")
