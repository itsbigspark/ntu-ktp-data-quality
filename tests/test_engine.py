"""
Tests for dq_engine -- the core validation engine.

Run: pytest tests/test_engine.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """A small DataFrame with deliberate data quality issues."""
    return pd.DataFrame({
        "customer_id": ["CUST-00001", "CUST-00002", "CUST-00003", "CUST-00004", "BAD_ID", "CUST-00006"],
        "email": ["alice@bank.com", "bob@bank.com", "not-an-email", "carol@bank.com", "dave@bank.com", "eve@bank.com"],
        "amount": [100.50, 250.00, -10.0, 999999.99, 50.0, None],
        "city": ["London", "Manchester", "Manchster", "Birmingham", "London", "Leeds"],
        "status": ["active", "active", "closed", "active", "frozen", "active"],
        "date_of_birth": ["1990-01-15", "1985-06-20", "1870-01-01", "1995-12-30", "2010-03-10", "1988-07-22"],
    })


@pytest.fixture
def reference_df():
    """Clean reference data."""
    return pd.DataFrame({
        "customer_id": ["CUST-00001", "CUST-00002", "CUST-00003"],
        "email": ["alice@bank.com", "bob@bank.com", "carol@bank.com"],
        "amount": [100.50, 250.00, 300.00],
        "city": ["London", "Manchester", "Birmingham"],
        "status": ["active", "active", "active"],
    })


# ---------------------------------------------------------------------------
# RulesWorkflow Tests
# ---------------------------------------------------------------------------
class TestRulesWorkflow:
    def test_import(self):
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        assert wf is not None

    def test_generate_from_data(self, sample_df):
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        rules = wf.generate(df_raw=sample_df, use_inferred=True)
        assert rules is not None
        assert isinstance(rules, dict)

    def test_generate_from_reference(self, sample_df, reference_df):
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        rules = wf.generate(df_raw=sample_df, df_ref=reference_df, use_inferred=True, use_reference=True)
        assert rules is not None

    def test_to_json(self, sample_df):
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        rules = wf.generate(df_raw=sample_df, use_inferred=True)
        json_str = wf.to_json(rules)
        assert isinstance(json_str, str)
        assert len(json_str) > 10

    def test_from_json_roundtrip(self, sample_df):
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        rules = wf.generate(df_raw=sample_df, use_inferred=True)
        json_str = wf.to_json(rules)
        loaded = wf.from_json(json_str)
        assert loaded is not None


# ---------------------------------------------------------------------------
# ValidationOrchestrator Tests
# ---------------------------------------------------------------------------
class TestValidationOrchestrator:
    def test_import(self):
        from dq_engine import ValidationOrchestrator, ValidationConfig, ValidationResult
        orch = ValidationOrchestrator()
        assert orch is not None

    def test_run_basic(self, sample_df):
        from dq_engine import ValidationOrchestrator, ValidationConfig
        config = ValidationConfig(use_rules=True, use_ml=False, use_corpus=False, normalize=False)
        orch = ValidationOrchestrator()

        # Generate rules first
        from dq_engine import RulesWorkflow
        wf = RulesWorkflow()
        rules = wf.generate(df_raw=sample_df, use_inferred=True)

        result = orch.run(sample_df, rules, config)
        assert result is not None
        assert result.total_issues >= 0
        assert len(result.steps_completed) > 0

    def test_run_without_rules(self, sample_df):
        """Should still work with empty rules."""
        from dq_engine import ValidationOrchestrator, ValidationConfig
        config = ValidationConfig(use_rules=True, use_ml=False, use_corpus=False, normalize=False)
        orch = ValidationOrchestrator()
        result = orch.run(sample_df, {}, config)
        assert result is not None

    def test_run_with_ml(self, sample_df):
        from dq_engine import ValidationOrchestrator, ValidationConfig
        config = ValidationConfig(use_rules=False, use_ml=True, use_corpus=False, normalize=False)
        orch = ValidationOrchestrator()
        result = orch.run(sample_df, {}, config)
        assert result is not None

    def test_result_has_timings(self, sample_df):
        from dq_engine import ValidationOrchestrator, ValidationConfig
        config = ValidationConfig(use_rules=True, use_ml=False, normalize=False)
        orch = ValidationOrchestrator()
        result = orch.run(sample_df, {}, config)
        assert isinstance(result.step_timings, dict)

    def test_progress_callback(self, sample_df):
        from dq_engine import ValidationOrchestrator, ValidationConfig
        progress_calls = []

        def on_progress(step, total, message):
            progress_calls.append((step, total, message))

        config = ValidationConfig(use_rules=True, use_ml=False, normalize=False, on_progress=on_progress)
        orch = ValidationOrchestrator()
        orch.run(sample_df, {}, config)
        assert len(progress_calls) > 0


# ---------------------------------------------------------------------------
# AIEnrichment Tests (no actual LLM calls)
# ---------------------------------------------------------------------------
class TestAIEnrichment:
    def test_import_providers(self):
        from dq_engine import OllamaProvider, AnthropicProvider, BedrockProvider
        assert OllamaProvider is not None
        assert AnthropicProvider is not None
        assert BedrockProvider is not None

    def test_enrichment_result_dataclass(self):
        from dq_engine.orchestrators.ai_enrichment import EnrichmentResult
        result = EnrichmentResult()
        assert result.smart_rules == []
        assert result.cross_column_issues == []
        assert result.explanations == []
        assert result.triage == []
        assert result.executive_summary == ""
        assert result.total_time_seconds == 0
        assert result.success is True

    def test_enrichment_result_with_errors(self):
        from dq_engine.orchestrators.ai_enrichment import EnrichmentResult
        result = EnrichmentResult(errors=["test error"])
        assert result.success is False

    def test_build_column_profile(self, sample_df):
        from dq_engine.orchestrators.ai_enrichment import _build_column_profile
        profile = _build_column_profile(sample_df)
        assert isinstance(profile, dict)
        assert "customer_id" in profile
        assert "amount" in profile
        assert profile["amount"]["dtype"] == "float64"
        assert "null_count" in profile["amount"]

    def test_build_issue_summary(self):
        from dq_engine.orchestrators.ai_enrichment import _build_issue_summary
        report = pd.DataFrame({
            "column": ["email", "email", "amount"],
            "issue_type": ["format", "format", "range"],
            "severity": ["high", "high", "medium"],
        })
        summary = _build_issue_summary(report)
        assert summary["total_issues"] == 3
        assert len(summary["affected_columns"]) == 2

    def test_build_cross_column_context(self, sample_df):
        from dq_engine.orchestrators.ai_enrichment import _build_cross_column_context
        relationships = _build_cross_column_context(sample_df)
        assert isinstance(relationships, list)

    def test_parse_json_response_direct(self):
        from dq_engine.orchestrators.ai_enrichment import _parse_json_response
        result = _parse_json_response('[{"key": "value"}]')
        assert isinstance(result, list)
        assert result[0]["key"] == "value"

    def test_parse_json_response_code_block(self):
        from dq_engine.orchestrators.ai_enrichment import _parse_json_response
        text = 'Here is the result:\n```json\n[{"key": "value"}]\n```\nDone.'
        result = _parse_json_response(text)
        assert isinstance(result, list)

    def test_parse_json_response_fallback(self):
        from dq_engine.orchestrators.ai_enrichment import _parse_json_response
        result = _parse_json_response("not json at all")
        assert isinstance(result, dict)
        assert "raw_response" in result


# ---------------------------------------------------------------------------
# Database Tests
# ---------------------------------------------------------------------------
class TestDatabase:
    def test_imports(self):
        from core.storage.database import (
            get_recent_batches, get_batch_issues, get_score_trend, get_batch_audit,
            get_ai_smart_rules, get_ai_cross_column, get_ai_explanations,
            get_ai_triage, get_ai_executive_summary, get_batches_with_ai,
            save_results_to_db, save_ai_enrichment,
        )

    def test_schema_creation(self, tmp_path):
        from core.storage.database import _init_schema_safe, _get_engine
        db_path = str(tmp_path / "test.db")
        config = {"output": {"database": {"engine": "sqlite", "path": db_path}}}
        engine = _get_engine(config)
        _init_schema_safe(engine, "sqlite")
        assert os.path.exists(db_path)

    def test_save_and_read(self, tmp_path):
        from core.storage.database import save_results_to_db, get_recent_batches, get_batch_issues
        db_path = str(tmp_path / "test.db")
        config = {"output": {"database": {"engine": "sqlite", "path": db_path}}}

        result = {
            "batch_id": "test_batch_001",
            "timestamp": "2026-01-01T00:00:00Z",
            "rows_processed": 100,
            "columns": ["col1", "col2"],
            "overall_score": 95.5,
            "pass": True,
            "pass_threshold": 85.0,
            "issues_count": 5,
            "quality_scores": {"completeness": 98, "uniqueness": 95, "consistency": 90, "validity": 96, "accuracy": 97, "timeliness": 100},
            "issues": pd.DataFrame({
                "row_id": [1, 2, 3],
                "column": ["col1", "col1", "col2"],
                "issue": ["missing", "format", "range"],
                "severity": ["high", "medium", "low"],
            }),
            "audit_trail": [
                {"step": "rules", "duration_ms": 100},
                {"step": "ml", "duration_ms": 200},
            ],
        }

        counts = save_results_to_db(result, config, source_file="test.csv")
        assert counts["batch_runs"] == 1
        assert counts["issues"] == 3

        batches = get_recent_batches(config)
        assert len(batches) == 1
        assert batches.iloc[0]["batch_id"] == "test_batch_001"

        issues = get_batch_issues(config, "test_batch_001")
        assert len(issues) == 3

    def test_save_ai_enrichment(self, tmp_path):
        from core.storage.database import save_results_to_db, save_ai_enrichment, get_ai_triage, get_ai_executive_summary
        from dq_engine.orchestrators.ai_enrichment import EnrichmentResult

        db_path = str(tmp_path / "test.db")
        config = {"output": {"database": {"engine": "sqlite", "path": db_path}}}

        # Create a batch first
        result = {
            "batch_id": "test_ai_batch",
            "timestamp": "2026-01-01T00:00:00Z",
            "rows_processed": 50,
            "columns": ["a"],
            "overall_score": 90.0,
            "pass": True,
            "pass_threshold": 85.0,
            "issues_count": 2,
            "quality_scores": {},
            "issues": pd.DataFrame(),
            "audit_trail": [],
        }
        save_results_to_db(result, config)

        # Save AI enrichment
        enrichment = EnrichmentResult(
            smart_rules=[{"column": "email", "rule_type": "format", "description": "test rule", "confidence": 0.9}],
            triage=[{"priority": 1, "column": "email", "issue_type": "format", "count": 5, "severity": "high", "reason": "test", "effort": "quick_fix", "recommendation": "fix it"}],
            executive_summary="Test summary.",
            provider="test",
            timings={"smart_rules": 1.0, "triage": 2.0},
        )

        counts = save_ai_enrichment("test_ai_batch", enrichment, config)
        assert counts["ai_smart_rules"] == 1
        assert counts["ai_triage"] == 1
        assert counts["ai_executive_summary"] == 1

        triage = get_ai_triage(config, "test_ai_batch")
        assert len(triage) == 1
        assert triage.iloc[0]["column_name"] == "email"

        summary = get_ai_executive_summary(config, "test_ai_batch")
        assert summary == "Test summary."


# ---------------------------------------------------------------------------
# Shared Module Tests
# ---------------------------------------------------------------------------
class TestSharedModules:
    def test_auth_hash(self):
        from app.shared.auth import _hash
        h = _hash("admin123")
        assert len(h) == 64  # SHA256 hex

    def test_state_init(self):
        """Verify init_state defines expected keys."""
        # Can't fully test without Streamlit runtime, but can import
        from app.shared.state import init_state
        assert callable(init_state)

    def test_theme_functions(self):
        from app.shared.theme import kpi_card, MATRIX_GREEN, MATRIX_CYAN
        html = kpi_card("42", "Test Label", "cyan")
        assert "42" in html
        assert "Test Label" in html
        assert "cyan" in html
