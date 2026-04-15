"""
ValidationOrchestrator - Sequences multiple validators and returns a unified report.

Extracted from data_quality_app.py lines 3705-3884.
No Streamlit dependency. No UI code. Pure logic.

Usage:
    from dq_engine import ValidationOrchestrator, ValidationConfig

    config = ValidationConfig(
        use_rules=True,
        use_bert=True,
        use_corpus=True,
        use_ml=False,
        use_api=False,
        normalize=True,
        bert_mode="critical",
        corpus_mappings={"bank_name": {"corpus": "banks", "type": "bank"}},
    )

    orchestrator = ValidationOrchestrator()
    result = orchestrator.run(df, rules, config)

    print(result.total_issues)
    print(result.report.head())
    print(result.scores)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    """Configuration for a validation run. Framework-agnostic."""

    use_rules: bool = True
    use_bert: bool = False
    use_corpus: bool = False
    use_ml: bool = False
    use_api: bool = False
    normalize: bool = True

    # BERT settings
    bert_mode: str = "critical"  # "critical", "all", "sample"
    bert_sample_size: int = 50

    # Corpus mappings: {column_name: {"corpus": name, "type": type}}
    corpus_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Corpus manager instance (injected, not created here)
    corpus_manager: Any = None

    # API settings
    api_config: Optional[Dict[str, str]] = None  # {"type": ..., "url": ..., "column": ...}

    # Reference dataframe (optional, for attach_suggestions)
    df_ref: Optional[pd.DataFrame] = None

    # AI Enrichment settings
    use_ai_enrichment: bool = False
    ai_provider: Any = None  # LLMProvider instance (OllamaProvider, BedrockProvider, etc.)

    # Progress callback: called with (step_number, total_steps, message)
    on_progress: Optional[Callable[[int, int, str], None]] = None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Immutable result of a validation run."""

    report: pd.DataFrame
    total_issues: int
    steps_completed: List[str]
    step_timings: Dict[str, float]  # step_name -> seconds
    errors: List[str]

    # Individual reports (for consumers that need them separately)
    rule_report: Optional[pd.DataFrame] = None
    corpus_report: Optional[pd.DataFrame] = None
    ml_report: Optional[pd.DataFrame] = None
    api_report: Optional[pd.DataFrame] = None

    # AI enrichment results (None if enrichment was not run)
    ai_enrichment: Any = None  # EnrichmentResult from ai_enrichment module

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_time_seconds(self) -> float:
        return sum(self.step_timings.values())


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------
def _detect_norm_config(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column normalization types based on column names."""
    config = {}
    for col in df.columns:
        col_lower = col.lower()
        if "email" in col_lower:
            config[col] = "email"
        elif "postcode" in col_lower or "postal" in col_lower:
            config[col] = "postcode"
        elif "phone" in col_lower or "tel" in col_lower:
            config[col] = "phone"
        else:
            config[col] = "text"
    return config


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ValidationOrchestrator:
    """
    Sequences validators and produces a unified report.

    This class contains zero UI code. It coordinates:
      1. Rule-based validation (with optional BERT explanations)
      2. Corpus validation
      3. ML anomaly detection
      4. External API validation
      5. Report unification
      6. Correction suggestion attachment
    """

    def run(
        self,
        df: pd.DataFrame,
        rules: list,
        config: Optional[ValidationConfig] = None,
    ) -> ValidationResult:
        """
        Execute the full validation pipeline.

        Args:
            df: The dataframe to validate.
            rules: Merged rule list from the Rules workflow.
            config: Validation configuration. Uses defaults if None.

        Returns:
            ValidationResult with unified report and metadata.
        """
        if config is None:
            config = ValidationConfig()

        steps_completed: List[str] = []
        step_timings: Dict[str, float] = {}
        errors: List[str] = []

        rule_report = None
        corpus_report = None
        ml_report = None
        api_report = None

        # Count active steps for progress reporting
        active_steps = sum([
            bool(config.use_rules),
            bool(config.use_corpus and config.corpus_mappings),
            bool(config.use_ml),
            bool(config.use_api and config.api_config),
        ])
        current_step = 0

        def _progress(message: str):
            nonlocal current_step
            current_step += 1
            if config.on_progress:
                config.on_progress(current_step, active_steps, message)
            logger.info("Step %d/%d: %s", current_step, active_steps, message)

        # ----- Prepare data -----
        df_to_validate = df.copy()

        if config.normalize:
            try:
                from core.corpus_validation import create_normalized_dataframe_for_validation
                norm_config = _detect_norm_config(df_to_validate)
                df_to_validate = create_normalized_dataframe_for_validation(
                    df_to_validate, norm_config
                )
                logger.info("Data normalized for validation.")
            except ImportError:
                logger.warning("corpus_validation module not available; skipping normalization.")
            except Exception as exc:
                logger.warning("Normalization failed: %s", exc)

        # ----- Step 1: Rule-based validation -----
        if config.use_rules:
            _progress("Running rule-based validation")
            t0 = time.time()
            try:
                rule_report = self._run_rules(df_to_validate, rules, config)
                elapsed = time.time() - t0
                step_timings["rule_validation"] = elapsed
                steps_completed.append("rule_validation")
                count = len(rule_report) if isinstance(rule_report, pd.DataFrame) else 0
                logger.info("Rule validation: %d issues in %.2fs", count, elapsed)
            except Exception as exc:
                elapsed = time.time() - t0
                step_timings["rule_validation"] = elapsed
                errors.append(f"Rule validation failed: {exc}")
                logger.error("Rule validation failed: %s", exc, exc_info=True)

        # ----- Step 2: Corpus validation -----
        if config.use_corpus and config.corpus_mappings:
            _progress("Running corpus validation")
            t0 = time.time()
            try:
                corpus_report = self._run_corpus(
                    df_to_validate, config.corpus_mappings, config.corpus_manager
                )
                elapsed = time.time() - t0
                step_timings["corpus_validation"] = elapsed
                steps_completed.append("corpus_validation")
                count = len(corpus_report) if isinstance(corpus_report, pd.DataFrame) else 0
                logger.info("Corpus validation: %d issues in %.2fs", count, elapsed)
            except Exception as exc:
                elapsed = time.time() - t0
                step_timings["corpus_validation"] = elapsed
                errors.append(f"Corpus validation failed: {exc}")
                logger.error("Corpus validation failed: %s", exc, exc_info=True)

        # ----- Step 3: ML anomaly detection -----
        if config.use_ml:
            _progress("Running ML anomaly detection")
            t0 = time.time()
            try:
                ml_report = self._run_ml(df_to_validate, rules)
                elapsed = time.time() - t0
                step_timings["ml_anomaly"] = elapsed
                steps_completed.append("ml_anomaly")
                count = len(ml_report) if isinstance(ml_report, pd.DataFrame) else 0
                logger.info("ML anomaly detection: %d issues in %.2fs", count, elapsed)
            except Exception as exc:
                elapsed = time.time() - t0
                step_timings["ml_anomaly"] = elapsed
                errors.append(f"ML anomaly detection failed: {exc}")
                logger.error("ML anomaly detection failed: %s", exc, exc_info=True)

        # ----- Step 4: API validation -----
        if config.use_api and config.api_config:
            _progress("Running API validation")
            t0 = time.time()
            try:
                api_report = self._run_api(df_to_validate, config.api_config)
                elapsed = time.time() - t0
                step_timings["api_validation"] = elapsed
                steps_completed.append("api_validation")
                count = len(api_report) if isinstance(api_report, pd.DataFrame) else 0
                logger.info("API validation: %d issues in %.2fs", count, elapsed)
            except Exception as exc:
                elapsed = time.time() - t0
                step_timings["api_validation"] = elapsed
                errors.append(f"API validation failed: {exc}")
                logger.error("API validation failed: %s", exc, exc_info=True)

        # ----- Unify reports -----
        t0 = time.time()
        try:
            from core.validation_ui_helpers import unify_validation_reports
            unified = unify_validation_reports(
                rule_report=rule_report,
                corpus_report=corpus_report,
                ml_report=ml_report,
                api_report=api_report,
            )
        except ImportError:
            # Fallback: concatenate whatever we have
            parts = [r for r in [rule_report, corpus_report, ml_report, api_report]
                     if isinstance(r, pd.DataFrame) and len(r) > 0]
            unified = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        step_timings["unify_reports"] = time.time() - t0
        steps_completed.append("unify_reports")

        # ----- Attach correction suggestions -----
        t0 = time.time()
        try:
            from core.validator.validate import attach_suggestions
            unified = attach_suggestions(unified, df_to_validate, df_ref=config.df_ref)
            steps_completed.append("attach_suggestions")
        except ImportError:
            logger.warning("attach_suggestions not available; skipping.")
        except Exception as exc:
            errors.append(f"Correction suggestions failed: {exc}")
            logger.warning("attach_suggestions failed: %s", exc)
        step_timings["attach_suggestions"] = time.time() - t0

        # ----- Step 5: AI Enrichment (the differentiator) -----
        ai_enrichment_result = None
        if config.use_ai_enrichment and config.ai_provider is not None:
            if config.on_progress:
                config.on_progress(current_step + 1, active_steps + 5, "Running AI enrichment (5 calls)")
            logger.info("Running AI enrichment using %s", config.ai_provider.name)
            t0 = time.time()
            try:
                from dq_engine.orchestrators.ai_enrichment import AIEnrichment

                enricher = AIEnrichment(config.ai_provider)
                ai_enrichment_result = enricher.run_all(
                    df=df_to_validate,
                    validation_report=unified,
                    on_progress=config.on_progress,
                )
                elapsed = time.time() - t0
                step_timings["ai_enrichment"] = elapsed
                steps_completed.append("ai_enrichment")
                logger.info(
                    "AI enrichment complete in %.2fs: %d rules, %d cross-col, %d explanations, %d triage items",
                    elapsed,
                    len(ai_enrichment_result.smart_rules),
                    len(ai_enrichment_result.cross_column_issues),
                    len(ai_enrichment_result.explanations),
                    len(ai_enrichment_result.triage),
                )
                # Propagate AI errors
                for err in ai_enrichment_result.errors:
                    errors.append(f"AI enrichment: {err}")
            except Exception as exc:
                step_timings["ai_enrichment"] = time.time() - t0
                errors.append(f"AI enrichment failed: {exc}")
                logger.error("AI enrichment failed: %s", exc, exc_info=True)

        return ValidationResult(
            report=unified,
            total_issues=len(unified) if isinstance(unified, pd.DataFrame) else 0,
            steps_completed=steps_completed,
            step_timings=step_timings,
            errors=errors,
            rule_report=rule_report,
            corpus_report=corpus_report,
            ml_report=ml_report,
            api_report=api_report,
            ai_enrichment=ai_enrichment_result,
        )

    # ------------------------------------------------------------------
    # Private step implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _run_rules(
        df: pd.DataFrame, rules: list, config: ValidationConfig
    ) -> Optional[pd.DataFrame]:
        """Run rule-based validation, optionally with BERT explanations."""

        if config.use_bert:
            from core.enhanced_validator import validate_dataframe

            kwargs = {"use_bert": True}
            if config.bert_mode == "critical":
                kwargs["explain_critical"] = True
            elif config.bert_mode == "all":
                kwargs["explain_all"] = True
            elif config.bert_mode == "sample":
                kwargs["explain_sample"] = config.bert_sample_size

            report = validate_dataframe(df, rules=rules, **kwargs)
        else:
            from core.validator.validate import validate_df
            report = validate_df(df, rules, enable_typo_detection=True)

        if isinstance(report, pd.DataFrame):
            return report
        return None

    @staticmethod
    def _run_corpus(
        df: pd.DataFrame,
        corpus_mappings: Dict[str, Dict[str, str]],
        corpus_manager: Any,
    ) -> Optional[pd.DataFrame]:
        """Run corpus validation using loaded corpora."""
        from core.corpus_validation import validate_all_corpus_mappings

        report = validate_all_corpus_mappings(
            df=df,
            corpus_mappings=corpus_mappings,
            corpus_manager=corpus_manager,
        )
        if isinstance(report, pd.DataFrame):
            return report
        return None

    @staticmethod
    def _run_ml(df: pd.DataFrame, rules: list) -> Optional[pd.DataFrame]:
        """Run ML-based anomaly detection."""
        from core.validator.validate import validate_and_anomaly_report

        report = validate_and_anomaly_report(df, rules, use_ml=True)
        if isinstance(report, pd.DataFrame):
            return report
        return None

    @staticmethod
    def _run_api(
        df: pd.DataFrame, api_config: Dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """Run external API validation."""
        from core.external_validators import validate_column_via_api

        col = api_config["column"]
        api_df = validate_column_via_api(
            df[[col]].copy(),
            column=col,
            api_type=api_config["type"],
            api_url=api_config["url"],
        )

        # Convert API result to issues format
        verified_col = f"{col}_verified"
        issues = []
        for idx, row in api_df.iterrows():
            if verified_col in api_df.columns:
                v = row[verified_col]
                is_invalid = (v == "\u274c") or (v is False) or (v == 0)
                if is_invalid:
                    issues.append({
                        "row_id": idx,
                        "column": col,
                        "issue_type": "api_validation_failed",
                        "value": row[col],
                        "suggested_fix": None,
                        "confidence": 0.95,
                        "source": "api",
                        "description": f"API validation failed for {api_config['type']}",
                    })

        if issues:
            return pd.DataFrame(issues)
        return None
