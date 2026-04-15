"""
dq_engine - Core orchestration package for the AI Powered DQ Investigator.

This package provides framework-agnostic orchestrators that coordinate
the validation, rules, deduplication, and session logic from core/.
No Streamlit or UI dependency.
"""

__version__ = "1.0.0"

from dq_engine.orchestrators.validation import ValidationOrchestrator, ValidationConfig, ValidationResult
from dq_engine.orchestrators.rules import RulesWorkflow
from dq_engine.orchestrators.ai_enrichment import (
    AIEnrichment,
    EnrichmentResult,
    OllamaProvider,
    BedrockProvider,
    AnthropicProvider,
    LLMProvider,
)
