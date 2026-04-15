"""
RulesWorkflow - Generates, merges, and manages validation rules.

Extracted from data_quality_app.py lines 1866-2262.
No Streamlit dependency. No UI code. Pure logic.

Usage:
    from dq_engine import RulesWorkflow

    workflow = RulesWorkflow()

    # Generate rules from data
    rules = workflow.generate(df_raw, df_ref=df_ref, rules_json=imported_json)

    # Or step by step
    inferred = workflow.infer_from_data(df_raw)
    reference = workflow.infer_from_reference(df_ref)
    merged = workflow.merge(rules_json=imported_json, reference=reference, inferred=inferred)

    # Select/filter rules
    final = workflow.finalize(merged, selections={"col_0": True, "col_1": False, ...})

    # Serialize for storage/download
    json_str = workflow.to_json(final)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RulesConfig:
    """Configuration for rule generation thresholds."""

    min_numeric_samples: int = 10
    min_samples_for_regex: int = 10
    min_regex_coverage: float = 0.80
    max_regex_patterns: int = 5
    low_cardinality_max_abs: int = 50

    def to_rule_config(self):
        """Convert to core.rule_generator_enhanced.RuleConfig if available."""
        try:
            from core.rule_generator_enhanced import RuleConfig
            rc = RuleConfig()
            rc.MIN_NUMERIC_SAMPLES = self.min_numeric_samples
            rc.MIN_SAMPLES_FOR_REGEX = self.min_samples_for_regex
            rc.MIN_REGEX_COVERAGE = self.min_regex_coverage
            rc.MAX_REGEX_PATTERNS = self.max_regex_patterns
            rc.LOW_CARDINALITY_MAX_ABS = self.low_cardinality_max_abs
            return rc
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------
def make_serializable(obj: Any) -> Any:
    """Convert Python objects (regex, numpy) to JSON-safe types."""
    if isinstance(obj, re.Pattern):
        return obj.pattern
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
class RulesWorkflow:
    """
    Generates, merges, selects, and serializes validation rules.

    This class coordinates rule generation from three sources:
      1. Imported JSON rules (highest priority)
      2. Reference dataset rules (medium priority)
      3. Inferred rules from the raw data (lowest priority)

    It then merges them with source precedence and allows selection/filtering.
    """

    def __init__(self, config: Optional[RulesConfig] = None):
        self.config = config or RulesConfig()
        self._use_enhanced = self._check_enhanced_available()

    @staticmethod
    def _check_enhanced_available() -> bool:
        """Check if the enhanced rule generator module is available."""
        try:
            from core.rule_generator_enhanced import generate_rules_with_metadata
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def infer_from_data(self, df: pd.DataFrame) -> Union[dict, Dict[str, list]]:
        """
        Infer validation rules from the raw (unclean) dataset.

        Returns:
            If enhanced module available: dict of {column: [rule_dicts]}
            Otherwise: standard rules dict {"columns": {...}}
        """
        if self._use_enhanced:
            from core.rule_generator_enhanced import generate_rules_with_metadata
            rc = self.config.to_rule_config()
            return generate_rules_with_metadata(df, rc, "inferred")
        else:
            from core.validator.discover import infer_rules_from_unclean
            return infer_rules_from_unclean(df)

    def infer_from_reference(self, df_ref: pd.DataFrame) -> Union[dict, Dict[str, list]]:
        """
        Generate validation rules from a clean reference dataset.

        Returns:
            Rules dict matching the format of infer_from_data().
        """
        if self._use_enhanced:
            from core.rule_generator_enhanced import generate_rules_with_metadata
            rc = self.config.to_rule_config()
            return generate_rules_with_metadata(df_ref, rc, "reference")
        else:
            from core.validator.discover import rules_from_reference
            return rules_from_reference(df_ref)

    def merge(
        self,
        rules_json: Optional[dict] = None,
        reference: Optional[dict] = None,
        inferred: Optional[dict] = None,
    ) -> dict:
        """
        Merge rules from all sources with precedence: JSON > reference > inferred.

        Args:
            rules_json: Imported JSON rules (highest priority).
            reference: Rules from reference dataset.
            inferred: Rules inferred from raw data.

        Returns:
            Merged rules dict.
        """
        if self._use_enhanced:
            from core.rule_generator_enhanced import merge_rules_from_sources
            rc = self.config.to_rule_config()
            return merge_rules_from_sources(
                rules_json,
                reference or {},
                inferred or {},
                rc,
            )
        else:
            from core.validator.discover import merge_rules
            from core.validator.rules_schema import normalize_rules
            merged = merge_rules(
                rules_json,
                reference or {"columns": {}},
                inferred or {"columns": {}},
            )
            return normalize_rules(merged)

    def generate(
        self,
        df_raw: pd.DataFrame,
        df_ref: Optional[pd.DataFrame] = None,
        rules_json: Optional[dict] = None,
        use_inferred: bool = True,
        use_reference: bool = True,
        use_json: bool = True,
    ) -> dict:
        """
        One-shot: generate and merge rules from all available sources.

        This is the convenience method that does everything in one call.

        Args:
            df_raw: The raw dataset to infer rules from.
            df_ref: Optional clean reference dataset.
            rules_json: Optional imported JSON rules.
            use_inferred: Whether to infer rules from df_raw.
            use_reference: Whether to generate rules from df_ref.
            use_json: Whether to include imported JSON rules.

        Returns:
            Merged rules dict ready for validation.
        """
        inferred = self.infer_from_data(df_raw) if use_inferred else {}
        reference = (
            self.infer_from_reference(df_ref)
            if (use_reference and df_ref is not None and not df_ref.empty)
            else {}
        )
        json_rules = rules_json if use_json else None

        merged = self.merge(
            rules_json=json_rules,
            reference=reference,
            inferred=inferred,
        )

        logger.info(
            "Rules generated: json=%s, reference=%s, inferred=%s",
            "yes" if json_rules else "no",
            "yes" if reference else "no",
            "yes" if inferred else "no",
        )

        return merged

    # ------------------------------------------------------------------
    # Selection / filtering
    # ------------------------------------------------------------------

    def finalize(
        self,
        generated_rules: Dict[str, list],
        selections: Optional[Dict[str, bool]] = None,
    ) -> dict:
        """
        Build the final rules dict from generated rules and user selections.

        If no selections provided, all rules are included.

        Args:
            generated_rules: Output of generate() when using enhanced module.
                Format: {column_name: [rule_dicts]}
            selections: Optional dict of {"{col}_{idx}": bool} for each rule.

        Returns:
            Standard rules dict: {"columns": {...}, "uniqueness": {}, "cross_field": []}
        """
        final_rules: Dict[str, Any] = {"columns": {}, "uniqueness": {}, "cross_field": []}

        for col_name, rules_list in generated_rules.items():
            col_rules: Dict[str, Any] = {}

            for idx, rule in enumerate(rules_list):
                rule_key = f"{col_name}_{idx}"

                # Check if selected (default: use rule's own 'selected' field, or True)
                if selections is not None:
                    is_selected = selections.get(rule_key, rule.get("selected", True))
                else:
                    is_selected = rule.get("selected", True)

                if not is_selected:
                    continue

                # Merge rule_data into column rules
                rule_data = rule.get("rule_data", {})
                for key, value in rule_data.items():
                    if key not in col_rules:
                        col_rules[key] = value
                    elif isinstance(col_rules[key], list):
                        if isinstance(value, list):
                            col_rules[key].extend(value)
                        else:
                            col_rules[key].append(value)

            if col_rules:
                final_rules["columns"][col_name] = col_rules

        return final_rules

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def get_stats(generated_rules: Dict[str, list], selections: Optional[Dict[str, bool]] = None) -> dict:
        """
        Compute summary statistics for generated rules.

        Returns:
            Dict with total_columns, total_rules, selected_rules, selection_pct.
        """
        total = sum(len(rules) for rules in generated_rules.values())
        selected = 0
        for col_name, rules_list in generated_rules.items():
            for idx, rule in enumerate(rules_list):
                key = f"{col_name}_{idx}"
                if selections is not None:
                    is_sel = selections.get(key, rule.get("selected", True))
                else:
                    is_sel = rule.get("selected", True)
                if is_sel:
                    selected += 1

        return {
            "total_columns": len(generated_rules),
            "total_rules": total,
            "selected_rules": selected,
            "selection_pct": (selected / total * 100) if total > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(rules: dict, indent: int = 2) -> str:
        """Serialize rules to JSON string, handling regex/numpy types."""
        return json.dumps(make_serializable(rules), indent=indent)

    @staticmethod
    def from_json(json_str: str) -> dict:
        """Load rules from a JSON string."""
        return json.loads(json_str)

    @staticmethod
    def from_file(path: str) -> dict:
        """Load rules from a JSON file."""
        with open(path, "r") as f:
            return json.load(f)
