# ==========================================================
# core/rule_generator_enhanced.py
# Enhanced Rule Generation with Detailed Metadata for Interactive UI
# ==========================================================

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import re
from collections import Counter
from datetime import datetime

# Import base discovery functions
from .validator.discover import (
    _non_null_str_series,
    _looks_numeric_series,
    _try_numeric_bounds,
    _is_date_series,
    _date_bounds_iso,
    _infer_date_formats,
    _compress_to_regex,
    COMMON_DATE_FORMATS,
)


# ==========================================================
# ENHANCED CONFIGURATION (User-adjustable)
# ==========================================================
class RuleConfig:
    """Configuration for rule generation - can be adjusted by user"""

    def __init__(self):
        # Numeric detection
        self.MAJORITY_COVERAGE_STRICT = 0.95
        self.MAJORITY_COVERAGE_MEDIUM = 0.80
        self.MAJORITY_COVERAGE_LOOSE = 0.60
        self.MIN_NUMERIC_SAMPLES = 10

        # Pattern detection
        self.MIN_SAMPLES_FOR_REGEX = 20
        self.MIN_PATTERN_MATCHES = 10
        self.MIN_REGEX_COVERAGE = 0.10
        self.MIN_REGEX_COVERAGE_STRICT = 0.20
        self.MAX_REGEX_PATTERNS = 10

        # Categorical detection
        self.LOW_CARDINALITY_MAX_ABS = 100
        self.LOW_CARDINALITY_MAX_RATIO = 0.05
        self.MIN_CATEGORICAL_OCCURRENCES = 2

        # Uniqueness detection
        self.UNIQUE_RATIO_STRICT = 1.00
        self.UNIQUE_RATIO_MEDIUM = 0.98
        self.UNIQUE_RATIO_LOOSE = 0.95
        self.MAX_ALLOWED_DUPLICATES = 3

        # Quality thresholds
        self.MIN_NON_NULL_RATIO = 0.30
        self.MAX_FREE_TEXT_UNIQUE_RATIO = 0.80
        self.MAX_FREE_TEXT_AVG_LENGTH = 100

    def to_dict(self):
        """Export config as dictionary"""
        return {k: v for k, v in self.__dict__.items() if k.isupper()}

    def from_dict(self, config_dict):
        """Import config from dictionary"""
        for k, v in config_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ==========================================================
# RULE METADATA STRUCTURE
# ==========================================================
def create_rule_metadata(
    rule_type: str,
    source: str,
    coverage: float,
    matched_count: int,
    total_count: int,
    examples_match: List[str],
    examples_fail: List[str],
    rule_data: Dict[str, Any],
    quality_score: float,
    warnings: List[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a rule

    Args:
        rule_type: Type of rule (type, regex, categorical, uniqueness, etc.)
        source: Source of rule (json, reference, inferred)
        coverage: Percentage coverage (0.0 to 1.0)
        matched_count: Number of rows matched
        total_count: Total rows
        examples_match: Example values that match the rule
        examples_fail: Example values that fail the rule
        rule_data: The actual rule definition
        quality_score: Quality score (0-100)
        warnings: List of warning messages
    """
    return {
        'rule_type': rule_type,
        'source': source,
        'coverage': coverage,
        'matched_count': matched_count,
        'total_count': total_count,
        'examples_match': examples_match[:3],  # Max 3 examples
        'examples_fail': examples_fail[:3],
        'rule_data': rule_data,
        'quality_score': quality_score,
        'warnings': warnings or [],
        'selected': True if quality_score > 60 else False,  # Auto-select high quality
    }


# ==========================================================
# ENHANCED RULE GENERATION
# ==========================================================
def generate_rules_with_metadata(
    df: pd.DataFrame,
    config: RuleConfig = None,
    source: str = "inferred"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate rules with comprehensive metadata for each column

    Returns:
        Dict mapping column_name -> list of rule metadata dicts
    """
    if config is None:
        config = RuleConfig()

    column_rules = {}

    for col in df.columns:
        rules_list = []
        s_full = df[col]
        s = _non_null_str_series(s_full)

        total_count = len(s_full)
        non_null_count = s_full.notna().sum()

        if total_count == 0:
            column_rules[col] = []
            continue

        # Skip if too many nulls
        non_null_ratio = non_null_count / total_count
        if non_null_ratio < config.MIN_NON_NULL_RATIO:
            warning_rule = create_rule_metadata(
                rule_type='warning',
                source=source,
                coverage=0,
                matched_count=0,
                total_count=total_count,
                examples_match=[],
                examples_fail=[],
                rule_data={},
                quality_score=0,
                warnings=[f"Column has {(1-non_null_ratio)*100:.1f}% nulls - too sparse for reliable rules"]
            )
            column_rules[col] = [warning_rule]
            continue

        # ===========================================
        # 1. NUMERIC TYPE DETECTION
        # ===========================================
        if _looks_numeric_series(s) and len(s) >= config.MIN_NUMERIC_SAMPLES:
            mn, mx = _try_numeric_bounds(s)
            if mn is not None and mx is not None:
                # Collect examples
                numeric_vals = []
                for v in s_full.dropna():
                    try:
                        numeric_vals.append(float(str(v).replace(",", "")))
                    except:
                        pass

                examples_match = [str(v) for v in s_full.dropna().head(3)]
                examples_fail = []

                # Count matches
                matched = len(numeric_vals)
                coverage = matched / non_null_count if non_null_count > 0 else 0

                quality_score = min(100, coverage * 100)

                rule = create_rule_metadata(
                    rule_type='type_numeric',
                    source=source,
                    coverage=coverage,
                    matched_count=matched,
                    total_count=non_null_count,
                    examples_match=examples_match,
                    examples_fail=examples_fail,
                    rule_data={'type': 'number', 'min': mn, 'max': mx},
                    quality_score=quality_score,
                )
                rules_list.append(rule)

        # ===========================================
        # 2. DATE TYPE DETECTION
        # ===========================================
        if _is_date_series(s_full):
            dmin, dmax = _date_bounds_iso(s_full)
            fmts = _infer_date_formats(s_full)

            examples_match = [str(v) for v in s_full.dropna().head(3)]

            rule = create_rule_metadata(
                rule_type='type_date',
                source=source,
                coverage=1.0,
                matched_count=non_null_count,
                total_count=non_null_count,
                examples_match=examples_match,
                examples_fail=[],
                rule_data={
                    'type': 'date',
                    'min_date': dmin,
                    'max_date': dmax,
                    'date_format': fmts
                },
                quality_score=95,
            )
            rules_list.append(rule)

        # ===========================================
        # 3. CATEGORICAL (LOW CARDINALITY)
        # ===========================================
        unique_count = s.nunique(dropna=True)
        unique_ratio = unique_count / len(s) if len(s) > 0 else 0

        is_categorical = (
            unique_count <= config.LOW_CARDINALITY_MAX_ABS or
            unique_ratio <= config.LOW_CARDINALITY_MAX_RATIO
        )

        if is_categorical and unique_count > 0:
            vc = s.value_counts()
            vals = vc.index.tolist()[:200]

            examples_match = vals[:3]

            total = int(vc.sum())
            coverage = 1.0

            # Quality based on number of categories
            if unique_count <= 10:
                quality_score = 95
            elif unique_count <= 25:
                quality_score = 85
            else:
                quality_score = 70

            rule = create_rule_metadata(
                rule_type='categorical',
                source=source,
                coverage=coverage,
                matched_count=total,
                total_count=total,
                examples_match=examples_match,
                examples_fail=[],
                rule_data={
                    'allowed_values': vals,
                    'unique_count': unique_count,
                },
                quality_score=quality_score,
            )
            rules_list.append(rule)

        # ===========================================
        # 4. UNIQUENESS DETECTION
        # ===========================================
        unique_ratio = s_full.nunique() / len(s_full) if len(s_full) > 0 else 0

        if unique_ratio >= config.UNIQUE_RATIO_LOOSE:
            duplicate_count = len(s_full) - s_full.nunique()

            # Determine severity
            if unique_ratio >= config.UNIQUE_RATIO_STRICT:
                quality_score = 100
                warnings = []
            elif unique_ratio >= config.UNIQUE_RATIO_MEDIUM:
                quality_score = 90
                warnings = [f"{duplicate_count} duplicates found (likely errors)"]
            else:
                quality_score = 75
                warnings = [f"{duplicate_count} duplicates found"]

            examples_match = [str(v) for v in s_full.dropna().head(3)]

            rule = create_rule_metadata(
                rule_type='uniqueness',
                source=source,
                coverage=unique_ratio,
                matched_count=s_full.nunique(),
                total_count=len(s_full),
                examples_match=examples_match,
                examples_fail=[],
                rule_data={'unique': True, 'duplicate_count': duplicate_count},
                quality_score=quality_score,
                warnings=warnings,
            )
            rules_list.append(rule)

        # ===========================================
        # 5. REGEX PATTERNS
        # ===========================================
        if len(s) >= config.MIN_SAMPLES_FOR_REGEX:
            # Skip free text
            if unique_ratio < config.MAX_FREE_TEXT_UNIQUE_RATIO:
                avg_len = s.str.len().mean()
                if avg_len < config.MAX_FREE_TEXT_AVG_LENGTH:
                    patterns = generate_regex_patterns_with_metadata(
                        s, config, source, total_count
                    )
                    rules_list.extend(patterns)

        # ===========================================
        # 6. REQUIRED FIELD DETECTION
        # ===========================================
        null_count = s_full.isna().sum()
        null_ratio = null_count / total_count

        if null_ratio < 0.01:  # Less than 1% nulls
            rule = create_rule_metadata(
                rule_type='required',
                source=source,
                coverage=1.0 - null_ratio,
                matched_count=non_null_count,
                total_count=total_count,
                examples_match=[],
                examples_fail=[],
                rule_data={'required': True, 'null_count': null_count},
                quality_score=95,
            )
            rules_list.append(rule)

        column_rules[col] = rules_list

    return column_rules


def generate_regex_patterns_with_metadata(
    s: pd.Series,
    config: RuleConfig,
    source: str,
    total_count: int
) -> List[Dict[str, Any]]:
    """Generate regex patterns with metadata"""
    patterns_list = []

    # Generate patterns
    pats = [_compress_to_regex(v) for v in s]
    cnt = Counter(pats)

    total_values = len(s)

    for pattern, count in cnt.most_common(config.MAX_REGEX_PATTERNS):
        coverage = count / total_values

        # Filter by coverage
        if coverage < config.MIN_REGEX_COVERAGE or count < config.MIN_PATTERN_MATCHES:
            continue

        # Find examples that match this pattern
        examples_match = []
        examples_fail = []

        try:
            regex_compiled = re.compile(pattern)
            for val in s.head(100):
                if regex_compiled.fullmatch(str(val)):
                    if len(examples_match) < 3:
                        examples_match.append(str(val))
                else:
                    if len(examples_fail) < 3:
                        examples_fail.append(str(val))
        except:
            continue

        # Quality score
        if coverage >= 0.80:
            quality_score = 95
        elif coverage >= 0.50:
            quality_score = 80
        elif coverage >= 0.20:
            quality_score = 65
        else:
            quality_score = 40

        # Warnings
        warnings = []
        if coverage < 0.20:
            warnings.append(f"Low coverage ({coverage*100:.1f}%) - might be noise")
        if len(pattern) > 100:
            warnings.append("Very long pattern - might be overfitted")

        rule = create_rule_metadata(
            rule_type='regex',
            source=source,
            coverage=coverage,
            matched_count=count,
            total_count=total_values,
            examples_match=examples_match,
            examples_fail=examples_fail,
            rule_data={'regex': pattern},
            quality_score=quality_score,
            warnings=warnings,
        )
        patterns_list.append(rule)

    return patterns_list


# ==========================================================
# MERGE RULES FROM MULTIPLE SOURCES
# ==========================================================
def merge_rules_from_sources(
    json_rules: Optional[Dict] = None,
    reference_rules: Optional[Dict] = None,
    inferred_rules: Optional[Dict] = None,
    config: RuleConfig = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge rules from multiple sources with precedence

    Precedence: JSON > Reference > Inferred
    """
    merged = {}

    # Get all columns — json_rules format is flat {col: rule_config}, not nested
    all_columns = set()
    if json_rules:
        all_columns.update(json_rules.keys())
    if reference_rules:
        all_columns.update(reference_rules.keys())
    if inferred_rules:
        all_columns.update(inferred_rules.keys())

    for col in all_columns:
        col_rules = []

        # Add inferred rules (lowest priority)
        if inferred_rules and col in inferred_rules:
            for rule in inferred_rules[col]:
                rule['source'] = 'inferred'
                rule['source_priority'] = 3
                col_rules.append(rule)

        # Add reference rules (medium priority)
        if reference_rules and col in reference_rules:
            for rule in reference_rules[col]:
                rule['source'] = 'reference'
                rule['source_priority'] = 2
                col_rules.append(rule)

        # Add JSON rules (highest priority) — convert flat rule config to metadata format
        if json_rules and col in json_rules:
            jcr = json_rules[col]
            converted = {
                'source': 'json',
                'source_priority': 1,
                'quality_score': 1.0,
                'column': col,
                'description': jcr.get('description', f'{col} JSON rule'),
                'severity': jcr.get('severity', 'error'),
                'auto_fix': jcr.get('auto_fix', False),
            }
            if 'pattern' in jcr:
                converted['rule_type'] = 'pattern'
                converted['pattern'] = jcr['pattern']
            elif 'allowed_values' in jcr:
                converted['rule_type'] = 'allowed_values'
                converted['allowed_values'] = jcr['allowed_values']
                converted['case_sensitive'] = jcr.get('case_sensitive', True)
            elif 'min' in jcr or 'max' in jcr:
                converted['rule_type'] = 'range'
                converted['min'] = jcr.get('min')
                converted['max'] = jcr.get('max')
            elif 'min_length' in jcr or 'max_length' in jcr:
                converted['rule_type'] = 'length'
                converted['min_length'] = jcr.get('min_length')
                converted['max_length'] = jcr.get('max_length')
            else:
                converted['rule_type'] = 'json'
            col_rules.append(converted)

        # Sort by source priority, then by quality score
        col_rules.sort(key=lambda x: (x.get('source_priority', 999), -x.get('quality_score', 0)))

        merged[col] = col_rules

    return merged
