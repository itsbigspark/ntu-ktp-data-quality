# ==========================================================
# core/rule_ui_helpers.py
# UI Helper Functions for Interactive Rules Tab
# ==========================================================

import streamlit as st
from typing import Dict, Any, List
import re


# ==========================================================
# THRESHOLD DESCRIPTIONS
# ==========================================================
THRESHOLD_INFO = {
    'MAJORITY_COVERAGE_STRICT': {
        'name': 'Numeric Coverage (Strict)',
        'description': '% of values that must be numeric before marking column as "number" type',
        'current_help': 'For clean data with < 5% nulls',
        'impact': 'Higher → Stricter (fewer false positives)\nLower → Permissive (more false positives)',
        'example': '95% → Allow only 5% bad data\n80% → Allow 20% bad data',
        'recommendation': 'Use 95% for clean datasets, 80% for moderate quality',
        'range': (0.5, 1.0),
        'default': 0.95,
    },
    'MAJORITY_COVERAGE_MEDIUM': {
        'name': 'Numeric Coverage (Medium)',
        'description': '% of values that must be numeric for moderate quality data',
        'current_help': 'For data with 5-20% nulls',
        'impact': 'Balanced threshold for typical datasets',
        'example': '80% is standard for most business data',
        'recommendation': 'Default for most use cases',
        'range': (0.5, 1.0),
        'default': 0.80,
    },
    'MAJORITY_COVERAGE_LOOSE': {
        'name': 'Numeric Coverage (Loose)',
        'description': '% of values that must be numeric for dirty data',
        'current_help': 'For data with > 20% nulls',
        'impact': 'Most permissive - use for very messy data',
        'example': '60% → Allow 40% bad data',
        'recommendation': 'Only for very dirty datasets',
        'range': (0.3, 0.9),
        'default': 0.60,
    },
    'MIN_NUMERIC_SAMPLES': {
        'name': 'Min Numeric Samples',
        'description': 'Minimum number of valid numeric values needed before creating numeric rule',
        'current_help': 'Prevents rules from tiny samples',
        'impact': 'Higher → More reliable rules, but might miss columns with few values',
        'example': '10 samples → Need at least 10 valid numbers',
        'recommendation': 'Keep at 10 for most cases',
        'range': (5, 100),
        'default': 10,
    },
    'MIN_SAMPLES_FOR_REGEX': {
        'name': 'Min Samples for Patterns',
        'description': 'Minimum rows needed before generating regex patterns',
        'current_help': 'CRITICAL: Was 1 (broken!), now 20 (much better)',
        'impact': 'Higher → Fewer but meaningful patterns\nLower → More patterns but potentially overfitted',
        'example': 'With 1: Overfitted patterns matching only 1 row\nWith 20: Patterns represent actual structure',
        'recommendation': 'Small datasets (<100): 10\nMedium (100-1K): 20\nLarge (>1K): 50',
        'range': (5, 100),
        'default': 20,
    },
    'MIN_PATTERN_MATCHES': {
        'name': 'Min Pattern Matches',
        'description': 'Pattern must match at least this many rows to be kept',
        'current_help': 'Works with coverage % to filter weak patterns',
        'impact': 'Prevents keeping patterns that only match a few rows',
        'example': '10 → Pattern must match at least 10 rows',
        'recommendation': 'Keep at 10',
        'range': (3, 50),
        'default': 10,
    },
    'MIN_REGEX_COVERAGE': {
        'name': 'Min Pattern Coverage',
        'description': 'Minimum % of values a pattern must match to be kept',
        'current_help': 'Was 5% (too low), now 10% (better)',
        'impact': 'Higher → Fewer, higher-quality patterns\nLower → More patterns, potentially noisy',
        'example': '10% → Pattern must match at least 10% of data\n5% → Pattern could match only 5% (often noise)',
        'recommendation': 'Clean data: 20%\nModerate: 10%\nDirty: 5%',
        'range': (0.05, 0.50),
        'default': 0.10,
    },
    'MAX_REGEX_PATTERNS': {
        'name': 'Max Patterns Per Column',
        'description': 'Maximum number of regex patterns to keep for each column',
        'current_help': 'Was 100 (way too high!), now 10 (reasonable)',
        'impact': 'Lower → Forces quality over quantity\nHigher → Might keep noise',
        'example': '10 patterns → Manageable, meaningful\n100 patterns → Column has no structure!',
        'recommendation': 'Ideal: 5\nMax: 10',
        'range': (3, 25),
        'default': 10,
    },
    'LOW_CARDINALITY_MAX_ABS': {
        'name': 'Max Unique Values (Categorical)',
        'description': 'Max unique values before treating column as categorical',
        'current_help': 'Was 25 (too low for states/countries), now 100',
        'impact': 'Determines what gets "allowed_values" list',
        'example': 'US States (50) → Need at least 50\nCountries (195) → Need at least 200',
        'recommendation': 'Standard: 100\nWith countries: 250',
        'range': (10, 500),
        'default': 100,
    },
    'LOW_CARDINALITY_MAX_RATIO': {
        'name': 'Max Unique Ratio (Categorical)',
        'description': 'Max % of unique values for categorical columns',
        'current_help': 'NEW: Relative threshold works with absolute',
        'impact': 'Prevents marking high-cardinality columns as categorical',
        'example': '5% → In 1000 rows, max 50 unique = categorical\n10% → In 1000 rows, max 100 unique = categorical',
        'recommendation': 'Keep at 5% for most cases',
        'range': (0.01, 0.20),
        'default': 0.05,
    },
    'UNIQUE_RATIO_STRICT': {
        'name': 'Uniqueness Threshold (Strict)',
        'description': 'For ID-like columns - require 100% unique',
        'current_help': 'No duplicates allowed for IDs',
        'impact': 'Enforces strict uniqueness for ID columns',
        'example': '100% → Zero duplicates allowed',
        'recommendation': 'Always 100% for IDs',
        'range': (0.95, 1.0),
        'default': 1.00,
    },
    'UNIQUE_RATIO_MEDIUM': {
        'name': 'Uniqueness Threshold (Medium)',
        'description': 'Allow minor errors - 98% unique',
        'current_help': 'Allows 2% duplicates (likely data errors)',
        'impact': 'Catches columns that should be unique but have few duplicates',
        'example': '98% → Allow 2% duplicates (20 in 1000 rows)',
        'recommendation': 'Good for email, username columns',
        'range': (0.90, 1.0),
        'default': 0.98,
    },
    'UNIQUE_RATIO_LOOSE': {
        'name': 'Uniqueness Threshold (Loose)',
        'description': 'Allow 5% duplicates',
        'current_help': 'Original threshold - most permissive',
        'impact': 'Fallback for detecting uniqueness',
        'example': '95% → Allow 5% duplicates',
        'recommendation': 'Use as minimum threshold',
        'range': (0.80, 0.99),
        'default': 0.95,
    },
}


def get_source_badge(source: str) -> str:
    """Get colored badge HTML for rule source"""
    badges = {
        'json': '<span style="background-color: #4169E1; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">🔵 JSON</span>',
        'reference': '<span style="background-color: #32CD32; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">🟢 Reference</span>',
        'inferred': '<span style="background-color: #FFD700; color: black; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">🟡 Inferred</span>',
    }
    return badges.get(source.lower(), source)


def get_quality_badge(quality_score: float) -> str:
    """Get colored badge HTML for quality score"""
    if quality_score >= 90:
        return f'<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">✅ {quality_score:.0f}%</span>'
    elif quality_score >= 70:
        return f'<span style="background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">⚠️ {quality_score:.0f}%</span>'
    else:
        return f'<span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">❌ {quality_score:.0f}%</span>'


def get_coverage_bar(coverage: float, width: int = 200) -> str:
    """Get HTML progress bar for coverage"""
    percentage = coverage * 100
    color = '#28a745' if coverage >= 0.80 else '#ffc107' if coverage >= 0.50 else '#dc3545'

    return f'''
    <div style="width: {width}px; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; height: 20px; text-align: center; color: white; font-size: 0.75em; line-height: 20px; font-weight: bold;">
            {percentage:.1f}%
        </div>
    </div>
    '''


def format_rule_summary(rule: Dict[str, Any]) -> str:
    """Format rule data as readable summary"""
    rule_type = rule.get('rule_type', '')
    rule_data = rule.get('rule_data', {})

    if rule_type == 'type_numeric':
        return f"**Type:** Number | **Range:** {rule_data.get('min', 'N/A')} - {rule_data.get('max', 'N/A')}"

    elif rule_type == 'type_date':
        return f"**Type:** Date | **Range:** {rule_data.get('min_date', 'N/A')} to {rule_data.get('max_date', 'N/A')}"

    elif rule_type == 'categorical':
        unique_count = rule_data.get('unique_count', 0)
        values = rule_data.get('allowed_values', [])
        preview = ', '.join([str(v) for v in values[:5]])
        if len(values) > 5:
            preview += f'... (+{len(values)-5} more)'
        return f"**Categorical:** {unique_count} unique values | **Values:** {preview}"

    elif rule_type == 'uniqueness':
        dup_count = rule_data.get('duplicate_count', 0)
        return f"**Uniqueness:** Required | **Duplicates:** {dup_count}"

    elif rule_type == 'regex':
        pattern = rule_data.get('regex', '')
        # Truncate long patterns
        if len(pattern) > 80:
            pattern = pattern[:77] + '...'
        return f"**Pattern:** `{pattern}`"

    elif rule_type == 'required':
        null_count = rule_data.get('null_count', 0)
        return f"**Required Field:** {null_count} nulls"

    else:
        return str(rule_data)


def test_regex_pattern(pattern: str, test_values: List[str]) -> Dict[str, List[str]]:
    """Test regex pattern against values"""
    matches = []
    failures = []

    try:
        regex = re.compile(pattern)
        for val in test_values:
            if regex.fullmatch(str(val)):
                matches.append(val)
            else:
                failures.append(val)
    except Exception as e:
        return {'error': str(e), 'matches': [], 'failures': []}

    return {'matches': matches, 'failures': failures}
