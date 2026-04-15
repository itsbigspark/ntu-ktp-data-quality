# core/validation_ui_helpers.py
"""
UI helper functions for the Validate & Fix tab.
Provides visual highlighting, unified validation, and corpus integration.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re


def style_dataframe_with_issues(df: pd.DataFrame, issues_report: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Style a dataframe to highlight cells with issues in red, valid cells in green.

    Args:
        df: Original dataframe
        issues_report: DataFrame with columns [row_id, column, issue_type, ...]

    Returns:
        Styled dataframe with color-coded cells
    """
    # Create a boolean mask of same shape as df
    # True = has issue (red), False = valid (green)
    issue_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Mark cells with issues
    for _, issue in issues_report.iterrows():
        row_id = issue.get('row_id')
        col = issue.get('column')

        if row_id is not None and col in df.columns:
            try:
                issue_mask.at[row_id, col] = True
            except (KeyError, IndexError):
                continue

    # Define styling function
    def highlight_cell(val, row_idx, col_name):
        if issue_mask.at[row_idx, col_name]:
            return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'  # Red
        else:
            return 'background-color: #ccffcc; color: #006600'  # Green

    # Apply styling
    styled = df.style.apply(
        lambda x: [highlight_cell(val, idx, x.name) for idx, val in x.items()],
        axis=0
    )

    return styled


def create_severity_badge(severity: str) -> str:
    """Create HTML badge for issue severity."""
    badges = {
        'critical': '<span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">🔴 CRITICAL</span>',
        'warning': '<span style="background-color: #ffc107; color: black; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">🟡 WARNING</span>',
        'info': '<span style="background-color: #0dcaf0; color: black; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold;">🔵 INFO</span>',
    }
    return badges.get(severity.lower(), severity)


def create_source_badge(source: str) -> str:
    """Create HTML badge for validation source."""
    badges = {
        'rules': '<span style="background-color: #0d6efd; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">📋 Rules</span>',
        'corpus': '<span style="background-color: #198754; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">📚 Corpus</span>',
        'ml': '<span style="background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">🤖 ML</span>',
        'api': '<span style="background-color: #6f42c1; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">🌐 API</span>',
    }
    return badges.get(source.lower(), source)


def create_confidence_badge(confidence: float) -> str:
    """Create HTML badge for confidence score."""
    if confidence >= 0.90:
        color = '#198754'  # Green
        icon = '✅'
        label = 'High'
    elif confidence >= 0.70:
        color = '#ffc107'  # Yellow
        icon = '⚠️'
        label = 'Medium'
    else:
        color = '#dc3545'  # Red
        icon = '❌'
        label = 'Low'

    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px;">{icon} {label} ({confidence:.0%})</span>'


def auto_detect_corpus_mappings(df: pd.DataFrame, corpus_list: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """
    Auto-detect which columns should use which corpus based on naming patterns.

    Args:
        df: The dataframe with columns to map
        corpus_list: List of dicts with keys: name, type

    Returns:
        Dict mapping column names to corpus config
        {
            "email": {
                "corpus": "email_domains",
                "confidence": 0.95,
                "type": "alias"
            },
            ...
        }
    """
    mappings = {}

    # Define pattern matching rules
    patterns = {
        'email': [r'email', r'e_?mail', r'contact', r'.*@.*'],
        'postcode': [r'post_?code', r'postal', r'zip', r'post.*code'],
        'address': [r'address', r'addr', r'street', r'location'],
        'company': [r'company', r'organisation', r'organization', r'business', r'firm'],
        'phone': [r'phone', r'telephone', r'tel', r'mobile', r'contact'],
        'domain': [r'domain', r'website', r'url', r'site'],
        'name': [r'name', r'first.*name', r'last.*name', r'full.*name'],
    }

    for col in df.columns:
        col_lower = col.lower()

        for corpus in corpus_list:
            corpus_name = corpus.get('corpus_name', '').lower()
            corpus_type = corpus.get('corpus_type', 'alias')

            # Try to match corpus name to column name
            best_confidence = 0.0
            matched_pattern = None

            # Check each pattern category
            for category, pattern_list in patterns.items():
                # If corpus name contains category keyword
                if category in corpus_name:
                    for pattern in pattern_list:
                        if re.search(pattern, col_lower, re.IGNORECASE):
                            confidence = 0.95  # High confidence for pattern match
                            if confidence > best_confidence:
                                best_confidence = confidence
                                matched_pattern = category

            # Direct name matching (lower confidence)
            if corpus_name in col_lower or col_lower in corpus_name:
                confidence = 0.80
                if confidence > best_confidence:
                    best_confidence = confidence
                    matched_pattern = 'direct'

            # If we found a match, add to mappings
            if best_confidence > 0:
                # Only keep highest confidence mapping per column
                if col not in mappings or best_confidence > mappings[col]['confidence']:
                    mappings[col] = {
                        'corpus': corpus.get('corpus_name', ''),
                        'confidence': best_confidence,
                        'type': corpus_type,
                        'pattern': matched_pattern
                    }

    return mappings


def unify_validation_reports(
    rule_report: Optional[pd.DataFrame],
    corpus_report: Optional[pd.DataFrame],
    ml_report: Optional[pd.DataFrame],
    api_report: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge all validation reports into a single unified report.

    Standard columns:
    - row_id: int
    - column: str
    - issue_type: str
    - value: str (current value)
    - suggested_fix: str (proposed fix)
    - confidence: float (0.0-1.0)
    - severity: str (critical, warning, info)
    - source: str (rules, corpus, ml, api)
    - description: str
    """
    all_reports = []

    # Process rule report
    if rule_report is not None and len(rule_report) > 0:
        df = rule_report.copy()
        if 'source' not in df.columns:
            df['source'] = 'rules'
        if 'severity' not in df.columns:
            df['severity'] = 'warning'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.85  # Default confidence for rules
        all_reports.append(df)

    # Process corpus report
    if corpus_report is not None and len(corpus_report) > 0:
        df = corpus_report.copy()
        if 'source' not in df.columns:
            df['source'] = 'corpus'
        if 'severity' not in df.columns:
            df['severity'] = 'info'
        all_reports.append(df)

    # Process ML report
    if ml_report is not None and len(ml_report) > 0:
        df = ml_report.copy()
        if 'source' not in df.columns:
            df['source'] = 'ml'
        if 'severity' not in df.columns:
            df['severity'] = 'warning'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.70  # ML often has lower confidence
        all_reports.append(df)

    # Process API report
    if api_report is not None and len(api_report) > 0:
        df = api_report.copy()
        if 'source' not in df.columns:
            df['source'] = 'api'
        if 'severity' not in df.columns:
            df['severity'] = 'info'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.95  # API validation is usually reliable
        all_reports.append(df)

    # Merge all reports
    if not all_reports:
        # Return empty dataframe with standard schema
        return pd.DataFrame(columns=[
            'row_id', 'column', 'issue_type', 'value', 'suggested_fix',
            'confidence', 'severity', 'source', 'description'
        ])

    unified = pd.concat(all_reports, ignore_index=True)

    # Ensure standard columns exist
    for col in ['row_id', 'column', 'issue_type', 'value', 'suggested_fix',
                'confidence', 'severity', 'source', 'description']:
        if col not in unified.columns:
            unified[col] = None

    # Fill missing values
    unified['confidence'] = unified['confidence'].fillna(0.5)
    unified['severity'] = unified['severity'].fillna('info')
    unified['description'] = unified['description'].fillna('')

    # Sort by severity (critical first), then confidence (high first)
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    unified['_severity_rank'] = unified['severity'].map(lambda x: severity_order.get(x.lower(), 3))
    unified = unified.sort_values(['_severity_rank', 'confidence'], ascending=[True, False])
    unified = unified.drop('_severity_rank', axis=1)

    return unified


def apply_fixes_to_dataframe(
    df: pd.DataFrame,
    issues_report: pd.DataFrame,
    selected_fix_indices: List[int]
) -> pd.DataFrame:
    """
    Apply selected fixes from the issues report to a copy of the dataframe.

    Args:
        df: Original dataframe
        issues_report: Unified issues report
        selected_fix_indices: List of row indices from issues_report to apply

    Returns:
        Copy of df with fixes applied
    """
    df_fixed = df.copy()

    for idx in selected_fix_indices:
        if idx >= len(issues_report):
            continue

        issue = issues_report.iloc[idx]
        row_id = issue.get('row_id')
        col = issue.get('column')
        fix = issue.get('suggested_fix')

        # Skip if any required field is missing
        if row_id is None or col is None or pd.isna(fix):
            continue

        # Skip if column doesn't exist
        if col not in df_fixed.columns:
            continue

        # Apply the fix
        try:
            df_fixed.at[row_id, col] = fix
        except (KeyError, IndexError):
            # Row or column doesn't exist, skip
            continue

    return df_fixed


def get_validation_summary_stats(issues_report: pd.DataFrame, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics from validation results.

    Returns:
        Dict with metrics like total_issues, by_severity, by_source, by_column, etc.
    """
    if len(issues_report) == 0:
        return {
            'total_issues': 0,
            'total_cells': len(df) * len(df.columns),
            'affected_rows': 0,
            'affected_columns': 0,
            'by_severity': {},
            'by_source': {},
            'by_column': {},
            'data_quality_score': 100.0
        }

    total_cells = len(df) * len(df.columns)
    total_issues = len(issues_report)

    # Count unique rows and columns affected
    affected_rows = issues_report['row_id'].nunique() if 'row_id' in issues_report.columns else 0
    affected_columns = issues_report['column'].nunique() if 'column' in issues_report.columns else 0

    # Group by severity
    by_severity = issues_report['severity'].value_counts().to_dict() if 'severity' in issues_report.columns else {}

    # Group by source
    by_source = issues_report['source'].value_counts().to_dict() if 'source' in issues_report.columns else {}

    # Group by column
    by_column = issues_report['column'].value_counts().to_dict() if 'column' in issues_report.columns else {}

    # Calculate data quality score (simple: % of cells without issues)
    data_quality_score = ((total_cells - total_issues) / total_cells * 100) if total_cells > 0 else 0.0

    return {
        'total_issues': total_issues,
        'total_cells': total_cells,
        'affected_rows': affected_rows,
        'affected_columns': affected_columns,
        'by_severity': by_severity,
        'by_source': by_source,
        'by_column': by_column,
        'data_quality_score': data_quality_score
    }


def create_downloadable_styled_excel(df: pd.DataFrame, issues_report: pd.DataFrame) -> bytes:
    """
    Create an Excel file with conditional formatting (green=valid, red=issues).

    Args:
        df: Original dataframe
        issues_report: Issues report

    Returns:
        Bytes of Excel file
    """
    try:
        from io import BytesIO
        import openpyxl
        from openpyxl.styles import PatternFill

        # Create issue mask
        issue_cells = set()
        for _, issue in issues_report.iterrows():
            row_id = issue.get('row_id')
            col = issue.get('column')
            if row_id is not None and col in df.columns:
                col_idx = df.columns.get_loc(col)
                issue_cells.add((row_id, col_idx))

        # Write to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')

            # Get the worksheet
            worksheet = writer.sheets['Data']

            # Define fills
            red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')
            green_fill = PatternFill(start_color='CCFFCC', end_color='CCFFCC', fill_type='solid')

            # Apply conditional formatting
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    # Excel is 1-indexed, and we have headers
                    cell = worksheet.cell(row=row_idx + 2, column=col_idx + 1)

                    if (row_idx, col_idx) in issue_cells:
                        cell.fill = red_fill
                    else:
                        cell.fill = green_fill

        return output.getvalue()

    except ImportError:
        # Fallback to CSV if openpyxl not available
        return df.to_csv(index=False).encode('utf-8')
