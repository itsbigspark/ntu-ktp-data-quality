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


_STYLE_ISSUE = 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
_STYLE_VALID = 'background-color: #ccffcc; color: #006600'

# Above this many cells we skip per-cell tooltips (the HTML they generate
# freezes the browser on large datasets). Colouring still applies.
_MAX_TOOLTIP_CELLS = 60000


def style_dataframe_with_issues(
    df: pd.DataFrame,
    issues_report: pd.DataFrame,
    include_citation: bool = True,
    max_tooltip_cells: int = _MAX_TOOLTIP_CELLS,
) -> pd.io.formats.style.Styler:
    """
    Style a dataframe: cells with an issue are red, valid cells are green.

    When the grid is small enough, each flagged cell also gets a hover tooltip
    showing the issue type, its description, and the regulatory citation
    (e.g. "missing - value is missing | BCBS 239 Principle 4 (Completeness)").

    Args:
        df: Original dataframe.
        issues_report: issues with columns [row_id, column, issue|issue_type,
            description?, regulatory_citation?].
        include_citation: append the regulatory citation to the tooltip text.
        max_tooltip_cells: above this cell count, tooltips are skipped for
            performance (colouring is unaffected).

    Returns:
        A pandas Styler with colour coding (and tooltips when feasible).
    """
    # Boolean mask + per-cell tooltip text, both shaped like df.
    issue_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    n_cells = max(len(df) * max(len(df.columns), 1), 1)
    want_tooltips = n_cells <= max_tooltip_cells
    tooltips = (
        pd.DataFrame("", index=df.index, columns=df.columns) if want_tooltips else None
    )

    issue_attr = "issue_type" if "issue_type" in issues_report.columns else "issue"

    for _, issue in issues_report.iterrows():
        row_id = issue.get("row_id")
        col = issue.get("column")
        if row_id is None or col not in df.columns:
            continue
        try:
            issue_mask.at[row_id, col] = True
            if want_tooltips:
                itype = str(issue.get(issue_attr, "") or "")
                desc = str(issue.get("description", "") or "")
                parts = [p for p in (itype, desc) if p]
                text = " - ".join(parts) if parts else "issue"
                if include_citation:
                    cite = issue.get("regulatory_citation")
                    if cite is not None and str(cite) not in ("", "None", "nan"):
                        text = f"{text} | {cite}"
                # Keep the first issue's tooltip if a cell already has one.
                if not tooltips.at[row_id, col]:
                    tooltips.at[row_id, col] = text
        except (KeyError, IndexError):
            continue

    # Build a style DataFrame in one shot (far faster than per-cell python).
    style_df = issue_mask.applymap(lambda flagged: _STYLE_ISSUE if flagged else _STYLE_VALID)
    styled = df.style.apply(lambda _: style_df, axis=None)

    if want_tooltips and tooltips is not None and (tooltips.values != "").any():
        try:
            styled = styled.set_tooltips(tooltips)
        except Exception:
            pass  # tooltips are a nicety; never fail the render over them

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
