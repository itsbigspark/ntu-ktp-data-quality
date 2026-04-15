"""
Enhanced Validator with BERT Explanations

This module integrates BERT-based natural language explanations
into the existing validation pipeline, making data quality issues
understandable for non-technical users.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

from core.validator.validate import validate_df
from core.validator.discover import infer_rules_from_unclean
from core.bert_explainer import create_bert_explainer, BERTExplainer


class EnhancedValidator:
    """
    Validator with BERT-powered natural language explanations.

    Combines:
    - Existing validation logic (precise, rule-based)
    - BERT explanations (human-friendly, contextual)
    - Domain knowledge learning (gets smarter over time)
    """

    def __init__(self, use_bert: bool = True, memory_path: str = "data/bert_memory.db"):
        """
        Initialize enhanced validator.

        Args:
            use_bert: Whether to use BERT explanations (set False for faster processing)
            memory_path: Path to BERT knowledge base
        """
        self.use_bert = use_bert
        if self.use_bert:
            self.explainer = create_bert_explainer(memory_path)
        else:
            self.explainer = None

    def validate_with_explanations(
        self,
        df: pd.DataFrame,
        rules: Optional[Dict] = None,
        explain_all: bool = False,
        explain_sample: int = 0
    ) -> pd.DataFrame:
        """
        Validate DataFrame and add natural language explanations.

        Args:
            df: DataFrame to validate
            rules: Validation rules (auto-inferred if None)
            explain_all: Add explanations to all issues
            explain_sample: Add explanations to first N issues (0 = none)

        Returns:
            DataFrame with validation results + explanations column
        """
        # Infer rules if not provided
        if rules is None:
            rules = infer_rules_from_unclean(df)

        # Run standard validation
        validation_report = validate_df(df, rules)

        # Add explanations if BERT is enabled
        if self.use_bert and (explain_all or explain_sample > 0):
            explanations = []

            limit = len(validation_report) if explain_all else min(explain_sample, len(validation_report))

            for idx, (_, issue) in enumerate(validation_report.iterrows()):
                if idx < limit:
                    explanation = self.explainer.explain_validation_issue(
                        column=issue['column'],
                        value=issue.get('value'),
                        issue=issue['issue'],
                        detail=issue['detail'],
                        expected=issue.get('expected'),
                        row=issue.get('row_id')
                    )
                else:
                    explanation = None

                explanations.append(explanation)

            validation_report['explanation'] = explanations

        return validation_report

    def validate_and_export(
        self,
        df: pd.DataFrame,
        output_path: str,
        rules: Optional[Dict] = None,
        explain_critical_only: bool = True
    ) -> Dict[str, Any]:
        """
        Validate DataFrame and export results with explanations.

        Args:
            df: DataFrame to validate
            output_path: Path to save results CSV
            rules: Validation rules (auto-inferred if None)
            explain_critical_only: Only explain high-severity issues

        Returns:
            Dictionary with summary statistics
        """
        # Infer rules if not provided
        if rules is None:
            rules = infer_rules_from_unclean(df)

        # Run validation
        validation_report = validate_df(df, rules)

        # Add explanations
        if self.use_bert:
            if explain_critical_only:
                # Only explain high-severity issues
                critical_mask = validation_report['severity'] == 'high'
                critical_issues = validation_report[critical_mask]

                explanations = []
                for _, issue in validation_report.iterrows():
                    if issue['severity'] == 'high':
                        explanation = self.explainer.explain_validation_issue(
                            column=issue['column'],
                            value=issue.get('value'),
                            issue=issue['issue'],
                            detail=issue['detail'],
                            expected=issue.get('expected'),
                            row=issue.get('row_id')
                        )
                    else:
                        explanation = None
                    explanations.append(explanation)

                validation_report['explanation'] = explanations
            else:
                # Explain all issues
                validation_report = self.validate_with_explanations(
                    df, rules, explain_all=True
                )

        # Save to CSV
        validation_report.to_csv(output_path, index=False)

        # Generate summary
        summary = {
            'total_rows': len(df),
            'total_issues': len(validation_report),
            'issue_rate': len(validation_report) / len(df) * 100,
            'issues_by_severity': validation_report['severity'].value_counts().to_dict(),
            'issues_by_type': validation_report['issue'].value_counts().to_dict(),
            'issues_by_column': validation_report['column'].value_counts().to_dict(),
            'output_file': output_path
        }

        return summary

    def teach_domain_knowledge(
        self,
        terms: List[Dict[str, str]] = None,
        equivalences: List[Dict[str, Any]] = None
    ):
        """
        Teach BERT about domain-specific knowledge.

        Args:
            terms: List of {"term": "...", "full_form": "...", "meaning": "...", "category": "..."}
            equivalences: List of {"text1": "...", "text2": "...", "are_equivalent": bool, "domain": "..."}
        """
        if not self.use_bert:
            print("BERT is disabled. Enable it to teach domain knowledge.")
            return

        if terms:
            for term_info in terms:
                self.explainer.teach_term(
                    term=term_info['term'],
                    full_form=term_info.get('full_form'),
                    meaning=term_info.get('meaning'),
                    category=term_info.get('category', 'general')
                )

        if equivalences:
            for eq_info in equivalences:
                self.explainer.teach_equivalence(
                    text1=eq_info['text1'],
                    text2=eq_info['text2'],
                    are_equivalent=eq_info.get('are_equivalent', True),
                    domain=eq_info.get('domain')
                )

    def get_summary_report(self, validation_report: pd.DataFrame) -> str:
        """
        Generate human-readable summary report.

        Args:
            validation_report: DataFrame from validate_with_explanations

        Returns:
            Formatted summary text
        """
        total_issues = len(validation_report)

        if total_issues == 0:
            return "✓ No data quality issues found! The data looks clean."

        report = []
        report.append(f"Data Quality Report")
        report.append("=" * 80)
        report.append(f"\nTotal Issues: {total_issues}")

        # By severity
        if 'severity' in validation_report.columns:
            report.append("\nBy Severity:")
            for severity, count in validation_report['severity'].value_counts().items():
                pct = count / total_issues * 100
                report.append(f"  {severity.upper()}: {count} ({pct:.1f}%)")

        # By type
        report.append("\nTop Issue Types:")
        for issue_type, count in validation_report['issue'].value_counts().head(5).items():
            pct = count / total_issues * 100
            report.append(f"  {issue_type}: {count} ({pct:.1f}%)")

        # By column
        report.append("\nMost Problematic Columns:")
        for column, count in validation_report['column'].value_counts().head(5).items():
            pct = count / total_issues * 100
            report.append(f"  {column}: {count} ({pct:.1f}%)")

        # Sample explanations
        if 'explanation' in validation_report.columns:
            explained = validation_report[validation_report['explanation'].notna()]
            if len(explained) > 0:
                report.append("\nSample Issues with Explanations:")
                report.append("-" * 80)
                for idx, (_, issue) in enumerate(explained.head(3).iterrows(), 1):
                    report.append(f"\n{idx}. Row {issue.get('row_id', 'N/A')}, Column '{issue['column']}'")
                    report.append(f"   Value: {issue.get('value', 'N/A')}")
                    report.append(f"   Issue: {issue['issue']}")
                    report.append(f"   Explanation: {issue['explanation']}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# Convenience functions
def validate_dataframe(
    df: pd.DataFrame,
    rules: dict = None,
    use_bert: bool = True,
    explain_critical: bool = True,
    explain_all: bool = False,
    explain_sample: int = 0
) -> pd.DataFrame:
    """
    Quick validation with optional BERT explanations.

    Args:
        df: DataFrame to validate
        rules: Validation rules dict (if None, inferred automatically)
        use_bert: Use BERT for explanations
        explain_critical: Only explain high-severity issues
        explain_all: Explain all issues
        explain_sample: Explain first N issues only
    Returns:
        Validation report DataFrame
    """
    validator = EnhancedValidator(use_bert=use_bert)

    if not rules:
        rules = infer_rules_from_unclean(df)

    # Determine which rows get BERT explanations
    # explain_all overrides explain_critical; explain_sample limits to first N
    report = validate_df(df, rules, enable_typo_detection=True)

    if use_bert:
        explanations = []
        explained_count = 0
        for i, (_, issue) in enumerate(report.iterrows()):
            should_explain = (
                explain_all or
                (explain_critical and issue.get('severity') == 'high') or
                (explain_sample > 0 and explained_count < explain_sample)
            )
            if should_explain:
                try:
                    explanation = validator.explainer.explain_validation_issue(
                        column=issue['column'],
                        value=issue.get('value'),
                        issue=issue['issue'],
                        detail=issue['detail'],
                        expected=issue.get('expected')
                    )
                    explained_count += 1
                except Exception:
                    explanation = None
            else:
                explanation = None
            explanations.append(explanation)
        report['explanation'] = explanations

    return report
