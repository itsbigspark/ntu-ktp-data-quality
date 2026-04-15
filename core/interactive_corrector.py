"""
Interactive Correction Interface

Allows users to review and accept/reject correction suggestions.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import pandas as pd
from core.correction_suggester import CorrectionSuggester, CorrectionSuggestion, format_suggestion_for_display


class InteractiveCorrector:
    """
    Interactive correction workflow with user approval.
    """

    def __init__(self, df: pd.DataFrame, validation_report: pd.DataFrame):
        """
        Initialize interactive corrector.

        Args:
            df: DataFrame to correct
            validation_report: Validation issues from enhanced_validator
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.validation_report = validation_report
        self.suggester = CorrectionSuggester(df)

        self.corrections_made = []
        self.corrections_skipped = []

    def correct_interactively(
        self,
        max_issues: Optional[int] = None,
        auto_apply_high_confidence: bool = False,
        confidence_threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Interactively correct issues with user approval.

        Args:
            max_issues: Maximum issues to process (None = all)
            auto_apply_high_confidence: Auto-apply suggestions above threshold
            confidence_threshold: Threshold for auto-application

        Returns:
            Corrected DataFrame
        """
        issues = self.validation_report.copy()

        if max_issues:
            issues = issues.head(max_issues)

        total_issues = len(issues)

        print("=" * 80)
        print(f"Interactive Correction - {total_issues} Issues Found")
        print("=" * 80)
        print()

        for idx, (_, issue) in enumerate(issues.iterrows(), 1):
            print(f"\nIssue {idx} of {total_issues}:")
            print("─" * 80)

            self._process_issue(
                issue,
                auto_apply_high_confidence,
                confidence_threshold
            )

        self._show_summary()

        return self.df

    def _process_issue(
        self,
        issue: pd.Series,
        auto_apply: bool,
        threshold: float
    ):
        """Process a single issue"""

        row_id = issue['row_id']
        column = issue['column']
        current_value = issue.get('value')
        issue_type = issue['issue']
        explanation = issue.get('explanation', issue.get('detail', ''))

        print(f"Row: {row_id}")
        print(f"Column: {column}")
        print(f"Current Value: \"{current_value}\"")
        print(f"\nProblem: {issue_type}")
        print(f"Explanation: {explanation}")

        # Get suggestions
        suggestions = self.suggester.get_all_suggestions(
            value=current_value,
            column=column,
            issue_type=self._map_issue_to_type(column, issue_type),
            top_k=3
        )

        if not suggestions:
            print("\n⚠ No automatic suggestions available.")
            print(f"Actions: [M] Manual correction  [S] Skip")

            choice = self._get_user_choice(['M', 'S'])

            if choice == 'M':
                self._manual_correction(row_id, column, current_value)
            else:
                self._skip_correction(row_id, column, current_value)
            return

        # Display suggestions
        print("\nSUGGESTED CORRECTIONS (ranked by confidence):\n")

        for rank, sugg in enumerate(suggestions, 1):
            print(format_suggestion_for_display(sugg, rank))

        # Auto-apply if high confidence
        if auto_apply and suggestions[0].confidence >= threshold:
            print(f"\n✓ Auto-applying suggestion #1 (confidence {suggestions[0].confidence:.0%} >= {threshold:.0%})")
            self._apply_correction(row_id, column, current_value, suggestions[0].suggested_value, suggestions[0])
            return

        # User choice
        print("\nActions:")
        print("  [1-" + str(len(suggestions)) + "] Accept suggestion #")
        print("  [M] Manual correction (type your own)")
        print("  [S] Skip (leave as-is)")
        print("  [A] Apply to all similar issues")

        valid_choices = [str(i) for i in range(1, len(suggestions) + 1)] + ['M', 'S', 'A']
        choice = self._get_user_choice(valid_choices)

        if choice == 'M':
            self._manual_correction(row_id, column, current_value)
        elif choice == 'S':
            self._skip_correction(row_id, column, current_value)
        elif choice == 'A':
            self._apply_to_all_similar(issue, suggestions[0])
        else:
            # Accept suggestion
            suggestion_idx = int(choice) - 1
            selected_suggestion = suggestions[suggestion_idx]
            self._apply_correction(
                row_id, column, current_value,
                selected_suggestion.suggested_value,
                selected_suggestion
            )

    def _get_user_choice(self, valid_choices: List[str]) -> str:
        """Get user input with validation"""
        while True:
            try:
                choice = input("\nYour choice: ").strip().upper()
                if choice in valid_choices:
                    return choice
                print(f"Invalid choice. Please choose from: {', '.join(valid_choices)}")
            except (EOFError, KeyboardInterrupt):
                print("\n\nInterrupted by user.")
                return 'S'

    def _manual_correction(self, row_id: int, column: str, current_value: Any):
        """Handle manual correction"""
        print(f"\nCurrent value: \"{current_value}\"")
        new_value = input("Enter corrected value: ").strip()

        if new_value:
            self._apply_correction(row_id, column, current_value, new_value, None)
        else:
            print("No value entered, skipping...")
            self._skip_correction(row_id, column, current_value)

    def _apply_correction(
        self,
        row_id: int,
        column: str,
        old_value: Any,
        new_value: Any,
        suggestion: Optional[CorrectionSuggestion]
    ):
        """Apply a correction"""
        self.df.loc[row_id, column] = new_value

        self.corrections_made.append({
            'row_id': row_id,
            'column': column,
            'old_value': old_value,
            'new_value': new_value,
            'method': suggestion.method if suggestion else 'manual',
            'confidence': suggestion.confidence if suggestion else 1.0
        })

        print(f"✓ Corrected: \"{old_value}\" → \"{new_value}\"")

    def _skip_correction(self, row_id: int, column: str, value: Any):
        """Skip a correction"""
        self.corrections_skipped.append({
            'row_id': row_id,
            'column': column,
            'value': value
        })

        print(f"○ Skipped (left as-is)")

    def _apply_to_all_similar(self, issue: pd.Series, suggestion: CorrectionSuggestion):
        """Apply correction to all similar issues"""
        column = issue['column']
        old_value = issue.get('value')

        # Find all rows with same value in same column
        similar_rows = self.df[self.df[column] == old_value].index

        print(f"\nFound {len(similar_rows)} rows with same value.")
        confirm = input(f"Apply correction to all {len(similar_rows)} rows? [Y/N]: ").strip().upper()

        if confirm == 'Y':
            for row_id in similar_rows:
                self._apply_correction(row_id, column, old_value, suggestion.suggested_value, suggestion)
            print(f"✓ Applied to {len(similar_rows)} rows")
        else:
            print("○ Cancelled")

    def _map_issue_to_type(self, column: str, issue: str) -> Optional[str]:
        """Map validation issue type to pattern type"""
        if 'email' in column.lower():
            return 'email'
        elif 'phone' in column.lower():
            return 'phone'
        elif 'postcode' in column.lower() or 'postal' in column.lower():
            return 'postcode_uk'
        elif 'date' in column.lower() or issue == 'invalid date':
            return 'date'
        return None

    def _show_summary(self):
        """Show correction summary"""
        print("\n" + "=" * 80)
        print("CORRECTION SUMMARY")
        print("=" * 80)
        print(f"Total corrections made: {len(self.corrections_made)}")
        print(f"Total skipped: {len(self.corrections_skipped)}")

        if self.corrections_made:
            print("\nCorrections by method:")
            methods = {}
            for corr in self.corrections_made:
                method = corr['method']
                methods[method] = methods.get(method, 0) + 1

            for method, count in methods.items():
                print(f"  {method}: {count}")

            print("\nChanges made:")
            for corr in self.corrections_made[:10]:  # Show first 10
                print(f"  Row {corr['row_id']}, {corr['column']}: "
                      f"\"{corr['old_value']}\" → \"{corr['new_value']}\"")

            if len(self.corrections_made) > 10:
                print(f"  ... and {len(self.corrections_made) - 10} more")

    def export_corrections_log(self, output_path: str):
        """Export log of all corrections made"""
        corrections_df = pd.DataFrame(self.corrections_made)
        corrections_df.to_csv(output_path, index=False)
        print(f"\n✓ Corrections log saved to: {output_path}")

    def get_corrected_dataframe(self) -> pd.DataFrame:
        """Get the corrected DataFrame"""
        return self.df


# Convenience function
def correct_data_interactively(
    df: pd.DataFrame,
    validation_report: pd.DataFrame,
    max_issues: Optional[int] = None,
    auto_high_confidence: bool = False
) -> pd.DataFrame:
    """
    Quick interactive correction.

    Args:
        df: DataFrame to correct
        validation_report: Validation issues
        max_issues: Max issues to process
        auto_high_confidence: Auto-apply 95%+ confidence suggestions

    Returns:
        Corrected DataFrame
    """
    corrector = InteractiveCorrector(df, validation_report)

    corrected_df = corrector.correct_interactively(
        max_issues=max_issues,
        auto_apply_high_confidence=auto_high_confidence
    )

    return corrected_df
