"""
Correction Suggestion System

Generates ranked correction suggestions with confidence scores.
Users can accept, reject, or manually override suggestions.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import re
import pandas as pd
from difflib import SequenceMatcher
from collections import Counter
import numpy as np


class CorrectionSuggestion:
    """A single correction suggestion with confidence score"""

    def __init__(
        self,
        original_value: Any,
        suggested_value: Any,
        confidence: float,
        method: str,
        reasoning: str
    ):
        self.original_value = original_value
        self.suggested_value = suggested_value
        self.confidence = confidence  # 0.0 to 1.0
        self.method = method
        self.reasoning = reasoning

    def __repr__(self):
        return (f"CorrectionSuggestion('{self.original_value}' → "
                f"'{self.suggested_value}', {self.confidence:.0%}, {self.method})")


class CorrectionSuggester:
    """
    Generates correction suggestions for data quality issues.

    Features:
    - Multiple correction methods (spell check, pattern matching, fuzzy matching)
    - Confidence scoring based on multiple factors
    - Ranked suggestions
    - Reasoning for each suggestion
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize suggester.

        Args:
            df: Optional DataFrame to learn patterns from
        """
        self.df = df
        self._build_vocabulary()
        self._build_pattern_library()

    def _build_vocabulary(self):
        """Build vocabulary from DataFrame for context-aware suggestions"""
        self.vocabulary = {}

        if self.df is not None:
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Count word frequency
                    words = []
                    for val in self.df[col].dropna():
                        if isinstance(val, str):
                            words.extend(val.split())

                    self.vocabulary[col] = Counter(words)

    def _build_pattern_library(self):
        """Common patterns for different data types"""
        self.patterns = {
            'email': {
                'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'common_fixes': {
                    'missing_@': lambda x: x.replace(' at ', '@'),
                    'missing_dot': lambda x: x + '.com' if '@' in x and '.' not in x.split('@')[1] else x
                }
            },
            'phone': {
                'regex': r'^\d{10}$|^\d{3}-\d{3}-\d{4}$',
                'common_fixes': {
                    'remove_spaces': lambda x: x.replace(' ', '').replace('-', ''),
                    'add_dashes': lambda x: f"{x[:3]}-{x[3:6]}-{x[6:]}" if len(x) == 10 else x
                }
            },
            'postcode_uk': {
                'regex': r'^[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}$',
                'common_fixes': {
                    'add_space': lambda x: f"{x[:-3]} {x[-3:]}" if len(x) >= 5 and ' ' not in x else x,
                    'uppercase': lambda x: x.upper()
                }
            },
            'date': {
                'regex': r'^\d{4}-\d{2}-\d{2}$',
                'common_fixes': {
                    'slash_to_dash': lambda x: x.replace('/', '-'),
                    'reorder': lambda x: self._reorder_date(x)
                }
            }
        }

    def _reorder_date(self, date_str: str) -> str:
        """Try to reorder date components"""
        # Simple heuristic: if day appears first, swap with year
        parts = re.split(r'[-/]', date_str)
        if len(parts) == 3:
            # If first part is small (likely day), assume DD-MM-YYYY
            if int(parts[0]) <= 31:
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return date_str

    def suggest_spell_correction(
        self,
        value: str,
        column: str,
        top_k: int = 3
    ) -> List[CorrectionSuggestion]:
        """
        Suggest spelling corrections using edit distance and word frequency.

        Args:
            value: Misspelled value
            column: Column name (for context)
            top_k: Number of suggestions to return

        Returns:
            List of correction suggestions
        """
        suggestions = []

        if column not in self.vocabulary:
            return suggestions

        # Get candidate words from vocabulary
        words = value.split()
        corrected_words = []

        for word in words:
            candidates = []

            # Find similar words in vocabulary
            for vocab_word, freq in self.vocabulary[column].items():
                if len(vocab_word) < 2:
                    continue

                # Calculate edit distance
                similarity = SequenceMatcher(None, word.lower(), vocab_word.lower()).ratio()

                if similarity > 0.7:  # At least 70% similar
                    # Confidence based on similarity and frequency
                    confidence = (similarity * 0.7) + (min(freq / 100, 1.0) * 0.3)

                    candidates.append((vocab_word, confidence, similarity, freq))

            # Sort by confidence
            candidates.sort(key=lambda x: x[1], reverse=True)

            if candidates:
                best_word, conf, sim, freq = candidates[0]
                corrected_words.append(best_word)

                # Only add if confidence is high enough
                if conf > 0.8:
                    corrected_value = ' '.join(corrected_words + words[len(corrected_words):])

                    reasoning = (f"Edit distance: {sim:.0%}, "
                               f"word appears {freq} times in dataset, "
                               f"common term in this column")

                    suggestions.append(CorrectionSuggestion(
                        original_value=value,
                        suggested_value=corrected_value,
                        confidence=conf,
                        method="spell_check",
                        reasoning=reasoning
                    ))
            else:
                corrected_words.append(word)

        return suggestions[:top_k]

    def suggest_pattern_fix(
        self,
        value: str,
        issue_type: str,
        column: str
    ) -> List[CorrectionSuggestion]:
        """
        Suggest fixes based on common patterns.

        Args:
            value: Current value
            issue_type: Type of issue (email, phone, postcode, date)
            column: Column name

        Returns:
            List of correction suggestions
        """
        suggestions = []

        if issue_type not in self.patterns:
            return suggestions

        pattern_info = self.patterns[issue_type]
        regex = pattern_info['regex']
        fixes = pattern_info['common_fixes']

        # Try each fix
        for fix_name, fix_func in fixes.items():
            try:
                fixed_value = fix_func(value)

                # Check if fix matches pattern
                if re.match(regex, str(fixed_value), re.IGNORECASE):
                    # Calculate confidence based on how much changed
                    original_len = len(str(value))
                    changes = sum(1 for a, b in zip(str(value), str(fixed_value)) if a != b)
                    confidence = 1.0 - (changes / max(original_len, 1)) * 0.5

                    reasoning = f"Applied '{fix_name}' fix, matches {issue_type} pattern"

                    suggestions.append(CorrectionSuggestion(
                        original_value=value,
                        suggested_value=fixed_value,
                        confidence=confidence,
                        method=f"pattern_{issue_type}",
                        reasoning=reasoning
                    ))
            except Exception:
                continue

        return suggestions

    def suggest_fuzzy_match(
        self,
        value: str,
        column: str,
        top_k: int = 3
    ) -> List[CorrectionSuggestion]:
        """
        Suggest corrections by fuzzy matching against existing values.

        Args:
            value: Current value
            column: Column name
            top_k: Number of suggestions

        Returns:
            List of correction suggestions
        """
        suggestions = []

        if self.df is None or column not in self.df.columns:
            return suggestions

        # Get unique values from column
        unique_values = self.df[column].dropna().unique()

        candidates = []
        for candidate in unique_values:
            if not isinstance(candidate, str):
                continue

            # Skip if exact match
            if candidate.lower() == str(value).lower():
                continue

            # Calculate similarity
            similarity = SequenceMatcher(None, str(value).lower(), candidate.lower()).ratio()

            if similarity > 0.75:  # At least 75% similar
                # Count frequency in dataset
                freq = (self.df[column] == candidate).sum()

                # Confidence based on similarity and frequency
                confidence = (similarity * 0.6) + (min(freq / len(self.df), 1.0) * 0.4)

                candidates.append((candidate, confidence, similarity, freq))

        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        for candidate, conf, sim, freq in candidates[:top_k]:
            reasoning = (f"Similarity: {sim:.0%}, "
                        f"appears {freq} times ({freq/len(self.df):.1%} of dataset), "
                        f"likely correct variant")

            suggestions.append(CorrectionSuggestion(
                original_value=value,
                suggested_value=candidate,
                confidence=conf,
                method="fuzzy_match",
                reasoning=reasoning
            ))

        return suggestions

    def suggest_format_standardization(
        self,
        value: str,
        column: str
    ) -> List[CorrectionSuggestion]:
        """
        Suggest format standardizations (whitespace, case, etc.)

        Args:
            value: Current value
            column: Column name

        Returns:
            List of correction suggestions
        """
        suggestions = []

        # Whitespace normalization (very high confidence)
        if value != value.strip():
            suggestions.append(CorrectionSuggestion(
                original_value=value,
                suggested_value=value.strip(),
                confidence=0.99,
                method="whitespace_trim",
                reasoning="Remove leading/trailing whitespace (safe operation)"
            ))

        # Multiple spaces to single space
        normalized = ' '.join(value.split())
        if normalized != value:
            suggestions.append(CorrectionSuggestion(
                original_value=value,
                suggested_value=normalized,
                confidence=0.95,
                method="whitespace_normalize",
                reasoning="Normalize multiple spaces to single space"
            ))

        # Title case (if column looks like a name)
        if 'name' in column.lower() and not value.istitle():
            suggestions.append(CorrectionSuggestion(
                original_value=value,
                suggested_value=value.title(),
                confidence=0.70,
                method="title_case",
                reasoning="Apply title case (column appears to be a name field)"
            ))

        # Uppercase (if column looks like a code)
        if any(x in column.lower() for x in ['code', 'id', 'postcode']):
            if not value.isupper() and value.isalpha():
                suggestions.append(CorrectionSuggestion(
                    original_value=value,
                    suggested_value=value.upper(),
                    confidence=0.75,
                    method="uppercase",
                    reasoning="Apply uppercase (column appears to be a code field)"
                ))

        return suggestions

    def get_all_suggestions(
        self,
        value: Any,
        column: str,
        issue_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[CorrectionSuggestion]:
        """
        Get all possible correction suggestions from all methods.

        Args:
            value: Current value
            column: Column name
            issue_type: Type of issue (email, phone, etc.)
            top_k: Maximum suggestions to return

        Returns:
            Ranked list of correction suggestions
        """
        all_suggestions = []

        if not isinstance(value, str):
            return all_suggestions

        # Collect suggestions from all methods
        all_suggestions.extend(self.suggest_spell_correction(value, column))
        all_suggestions.extend(self.suggest_fuzzy_match(value, column))
        all_suggestions.extend(self.suggest_format_standardization(value, column))

        if issue_type:
            all_suggestions.extend(self.suggest_pattern_fix(value, issue_type, column))

        # Remove duplicates (same suggestion from different methods)
        seen = {}
        unique_suggestions = []

        for sugg in all_suggestions:
            key = str(sugg.suggested_value)
            if key not in seen or sugg.confidence > seen[key].confidence:
                seen[key] = sugg

        unique_suggestions = list(seen.values())

        # Sort by confidence
        unique_suggestions.sort(key=lambda x: x.confidence, reverse=True)

        return unique_suggestions[:top_k]


def format_suggestion_for_display(
    suggestion: CorrectionSuggestion,
    rank: int
) -> str:
    """
    Format a suggestion for user-friendly display.

    Args:
        suggestion: The suggestion to format
        rank: Rank (1, 2, 3, ...)

    Returns:
        Formatted string
    """
    symbol = "✓" if suggestion.confidence > 0.8 else "○"

    output = f"  {rank}. {symbol} \"{suggestion.suggested_value}\"  "
    output += f"[{suggestion.confidence:.0%} confident]\n"
    output += f"     Reason: {suggestion.reasoning}\n"
    output += f"     Method: {suggestion.method}\n"

    return output
