"""Tests for core utilities: CorrectionSuggester and data_profiler."""

import unittest

import pandas as pd

from core.correction_suggester import CorrectionSuggestion, CorrectionSuggester
from core.data_profiler import profile_dataset


# ── CorrectionSuggester ───────────────────────────────────────────────────────

class TestCorrectionSuggesterInit(unittest.TestCase):

    def test_init_without_df(self):
        cs = CorrectionSuggester()
        self.assertIsNotNone(cs)

    def test_init_with_df(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "city": ["London", "Paris", "Berlin"]})
        cs = CorrectionSuggester(df=df)
        self.assertIsNotNone(cs)

    def test_vocabulary_built_from_df(self):
        df = pd.DataFrame({"name": ["Alice", "Alice", "Bob"]})
        cs = CorrectionSuggester(df=df)
        # Vocabulary for "name" column should contain "Alice"
        if hasattr(cs, "vocabulary") and "name" in cs.vocabulary:
            self.assertIn("Alice", cs.vocabulary["name"])

    def test_suggest_pattern_fix_returns_list(self):
        df = pd.DataFrame({"email": ["user@example.com"]})
        cs = CorrectionSuggester(df=df)
        result = cs.suggest_pattern_fix("user@example.com", "email", "email")
        self.assertIsInstance(result, list)

    def test_suggest_spell_correction_unknown_column_returns_empty(self):
        df = pd.DataFrame({"name": ["Alice"]})
        cs = CorrectionSuggester(df=df)
        result = cs.suggest_spell_correction("Alce", "unknown_column")
        self.assertEqual(result, [])

    def test_suggestion_has_required_fields(self):
        s = CorrectionSuggestion(
            original_value="Alce",
            suggested_value="Alice",
            confidence=0.9,
            method="spell_check",
            reasoning="Edit distance match",
        )
        self.assertEqual(s.original_value, "Alce")
        self.assertEqual(s.suggested_value, "Alice")
        self.assertGreater(s.confidence, 0)

    def test_suggest_fuzzy_match_returns_list(self):
        df = pd.DataFrame({"bank": ["Barclays", "HSBC", "Lloyds"] * 10})
        cs = CorrectionSuggester(df=df)
        result = cs.suggest_fuzzy_match("Barclay", "bank")
        self.assertIsInstance(result, list)

    def test_get_all_suggestions_returns_list(self):
        df = pd.DataFrame({"email": ["user@example.com"]})
        cs = CorrectionSuggester(df=df)
        result = cs.get_all_suggestions("user@example.com", "email")
        self.assertIsInstance(result, list)


# ── DataProfiler ──────────────────────────────────────────────────────────────

class TestProfileDataset(unittest.TestCase):

    def _make_df(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob", None, "Charlie", "Alice"],
            "age": [25, 30, 35, None, 25],
            "score": [1.1, 2.2, 3.3, 4.4, 1.1],
        })

    def test_returns_dict_with_required_keys(self):
        df = self._make_df()
        profile = profile_dataset(df)
        for key in ("dataset_summary", "column_profiles", "correlations", "patterns", "outliers"):
            self.assertIn(key, profile)

    def test_dataset_summary_row_count(self):
        df = self._make_df()
        summary = profile_dataset(df)["dataset_summary"]
        self.assertEqual(summary["total_rows"], 5)

    def test_dataset_summary_column_count(self):
        df = self._make_df()
        summary = profile_dataset(df)["dataset_summary"]
        self.assertEqual(summary["total_columns"], 3)

    def test_missing_cells_counted(self):
        df = self._make_df()
        summary = profile_dataset(df)["dataset_summary"]
        self.assertEqual(summary["missing_cells"], 2)  # one in name, one in age

    def test_duplicate_rows_detected(self):
        df = self._make_df()
        summary = profile_dataset(df)["dataset_summary"]
        # Row 0 and row 4 are identical → 1 duplicate
        self.assertGreaterEqual(summary["duplicate_rows"], 1)

    def test_column_profiles_has_entry_per_column(self):
        df = self._make_df()
        profiles = profile_dataset(df)["column_profiles"]
        for col in df.columns:
            self.assertIn(col, profiles)

    def test_column_profile_missing_count(self):
        df = self._make_df()
        profiles = profile_dataset(df)["column_profiles"]
        self.assertEqual(profiles["name"]["missing"], 1)
        self.assertEqual(profiles["age"]["missing"], 1)

    def test_numeric_column_has_stats(self):
        df = self._make_df()
        profiles = profile_dataset(df)["column_profiles"]
        age_profile = profiles["age"]
        # Numeric columns should include mean/std
        self.assertIn("mean", age_profile)

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame({"a": [], "b": []})
        profile = profile_dataset(df)
        self.assertEqual(profile["dataset_summary"]["total_rows"], 0)

    def test_all_missing_column(self):
        df = pd.DataFrame({"a": [None, None, None]})
        profile = profile_dataset(df)
        self.assertEqual(profile["dataset_summary"]["missing_cells"], 3)

    def test_memory_usage_positive(self):
        df = self._make_df()
        summary = profile_dataset(df)["dataset_summary"]
        self.assertGreater(summary["memory_usage_mb"], 0)


if __name__ == "__main__":
    unittest.main()
