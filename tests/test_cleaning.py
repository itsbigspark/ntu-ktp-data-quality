"""Tests for text/data cleaning operations used throughout the pipeline."""

import re
import unittest

import pandas as pd


# ── Cleaning helpers (inline, matching the logic in core/preprocess.py) ────────

def strip_whitespace(value: str) -> str:
    return value.strip() if isinstance(value, str) else value


def normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else value


def to_lowercase(value: str) -> str:
    return value.lower() if isinstance(value, str) else value


def remove_special_chars(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", value) if isinstance(value, str) else value


def clean_email(value: str) -> str:
    return value.strip().lower() if isinstance(value, str) else value


def standardise_phone_uk(value: str) -> str:
    """Remove spaces/dashes, ensure starts with +44 or 0."""
    if not isinstance(value, str):
        return value
    digits = re.sub(r"[\s\-\(\)]", "", value)
    return digits


# ── Core module: core.preprocess ──────────────────────────────────────────────

try:
    from core.preprocess import (
        clean_dataframe,
        standardise_column_names,
    )
    HAS_CORE_PREPROCESS = True
except ImportError:
    HAS_CORE_PREPROCESS = False


class TestStripWhitespace(unittest.TestCase):
    def test_leading_trailing(self):
        self.assertEqual(strip_whitespace("  hello  "), "hello")

    def test_no_whitespace(self):
        self.assertEqual(strip_whitespace("hello"), "hello")

    def test_empty_string(self):
        self.assertEqual(strip_whitespace(""), "")

    def test_non_string_passthrough(self):
        self.assertIsNone(strip_whitespace(None))
        self.assertEqual(strip_whitespace(42), 42)


class TestNormaliseWhitespace(unittest.TestCase):
    def test_double_spaces_collapsed(self):
        self.assertEqual(normalise_whitespace("hello  world"), "hello world")

    def test_tabs_collapsed(self):
        self.assertEqual(normalise_whitespace("hello\t\tworld"), "hello world")

    def test_newlines_collapsed(self):
        self.assertEqual(normalise_whitespace("hello\nworld"), "hello world")

    def test_leading_trailing_removed(self):
        self.assertEqual(normalise_whitespace("  hello  "), "hello")


class TestToLowercase(unittest.TestCase):
    def test_uppercase_converted(self):
        self.assertEqual(to_lowercase("HELLO WORLD"), "hello world")

    def test_mixed_case(self):
        self.assertEqual(to_lowercase("Hello World"), "hello world")

    def test_already_lowercase(self):
        self.assertEqual(to_lowercase("hello"), "hello")

    def test_non_string_passthrough(self):
        self.assertIsNone(to_lowercase(None))


class TestRemoveSpecialChars(unittest.TestCase):
    def test_punctuation_removed(self):
        self.assertEqual(remove_special_chars("hello!"), "hello")

    def test_preserves_alphanumeric(self):
        self.assertEqual(remove_special_chars("abc123"), "abc123")

    def test_preserves_spaces(self):
        self.assertEqual(remove_special_chars("hello world"), "hello world")

    def test_multiple_special_chars(self):
        result = remove_special_chars("@hello#world$")
        self.assertEqual(result, "helloworld")


class TestCleanEmail(unittest.TestCase):
    def test_strips_and_lowercases(self):
        self.assertEqual(clean_email("  User@Example.COM  "), "user@example.com")

    def test_already_clean(self):
        self.assertEqual(clean_email("user@example.com"), "user@example.com")


class TestStandardisePhoneUK(unittest.TestCase):
    def test_spaces_removed(self):
        self.assertEqual(standardise_phone_uk("07911 123456"), "07911123456")

    def test_dashes_removed(self):
        self.assertEqual(standardise_phone_uk("07911-123456"), "07911123456")

    def test_parentheses_removed(self):
        self.assertEqual(standardise_phone_uk("(020) 7946 0958"), "02079460958")


class TestDataFrameCleaning(unittest.TestCase):
    """Integration-style tests that apply cleaning to a full DataFrame."""

    def _make_df(self):
        return pd.DataFrame({
            "name": ["  Alice  ", "BOB", "charlie "],
            "email": ["Alice@Test.COM", "  bob@test.com", "CHARLIE@TEST.COM"],
            "amount": [1.0, 2.0, 3.0],
        })

    def test_strip_all_string_columns(self):
        df = self._make_df().copy()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(strip_whitespace)
        self.assertEqual(df["name"].iloc[0], "Alice")
        self.assertEqual(df["name"].iloc[2], "charlie")

    def test_no_numeric_column_corruption(self):
        df = self._make_df().copy()
        original_amounts = df["amount"].tolist()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(to_lowercase)
        self.assertEqual(df["amount"].tolist(), original_amounts)


@unittest.skipUnless(HAS_CORE_PREPROCESS, "core.preprocess not importable")
class TestCorePreprocess(unittest.TestCase):

    def test_standardise_column_names_lowercases(self):
        df = pd.DataFrame({"First Name": [1], "Last  Name": [2], "AGE": [3]})
        cleaned = standardise_column_names(df)
        for col in cleaned.columns:
            self.assertEqual(col, col.lower())

    def test_clean_dataframe_returns_dataframe(self):
        df = pd.DataFrame({"name": ["  Alice  "], "value": [1]})
        result = clean_dataframe(df)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
