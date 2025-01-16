import unittest
import pandas as pd
from data_validator.cleaning import create_cleaning_rules, apply_cleaning_rules

class TestCleaning(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.unclean_data = pd.DataFrame({
            "Column1": ["123", "abc", "456"],
            "Column2": ["2023-01-01", "INVALID", "2023-01-02"]
        })
        self.patterns = {"Column1": r"^\d+$", "Column2": r"^\d{4}-\d{2}-\d{2}$"}
        self.metrics = {"Column1": {"valid_percentage": 80}, "Column2": {"valid_percentage": 70}}

    def test_create_cleaning_rules(self):
        cleaning_rules = create_cleaning_rules(self.patterns, self.metrics)
        self.assertIn("Column1", cleaning_rules)
        self.assertIn("Column2", cleaning_rules)

    def test_apply_cleaning_rules(self):
        cleaning_rules = create_cleaning_rules(self.patterns, self.metrics)
        cleaned_data, highlighted_issues = apply_cleaning_rules(self.unclean_data, cleaning_rules)

        # Check cleaned data
        self.assertEqual(len(cleaned_data), len(self.unclean_data))

        # Check for highlighted issues
        self.assertIn("Column2", highlighted_issues.columns)
        self.assertTrue((highlighted_issues["Column2"] == "INVALID").any())

if __name__ == "__main__":
    unittest.main()


