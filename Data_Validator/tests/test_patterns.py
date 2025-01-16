import unittest
import pandas as pd
from data_validator.patterns import infer_regex_patterns

class TestPatterns(unittest.TestCase):
    def setUp(self):
        # Sample clean data
        self.clean_data = pd.DataFrame({
            "Column1": ["123", "456", "789"],
            "Column2": ["2023-01-01", "2023-01-02", "2023-01-03"]
        })

    def test_infer_regex_patterns(self):
        patterns = infer_regex_patterns(self.clean_data)

        # Check that patterns contain all columns
        self.assertIn("Column1", patterns)
        self.assertIn("Column2", patterns)

        # Check the inferred pattern
        self.assertEqual(patterns["Column1"], r"^\d+$")
        self.assertEqual(patterns["Column2"], r"^\d{4}-\d{2}-\d{2}$")

if __name__ == "__main__":
    unittest.main()
