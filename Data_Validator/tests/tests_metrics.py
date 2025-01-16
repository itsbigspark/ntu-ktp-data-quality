import unittest
import pandas as pd
from data_validator.metrics import calculate_metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample clean data
        self.clean_data = pd.DataFrame({
            "Column1": ["123", "456", "789"],
            "Column2": ["2023-01-01", "2023-01-02", "2023-01-03"]
        })

    def test_calculate_metrics(self):
        metrics = calculate_metrics(self.clean_data)

        # Check that metrics contain all columns
        self.assertIn("Column1", metrics)
        self.assertIn("Column2", metrics)

        # Check valid percentage
        self.assertGreaterEqual(metrics["Column1"]["valid_percentage"], 100)
        self.assertGreaterEqual(metrics["Column2"]["valid_percentage"], 100)

if __name__ == "__main__":
    unittest.main()
