import numpy as np
import pandas as pd

def generate_synthetic_data(metrics, patterns, num_rows):
    """Generate synthetic data based on AVH metrics and regex patterns."""
    synthetic_data = {}
    for col, metric in metrics.items():
        if "mean" in metric and "std" in metric:
            synthetic_data[col] = np.random.normal(metric["mean"], metric["std"], num_rows)
        elif "entropy" in metric:
            synthetic_data[col] = ["SYNTHETIC"] * num_rows
    return pd.DataFrame(synthetic_data)
