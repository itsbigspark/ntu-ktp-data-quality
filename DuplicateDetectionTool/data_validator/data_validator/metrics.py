import re
import pandas as pd
from scipy.stats import entropy

def calculate_metrics(data):
    """Calculate AVH metrics for a dataset."""
    metrics = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            metrics[col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "completeness_ratio": data[col].notna().mean(),
            }
        else:
            freq_dist = data[col].value_counts(normalize=True)
            metrics[col] = {
                "entropy": entropy(freq_dist),
                "unique_ratio": data[col].nunique() / len(data),
                "completeness_ratio": data[col].notna().mean(),
            }
    return metrics
