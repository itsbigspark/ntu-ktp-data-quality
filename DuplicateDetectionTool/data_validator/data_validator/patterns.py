

import re

def infer_regex_patterns(data):
    """Generate regex patterns for dataset columns."""
    regex_patterns = {}
    for col in data.columns:
        sample_values = data[col].dropna().astype(str)
        if sample_values.empty:
            regex_patterns[col] = r".*"
            continue

        if sample_values.str.isdigit().all():
            regex_patterns[col] = r"^\d+$"
        elif sample_values.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
            regex_patterns[col] = r"^\d{4}-\d{2}-\d{2}$"
        else:
            regex_patterns[col] = r".*"
    return regex_patterns
