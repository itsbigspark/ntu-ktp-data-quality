import re
import pandas as pd

def create_cleaning_rules(regex_patterns):
    """Generate cleaning rules based on regex patterns."""
    cleaning_rules = {}
    for col, pattern in regex_patterns.items():
        if pattern == r"^\d+$":
            cleaning_rules[col] = lambda x: re.sub(r"\D", "", str(x))
        elif pattern == r"^\d{4}-\d{2}-\d{2}$":
            cleaning_rules[col] = lambda x: x if re.match(pattern, str(x)) else None
        else:
            cleaning_rules[col] = lambda x: x
    return cleaning_rules

def apply_cleaning_rules(data, cleaning_rules):
    """Apply cleaning rules to the dataset."""
    cleaned_data = data.copy()
    for col, rule in cleaning_rules.items():
        if col in cleaned_data.columns:
            cleaned_data[col] = cleaned_data[col].apply(lambda x: rule(x) if pd.notnull(x) else x)
    return cleaned_data


