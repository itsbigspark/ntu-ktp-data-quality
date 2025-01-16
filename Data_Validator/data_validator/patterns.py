import re
import pandas as pd


def infer_regex_patterns(data, sample_size=100):
    """
    Infer regex patterns for each column in the dataset.
    This function samples a limited number of rows (default: 100) to infer patterns efficiently.

    Parameters:
        data (pd.DataFrame): The dataset to infer patterns from.
        sample_size (int): Number of rows to sample for pattern inference.

    Returns:
        dict: A dictionary where keys are column names and values are inferred regex patterns.
    """
    patterns = {}

    for col in data.columns:
        # Use all non-null values if the column has fewer than `sample_size` rows
        sample_values = data[col].dropna().astype(str)
        if len(sample_values) > sample_size:
            sample_values = sample_values.sample(n=sample_size, random_state=42)

        inferred_patterns = set()

        for value in sample_values:
            inferred_patterns.add(infer_pattern_for_value(value))

        # Consolidate inferred patterns for the column
        if len(inferred_patterns) == 1:
            # If all values match a single pattern, use that pattern
            patterns[col] = inferred_patterns.pop()
        else:
            # If multiple patterns, create a combined regex
            patterns[col] = combine_patterns(list(inferred_patterns))

    return patterns


def infer_pattern_for_value(value):
    """
    Infer regex pattern for a single value.

    Parameters:
        value (str): The value to analyze.

    Returns:
        str: Inferred regex pattern for the value.
    """
    # Strict patterns for numbers
    if re.fullmatch(r'^\d+$', value):
        return r'^\d+$'  # Integers
    elif re.fullmatch(r'^\d+\.\d+$', value):
        return r'^\d+\.\d+$'  # Decimals

    # Patterns for dates and times
    elif re.fullmatch(r'^\d{4}-\d{2}-\d{2}$', value):
        return r'^\d{4}-\d{2}-\d{2}$'  # ISO format dates
    elif re.fullmatch(r'^\d{2}/\d{2}/\d{4}$', value):
        return r'^\d{2}/\d{2}/\d{4}$'  # DD/MM/YYYY dates
    elif re.fullmatch(r'^\d{2}:\d{2}(:\d{2})?$', value):
        return r'^\d{2}:\d{2}(:\d{2})?$'  # Time formats HH:MM or HH:MM:SS

    # Alphanumeric patterns
    elif re.fullmatch(r'^[A-Za-z\s]+$', value):
        return r'^[A-Za-z\s]+$'  # Letters and spaces only
    elif re.fullmatch(r'^[A-Za-z0-9\s]+$', value):
        return r'^[A-Za-z0-9\s]+$'  # Alphanumeric and spaces
    elif re.fullmatch(r'^[A-Za-z0-9\s,.\-]+$', value):
        return r'^[A-Za-z0-9\s,.\-]+$'  # Alphanumeric with punctuation
    elif re.fullmatch(r'^[\w\-_]+$', value):
        return r'^[\w\-_]+$'  # Mixed alphanumeric with underscores/hyphens

    # Postal code patterns (e.g., UK postal codes)
    elif re.fullmatch(r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$', value):
        return r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$'

    # Default fallback pattern
    return r'.*'


def validate_regex_patterns(data, patterns):
    """
    Validate dataset values against inferred regex patterns.

    Parameters:
        data (pd.DataFrame): The dataset to validate.
        patterns (dict): Dictionary of regex patterns inferred for the dataset.

    Returns:
        dict: A dictionary of validation results (True/False) for each column.
    """
    validation_results = {}

    for col, pattern in patterns.items():
        regex = re.compile(pattern)
        validation_results[col] = data[col].apply(
            lambda x: pd.isnull(x) or bool(regex.fullmatch(str(x)))  # Treat NaN as valid
        )

    return validation_results


def combine_patterns(pattern_list):
    """
    Combine multiple regex patterns into a single regex pattern.

    Parameters:
        pattern_list (list): A list of regex patterns to combine.

    Returns:
        str: A single regex pattern that matches any of the patterns in the list.
    """
    # Validate input patterns
    valid_patterns = [pat for pat in pattern_list if isinstance(pat, str) and is_valid_regex(pat)]

    if not valid_patterns:
        return r'.*'  # Fallback for empty or invalid pattern list
    return f"({'|'.join(valid_patterns)})"


def is_valid_regex(pattern):
    """
    Check if a string is a valid regex pattern.

    Parameters:
        pattern (str): The regex pattern to validate.

    Returns:
        bool: True if the pattern is valid, False otherwise.
    """
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False
