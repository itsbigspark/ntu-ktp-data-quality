import pandas as pd
import re


def apply_cleaning_rules(data, cleaning_rules):
    """
    Apply cleaning rules to the dataset.
    - Missing values are labeled as 'MISSING' in the highlighted dataset.
    - Invalid values are labeled as 'INVALID' in the highlighted dataset.

    Parameters:
        data (pd.DataFrame): Input dataset to be cleaned.
        cleaning_rules (dict): Dictionary of cleaning rules (regex patterns or transformation functions).

    Returns:
        tuple: (cleaned_data, highlighted_issues)
            - cleaned_data (pd.DataFrame): Dataset with cleaned values.
            - highlighted_issues (pd.DataFrame): Dataset highlighting "MISSING" and "INVALID" values.
    """
    cleaned_data = data.copy()
    highlighted_issues = data.copy()

    for col, rule in cleaning_rules.items():
        if col in data.columns:
            def clean_value(value):
                # Handle missing values
                if pd.isnull(value):
                    return "MISSING"
                value = str(value)  # Ensure value is a string for regex matching

                # Check for invalid values using regex or transformation function
                if isinstance(rule, re.Pattern):
                    if not rule.fullmatch(value):
                        return "INVALID"
                elif callable(rule):
                    try:
                        return rule(value)  # Apply transformation
                    except Exception:
                        return "INVALID"
                return value  # Return valid value as-is

            # Apply the cleaning rule and highlight issues
            cleaned_data[col] = data[col].apply(lambda x: clean_value(x) if pd.notnull(x) else "MISSING")
            highlighted_issues[col] = cleaned_data[col]

    return cleaned_data, highlighted_issues


def create_cleaning_rules(patterns, avh_constraints=None):
    """
    Create cleaning rules based on regex patterns and optional AVH constraints.

    Parameters:
        patterns (dict): Dictionary of regex patterns (as strings or compiled regex).
        avh_constraints (dict, optional): Dictionary of AVH constraints.

    Returns:
        dict: Cleaning rules combining regex patterns and AVH constraints.
    """
    cleaning_rules = {}
    for col, pattern in patterns.items():
        if avh_constraints and col in avh_constraints:
            constraints = avh_constraints[col]
            if isinstance(constraints, dict):
                cleaning_rules[col] = lambda x: apply_avh_constraints(
                    re.sub(r'[^A-Za-z0-9\s]', '', str(x)), constraints
                )
            else:
                raise TypeError(f"AVH constraints for column '{col}' must be a dictionary. Got {type(constraints)}")
        else:
            try:
                # Compile regex pattern if provided as a string
                cleaning_rules[col] = re.compile(pattern) if isinstance(pattern, str) else pattern
            except re.error as e:
                raise ValueError(f"Invalid regex pattern for column '{col}': {pattern}. Error: {e}")

    return cleaning_rules


def apply_avh_constraints(value, constraints):
    """
    Apply AVH constraints to a value.

    Parameters:
        value (str or numeric): The value to validate or adjust.
        constraints (dict): AVH constraints (e.g., {"min": <min_value>, "max": <max_value>}).

    Returns:
        str: Adjusted value if it violates constraints, or the original value otherwise.
    """
    try:
        numeric_value = float(value)  # Convert value to numeric if possible
    except ValueError:
        numeric_value = None

    if isinstance(constraints, dict):
        # Apply numeric constraints
        if numeric_value is not None:
            if "min" in constraints and numeric_value < constraints["min"]:
                return str(constraints["min"])
            if "max" in constraints and numeric_value > constraints["max"]:
                return str(constraints["max"])
        else:
            # Apply non-numeric constraints (e.g., mode)
            if "mode" in constraints and value != constraints["mode"]:
                return constraints["mode"]
    else:
        raise TypeError(f"Constraints for value '{value}' must be a dictionary, but got {type(constraints)}.")

    return value


def validate_data(data, cleaning_rules):
    """
    Validate the dataset against the cleaning rules.

    Parameters:
        data (pd.DataFrame): Dataset to validate.
        cleaning_rules (dict): Cleaning rules (regex patterns or transformation functions).

    Returns:
        list: List of tuples (row_index, column_name, issue_type) for invalid entries.
    """
    invalid_entries = []

    for col, rule in cleaning_rules.items():
        if col in data.columns:
            for idx, value in data[col].dropna().items():
                value = str(value)  # Ensure value is a string for regex matching
                if isinstance(rule, re.Pattern):
                    if not rule.fullmatch(value):
                        invalid_entries.append((idx, col, "INVALID"))
                elif callable(rule):
                    try:
                        rule(value)
                    except Exception:
                        invalid_entries.append((idx, col, "INVALID"))

    return invalid_entries
