import pandas as pd
import re
from scipy.stats import entropy


def calculate_metrics(data):
    """
    Calculate metrics for the dataset.
    Metrics include:
    - For numerical columns: mean, standard deviation, unique ratio, completeness ratio.
    - For non-numerical columns: mode, entropy, unique ratio, completeness ratio.

    Parameters:
        data (pd.DataFrame): The input dataset.

    Returns:
        dict: A dictionary where keys are column names and values are metric dictionaries.
    """
    metrics = {}
    for col in data.columns:
        try:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Metrics for numeric columns
                metrics[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'unique_ratio': data[col].nunique() / len(data) if len(data) > 0 else 0,
                    'completeness_ratio': data[col].notnull().mean()
                }
            else:
                # Metrics for non-numeric columns
                freq_dist = data[col].value_counts(normalize=True)
                metrics[col] = {
                    'mode': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'entropy': entropy(freq_dist) if len(freq_dist) > 0 else 0,
                    'unique_ratio': data[col].nunique() / len(data) if len(data) > 0 else 0,
                    'completeness_ratio': data[col].notnull().mean()
                }
        except Exception as e:
            print(f"Error calculating metrics for column '{col}': {e}")
            metrics[col] = {
                'mean': None, 'std': None, 'unique_ratio': None, 'completeness_ratio': None,
                'mode': None, 'entropy': None
            }
    return metrics


def validate_patterns(data, patterns):
    """
    Validate data columns against the provided regex patterns.
    Returns a list of tuples indicating invalid entries: (index, column, value).

    Parameters:
        data (pd.DataFrame): The dataset to validate.
        patterns (dict): A dictionary where keys are column names and values are regex patterns.

    Returns:
        list: A list of tuples containing invalid entries as (index, column, value).
    """
    invalid_entries = []

    for col, pattern in patterns.items():
        if col in data.columns:
            try:
                # Compile the regex pattern
                regex = re.compile(pattern) if isinstance(pattern, str) else pattern

                # Validate column values
                for idx, value in data[col].dropna().items():  # Safe iteration
                    if not regex.fullmatch(str(value)):
                        invalid_entries.append((idx, col, value))
            except re.error as e:
                print(f"Invalid regex pattern for column '{col}': {e}")
            except Exception as e:
                print(f"Error validating column '{col}': {e}")

    return invalid_entries


def summarize_invalid_entries(invalid_entries):
    """
    Summarize the invalid entries by column.

    Parameters:
        invalid_entries (list): A list of tuples containing invalid entries as (index, column, value).

    Returns:
        dict: A summary dictionary where keys are column names and values are counts of invalid entries.
    """
    summary = {}
    for _, col, _ in invalid_entries:
        summary[col] = summary.get(col, 0) + 1
    return summary


def validate_avh_constraints(data, metrics, avh_constraints):
    """
    Validate data columns against AVH constraints (min, max, mode).

    Parameters:
        data (pd.DataFrame): The dataset to validate.
        metrics (dict): The AVH metrics for the dataset.
        avh_constraints (dict): The AVH constraints (e.g., min/max values for each column).

    Returns:
        dict: A dictionary where keys are column names and values are lists of invalid rows.
    """
    invalid_entries = {}

    for col, constraints in avh_constraints.items():
        if col in data.columns and col in metrics:
            invalid_rows = []

            for idx, value in data[col].dropna().items():
                try:
                    numeric_value = float(value) if pd.api.types.is_numeric_dtype(data[col]) else None
                    if numeric_value is not None:
                        # Validate numeric constraints
                        if 'min' in constraints and numeric_value < constraints['min']:
                            invalid_rows.append(idx)
                        if 'max' in constraints and numeric_value > constraints['max']:
                            invalid_rows.append(idx)
                    else:
                        # Validate mode for non-numeric data
                        if 'mode' in constraints and value != constraints['mode']:
                            invalid_rows.append(idx)
                except Exception as e:
                    print(f"Error validating value '{value}' in column '{col}': {e}")

            invalid_entries[col] = invalid_rows

    return invalid_entries
