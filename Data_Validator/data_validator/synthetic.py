import numpy as np
import pandas as pd
import random
import re


def generate_synthetic_data(avh_metrics, patterns, num_rows=100):
    """
    Generate synthetic data based on AVH metrics and regex patterns.

    Parameters:
        avh_metrics (dict): Dictionary containing column metrics like mean, std, mode, etc.
        patterns (dict): Dictionary containing regex patterns for each column.
        num_rows (int): Number of rows to generate.

    Returns:
        pd.DataFrame: A synthetic dataset.
    """
    synthetic_data = pd.DataFrame()

    for col, metrics in avh_metrics.items():
        try:
            if 'mean' in metrics and 'std' in metrics:
                # Generate numeric data based on AVH metrics
                synthetic_data[col] = np.random.normal(loc=metrics['mean'], scale=metrics['std'], size=num_rows)
                synthetic_data[col] = synthetic_data[col].round(2)  # Round numeric values to 2 decimal places
            elif 'mode' in metrics:
                # Generate categorical data based on mode
                synthetic_data[col] = [metrics['mode']] * num_rows
            else:
                synthetic_data[col] = [np.nan] * num_rows  # Fallback for undefined metrics
        except Exception as e:
            print(f"Error generating synthetic data for column '{col}': {e}")
            synthetic_data[col] = [np.nan] * num_rows

    for col, pattern in patterns.items():
        if col not in synthetic_data.columns:
            synthetic_data[col] = []
            for _ in range(num_rows):
                try:
                    synthetic_value = generate_value_from_pattern(pattern)
                    synthetic_data[col].append(synthetic_value)
                except Exception as e:
                    print(f"Error generating value for column '{col}' with pattern '{pattern}': {e}")
                    synthetic_data[col].append('SyntheticValue')

    return synthetic_data


def generate_value_from_pattern(pattern):
    """
    Generate a synthetic value based on a regex pattern.

    Parameters:
        pattern (str): A regex pattern.

    Returns:
        str: A synthetic value matching the given pattern.
    """
    try:
        if pattern == r'^\d+$':
            return str(random.randint(1000, 9999))  # Random integer
        elif pattern == r'^\d+\.\d+$':
            return f"{random.randint(1, 1000)}.{random.randint(0, 99):02}"  # Random decimal
        elif pattern == r'^\d{4}-\d{2}-\d{2}$':
            return f"{random.randint(2000, 2023)}-{random.randint(1, 12):02}-{random.randint(1, 28):02}"  # Random ISO date
        elif pattern == r'^[A-Za-z\s]+$':
            return random.choice(['Example', 'Test', 'Synthetic'])  # Random words
        elif pattern == r'^[A-Za-z0-9\s]+$':
            return random.choice(['Data123', 'Sample45', 'Synthetic'])  # Alphanumeric
        elif pattern == r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$':
            return random.choice(['EC1A 1BB', 'W1A 0AX', 'M1 1AE'])  # UK postal codes
        elif pattern == r'^[A-Za-z0-9\s,.\-]+$':
            return random.choice(['123 Main St.', 'Unit 45, Example Rd', 'Synthetic Value'])
        else:
            return 'SyntheticValue'  # Default fallback value
    except Exception as e:
        print(f"Error in pattern generation: {e}")
        return 'SyntheticValue'


def suggest_constraints_from_metrics(avh_metrics, tolerance=0.1):
    """
    Suggest constraints for data generation based on AVH metrics.

    Parameters:
        avh_metrics (dict): Historical metrics for each column.
        tolerance (float): Percentage tolerance for numeric ranges.

    Returns:
        dict: Constraints for synthetic data generation.
    """
    constraints = {}
    for col, metrics in avh_metrics.items():
        constraints[col] = {}
        for metric, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    constraints[col][metric] = {
                        'min': value * (1 - tolerance),
                        'max': value * (1 + tolerance)
                    }
                else:
                    constraints[col][metric] = value  # For non-numeric constraints like mode
            except Exception as e:
                print(f"Error suggesting constraints for column '{col}', metric '{metric}': {e}")
                constraints[col][metric] = None
    return constraints


def apply_constraints(value, constraints):
    """
    Apply constraints to adjust a value if it violates the specified rules.

    Parameters:
        value (numeric or str): The value to adjust.
        constraints (dict): Constraints (e.g., {"min": <min_value>, "max": <max_value>}).

    Returns:
        numeric or str: Adjusted value if it violates constraints, or the original value otherwise.
    """
    try:
        if isinstance(value, (int, float)) and isinstance(constraints, dict):
            if "min" in constraints and value < constraints["min"]:
                return constraints["min"]
            if "max" in constraints and value > constraints["max"]:
                return constraints["max"]
        return value
    except Exception as e:
        print(f"Error applying constraints to value '{value}': {e}")
        return value
