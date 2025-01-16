import pandas as pd
import numpy as np
from fuzzywuzzy import process

class ErrorCorrector:
    def __init__(self, reference_data):
        """
        Initialize the ErrorCorrector with reference data.
        :param reference_data: The clean reference dataset as a Pandas DataFrame.
        """
        self.reference_data = reference_data

    def correct_missing_values(self, data, strategy="mean"):
        """
        Correct missing values using a specified strategy.
        :param data: The dataset with missing values.
        :param strategy: Imputation strategy ('mean', 'median', 'mode').
        :return: The dataset with missing values corrected.
        """
        for column in data.columns:
            if data[column].isnull().any():
                if pd.api.types.is_numeric_dtype(data[column]):
                    # Numeric column handling
                    if strategy == "mean":
                        fill_value = self.reference_data[column].mean()
                    elif strategy == "median":
                        fill_value = self.reference_data[column].median()
                    elif strategy == "mode":
                        fill_value = self.reference_data[column].mode()[0]
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                else:
                    # Non-numeric column handling
                    fill_value = self.reference_data[column].mode()[0] if not self.reference_data[column].empty else "missing"

                print(f"Filling missing values in column '{column}' with: {fill_value}")
                data[column].fillna(fill_value, inplace=True)
        return data

    def correct_typographical_errors(self, data, column):
        """
        Correct typographical errors by matching against reference data using fuzzy string matching.
        :param data: The dataset with potential typographical errors.
        :param column: The column to correct.
        :return: The dataset with corrected typographical errors.
        """
        reference_values = set(self.reference_data[column].dropna().unique())
        if not reference_values:
            print(f"No reference values available for column '{column}'. Skipping correction.")
            return data

        def correct_value(value):
            if pd.isnull(value) or value in reference_values:
                return value
            corrected_value = process.extractOne(value, reference_values)[0]
            print(f"Correcting '{value}' to '{corrected_value}' in column '{column}'")
            return corrected_value

        data[column] = data[column].apply(correct_value)
        return data

    def correct_out_of_range(self, data, column, min_val=None, max_val=None):
        """
        Correct out-of-range values by capping them within the acceptable range.
        :param data: The dataset with potential out-of-range values.
        :param column: The column to correct.
        :param min_val: Minimum acceptable value (optional).
        :param max_val: Maximum acceptable value (optional).
        :return: The dataset with corrected out-of-range values.
        """
        if not pd.api.types.is_numeric_dtype(data[column]):
            print(f"Column '{column}' is not numeric. Skipping out-of-range correction.")
            return data

        if min_val is None:
            min_val = self.reference_data[column].min()
        if max_val is None:
            max_val = self.reference_data[column].max()

        print(f"Capping values in column '{column}' to range ({min_val}, {max_val})")
        data[column] = data[column].clip(lower=min_val, upper=max_val)
        return data
