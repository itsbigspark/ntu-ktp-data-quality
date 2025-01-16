import pandas as pd

class ErrorDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    def detect_missing_values(self, data):
        return data.isnull()

    def detect_typographical_errors(self, data, column):
        reference_values = set(self.reference_data[column].dropna().unique())
        return ~data[column].isin(reference_values)

    def detect_out_of_range(self, data, column, min_val=None, max_val=None):
        if min_val is None:
            min_val = self.reference_data[column].min()
        if max_val is None:
            max_val = self.reference_data[column].max()
        return (data[column] < min_val) | (data[column] > max_val)
