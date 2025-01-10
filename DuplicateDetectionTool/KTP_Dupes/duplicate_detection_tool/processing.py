#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

def detect_column_types(df):
    """Automatically detect and categorize columns as text, numeric, or date."""
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return text_cols, numeric_cols, date_cols

def preprocess_text(text):
    """Normalize text by removing special characters and converting to lowercase."""
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower().strip())

def process_data(df, text_columns=None, numeric_columns=None, date_columns=None):
    """
    Preprocess the DataFrame by handling missing values, processing dates,
    normalizing text, and scaling numeric columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to process.
    - text_columns (list, optional): List of user-specified text columns.
    - numeric_columns (list, optional): List of user-specified numeric columns.
    - date_columns (list, optional): List of user-specified date columns.

    Returns:
    - pd.DataFrame: The processed DataFrame with normalized and scaled columns.
    - list: List of identified text columns.
    - list: List of identified numeric columns.
    - list: List of identified date columns.
    """
    # Fill any completely empty cells to maintain consistency
    df = df.fillna('MISSING')
    
    # Use provided columns if available, else detect automatically
    if text_columns is None or numeric_columns is None or date_columns is None:
        detected_text_cols, detected_numeric_cols, detected_date_cols = detect_column_types(df)
        text_columns = text_columns or detected_text_cols
        numeric_columns = numeric_columns or detected_numeric_cols
        date_columns = date_columns or detected_date_cols

    # Process date columns and create new features
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_year'] = df[col].dt.year.fillna(0).astype(int)
        df[f'{col}_month'] = df[col].dt.month.fillna(0).astype(int)
        df[f'{col}_day'] = df[col].dt.day.fillna(0).astype(int)

    # Normalize all text columns at once
    normalized_text_cols = {f'normalized_{col}': df[col].apply(preprocess_text) for col in text_columns}
    normalized_text_df = pd.DataFrame(normalized_text_cols, index=df.index)

    # Ensure all values in numeric columns are numeric and filter non-numeric columns
    valid_numeric_columns = [col for col in numeric_columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Handle empty numeric column case
    if not valid_numeric_columns:
        print("Warning: No valid numeric columns found for scaling.")
    else:
        # Fill missing values in numeric columns and standardize
        scaler = StandardScaler()
        df[valid_numeric_columns] = df[valid_numeric_columns].fillna(0)
        scaled_numerical = scaler.fit_transform(df[valid_numeric_columns])
        scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=[f'{col}_scaled' for col in valid_numeric_columns], index=df.index)
        df = pd.concat([df, scaled_numerical_df], axis=1)

    # Concatenate normalized text columns to the DataFrame
    df = pd.concat([df, normalized_text_df], axis=1)

    return df, text_columns, valid_numeric_columns, date_columns
