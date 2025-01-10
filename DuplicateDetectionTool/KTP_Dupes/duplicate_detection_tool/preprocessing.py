#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

def detect_column_types(df):
    """
    Detects column types (text, numeric, date) in the dataframe.
    """
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return text_cols, numeric_cols, date_cols

def preprocess_text(text):
    """
    Cleans and normalizes text data by removing special characters and converting to lowercase.
    """
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower().strip())

def process_data(df):
    """
    Main preprocessing function that handles missing values, converts dates, normalizes text, and scales numeric data.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
    - df (pd.DataFrame): The processed DataFrame with new columns.
    - text_cols (list): List of identified text columns.
    - numeric_cols (list): List of identified numeric columns.
    - date_cols (list): List of identified date columns.
    """
    # Step 1: Fill missing values with a placeholder for consistency
    df = df.fillna('MISSING')

    # Step 2: Attempt to convert text columns to dates and re-categorize column types
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            continue  # Skip columns that cannot be converted

    # Re-detect column types after potential date conversion
    text_cols, numeric_cols, date_cols = detect_column_types(df)

    # Step 3: Text Processing - Clean and normalize text data
    for col in text_cols:
        df[f'normalized_{col}'] = df[col].astype(str).apply(preprocess_text)

    # Step 4: Numeric Processing - Fill missing values and scale numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = df[numeric_cols].replace('MISSING', 0).astype(float)  # Handle 'MISSING' placeholder in numeric data
    scaled_numerical = scaler.fit_transform(df[numeric_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=[f'{col}_scaled' for col in numeric_cols], index=df.index)
    df = pd.concat([df, scaled_numerical_df], axis=1)

    # Step 5: Date Processing - Extract year, month, and day for date columns
    for col in date_cols:
        df[f'{col}_year'] = df[col].dt.year.fillna(0).astype(int)
        df[f'{col}_month'] = df[col].dt.month.fillna(0).astype(int)
        df[f'{col}_day'] = df[col].dt.day.fillna(0).astype(int)

    return df, text_cols, numeric_cols, date_cols
