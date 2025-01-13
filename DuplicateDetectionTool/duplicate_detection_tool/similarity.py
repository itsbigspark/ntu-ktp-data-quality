# similarity.py

import pandas as pd
from fuzzywuzzy import fuzz
from metaphone import doublemetaphone
import numpy as np

# Helper functions for similarity calculations
def fuzzy_similarity(s1, s2):
    return fuzz.token_set_ratio(s1, s2)

def phonetic_similarity(s1, s2):
    # Ensure values are strings
    s1 = str(s1) if pd.notna(s1) else ""
    s2 = str(s2) if pd.notna(s2) else ""
    phonetic_s1 = doublemetaphone(s1)[0] if s1 else ''
    phonetic_s2 = doublemetaphone(s2)[0] if s2 else ''
    return fuzz.token_set_ratio(phonetic_s1, phonetic_s2)

def numerical_similarity(val1, val2, threshold=0.02):
    # Ensure val1 and val2 are numbers, otherwise treat as dissimilar
    if pd.isna(val1) or pd.isna(val2):
        return 0
    return abs(val1 - val2) <= threshold * max(abs(val1), abs(val2))

def cumulative_similarity(row1, row2, text_cols, numeric_cols):
    total_score, field_count = 0, 0
    contrib_cols = []
    
    for col in text_cols:
        text_sim = fuzzy_similarity(row1[col], row2[col])
        phonetic_sim = phonetic_similarity(row1[col], row2[col])
        similarity_score = (text_sim + phonetic_sim) / 2
        total_score += similarity_score
        field_count += 1
        if similarity_score > 60:  # Threshold to consider the column as a contributor
            contrib_cols.append(col)

    for col in numeric_cols:
        num_sim = numerical_similarity(row1[col], row2[col]) * 100
        total_score += num_sim
        field_count += 1
        if num_sim > 60:  # Threshold for significant numerical similarity
            contrib_cols.append(col)

    avg_similarity = total_score / field_count if field_count > 0 else 0
    return avg_similarity, contrib_cols

def detect_duplicates_and_similars(df, text_cols, numeric_cols, threshold_duplicate=80, threshold_similar=60):
    duplicates = []
    similar_records = []
    unique_records = []
    
    for i in range(len(df)):
        for j in range(i + 1, min(i + 6, len(df))):
            score, contrib_cols = cumulative_similarity(df.loc[i], df.loc[j], text_cols, numeric_cols)
            if score >= threshold_duplicate:
                duplicates.append((i, j, score, contrib_cols))
            elif score >= threshold_similar:
                similar_records.append((i, j, score, contrib_cols))
    
    # Separate unique records
    all_similar_indices = {i for i, j, score, cols in duplicates + similar_records}.union(
                          {j for i, j, score, cols in duplicates + similar_records})
    unique_indices = [i for i in range(len(df)) if i not in all_similar_indices]
    unique_records = [(idx, 100) for idx in unique_indices]  # Assign maximum uniqueness score

    # Assign classes
    record_class = [0] * len(df)
    for i, j, score, _ in duplicates:
        record_class[i] = record_class[j] = 2
    for i, j, score, _ in similar_records:
        record_class[i] = record_class[j] = 1
    df['record_class'] = record_class

    return df, duplicates, similar_records, unique_records
