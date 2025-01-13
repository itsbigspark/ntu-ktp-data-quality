#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def run_meta_classification(df):
    """
    Perform meta-classification to classify records as unique, overly similar, or duplicates.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with clustering and similarity features.
    
    Returns:
    - dict: Dictionary containing performance metrics for each classifier.
    """
    # Define the meta features and labels
    X_meta = df[['kmeans_cluster', 'dbscan_cluster', 'cosine_similarity_listing_title', 
                 'cosine_similarity_status', 'euclidean_dist_to_centroid']]
    y_meta = df['record_class']  # Labels: 0 for Unique, 1 for Overly Similar, 2 for Duplicate
    
    # Split data into train and test sets, stratifying to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X_meta, y_meta, test_size=0.3, random_state=42, stratify=y_meta)
    
    # Check class distribution and apply SMOTE only if each class has enough samples
    class_counts = Counter(y_train)
    min_class_count = min(class_counts.values())

    if min_class_count >= 6:  # Enough samples for default SMOTE `k_neighbors=5`
        smote = SMOTE(random_state=42)
    else:
        # Set `k_neighbors=min(min_class_count - 1, 1)` for SMOTE to avoid errors with small classes
        smote = SMOTE(random_state=42, k_neighbors=min(min_class_count - 1, 1))

    # Apply SMOTE only if we have enough samples in the smallest class
    if min_class_count > 1:
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    else:
        print("Warning: Insufficient samples for SMOTE; using original training data without resampling.")
        X_train_smote, y_train_smote = X_train, y_train

    # Define the classifiers to use in meta-classification
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, max_depth=10, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=6)
    }
    
    # Dictionary to store the performance metrics for each classifier
    performance_metrics = {}

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_smote, y_train_smote)
        y_pred = clf.predict(X_test)
        
        # Calculate evaluation metrics for test set
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store the metrics for this classifier
        performance_metrics[name] = {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
        
        # Print metrics to give immediate feedback on each classifier's performance
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    return performance_metrics
