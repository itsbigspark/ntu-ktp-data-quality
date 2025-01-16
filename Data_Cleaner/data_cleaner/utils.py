def detect_column_types(data):
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    return numeric_features, categorical_features
