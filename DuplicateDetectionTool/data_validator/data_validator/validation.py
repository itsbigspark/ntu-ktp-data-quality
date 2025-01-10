def validate_patterns(data, regex_patterns):
    """Validate dataset columns using inferred regex patterns."""
    invalid_entries = []
    for col, pattern in regex_patterns.items():
        if col in data.columns:
            invalid_rows = data[~data[col].astype(str).str.match(pattern, na=False)]
            for idx, value in invalid_rows[col].items():
                invalid_entries.append((idx, col, value))
    return invalid_entries
