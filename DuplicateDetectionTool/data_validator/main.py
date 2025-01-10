import pandas as pd
import sys
from data_validator import *

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <clean_data.csv> <unclean_data.csv>")
        sys.exit(1)

    clean_file, unclean_file = sys.argv[1:3]
    clean_data = pd.read_csv(clean_file)
    unclean_data = pd.read_csv(unclean_file)

    patterns = infer_regex_patterns(clean_data)
    metrics = calculate_metrics(clean_data)
    cleaning_rules = create_cleaning_rules(patterns)

    # Validate and clean the data
    invalid_entries = validate_patterns(unclean_data, patterns)
    cleaned_data = apply_cleaning_rules(unclean_data, cleaning_rules)

    # Save outputs
    cleaned_data.to_csv("outputs/cleaned_data.csv", index=False)

    # Highlight issues
    highlighted = unclean_data.copy()
    for idx, col, _ in invalid_entries:
        highlighted.at[idx, col] = "INVALID"
    highlighted.to_csv("outputs/highlighted_issues.csv", index=False)

    # Save metrics
    with open("outputs/metrics.json", "w") as f:
        json.dump({"patterns": patterns, "metrics": metrics}, f)

    print("Processing complete. Outputs saved in 'outputs/'.")

if __name__ == "__main__":
    main()
