import argparse
import pandas as pd
from duplicate_detection_tool.preprocessing import preprocess_text
from duplicate_detection_tool.processing import process_data
from duplicate_detection_tool.similarity import detect_duplicates_and_similars
from duplicate_detection_tool.clustering import run_clustering, cluster_analysis
from duplicate_detection_tool.meta_classifier import run_meta_classification

def main(input_file):
    # Load the data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded '{input_file}' with {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    # Step 1: Data Preprocessing
    print("\n--- Step 1: Data Preprocessing ---")
    df, text_cols, numeric_cols, date_cols = process_data(df)
    print("Data preprocessing completed.")
    print(f"Identified text columns: {text_cols}")
    print(f"Identified numeric columns: {numeric_cols}")
    print(f"Identified date columns: {date_cols}")

    # Step 2: Non-ML Detection for Duplicates, Overly Similar, and Unique Records
    print("\n--- Step 2: Non-ML Duplicate Detection ---")
    df, duplicate_pairs, similar_pairs, unique_records = detect_duplicates_and_similars(
        df, text_cols, numeric_cols, threshold_duplicate=80, threshold_similar=60
    )
    print(f"Records classified: {len(duplicate_pairs)} duplicates, "
          f"{len(similar_pairs)} overly similar, "
          f"{len(unique_records)} unique records.")

    # Step 3: Clustering and Meta-Classification
    print("\n--- Step 3: Clustering and Meta-Classification ---")
    df, clustering_metrics = run_clustering(df, text_cols, numeric_cols)
    classifiers_performance = run_meta_classification(df)
    
    # Output Summary of Duplicate Detection and Similarity Results
    print("\n--- Summary of Record Classification ---")
    print(f"Total Duplicates: {len(df[df['record_class'] == 2])}")
    print(f"Total Overly Similar: {len(df[df['record_class'] == 1])}")
    print(f"Total Unique: {len(df[df['record_class'] == 0])}")

    # Output Clustering Performance Metrics
    print("\n--- Clustering Performance ---")
    for metric, value in clustering_metrics.items():
        print(f"{metric}: {value}")

    # Output Meta-Classifier Performance
    print("\n--- Meta-Classifier Model Performance ---")
    for model_name, metrics in classifiers_performance.items():
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1']:.4f}")

    # Step 4: Cluster Analysis
    print("\n--- Step 4: Cluster Analysis ---")
    cluster_analysis(df, 'kmeans_cluster', df[numeric_cols].values)
    cluster_analysis(df, 'dbscan_cluster', df[numeric_cols].values)
    
    # Prompt the user to view or save records
    view_or_save = input("\nWould you like to view or save the records? Enter 'view' to print, 'save' to save as CSV, or 'no' to skip: ").strip().lower()
    
    if view_or_save == 'view':
        print("\n--- Duplicates ---")
        print(pd.DataFrame(duplicate_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"]))

        print("\n--- Overly Similar ---")
        print(pd.DataFrame(similar_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"]))

        print("\n--- Unique Records ---")
        print(pd.DataFrame(unique_records, columns=["Record", "Uniqueness Score"]))
    elif view_or_save == 'save':
        pd.DataFrame(duplicate_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"]).to_csv("duplicates.csv", index=False)
        pd.DataFrame(similar_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"]).to_csv("overly_similar.csv", index=False)
        pd.DataFrame(unique_records, columns=["Record", "Uniqueness Score"]).to_csv("unique_records.csv", index=False)
        print("\nRecords saved to 'duplicates.csv', 'overly_similar.csv', and 'unique_records.csv'.")
    else:
        print("No output selected.")

    # Option to inspect specific pairs
    while True:
        inspect_choice = input("\nWould you like to manually inspect specific records? (yes/no): ").strip().lower()
        if inspect_choice == 'yes':
            record_type = input("Enter 'duplicates' to check duplicates or 'similar' for overly similar records: ").strip().lower()
            
            if record_type == 'duplicates':
                pair_df = pd.DataFrame(duplicate_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"])
            elif record_type == 'similar':
                pair_df = pd.DataFrame(similar_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"])
            else:
                print("Invalid choice. Please enter 'duplicates' or 'similar'.")
                continue
            
            print(pair_df)
            try:
                rec1 = int(input("Enter the ID of Record 1 to inspect: "))
                rec2 = int(input("Enter the ID of Record 2 to inspect: "))
                
                print("\n--- Record 1 ---")
                print(df.loc[rec1])
                print("\n--- Record 2 ---")
                print(df.loc[rec2])
            except (ValueError, KeyError):
                print("Invalid record ID entered. Please try again with valid record numbers from the table.")
        elif inspect_choice == 'no':
            break
        else:
            print("Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Duplicate Detection Tool on input CSV data.")
    parser.add_argument("input_file", type=str, help="Path to the CSV input file")
    args = parser.parse_args()
    main(args.input_file)
