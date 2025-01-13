import streamlit as st
import pandas as pd
import numpy as np
from duplicate_detection_tool.processing import process_data
from duplicate_detection_tool.similarity import detect_duplicates_and_similars
from duplicate_detection_tool.clustering import run_clustering
from duplicate_detection_tool.meta_classifier import run_meta_classification
from datetime import datetime

# --- Utility Functions ---
def generate_dataset_summary(df):
    """Generate statistical and non-statistical descriptions of the dataset."""
    summary = {
        "Number of Records": len(df),
        "Number of Columns": len(df.columns),
        "Column Types": df.dtypes.value_counts().to_dict(),
        "Missing Values (%)": df.isnull().mean() * 100,
        "Basic Statistics": df.describe().to_dict()
    }
    return summary

def save_diagnostic_report(summary, performance_metrics, output_file="diagnostic_report.json"):
    """Save metadata and performance metrics to a diagnostic JSON file."""
    report = {
        "Dataset Summary": summary,
        "Performance Metrics": performance_metrics,
        "Timestamp": datetime.now().isoformat()
    }
    pd.Series(report).to_json(output_file)
    return output_file

# --- Streamlit App ---
st.set_page_config(page_title="🛠️ Duplicate Detection Tool", layout="wide")



# Add Company Images
#st.image(["images/bigspark_logo.png", "images/UKRI_logo.png", "images/NTU_Primary_logo.png" ], width=200, caption=["Company 1", "Company 2", "Company 3"])

# Add Company Images with Spacing
#st.write("### Partners & Sponsors")
col1, col2, col3 = st.columns([1, 1, 1])  # Three equal-width columns

with col1:
    st.image("images/bigspark_logo.png", width=200)
with col2:
    st.image("images/UKRI_logo.png", width=300)
with col3:
    st.image("images/NTU_Primary_logo.png", width=200)

# Sidebar for file upload
st.sidebar.title("Duplicate Detection Tool")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    st.sidebar.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

    # --- Step 1: Data Preprocessing ---
    st.header("Step 1: Data Preprocessing")
    df, text_cols, numeric_cols, date_cols = process_data(df)
    st.write("Data Preprocessing Completed!")
    st.write(f"Identified Text Columns: {text_cols}")
    st.write(f"Identified Numeric Columns: {numeric_cols}")
    st.write(f"Identified Date Columns: {date_cols}")

    # Display dataset summary
    st.subheader("Dataset Summary")
    dataset_summary = generate_dataset_summary(df)
    st.json(dataset_summary)

    # --- Step 2: Non-ML Duplicate Detection ---
    st.header("Step 2: Non-ML Duplicate Detection")
    df, duplicate_pairs, similar_pairs, unique_records = detect_duplicates_and_similars(
        df, text_cols, numeric_cols, threshold_duplicate=80, threshold_similar=60
    )
    st.write(f"Records Classified: {len(duplicate_pairs)} duplicates, "
             f"{len(similar_pairs)} overly similar, "
             f"{len(unique_records)} unique records.")

    # Display options for duplicates and similar records
    analysis_choice = st.radio(
        "Select what you want to analyze:",
        ("Duplicates", "Overly Similar", "Unique Records")
    )

    if analysis_choice == "Duplicates":
        st.subheader("Duplicate Records")
        st.write("Duplicate Record Pairs with Similarity Scores:")
        duplicates_df = pd.DataFrame(duplicate_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"])
        st.dataframe(duplicates_df)

        # Select records for manual inspection
        st.write("Manually Inspect Records:")
        record1 = st.number_input("Enter Record 1 ID", min_value=0, max_value=len(df) - 1)
        record2 = st.number_input("Enter Record 2 ID", min_value=0, max_value=len(df) - 1)
        if st.button("View Selected Records"):
            st.write(df.iloc[[record1, record2]])

    elif analysis_choice == "Overly Similar":
        st.subheader("Overly Similar Records")
        st.write("Overly Similar Record Pairs with Similarity Scores:")
        similar_df = pd.DataFrame(similar_pairs, columns=["Record 1", "Record 2", "Similarity Score", "Contributing Columns"])
        st.dataframe(similar_df)

        # Select records for manual inspection
        st.write("Manually Inspect Records:")
        record1 = st.number_input("Enter Record 1 ID", min_value=0, max_value=len(df) - 1)
        record2 = st.number_input("Enter Record 2 ID", min_value=0, max_value=len(df) - 1)
        if st.button("View Selected Records"):
            st.write(df.iloc[[record1, record2]])

    elif analysis_choice == "Unique Records":
        st.subheader("Unique Records")
        st.write(f"Total Unique Records: {len(unique_records)}")
        st.write("Unique Records by Ranking:")
        unique_df = pd.DataFrame(unique_records, columns=["Record ID", "Uniqueness Score"]).sort_values(by="Uniqueness Score", ascending=False)
        st.dataframe(unique_df)

    # --- Step 3: Clustering and Meta-Classification ---
    st.header("Step 3: Clustering and Meta-Classification")
    df, clustering_metrics = run_clustering(df, text_cols, numeric_cols)
    st.write("Clustering Metrics:")
    st.json(clustering_metrics)

    classifiers_performance = run_meta_classification(df)
    st.write("Meta-Classifier Performance:")
    st.json(classifiers_performance)

    # --- Step 4: Save Results ---
    st.header("Step 4: Save Results")
    st.write("You can download the classified records below:")

    duplicates_csv = "duplicates.csv"
    overly_similar_csv = "overly_similar.csv"
    unique_csv = "unique_records.csv"
    pd.DataFrame(duplicate_pairs).to_csv(duplicates_csv, index=False)
    pd.DataFrame(similar_pairs).to_csv(overly_similar_csv, index=False)
    pd.DataFrame(unique_records).to_csv(unique_csv, index=False)

    st.download_button("Download Duplicates", open(duplicates_csv, "rb").read(), duplicates_csv)
    st.download_button("Download Overly Similar Records", open(overly_similar_csv, "rb").read(), overly_similar_csv)
    st.download_button("Download Unique Records", open(unique_csv, "rb").read(), unique_csv)

    # Save and download diagnostic report
    diagnostic_report_file = save_diagnostic_report(dataset_summary, classifiers_performance)
    st.download_button("Download Diagnostic Report", open(diagnostic_report_file, "rb").read(), diagnostic_report_file)

    # --- Feedback Section ---
    st.header("Feedback")
    st.text_area("Share your feedback about the tool:", placeholder="Enter your feedback here...")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
