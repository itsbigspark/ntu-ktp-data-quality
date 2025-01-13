import streamlit as st
import pandas as pd
from duplicate_detection_tool.processing import process_data
from duplicate_detection_tool.similarity import detect_duplicates_and_similars
from duplicate_detection_tool.clustering import run_clustering, cluster_analysis
from duplicate_detection_tool.meta_classifier import run_meta_classification

# Streamlit app configuration
st.set_page_config(
    page_title="Duplicate Detection Tool",
    page_icon="🔍",
    layout="wide"
)

# Load and process data
@st.cache_data
def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        # Preprocess data
        df, text_cols, numeric_cols, date_cols = process_data(df)
        return df, text_cols, numeric_cols, date_cols
    else:
        return None, None, None, None

# Combine Non-ML and ML detection results
def combine_results(non_ml_pairs, ml_pairs, weight_non_ml=0.7, weight_ml=0.3):
    """
    Combine Non-ML and ML results with weightage towards Non-ML results.
    """
    combined = {}
    for i, j, score, cols in non_ml_pairs + ml_pairs:
        key = tuple(sorted((i, j)))  # Ensure consistent pair ordering
        if key not in combined:
            combined[key] = {
                "score": 0,
                "contributing_columns": set(),
                "detection_sources": []
            }
        if (i, j, score, cols) in non_ml_pairs:
            combined[key]["score"] += score * weight_non_ml
            combined[key]["detection_sources"].append("Non-ML")
        if (i, j, score, cols) in ml_pairs:
            combined[key]["score"] += score * weight_ml
            combined[key]["detection_sources"].append("ML")
        combined[key]["contributing_columns"].update(cols)
    # Sort combined results by score
    combined_sorted = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)
    return combined_sorted

# App layout
def main():
    st.sidebar.title("🔍 Duplicate Detection Tool")
    st.sidebar.image("duplicate_detection_tool/images/bigspark_logo.png", use_column_width=True)
    st.sidebar.image("duplicate_detection_tool/images/UKRI_logo.png", use_column_width=True)
    st.sidebar.image("duplicate_detection_tool/images/NTU_Primary_logo.png", use_column_width=True)

    # Step 1: File Upload
    st.title("Detect Duplicates and Overly Similar Records")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format):", type=["csv"])

    if uploaded_file is not None:
        # Load and process the data
        st.write("Processing your dataset...")
        df, text_cols, numeric_cols, date_cols = load_and_process_data(uploaded_file)

        if df is not None:
            st.success("Data loaded successfully!")
            st.write(f"**Dataset Shape:** {df.shape}")
            st.write(f"**Text Columns:** {text_cols}")
            st.write(f"**Numeric Columns:** {numeric_cols}")
            st.write(f"**Date Columns:** {date_cols}")

            # Step 2: Perform Non-ML Duplicate and Similar Detection
            st.subheader("Step 2: Non-ML Detection")
            df, non_ml_duplicates, non_ml_similar, unique_records = detect_duplicates_and_similars(
                df, text_cols, numeric_cols, threshold_duplicate=80, threshold_similar=60
            )
            st.write(f"**Non-ML Detection Results:**")
            st.write(f"- Duplicates detected: {len(non_ml_duplicates)}")
            st.write(f"- Overly similar records: {len(non_ml_similar)}")
            st.write(f"- Unique records: {len(unique_records)}")

            # Step 3: Clustering and ML-Based Detection
            st.subheader("Step 3: ML-Based Clustering and Detection")
            df, clustering_metrics = run_clustering(df, text_cols, numeric_cols)
            classifiers_performance = run_meta_classification(df)
            st.write("**Clustering Metrics:**", clustering_metrics)
            st.write("**Meta-Classifier Performance:**")
            for model_name, metrics in classifiers_performance.items():
                st.write(f"{model_name}: {metrics}")

            # Combine Non-ML and ML results
            combined_duplicates = combine_results(non_ml_duplicates, [])
            combined_similar = combine_results(non_ml_similar, [])

            # Step 4: User Action
            st.subheader("Step 4: Inspect Results")
            action = st.radio(
                "What would you like to do?",
                options=["Check Duplicates", "Check Overly Similar Records", "Inspect Unique Records", "Export Results"]
            )

            if action == "Check Duplicates":
                st.write("**Duplicate Pairs:**")
                for (i, j), details in combined_duplicates:
                    st.write(f"**Record Pair:** {i} and {j}")
                    st.write(f"- Similarity Score: {details['score']:.2f}")
                    st.write(f"- Contributing Columns: {', '.join(details['contributing_columns'])}")
                    st.write(f"- Detection Sources: {', '.join(details['detection_sources'])}")

            elif action == "Check Overly Similar Records":
                st.write("**Overly Similar Groups:**")
                for (i, j), details in combined_similar:
                    st.write(f"**Record Pair:** {i} and {j}")
                    st.write(f"- Similarity Score: {details['score']:.2f}")
                    st.write(f"- Contributing Columns: {', '.join(details['contributing_columns'])}")
                    st.write(f"- Detection Sources: {', '.join(details['detection_sources'])}")

            elif action == "Inspect Unique Records":
                st.write("**Unique Records:**")
                for idx, score in unique_records:
                    st.write(f"**Record Index:** {idx}")
                    st.write(f"- Uniqueness Score: {score:.2f}")

            elif action == "Export Results":
                df[df['record_class'] == 2].to_csv("duplicates.csv", index=False)
                df[df['record_class'] == 1].to_csv("overly_similar.csv", index=False)
                df[df['record_class'] == 0].to_csv("unique_records.csv", index=False)
                st.success("Results exported to 'duplicates.csv', 'overly_similar.csv', and 'unique_records.csv'.")

    else:
        st.info("Please upload a dataset to get started.")

if __name__ == "__main__":
    main()
