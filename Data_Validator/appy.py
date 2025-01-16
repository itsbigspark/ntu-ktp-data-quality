import streamlit as st
import pandas as pd
import json
import os
from data_validator.patterns import infer_regex_patterns
from data_validator.metrics import calculate_metrics
from data_validator.cleaning import create_cleaning_rules, apply_cleaning_rules
from data_validator.synthetic import generate_synthetic_data
from data_validator.visualization import visualize_issues, visualize_metrics

# Debug function to inspect DataFrame
def debug_dataframe(df, name="DataFrame"):
    st.write(f"Debug info for {name}:")
    st.write(f"Columns: {list(df.columns)}")
    st.write(f"Shape: {df.shape}")
    st.write("First few rows:")
    st.write(df.head())
    st.write("Data types:")
    st.write(df.dtypes)

# Helper Function: Apply Conditional Formatting to DataFrame
def style_dataframe(df):
    """
    Apply conditional formatting to a pandas DataFrame.
    Highlight "MISSING" in yellow and "INVALID" in red.
    Returns the styled DataFrame or the original if styling fails.
    """
    try:
        def highlight_cells(val):
            if pd.isna(val) or val is None:
                return ""
            val_str = str(val).upper()
            if val_str == "MISSING":
                return "background-color: yellow; color: black;"
            elif val_str == "INVALID":
                return "background-color: red; color: white;"
            return ""

        return df.style.applymap(highlight_cells)
    except Exception as e:
        st.warning(f"Warning: Could not apply styling to DataFrame: {str(e)}")
        return df

def process_data(error_data, reference_data, custom_rules):
    st.markdown("## Processing Files...")
    
    # Debug information for input data
    st.markdown("### Debug Information")
    debug_dataframe(error_data, "Error Data")
    if reference_data is not None:
        debug_dataframe(reference_data, "Reference Data")
    
    with st.spinner("Processing... Please wait."):
        try:
            # Infer Patterns and Calculate Metrics
            st.markdown("### Step 2: Inferring Patterns and Calculating Metrics")
            patterns = {}
            metrics = {}
            
            if reference_data is not None:
                patterns = infer_regex_patterns(reference_data)
                metrics = calculate_metrics(reference_data)
                st.write("Patterns and metrics inferred from reference data")
            
            if custom_rules:
                st.write("Updating with custom rules")
                patterns.update(custom_rules.get("patterns", {}))
                metrics.update(custom_rules.get("metrics", {}))

            st.json({"Patterns": patterns, "Metrics": metrics})

            # Apply Cleaning Rules
            st.markdown("### Step 3: Applying Cleaning Rules")
            try:
                cleaning_rules = create_cleaning_rules(patterns, metrics)
                st.write("Cleaning rules created successfully")
                cleaned_data, highlighted_issues = apply_cleaning_rules(error_data, cleaning_rules)
                st.write("Cleaning rules applied successfully")
            except Exception as e:
                st.error(f"Error in cleaning process: {str(e)}")
                st.write("Cleaning rules or application failed")
                return

            # Display Cleaned Data
            st.markdown("#### Cleaned Data")
            st.dataframe(cleaned_data.head())
            st.download_button(
                "Download Cleaned Data",
                cleaned_data.to_csv(index=False),
                file_name="cleaned_data.csv"
            )

            # Display Highlighted Issues
            st.markdown("#### Highlighted Issues")
            if highlighted_issues is not None and not highlighted_issues.empty:
                try:
                    # First display raw data for debugging
                    st.write("Raw Highlighted Issues:")
                    st.dataframe(highlighted_issues)
                    
                    # Then try to apply styling
                    styled_issues = style_dataframe(highlighted_issues)
                    st.write("Styled Highlighted Issues:")
                    st.dataframe(styled_issues)
                    
                    st.download_button(
                        "Download Highlighted Issues",
                        highlighted_issues.to_csv(index=False),
                        file_name="highlighted_issues.csv"
                    )
                except Exception as e:
                    st.error(f"Error displaying highlighted issues: {str(e)}")
                    st.write("Displaying raw issues instead:")
                    st.dataframe(highlighted_issues)
            else:
                st.info("No issues detected.")

            # Generate Synthetic Data
            try:
                st.markdown("### Step 4: Generating Synthetic Data")
                synthetic_data = generate_synthetic_data(metrics, patterns, num_rows=100)
                st.dataframe(synthetic_data.head())
                st.download_button(
                    "Download Synthetic Data",
                    synthetic_data.to_csv(index=False),
                    file_name="synthetic_data.csv"
                )
            except Exception as e:
                st.error(f"Error generating synthetic data: {str(e)}")

            # Generate Visualizations
            try:
                st.markdown("### Step 5: Visualizations")
                if not highlighted_issues.empty:
                    visualize_issues(highlighted_issues, "outputs/visualizations/issue_distribution.png")
                    if os.path.exists("outputs/visualizations/issue_distribution.png"):
                        st.image("outputs/visualizations/issue_distribution.png", caption="Issue Distribution")
                
                if metrics:
                    visualize_metrics(metrics, "outputs/visualizations/metrics_plot.png")
                    if os.path.exists("outputs/visualizations/metrics_plot.png"):
                        st.image("outputs/visualizations/metrics_plot.png", caption="Metrics Plot")
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")

            st.success("Processing complete!")
            
        except KeyError as ke:
            st.error(f"KeyError: Missing required column: {str(ke)}")
            st.write("Please ensure your data contains all required columns.")
            if reference_data is not None:
                st.write("Expected columns (from reference data):", list(reference_data.columns))
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.write("Please check your input data and try again.")

# Main App Function (rest of the code remains the same)
def main():
    st.markdown("<h1 style='text-align: center;'>🔧 Data Validation Tool</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload datasets to detect and correct errors automatically</h4>", unsafe_allow_html=True)

    try:
        col1, col2, col3 = st.columns([1.3, 2, 1.5])
        with col1:
            st.image("images/bigspark_logo.png", use_column_width=True)
        with col2:
            st.image("images/UKRI_logo.png", use_column_width=True)
        with col3:
            st.image("images/NTU_Primary_logo.png", use_column_width=True)
    except Exception:
        st.warning("Note: Some logos could not be loaded.")

    st.write("---")

    st.markdown("### Step 1: Upload Files")
    error_file = st.file_uploader("Upload Error Data File (.csv)", type="csv", key="error_file")
    reference_file = st.file_uploader("Upload Reference Data File (.csv) (Optional)", type="csv", key="reference_file")
    custom_rules_file = st.file_uploader("Upload Custom Rules (JSON) (Optional)", type="json", key="custom_rules_file")

    error_data, reference_data, custom_rules = None, None, None

    if error_file:
        try:
            error_data = pd.read_csv(error_file, dtype=str, na_filter=False)
            st.success("Error Data File Uploaded Successfully!")
            st.write("Error Data Preview:")
            st.dataframe(error_data.head())
        except Exception as e:
            st.error(f"Error reading Error Data file: {e}")

    if reference_file:
        try:
            reference_data = pd.read_csv(reference_file, dtype=str, na_filter=False)
            st.success("Reference Data File Uploaded Successfully!")
            st.write("Reference Data Preview:")
            st.dataframe(reference_data.head())
        except Exception as e:
            st.error(f"Error reading Reference Data file: {e}")

    if custom_rules_file:
        try:
            custom_rules = json.load(custom_rules_file)
            st.success("Custom Rules File Uploaded Successfully!")
            st.write("Custom Rules Preview:")
            st.json(custom_rules)
        except Exception as e:
            st.error(f"Error reading Custom Rules file: {e}")

    if st.button("Start Validation and Cleaning"):
        if error_data is not None:
            process_data(error_data, reference_data, custom_rules)
        else:
            st.error("Please upload an Error Data File to proceed.")

if __name__ == "__main__":
    main()