# app.py (Enhanced with Aggregated Preview before Download)

import streamlit as st
import pandas as pd
from modules.preprocessing import standardize_datasets
from modules.resolution import run_entity_resolution
from modules.aggregation import aggregate_resolved_entities
from io import StringIO
import yaml
from difflib import get_close_matches
import matplotlib.pyplot as plt

st.set_page_config(page_title="Entity Resolution App", layout="wide")
st.title("🔗 Entity Resolution for Multiple Datasets")

# Step 1: Upload CSV files
st.header("1. Upload CSV Files")
uploaded_files = st.file_uploader("Upload multiple CSV files", accept_multiple_files=True, type="csv")

if uploaded_files and len(uploaded_files) >= 2:
    dfs = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        df.columns = [col.strip().lower() for col in df.columns]
        dfs[file.name] = df

    st.success(f"{len(dfs)} file(s) loaded.")

    # Step 2: Show previews
    st.header("2. Preview Datasets")
    for name, df in dfs.items():
        st.subheader(f"Preview: {name}")
        st.dataframe(df.head())

    # Step 3: Find Common Columns
    st.header("3. Standardization Setup")
    all_columns = [set(df.columns) for df in dfs.values()]
    common_columns = sorted(list(set.intersection(*all_columns)))

    if not common_columns:
        st.warning("No exact common columns found. Please map similar columns manually.")
        st.stop()

    mode = st.radio("Choose Standardization Mode:", [
        "Select a dataset as standard",
        "Manually define standardization rules",
        "Select per-column source",
        "Use majority voting per column"
    ])

    standard_dataset = None
    manual_rules = {}
    abbr_dict = {}
    column_sources = {}

    if mode == "Select a dataset as standard":
        standard_dataset = st.selectbox("Choose the standard reference dataset:", list(dfs.keys()))

    elif mode == "Manually define standardization rules":
        st.subheader("Manual Rule Configuration")
        for col in common_columns:
            with st.expander(f"Rules for '{col}'"):
                lowercase = st.checkbox("Lowercase", key=f"lower_{col}")
                remove_punct = st.checkbox("Remove punctuation", key=f"punct_{col}")
                strip_currency = st.checkbox("Strip currency symbols", key=f"curr_{col}")
                manual_rules[col] = {
                    "lowercase": lowercase,
                    "remove_punctuation": remove_punct,
                    "strip_currency": strip_currency
                }

    elif mode == "Select per-column source":
        for col in common_columns:
            dataset_choice = st.selectbox(f"Choose source for column '{col}':", list(dfs.keys()), key=f"src_{col}")
            column_sources[col] = dataset_choice

    # Step 4: Abbreviation YAML
    st.header("4. Load Abbreviations (YAML)")
    yaml_file = st.file_uploader("Upload abbreviation mapping (YAML format)", type="yaml")
    if yaml_file:
        abbr_dict = yaml.safe_load(yaml_file)
        st.success("Abbreviation mapping loaded.")

    # Step 5: Apply Standardization
    if st.button("✅ Apply Standardization"):
        cleaned_dfs, preview_samples = standardize_datasets(
            dfs, common_columns, mode,
            standard_dataset=standard_dataset,
            manual_rules=manual_rules,
            abbr_dict=abbr_dict,
            column_sources=column_sources
        )

        st.success("Standardization complete.")
        st.header("5. Preview Standardized Columns")
        for col, samples in preview_samples.items():
            st.subheader(f"Column: {col}")
            st.dataframe(pd.DataFrame(samples).rename(columns={"original": "Before", "standardized": "After"}))

        # Step 6: Conflict Strategy
        st.header("6. Set Conflict Resolution Strategy")
        conflict_strategies = {}
        for col in common_columns:
            strategy = st.selectbox(
                f"Conflict strategy for '{col}':",
                ["prefer_dataset_1", "prefer_dataset_2", "prefer_non_null"],
                key=f"conflict_{col}"
            )
            conflict_strategies[col] = strategy

        # Step 7: Run Multi-Dataset Resolution
        st.header("7. Run Multi-Dataset Entity Resolution")

        all_dataset_names = list(cleaned_dfs.keys())
        base_df = cleaned_dfs[all_dataset_names[0]].copy()
        base_name = all_dataset_names[0]

        all_matches = []

        for next_name in all_dataset_names[1:]:
            next_df = cleaned_dfs[next_name]
            pair_dfs = {"base": base_df, "incoming": next_df}

            matches_df = run_entity_resolution(pair_dfs)
            all_matches.append(matches_df)

            # Merge matched rows
            base_df = aggregate_resolved_entities(matches_df, pair_dfs, conflict_strategies)
            base_name = f"resolved_{base_name}_{next_name}"

        st.success("✅ Final Resolved Dataset")

        # Slider to filter by Final Similarity
        sim_threshold = st.slider("Filter by minimum Final Similarity", 0.0, 1.0, 0.5, 0.01)
        filtered_df = base_df[base_df["Final_Similarity"] >= sim_threshold].copy()

        # 📌 Show final aggregated preview
        st.subheader("📦 Aggregated Final Resolved Dataset")
        st.dataframe(filtered_df.head(50))

        # Show distribution of similarity
        st.subheader("📊 Similarity Score Distribution")
        fig, ax = plt.subplots()
        filtered_df["Final_Similarity"].hist(ax=ax, bins=20)
        ax.set_xlabel("Final Similarity")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Show pairwise previews if available
        if all_matches:
            st.subheader("🧬 Side-by-Side Previews")
            preview_cols = [col for col in filtered_df.columns if col not in ["Row_Dataset1", "Row_Dataset2", "Final_Similarity"]]
            for i, row in filtered_df.iterrows():
                with st.expander(f"Match {i} - Similarity: {row['Final_Similarity']}"):
                    st.write({col: row[col] for col in preview_cols})

        # Step 8: Download Final Output
        st.download_button("📥 Download Final Resolved Dataset", filtered_df.to_csv(index=False).encode("utf-8"), file_name="resolved_entities.csv", mime="text/csv")

else:
    st.info("Upload at least two CSV files to begin.")