# modules/aggregation.py
import pandas as pd

def resolve_conflict(val1, val2, strategy):
    if strategy == "prefer_dataset_1":
        return val1
    elif strategy == "prefer_dataset_2":
        return val2
    elif strategy == "prefer_non_null":
        return val1 if pd.notnull(val1) and val1 != "" else val2
    else:
        return val1

def aggregate_resolved_entities(matches_df, dfs, conflict_strategies):
    df1, df2 = dfs["base"].copy(), dfs["incoming"].copy()
    name1, name2 = dfs["base"].attrs.get("source_name", "base"), dfs["incoming"].attrs.get("source_name", "incoming")

    resolved_entities = []

    for _, row in matches_df.iterrows():
        i1, i2 = row["Row_Dataset1"], row["Row_Dataset2"]
        row1 = df1.loc[i1]
        row2 = df2.loc[i2]

        resolved_row = {}
        for col in set(df1.columns).intersection(df2.columns):
            resolved_val = resolve_conflict(row1[col], row2[col], conflict_strategies.get(col, "prefer_dataset_1"))
            resolved_row[col] = resolved_val
            resolved_row[f"original_{col}_from_{name1}"] = row1[col]
            resolved_row[f"original_{col}_from_{name2}"] = row2[col]

        resolved_row["Row_Dataset1"] = i1
        resolved_row["Row_Dataset2"] = i2
        resolved_row["Final_Similarity"] = row["Final_Similarity"]
        resolved_row["Source_Chain"] = (
            row1.get("Source_Chain", name1) + " + " + row2.get("Source_Chain", name2)
        )
        resolved_entities.append(resolved_row)

    resolved_df = pd.DataFrame(resolved_entities)

    matched_1 = matches_df["Row_Dataset1"].unique()
    matched_2 = matches_df["Row_Dataset2"].unique()

    unmatched_1 = df1.drop(index=matched_1, errors="ignore").copy()
    unmatched_1["Source_Chain"] = name1
    unmatched_1["Final_Similarity"] = None
    unmatched_1["Row_Dataset1"] = unmatched_1.index
    unmatched_1["Row_Dataset2"] = None

    unmatched_2 = df2.drop(index=matched_2, errors="ignore").copy()
    unmatched_2["Source_Chain"] = name2
    unmatched_2["Final_Similarity"] = None
    unmatched_2["Row_Dataset1"] = None
    unmatched_2["Row_Dataset2"] = unmatched_2.index

    all_cols = resolved_df.columns.union(unmatched_1.columns).union(unmatched_2.columns)
    resolved_df = resolved_df.reindex(columns=all_cols, fill_value=None)
    unmatched_1 = unmatched_1.reindex(columns=all_cols, fill_value=None)
    unmatched_2 = unmatched_2.reindex(columns=all_cols, fill_value=None)

    final_df = pd.concat([resolved_df, unmatched_1, unmatched_2], ignore_index=True)
    return final_df
