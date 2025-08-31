import pandas as pd

def resolve_conflict(val1, val2, strategy):
    if strategy == "prefer_dataset_1":
        return val1
    elif strategy == "prefer_dataset_2":
        return val2
    elif strategy == "prefer_non_null":
        if pd.notna(val1) and val1 != "":
            return val1
        elif pd.notna(val2) and val2 != "":
            return val2
        else:
            return val1  # fallback
    else:
        return val1  # default

def aggregate_resolved_entities(matches_df, dfs, conflict_strategies):
    dataset_names = list(dfs.keys())
    df1, df2 = dfs[dataset_names[0]], dfs[dataset_names[1]]
    resolved_rows = []

    for _, match in matches_df.iterrows():
        row1 = df1.iloc[int(match["Row_Dataset1"])]
        row2 = df2.iloc[int(match["Row_Dataset2"])]

        resolved_row = {}

        for col in conflict_strategies.keys():
            val1 = row1[col] if col in row1 else None
            val2 = row2[col] if col in row2 else None
            resolved_value = resolve_conflict(val1, val2, conflict_strategies[col])
            resolved_row[col] = resolved_value

        # Add traceability + similarity
        resolved_row["Row_Dataset1"] = int(match["Row_Dataset1"])
        resolved_row["Row_Dataset2"] = int(match["Row_Dataset2"])
        resolved_row["Final_Similarity"] = round(match["Final_Similarity"], 4)

        resolved_rows.append(resolved_row)

    return pd.DataFrame(resolved_rows)