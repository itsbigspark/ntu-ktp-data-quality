"""
run_evaluation.py
─────────────────
Runs the full DQ engine on TEST3_DATA and computes precision / recall / F1
against the ground truth extracted by extract_ground_truth_test3.py.

Components evaluated:
  Layer 1 — Rule-based validation    (validate_df)
  Layer 2 — Heuristic anomaly        (detect_anomalies)
  Layer 3 — Corpus validation        (CSV-based, no Redis needed)
  Layer 4 — Duplicate detection      (pandas exact + near-dup)
  Layer 5 — ML anomaly               (ml_anomaly_report, 3-component ensemble)

Ablation table produced:
  Rule only → +Heuristic → +Corpus → +Duplicate → +ML → Full engine

Outputs:
  scripts/output/evaluation/
  ├── detected_issues_all.csv          — every issue the engine found
  ├── detected_issues_per_layer.csv    — issues tagged by which layer found them
  ├── pr_f1_overall.csv                — overall P / R / F1 per layer config
  ├── pr_f1_by_issue_type.csv          — P / R / F1 per issue_type (full engine)
  ├── pr_f1_by_column.csv              — P / R / F1 per column (full engine)
  └── evaluation_report.txt           — plain-text summary for the paper

Run:
    python3 scripts/run_evaluation.py
"""

import os, sys, re, json, warnings
from difflib import SequenceMatcher
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(BASE, "TEST3_DATA", "main_data", "insurance_claims.csv")
RULES_PATH  = os.path.join(BASE, "TEST3_DATA", "rules", "validation_rules.json")
CORPUS_DIR  = os.path.join(BASE, "TEST3_DATA", "corpus")
GT_PATH     = os.path.join(BASE, "scripts", "output", "ground_truth_test3.csv")
OUT_DIR     = os.path.join(BASE, "scripts", "output", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Issue type mapping: engine issue names → ground truth categories
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_TO_GT = {
    # ── Rule layer ────────────────────────────────────────────────────────────
    "missing":              "missing_value",
    "missing_value":        "missing_value",
    "format":               "format_error",
    "pattern":              "format_error",
    "type_mismatch":        "format_error",
    "range":                "range_error",
    "out_of_range":         "range_error",
    "allowed_values":       "standardisation_error",
    "invalid_category":     "standardisation_error",
    "cross_column":         "logical_error",
    "logical":              "logical_error",
    "rule_violation":       "logical_error",
    # ── Heuristic layer ───────────────────────────────────────────────────────
    # NOTE: rare_category is a discovery signal — it flags unusual values but
    # does NOT correspond 1:1 to corpus_mismatch. Mapping it to corpus_mismatch
    # would create thousands of false positives. Map to "discovery" which is
    # excluded from exact-match P/R (discovery layer evaluated separately).
    "numeric_outlier":      "range_error",
    "rare_category":        "discovery",        # excluded from exact P/R
    "text_length_outlier":  "discovery",        # excluded from exact P/R
    # ── Corpus layer ──────────────────────────────────────────────────────────
    "corpus_mismatch":      "corpus_mismatch",
    "standardisation_error":"standardisation_error",
    "not_canonical":        "corpus_mismatch",
    # ── Duplicate layer ───────────────────────────────────────────────────────
    "duplicate":            "duplicate",
    "duplicate_value":      "duplicate",
    "near_duplicate":       "near_duplicate",
    # ── ML layer ──────────────────────────────────────────────────────────────
    # ml_anomaly is row-level — excluded from column/type exact P/R;
    # evaluated via row-level recall and lift metrics separately.
    "ml_anomaly":           "ml_anomaly",       # excluded from exact P/R
}

# ─────────────────────────────────────────────────────────────────────────────
# Load data + config
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data…")
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
print(f"  {len(df)} rows × {len(df.columns)} columns")

with open(RULES_PATH) as f:
    rules = json.load(f)

gt = pd.read_csv(GT_PATH, dtype=str)
gt["row_id"] = gt["row_id"].astype(int)
print(f"  Ground truth: {len(gt)} records, {gt['row_id'].nunique()} unique error rows")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Rule-based + heuristic (via validate_and_anomaly_report)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Layer 1+2] Rule-based + heuristic validation…")
from core.validator.validate import validate_and_anomaly_report, validate_df
from core.anomaly import detect_anomalies, ml_anomaly_report

rule_report = validate_df(df, rules, enable_typo_detection=False)
rule_report["layer"] = "rule"

heur_report = detect_anomalies(df, rules)
heur_report["layer"] = "heuristic"

print(f"  Rule issues: {len(rule_report)}")
print(f"  Heuristic issues: {len(heur_report)}")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Corpus validation (CSV-based, no Redis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Layer 3] Corpus validation…")

def load_corpus_alias(filename):
    path = os.path.join(CORPUS_DIR, filename)
    c = pd.read_csv(path, dtype=str)
    alias_map  = {}
    for _, r in c.iterrows():
        alias  = str(r.get("alias", "")).strip()
        canon  = str(r.get("canonical", "")).strip()
        if alias and canon:
            alias_map[alias] = canon
            alias_map[alias.lower()] = canon
    canonicals = set(c["canonical"].dropna().str.strip().unique())
    return alias_map, canonicals

def load_corpus_codes(filename, code_col="code"):
    path = os.path.join(CORPUS_DIR, filename)
    c = pd.read_csv(path, dtype=str)
    return set(c[code_col].dropna().str.strip().unique())

insurer_aliases, insurer_canonicals   = load_corpus_alias("corpus_insurer_names.csv")
status_aliases,  status_canonicals    = load_corpus_alias("corpus_claim_status.csv")
valid_diag_codes                       = load_corpus_codes("corpus_diagnosis_codes.csv")

corpus_rows = []
for idx, row in df.iterrows():
    # Insurer: not canonical → corpus_mismatch
    ins = str(row.get("insurer", "")).strip()
    if ins and ins not in insurer_canonicals:
        corpus_rows.append(dict(
            row_id=idx, column="insurer", issue="corpus_mismatch",
            detail=f"Insurer '{ins}' not in canonical form",
            severity="medium", value=ins, expected=None, rule="corpus_insurer",
            layer="corpus",
        ))

    # Claim status: not canonical → standardisation_error
    st = str(row.get("claim_status", "")).strip()
    if st and st not in status_canonicals:
        corpus_rows.append(dict(
            row_id=idx, column="claim_status", issue="standardisation_error",
            detail=f"Claim status '{st}' not standardised",
            severity="medium", value=st, expected=None, rule="corpus_claim_status",
            layer="corpus",
        ))

    # Diagnosis code: non-empty and not valid ICD-10
    diag = str(row.get("diagnosis_code", "")).strip()
    if diag and diag not in valid_diag_codes:
        corpus_rows.append(dict(
            row_id=idx, column="diagnosis_code", issue="corpus_mismatch",
            detail=f"Diagnosis code '{diag}' not in ICD-10 list",
            severity="medium", value=diag, expected=None, rule="corpus_diagnosis",
            layer="corpus",
        ))

corpus_report = pd.DataFrame(corpus_rows) if corpus_rows else pd.DataFrame(
    columns=["row_id","column","issue","detail","severity","value","expected","rule","layer"]
)
print(f"  Corpus issues: {len(corpus_report)}")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Layer 4] Duplicate detection…")
dup_rows = []

# Exact duplicates
dup_mask = df.duplicated(keep=False)
for idx in df[dup_mask].index:
    dup_rows.append(dict(
        row_id=idx, column="all_columns", issue="duplicate",
        detail="Exact duplicate row", severity="high",
        value=None, expected=None, rule="exact_duplicate", layer="duplicate",
    ))

# Near-duplicates: same policy_id, similar claimant_name
grouped = df.groupby("policy_id")
near_seen = set()
for pol_id, grp in grouped:
    if len(grp) < 2:
        continue
    names   = grp["claimant_name"].tolist()
    indices = grp.index.tolist()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ratio = SequenceMatcher(
                None,
                str(names[i]).strip().lower(),
                str(names[j]).strip().lower(),
            ).ratio()
            if 0.70 < ratio < 1.0:
                for ix in (indices[i], indices[j]):
                    if ix not in near_seen:
                        near_seen.add(ix)
                        dup_rows.append(dict(
                            row_id=ix, column="claimant_name",
                            issue="near_duplicate",
                            detail="Near-duplicate: same policy_id, similar name",
                            severity="medium", value=df.at[ix, "claimant_name"],
                            expected=None, rule="near_duplicate", layer="duplicate",
                        ))

dup_report = pd.DataFrame(dup_rows) if dup_rows else pd.DataFrame(
    columns=["row_id","column","issue","detail","severity","value","expected","rule","layer"]
)
print(f"  Duplicate issues: {len(dup_report)}")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — ML anomaly
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Layer 5] ML anomaly detection (3-component ensemble)…")
ml_raw = ml_anomaly_report(df, None, False, ["IsolationForest"], ensemble=True)
ml_raw["layer"] = "ml"
print(f"  ML issues: {len(ml_raw)}")

# ─────────────────────────────────────────────────────────────────────────────
# Combine all layers
# ─────────────────────────────────────────────────────────────────────────────
COLS = ["row_id", "column", "issue", "detail", "severity", "value", "expected", "rule", "layer"]
for rep in [rule_report, heur_report, corpus_report, dup_report, ml_raw]:
    for c in COLS:
        if c not in rep.columns:
            rep[c] = None

all_issues = pd.concat(
    [rule_report[COLS], heur_report[COLS], corpus_report[COLS],
     dup_report[COLS], ml_raw[COLS]],
    ignore_index=True,
)
all_issues["row_id"] = all_issues["row_id"].astype(int)

# Map to ground truth issue types
all_issues["gt_issue_type"] = (
    all_issues["issue"].str.lower().str.strip()
    .map(ENGINE_TO_GT)
    .fillna("other")
)

all_issues.to_csv(os.path.join(OUT_DIR, "detected_issues_all.csv"), index=False)
print(f"\nTotal detected issues: {len(all_issues)}")
print(f"Layer breakdown:")
print(all_issues["layer"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# P / R / F1 computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_prf(detected_set, gt_set):
    """
    detected_set, gt_set: sets of tuples.
    Returns (precision, recall, f1, tp, fp, fn).
    Matching: exact tuple equality.
    """
    tp = len(detected_set & gt_set)
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return round(P, 4), round(R, 4), round(F1, 4), tp, fp, fn


# Ground truth as set of (row_id, column, issue_type)
gt_set_full = set(zip(
    gt["row_id"].astype(int),
    gt["column"].str.strip(),
    gt["issue_type"].str.strip(),
))

# Ground truth at row-level only (for row-level recall)
gt_rows = set(gt["row_id"].astype(int))

# ── Helper: build detected set from a subset of layers ───────────────────────
# Issue types excluded from exact-match P/R:
# - "discovery": heuristic signals (rare_category, text_length_outlier) — evaluated
#   separately as discovery recall, not exact detection
# - "ml_anomaly": row-level only — evaluated via row-level recall + lift
# - "other": unmapped engine issues
EXCLUDED_FROM_EXACT_PR = {"discovery", "ml_anomaly", "other"}

def detected_set_from(issues_df):
    """
    (row_id, column, gt_issue_type) tuples for exact P/R evaluation.
    Excludes discovery, ml_anomaly, and other unmapped types.
    """
    out = set()
    for _, r in issues_df.iterrows():
        itype = str(r["gt_issue_type"])
        if itype in EXCLUDED_FROM_EXACT_PR:
            continue
        col = "all_columns" if str(r["column"]) == "__row__" else str(r["column"])
        out.add((int(r["row_id"]), col, itype))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ablation table: cumulative layers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ABLATION TABLE — cumulative layers")
print("=" * 65)

layer_order = ["rule", "heuristic", "corpus", "duplicate", "ml"]
ablation_rows = []
cumulative_issues = pd.DataFrame(columns=COLS)

print(f"{'Config':<30} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
print("-" * 70)

for i, layer in enumerate(layer_order):
    new_layer    = all_issues[all_issues["layer"] == layer]
    cumulative_issues = pd.concat([cumulative_issues, new_layer], ignore_index=True)
    det_set      = detected_set_from(cumulative_issues)
    P, R, F1, tp, fp, fn = compute_prf(det_set, gt_set_full)

    config_name = " + ".join(layer_order[:i + 1])
    ablation_rows.append({
        "config": config_name, "P": P, "R": R, "F1": F1,
        "TP": tp, "FP": fp, "FN": fn,
        "n_detected": len(det_set),
    })
    print(f"  {config_name:<28} {P:>7.3f} {R:>7.3f} {F1:>7.3f} {tp:>6} {fp:>6} {fn:>6}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(os.path.join(OUT_DIR, "pr_f1_overall.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Per issue_type breakdown (full engine)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PER ISSUE TYPE — full engine")
print("=" * 65)

all_gt_types = gt["issue_type"].unique().tolist()
type_rows = []

print(f"{'Issue type':<30} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'GT':>5}")
print("-" * 68)

for itype in sorted(all_gt_types):
    gt_sub     = set(zip(
        gt[gt["issue_type"] == itype]["row_id"].astype(int),
        gt[gt["issue_type"] == itype]["column"].str.strip(),
        gt[gt["issue_type"] == itype]["issue_type"].str.strip(),
    ))
    det_sub    = {t for t in detected_set_from(all_issues) if t[2] == itype}
    P, R, F1, tp, fp, fn = compute_prf(det_sub, gt_sub)
    type_rows.append({
        "issue_type": itype, "P": P, "R": R, "F1": F1,
        "TP": tp, "FP": fp, "FN": fn, "GT_count": len(gt_sub),
    })
    print(f"  {itype:<28} {P:>7.3f} {R:>7.3f} {F1:>7.3f} {tp:>5} {fp:>5} {fn:>5} {len(gt_sub):>5}")

type_df = pd.DataFrame(type_rows)
type_df.to_csv(os.path.join(OUT_DIR, "pr_f1_by_issue_type.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Per column breakdown (full engine)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PER COLUMN — full engine")
print("=" * 65)

all_gt_cols = gt["column"].unique().tolist()
col_rows = []

print(f"{'Column':<25} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'GT':>5}")
print("-" * 55)

for col in sorted(all_gt_cols):
    gt_sub  = set(zip(
        gt[gt["column"] == col]["row_id"].astype(int),
        gt[gt["column"] == col]["column"].str.strip(),
        gt[gt["column"] == col]["issue_type"].str.strip(),
    ))
    det_sub = {t for t in detected_set_from(all_issues) if t[1] == col}
    P, R, F1, tp, fp, fn = compute_prf(det_sub, gt_sub)
    col_rows.append({
        "column": col, "P": P, "R": R, "F1": F1,
        "TP": tp, "FP": fp, "FN": fn, "GT_count": len(gt_sub),
    })
    print(f"  {col:<23} {P:>7.3f} {R:>7.3f} {F1:>7.3f} {tp:>5} {len(gt_sub):>5}")

col_df = pd.DataFrame(col_rows)
col_df.to_csv(os.path.join(OUT_DIR, "pr_f1_by_column.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Row-level recall (did we touch the right rows at all?)
# ─────────────────────────────────────────────────────────────────────────────

# Deterministic-only rows (rule + corpus + duplicate) — no heuristic noise
det_only = all_issues[all_issues["layer"].isin(["rule", "corpus", "duplicate"])]
det_rows_strict = set(det_only["row_id"].astype(int))
strict_tp = len(det_rows_strict & gt_rows)
strict_P  = strict_tp / len(det_rows_strict) if det_rows_strict else 0
strict_R  = strict_tp / len(gt_rows) if gt_rows else 0
strict_F1 = 2*strict_P*strict_R/(strict_P+strict_R) if (strict_P+strict_R) > 0 else 0

# All layers
detected_rows = set(all_issues["row_id"].astype(int))
row_tp = len(detected_rows & gt_rows)
row_P  = row_tp / len(detected_rows) if detected_rows else 0
row_R  = row_tp / len(gt_rows) if gt_rows else 0
row_F1 = 2*row_P*row_R/(row_P+row_R) if (row_P+row_R) > 0 else 0

print(f"\nROW-LEVEL RECALL:")
print(f"  Deterministic only (rule+corpus+dup): {len(det_rows_strict)} rows flagged")
print(f"    P={strict_P:.3f}  R={strict_R:.3f}  F1={strict_F1:.3f}")
print(f"  All layers: {len(detected_rows)} rows flagged")
print(f"    P={row_P:.3f}  R={row_R:.3f}  F1={row_F1:.3f}")

# ── Discovery recall: what fraction of error rows does heuristic touch? ──────
heur_rows = set(all_issues[all_issues["layer"] == "heuristic"]["row_id"].astype(int))
disc_tp   = len(heur_rows & gt_rows)
disc_rec  = disc_tp / len(gt_rows) if gt_rows else 0
disc_pre  = disc_tp / len(heur_rows) if heur_rows else 0
print(f"\nDISCOVERY (heuristic layer — not exact P/R):")
print(f"  Touches {len(heur_rows)} rows, {disc_tp} of which have errors")
print(f"  Discovery recall={disc_rec:.3f}  Discovery precision={disc_pre:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plain-text report
# ─────────────────────────────────────────────────────────────────────────────
full_engine = ablation_df.iloc[-1]

report_lines = [
    "=" * 65,
    "DQ ENGINE EVALUATION REPORT — TEST3_DATA (Insurance Claims)",
    "=" * 65,
    "",
    f"Dataset:       TEST3_DATA/main_data/insurance_claims.csv",
    f"Rows:          {len(df)}",
    f"Columns:       {len(df.columns)}",
    f"GT records:    {len(gt)} ({gt['row_id'].nunique()} unique error rows)",
    f"GT issue types:{', '.join(sorted(gt['issue_type'].unique()))}",
    "",
    "FULL ENGINE RESULT",
    "-" * 40,
    f"  Precision:  {full_engine['P']:.3f}",
    f"  Recall:     {full_engine['R']:.3f}",
    f"  F1:         {full_engine['F1']:.3f}",
    f"  TP: {int(full_engine['TP'])}  FP: {int(full_engine['FP'])}  FN: {int(full_engine['FN'])}",
    "",
    "ROW-LEVEL RECALL",
    "-" * 40,
    f"  P={row_P:.3f}  R={row_R:.3f}  F1={row_F1:.3f}",
    f"  (Detected {len(detected_rows)} rows, {len(gt_rows)} had errors)",
    "",
    "ABLATION TABLE",
    "-" * 40,
]
for _, ar in ablation_df.iterrows():
    report_lines.append(
        f"  {ar['config']:<28} P={ar['P']:.3f}  R={ar['R']:.3f}  F1={ar['F1']:.3f}"
    )

report_lines += [
    "",
    "PER ISSUE TYPE (full engine)",
    "-" * 40,
]
for _, tr in type_df.sort_values("F1", ascending=False).iterrows():
    report_lines.append(
        f"  {tr['issue_type']:<28} P={tr['P']:.3f}  R={tr['R']:.3f}  F1={tr['F1']:.3f}  (GT={tr['GT_count']})"
    )

report_lines += [
    "",
    "ML COMPONENT CONTRIBUTION",
    "-" * 40,
    f"  ML flagged {len(ml_raw)} rows ({len(ml_raw)/len(df)*100:.1f}%)",
    f"  ML lift vs random: ~1.35x (measured in ml_diagnostic.py)",
    "",
    "=" * 65,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(os.path.join(OUT_DIR, "evaluation_report.txt"), "w") as f:
    f.write(report_text)

print(f"\nAll outputs saved to: {OUT_DIR}/")
