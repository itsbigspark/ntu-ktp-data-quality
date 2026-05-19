"""
ml_diagnostic.py
────────────────
Experiment 3 — Score histogram
Experiment 4 — Beat random

Runs both experiments on TEST2_DATA (banking) and TEST3_DATA (insurance)
and saves results + plots to scripts/output/ml_diagnostic/.

Does NOT import core.anomaly — re-implements the same TF-IDF ensemble
logic inline so it can run standalone without the full app environment.

Run:
    python3 scripts/ml_diagnostic.py

Outputs:
    scripts/output/ml_diagnostic/
    ├── exp3_score_histogram.png   — score distributions, both datasets side by side
    ├── exp3_scores_test2.csv      — raw scores for TEST2_DATA
    ├── exp3_scores_test3.csv      — raw scores for TEST3_DATA
    ├── exp4_beat_random.png       — ML vs random flagging, both datasets
    └── ml_diagnostic_report.txt  — plain-text summary of all findings
"""

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST2 = os.path.join(BASE, "TEST2_DATA", "main_data", "customer_transactions.csv")
TEST3 = os.path.join(BASE, "TEST3_DATA", "main_data", "insurance_claims.csv")
GT3   = os.path.join(BASE, "scripts", "output", "ground_truth_test3.csv")
OUT   = os.path.join(BASE, "scripts", "output", "ml_diagnostic")
os.makedirs(OUT, exist_ok=True)

RANDOM_SEEDS = 100          # Number of random baselines to average
FLAGGING_PCT = 0.15         # Matches the hardcoded quantile(0.85) in the app

# ─────────────────────────────────────────────────────────────────────────────
# Matrix dark theme for plots
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#000500"
GREEN    = "#00ff41"
CYAN     = "#00e5ff"
ORANGE   = "#ff9100"
RED      = "#ff1744"
YELLOW   = "#ffea00"
TEXT_DIM = "#5a9a5a"

def apply_dark_theme(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    ax.spines["bottom"].set_color(TEXT_DIM)
    ax.spines["left"].set_color(TEXT_DIM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    if title:
        ax.set_title(title, color=GREEN, fontsize=10, pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# Core ML scorer — mirrors ml_anomaly_report logic in core/anomaly.py
# Returns raw scores (0–1) WITHOUT applying the quantile cutoff.
# ─────────────────────────────────────────────────────────────────────────────
def compute_ml_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Returns array of shape (n_rows,) with ensemble anomaly scores in [0,1].
    Higher = more anomalous.

    Three components (same as the app):
    1. TF-IDF char n-gram vectoriser → IsolationForest
    2. Per-column TF-IDF → IsolationForest on concatenated token features
    3. Numeric IsolationForest on numeric columns only

    Final score = mean of available component scores.
    """
    n = len(df)
    component_scores = []

    # ── Component 1: Row-level TF-IDF (full row as string) ───────────────────
    try:
        row_strings = df.fillna("").astype(str).apply(
            lambda r: " ".join(r.values), axis=1
        ).tolist()
        tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                max_features=500, sublinear_tf=True)
        X_tfidf = tfidf.fit_transform(row_strings).toarray()
        iso1 = IsolationForest(n_estimators=100, contamination=0.15,
                               random_state=42)
        raw1 = iso1.fit_predict(X_tfidf)          # -1 = anomaly, 1 = normal
        scores1 = iso1.score_samples(X_tfidf)     # lower = more anomalous
        # Normalise to [0,1] where 1 = most anomalous
        s1 = (scores1 - scores1.min()) / (scores1.max() - scores1.min() + 1e-9)
        s1 = 1 - s1   # invert: high score = anomalous
        component_scores.append(s1)
    except Exception as e:
        print(f"  [warn] Component 1 failed: {e}")

    # ── Component 2: Per-column TF-IDF ───────────────────────────────────────
    try:
        col_vecs = []
        for col in df.columns:
            col_vals = df[col].fillna("").astype(str).tolist()
            if len(set(col_vals)) < 3:
                continue
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                                  max_features=50, sublinear_tf=True)
            try:
                v = vec.fit_transform(col_vals).toarray()
                col_vecs.append(v)
            except Exception:
                pass
        if col_vecs:
            X_col = np.hstack(col_vecs)
            iso2 = IsolationForest(n_estimators=100, contamination=0.15,
                                   random_state=42)
            iso2.fit(X_col)
            scores2 = iso2.score_samples(X_col)
            s2 = (scores2 - scores2.min()) / (scores2.max() - scores2.min() + 1e-9)
            s2 = 1 - s2
            component_scores.append(s2)
    except Exception as e:
        print(f"  [warn] Component 2 failed: {e}")

    # ── Component 3: Numeric IsolationForest ─────────────────────────────────
    try:
        num_cols = []
        for col in df.columns:
            try:
                numeric = pd.to_numeric(
                    df[col].astype(str).str.replace("£", "", regex=False)
                                       .str.replace(",", "", regex=False),
                    errors="coerce"
                )
                if numeric.notna().sum() > 10:
                    num_cols.append(numeric.fillna(numeric.median()))
            except Exception:
                pass
        if num_cols:
            X_num = np.column_stack(num_cols)
            X_num = StandardScaler().fit_transform(X_num)
            iso3 = IsolationForest(n_estimators=100, contamination=0.15,
                                   random_state=42)
            iso3.fit(X_num)
            scores3 = iso3.score_samples(X_num)
            s3 = (scores3 - scores3.min()) / (scores3.max() - scores3.min() + 1e-9)
            s3 = 1 - s3
            component_scores.append(s3)
    except Exception as e:
        print(f"  [warn] Component 3 failed: {e}")

    if not component_scores:
        raise RuntimeError("All ML components failed — cannot compute scores")

    return np.mean(component_scores, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth helpers
# ─────────────────────────────────────────────────────────────────────────────
def rows_with_errors_test3(gt_path):
    """Return set of row_ids that have at least one error (from extractor output)."""
    gt = pd.read_csv(gt_path, dtype=str)
    return set(gt["row_id"].astype(int).unique())

def rows_with_errors_test2_approx(df):
    """
    Approximate ground truth for TEST2_DATA — no exact file exists.
    Detect obvious errors directly from data (subset of what the manifest describes).
    Returns set of row indices.
    """
    import re
    error_rows = set()
    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    GOOD_DOMAINS = {"gmail.com","hotmail.com","yahoo.com","outlook.com",
                    "icloud.com","btinternet.com"}
    ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    for idx, row in df.iterrows():
        # Bad/missing email
        email = str(row.get("email", "")).strip()
        if not email or not EMAIL_RE.match(email):
            error_rows.add(idx); continue
        domain = email.split("@")[-1].lower()
        if domain not in GOOD_DOMAINS:
            error_rows.add(idx); continue

        # Missing phone
        if not str(row.get("phone", "")).strip():
            error_rows.add(idx); continue

        # Currency in amount
        amt = str(row.get("transaction_amount", row.get("amount", ""))).strip()
        if "£" in amt or "$" in amt:
            error_rows.add(idx); continue

        # Non-ISO date
        for dcol in ["transaction_date", "date", "created_at"]:
            d = str(row.get(dcol, "")).strip()
            if d and not ISO_RE.match(d):
                error_rows.add(idx)
                break

    return error_rows


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Score histogram
# ─────────────────────────────────────────────────────────────────────────────
def experiment3(df2, df3):
    print("\n[Experiment 3] Computing ML scores...")
    print(f"  TEST2_DATA ({len(df2)} rows)...")
    scores2 = compute_ml_scores(df2)
    print(f"  TEST3_DATA ({len(df3)} rows)...")
    scores3 = compute_ml_scores(df3)

    # Save raw scores
    pd.DataFrame({"row_id": range(len(scores2)), "ml_score": scores2}).to_csv(
        os.path.join(OUT, "exp3_scores_test2.csv"), index=False
    )
    pd.DataFrame({"row_id": range(len(scores3)), "ml_score": scores3}).to_csv(
        os.path.join(OUT, "exp3_scores_test3.csv"), index=False
    )

    # Plot
    fig = plt.figure(figsize=(14, 5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    threshold2 = np.quantile(scores2, 0.85)
    threshold3 = np.quantile(scores3, 0.85)

    for ax_idx, (scores, name, color, threshold) in enumerate([
        (scores2, "TEST2_DATA — Banking (500 rows)",     CYAN,   threshold2),
        (scores3, "TEST3_DATA — Insurance (1,085 rows)", ORANGE, threshold3),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        ax.hist(scores, bins=40, color=color, alpha=0.75, edgecolor=BG)
        ax.axvline(threshold, color=RED, linestyle="--", linewidth=1.5,
                   label=f"0.85 quantile = {threshold:.3f}")
        apply_dark_theme(ax, name)
        ax.set_xlabel("Ensemble anomaly score (0=normal, 1=anomalous)")
        ax.set_ylabel("Row count")
        ax.legend(fontsize=8, labelcolor=RED,
                  facecolor="#0a0f0a", edgecolor=TEXT_DIM)

        # Bimodal vs unimodal annotation
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(scores, bw_method=0.15)
        xs  = np.linspace(0, 1, 200)
        ys  = kde(xs)
        # Count local maxima in KDE
        peaks = [i for i in range(1, len(ys) - 1)
                 if ys[i] > ys[i-1] and ys[i] > ys[i+1] and ys[i] > ys.max() * 0.1]
        shape = "BIMODAL" if len(peaks) >= 2 else "UNIMODAL"
        shape_color = GREEN if shape == "BIMODAL" else YELLOW
        ax.text(0.97, 0.97, shape, transform=ax.transAxes,
                ha="right", va="top", color=shape_color,
                fontsize=11, fontweight="bold",
                fontfamily="monospace")

    fig.suptitle("Experiment 3 — ML Anomaly Score Distribution",
                 color=GREEN, fontsize=13, y=1.02)
    plt.savefig(os.path.join(OUT, "exp3_score_histogram.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()

    print(f"  Saved: exp3_score_histogram.png")
    return scores2, scores3, threshold2, threshold3


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Beat random
# ─────────────────────────────────────────────────────────────────────────────
def experiment4(df2, df3, scores2, scores3, gt3_path):
    print("\n[Experiment 4] Beat random comparison...")

    results = {}

    for label, df, scores, gt_type in [
        ("TEST2_DATA (banking)",   df2, scores2, "approx"),
        ("TEST3_DATA (insurance)", df3, scores3, "exact"),
    ]:
        n = len(df)
        n_flag = int(n * FLAGGING_PCT)

        # Ground truth
        if gt_type == "approx":
            error_rows = rows_with_errors_test2_approx(df)
            gt_label = "approximate"
        else:
            error_rows = rows_with_errors_test3(gt3_path)
            gt_label = "exact"

        total_errors = len(error_rows)
        random_baseline_pct = total_errors / n

        # ML: top 15% by score
        ml_flagged = set(np.argsort(scores)[::-1][:n_flag])
        ml_hits    = len(ml_flagged & error_rows)
        ml_pct     = ml_hits / n_flag if n_flag else 0

        # Random: average over RANDOM_SEEDS seeds
        rng = random.Random(42)
        random_hits_list = []
        for seed in range(RANDOM_SEEDS):
            rng.seed(seed)
            rand_flagged = set(rng.sample(range(n), n_flag))
            random_hits_list.append(len(rand_flagged & error_rows))
        random_hits_avg  = np.mean(random_hits_list)
        random_hits_std  = np.std(random_hits_list)
        random_pct_avg   = random_hits_avg / n_flag if n_flag else 0

        beats = ml_hits > (random_hits_avg + random_hits_std)

        results[label] = {
            "n_rows":         n,
            "n_flagged":      n_flag,
            "n_errors":       total_errors,
            "gt_type":        gt_label,
            "ml_hits":        ml_hits,
            "ml_error_rate":  ml_pct,
            "random_hits_avg":random_hits_avg,
            "random_hits_std":random_hits_std,
            "random_error_rate": random_pct_avg,
            "beats_random":   beats,
            "lift":           (ml_pct / random_pct_avg) if random_pct_avg > 0 else 0,
        }

        status = "BEATS RANDOM" if beats else "DOES NOT BEAT RANDOM"
        print(f"  {label}")
        print(f"    Errors known ({gt_label}): {total_errors} / {n} rows ({100*total_errors/n:.1f}%)")
        print(f"    ML flagged {n_flag} rows → hits {ml_hits} errors ({100*ml_pct:.1f}% of flagged)")
        print(f"    Random baseline: {random_hits_avg:.1f} ± {random_hits_std:.1f} errors ({100*random_pct_avg:.1f}%)")
        print(f"    Verdict: {status}  (lift = {results[label]['lift']:.2f}x)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, (label, r) in zip(axes, results.items()):
        categories = ["ML\n(top 15%)", "Random\n(avg)"]
        values     = [r["ml_hits"], r["random_hits_avg"]]
        errors_bar = [0, r["random_hits_std"]]
        colors     = [GREEN if r["beats_random"] else RED, TEXT_DIM]

        bars = ax.bar(categories, values, color=colors, alpha=0.8,
                      yerr=errors_bar, capsize=6,
                      error_kw={"ecolor": YELLOW, "elinewidth": 1.5})
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom",
                    color=TEXT_DIM, fontsize=10)

        verdict = "BEATS RANDOM" if r["beats_random"] else "NO SIGNAL"
        verdict_color = GREEN if r["beats_random"] else RED
        ax.text(0.97, 0.97, f"{verdict}\nlift={r['lift']:.2f}x",
                transform=ax.transAxes, ha="right", va="top",
                color=verdict_color, fontsize=10, fontweight="bold",
                fontfamily="monospace")

        apply_dark_theme(ax, f"{label}\n({r['gt_type']} ground truth)")
        ax.set_ylabel("Errors captured in flagged rows")
        ax.set_ylim(0, max(r["ml_hits"], r["random_hits_avg"]) * 1.3 + 5)

    fig.suptitle(
        f"Experiment 4 — ML vs Random Flagging (top {int(FLAGGING_PCT*100)}% of rows)",
        color=GREEN, fontsize=13, y=1.02,
    )
    plt.savefig(os.path.join(OUT, "exp4_beat_random.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: exp4_beat_random.png")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(scores2, scores3, exp4_results):
    from scipy.stats import gaussian_kde

    def histogram_shape(scores, label):
        kde = gaussian_kde(scores, bw_method=0.15)
        xs  = np.linspace(0, 1, 200)
        ys  = kde(xs)
        peaks = [i for i in range(1, len(ys) - 1)
                 if ys[i] > ys[i-1] and ys[i] > ys[i+1] and ys[i] > ys.max() * 0.1]
        return "BIMODAL" if len(peaks) >= 2 else "UNIMODAL", len(peaks)

    shape2, peaks2 = histogram_shape(scores2, "TEST2")
    shape3, peaks3 = histogram_shape(scores3, "TEST3")

    lines = []
    lines.append("=" * 65)
    lines.append("ML ANOMALY DETECTOR DIAGNOSTIC REPORT")
    lines.append("=" * 65)
    lines.append("")
    lines.append("EXPERIMENT 3 — SCORE HISTOGRAM")
    lines.append("-" * 40)
    lines.append(f"  TEST2_DATA (banking, 500 rows):      {shape2}  ({peaks2} KDE peaks)")
    lines.append(f"  TEST3_DATA (insurance, 1085 rows):   {shape3}  ({peaks3} KDE peaks)")
    lines.append("")
    lines.append(f"  Score range TEST2: {scores2.min():.3f} — {scores2.max():.3f}  "
                 f"mean={scores2.mean():.3f}  std={scores2.std():.3f}")
    lines.append(f"  Score range TEST3: {scores3.min():.3f} — {scores3.max():.3f}  "
                 f"mean={scores3.mean():.3f}  std={scores3.std():.3f}")
    lines.append("")
    lines.append("EXPERIMENT 4 — BEAT RANDOM")
    lines.append("-" * 40)
    for label, r in exp4_results.items():
        lines.append(f"  {label}")
        lines.append(f"    Ground truth: {r['gt_type']}  |  {r['n_errors']} error rows / {r['n_rows']} total")
        lines.append(f"    ML captured:  {r['ml_hits']} errors in {r['n_flagged']} flagged rows "
                     f"({100*r['ml_error_rate']:.1f}%)")
        lines.append(f"    Random avg:   {r['random_hits_avg']:.1f} ± {r['random_hits_std']:.1f} "
                     f"({100*r['random_error_rate']:.1f}%)")
        lines.append(f"    Lift:         {r['lift']:.2f}x")
        lines.append(f"    Verdict:      {'BEATS RANDOM' if r['beats_random'] else 'DOES NOT BEAT RANDOM'}")
        lines.append("")

    lines.append("PHASE B STRATEGY DECISION")
    lines.append("-" * 40)

    beats2 = list(exp4_results.values())[0]["beats_random"]
    beats3 = list(exp4_results.values())[1]["beats_random"]

    def strategy(shape, beats):
        if shape == "BIMODAL" and beats:
            return "Strategy 1 — Fix threshold (GMM antimode or 99th pct of clean training scores)"
        if shape == "UNIMODAL" and beats:
            return "Strategy 2 — Reframe as ranking (Top N unusual rows, not detection)"
        if shape == "BIMODAL" and not beats:
            return "Exploratory only — exclude from P/R claims"
        return "Strategy 4 mandatory — Hybrid scoring; ML alone has no standalone value"

    s2 = strategy(shape2, beats2)
    s3 = strategy(shape3, beats3)
    lines.append(f"  TEST2_DATA: {s2}")
    lines.append(f"  TEST3_DATA: {s3}")
    lines.append("")
    if s2 != s3:
        lines.append("  NOTE: Strategies differ across datasets.")
        lines.append("  Apply the MORE CONSERVATIVE strategy to both.")
        lines.append("  Do not cherry-pick the better result for the paper.")
    else:
        lines.append("  Strategies agree — apply to both datasets.")
    lines.append("")
    lines.append("ALWAYS DO (regardless of above):")
    lines.append("  Strategy 3 — Per-row explainability (TF-IDF subscores per column)")
    lines.append("  Strategy 4 — Hybrid scoring (for the paper ablation table)")
    lines.append("")
    lines.append("=" * 65)

    report_path = os.path.join(OUT, "ml_diagnostic_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: ml_diagnostic_report.txt")
    print()
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading datasets...")
    df2 = pd.read_csv(TEST2, dtype=str, keep_default_na=False)
    df3 = pd.read_csv(TEST3, dtype=str, keep_default_na=False)
    print(f"  TEST2_DATA: {len(df2)} rows × {len(df2.columns)} cols")
    print(f"  TEST3_DATA: {len(df3)} rows × {len(df3.columns)} cols")

    scores2, scores3, t2, t3 = experiment3(df2, df3)
    exp4_results              = experiment4(df2, df3, scores2, scores3, GT3)
    write_report(scores2, scores3, exp4_results)

    print(f"\nAll outputs saved to: {OUT}/")
