# core/report_builder.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.quality_scores import compute_quality_scores


DIMENSIONS_ORDER = [
    "accuracy",
    "completeness",
    "consistency",
    "validity",
    "uniqueness",
    "timeliness",
]

DIM_COLORS = {
    "accuracy": "#2a9d8f",
    "completeness": "#e9c46a",
    "consistency": "#2a9d8f",
    "validity": "#2a9d8f",
    "uniqueness": "#f4a261",
    "timeliness": "#f97316",
}


def _safe_score(d: Dict[str, float], key: str) -> float:
    try:
        v = float(d.get(key, 0.0))
    except Exception:
        v = 0.0
    return max(0.0, min(100.0, v))


def build_html_report(
    df_after: pd.DataFrame,
    report_path: Path,
    scores_before: Optional[Dict[str, float]] = None,
    scores_after: Optional[Dict[str, float]] = None,
    artifacts: Optional[List[str]] = None,
) -> Path:
    """
    Build a dark-themed HTML Data Quality report similar to your original dq_report.html.

    - df_after: final cleaned / standardized dataset
    - report_path: where to write the HTML
    - scores_before / scores_after: dicts from compute_quality_scores()
    - artifacts: list of artifact filenames we expect in auto_artifacts/
    """
    # ------------------------------------------
    # 1) Compute / ensure scores
    # ------------------------------------------
    if scores_before is None:
        scores_before = compute_quality_scores(df_after)  # fallback

    if scores_after is None:
        scores_after = compute_quality_scores(df_after)

    # Dataset summary
    n_rows, n_cols = df_after.shape
    missing_total = int(df_after.isna().sum().sum())
    missing_pct = 100.0 * missing_total / (max(1, n_rows * n_cols))
    dup_rows = int(df_after.duplicated().sum())

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------
    # 2) Build rows for the dimension table
    # ------------------------------------------
    dim_rows_html = []
    for dim in DIMENSIONS_ORDER:
        b = _safe_score(scores_before, dim)
        a = _safe_score(scores_after, dim)
        delta = round(a - b, 1)
        delta_class = "pos" if delta > 0 else ("neg" if delta < 0 else "")
        color = DIM_COLORS.get(dim, "#2a9d8f")
        width = max(0.0, min(100.0, a))

        dim_rows_html.append(
            f"""
          <tr>
            <td class="dim">{dim.capitalize()}</td>
            <td class="num">{b:.1f}</td>
            <td class="num">{a:.1f}</td>
            <td class="num {delta_class}">{delta:+.1f}</td>
            <td>
              <div class="barwrap">
                <div class="bar" style="width:{width}%; background:{color};"></div>
                <div class="bartext">{a:.1f}</div>
              </div>
            </td>
          </tr>
        """
        )

    dims_table_html = "\n".join(dim_rows_html)

    # ------------------------------------------
    # 3) Artifacts section
    # ------------------------------------------
    if artifacts is None:
        artifacts = [
            "rules_merged.json",
            "validation_report.csv",
            "validated_dataset.csv",
            "cleaned_dataset.csv",
            "trimmed_dataset.csv",
            "final_kept_columns.csv",
            "pairs_scored.csv",
            "pairs_side_by_side.csv",
            "validated_dataset.xlsx",
            "cleaned_dataset.xlsx",
            "dq_report.html",
        ]

    artifacts_rows = []
    for name in artifacts:
        status = "Generated"
        artifacts_rows.append(
            f"""
          <tr>
            <td>{name}</td>
            <td class="num">{status}</td>
          </tr>
        """
        )

    artifacts_table_html = "\n".join(artifacts_rows)

    # ------------------------------------------
    # 4) Build final HTML (dark UI dashboard style)
    # ------------------------------------------
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Data Quality Report — Auto Pipeline</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #111827;
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #22c55e;
    --warn: #f59e0b;
    --danger: #ef4444;
  }}
  body {{
    margin:0; padding:0; background:var(--bg); color:var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 24px auto 80px; padding: 0 16px; }}
  header {{
    display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
    margin-bottom: 18px;
  }}
  h1 {{ font-size: 24px; margin: 0 0 8px 0; }}
  h2 {{ font-size: 18px; margin: 18px 0 8px 0; }}
  .muted {{ color: var(--muted); font-size: 13px; }}
  .panel {{
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid #263045; border-radius: 14px; padding: 16px; margin: 14px 0;
  }}
  table {{ width:100%; border-collapse: collapse; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom: 1px solid #1f2937; }}
  th {{ color:#a6b1c2; font-weight:600; font-size:13px; letter-spacing: .02em; }}
  td.num {{ text-align:right; width: 90px; }}
  td.dim {{ width: 160px; }}
  td.pos {{ color: #22c55e; }}
  td.neg {{ color: #ef4444; }}
  .grid2 {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .barwrap {{
    position: relative; background: #0b1220; border: 1px solid #1f2937; border-radius: 8px;
    height: 18px; overflow: hidden;
  }}
  .bar {{ height: 100%; }}
  .bartext {{
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    font-size: 11px; color:#f9fafb;
  }}
  .pill {{
    display:inline-flex; align-items:center; gap:6px; padding:3px 8px; border-radius:999px;
    background:#111827; border:1px solid #1f2937; color:#e5e7eb; font-size:11px;
  }}
  .pill span.dot {{
    width:8px; height:8px; border-radius:999px; background:var(--accent);
  }}
  .tag {{
    display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px;
    border:1px solid #374151; color:#d1d5db;
  }}
  .muted-sm {{ color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Data Quality Report — Uploaded dataset</h1>
      <div class="muted">Generated at {generated_at}</div>
      <div class="muted-sm">Auto mode pipeline · ntu_ktp_data_quality</div>
    </div>
    <div style="display:flex; flex-direction:column; gap:8px; align-items:flex-end;">
      <div class="pill"><span class="dot"></span><span>Auto report</span></div>
      <div class="tag">{n_rows} rows · {n_cols} columns</div>
    </div>
  </header>

  <section class="panel">
    <h2>Dataset Snapshot</h2>
    <p class="muted-sm">
      Rows: <strong>{n_rows}</strong> · Columns: <strong>{n_cols}</strong> ·
      Missing cells: <strong>{missing_total}</strong> ({missing_pct:.1f}% of all values) ·
      Duplicate rows: <strong>{dup_rows}</strong>
    </p>
  </section>

  <section class="panel">
    <h2>Quality Dimensions (Before → After)</h2>
    <p class="muted-sm">
      Scores are reported on six dimensions: Accuracy, Completeness, Consistency,
      Validity, Uniqueness, and Timeliness. Values are expressed as percentages (0–100).
    </p>
    <table>
      <thead>
        <tr>
          <th class="dim">Dimension</th>
          <th class="num">Before</th>
          <th class="num">After</th>
          <th class="num">Δ</th>
          <th>Visual</th>
        </tr>
      </thead>
      <tbody>
        {dims_table_html}
      </tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Generated Artifacts</h2>
    <p class="muted-sm">
      The auto-pipeline produced the following outputs in the <code>auto_artifacts/</code> folder.
      Exact contents will depend on your configuration and dataset size.
    </p>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th class="num">Status</th>
        </tr>
      </thead>
      <tbody>
        {artifacts_table_html}
      </tbody>
    </table>
  </section>

  <section class="panel">
    <h2>Notes</h2>
    <p class="muted-sm">
      This HTML report is generated automatically by the Data Quality Auto Pipeline.
      It is intended as a high-level overview of data health, not a replacement for
      detailed, column-level inspection.
    </p>
  </section>
</div>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path