# core/report_html.py
from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime
import html

def _bar_cell(score: float) -> str:
    # CSS bar using linear-gradient; works offline
    width = max(0, min(100, float(score)))
    color = "#2a9d8f" if width >= 80 else ("#e9c46a" if width >= 60 else "#e76f51")
    return f"""
    <div class="barwrap">
      <div class="bar" style="width:{width}%; background:{color};"></div>
      <div class="bartext">{width:.1f}</div>
    </div>
    """

def build_html_report(
    dataset_name: str,
    scores: Dict[str, Any],      # result of compute_quality_scores(...)
    artifacts: Dict[str, str],   # paths from auto_pipeline.run_auto_pipeline(...)
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    notes: Optional[str] = None,
) -> str:
    dims_order = ["accuracy", "completeness", "consistency", "validity", "uniqueness", "timeliness"]

    before = scores.get("before", {})
    after  = scores.get("after", {})

    # Simple rows for table
    rows = []
    for d in dims_order:
        b = float(before.get(d, 0.0))
        a = float(after.get(d, 0.0))
        delta = a - b
        delta_cls = "pos" if delta >= 0 else "neg"
        rows.append(f"""
          <tr>
            <td class="dim">{d.title()}</td>
            <td class="num">{b:.1f}</td>
            <td class="num">{a:.1f}</td>
            <td class="num {delta_cls}">{delta:+.1f}</td>
            <td>{_bar_cell(a)}</td>
          </tr>
        """)

    art_rows = []
    for k, v in (artifacts or {}).items():
        if not v:
            continue
        k_esc = html.escape(str(k))
        v_esc = html.escape(str(v))
        art_rows.append(f"<tr><td>{k_esc}</td><td><code>{v_esc}</code></td></tr>")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_before = f"{before_shape[0]:,} × {before_shape[1]:,}"
    n_after  = f"{after_shape[0]:,} × {after_shape[1]:,}"

    glossary = """
    <ul>
      <li><b>Accuracy</b> — Values adhere to formats and bounds (regex, min/max, valid dates).</li>
      <li><b>Completeness</b> — Share of non-missing cells.</li>
      <li><b>Consistency</b> — Type/parsing consistency (e.g., numbers parse as numbers, dates parse as dates).</li>
      <li><b>Validity</b> — Values are within allowed sets where defined.</li>
      <li><b>Uniqueness</b> — Few rows participate in duplicate pairs.</li>
      <li><b>Timeliness</b> — Date values are recent (within last 365 days) and not in the future.</li>
    </ul>
    """

    notes_html = f"<p>{html.escape(notes)}</p>" if notes else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Data Quality Report — {html.escape(dataset_name)}</title>
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
    --grid: #1f2937;
    --card: #0b1220;
  }}
  body {{
    margin:0; padding:0; background:var(--bg); color:var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Inter, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
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
    position:absolute; top:0; left:8px; line-height:18px; font-size:12px; color:#d1d5db; text-shadow: 0 1px 1px rgba(0,0,0,.4);
  }}
  code {{ background:#0b1220; border:1px solid #1f2937; padding:2px 6px; border-radius:6px; }}
  @media (max-width: 820px) {{
    .grid2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Data Quality Report — {html.escape(dataset_name)}</h1>
        <div class="muted">Generated at {now}</div>
      </div>
      <div class="muted">Before: {n_before}<br/>After: {n_after}</div>
    </header>

    <section class="panel">
      <h2>Six Dimensions (0–100)</h2>
      <table>
        <thead>
          <tr><th>Dimension</th><th>Before</th><th>After</th><th>Δ</th><th>Visual</th></tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Artifacts</h2>
      <table>
        <thead><tr><th>Item</th><th>Path</th></tr></thead>
        <tbody>
          {''.join(art_rows)}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>What each dimension means</h2>
      {glossary}
    </section>

    {notes_html}
  </div>
</body>
</html>
"""