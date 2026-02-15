"""
report.py
──────────
Auto-generate an HTML summary report with:
  - Model metrics table
  - All saved plots embedded inline
  - Database statistics
  - Physics validation results
"""

import base64
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger("sqcyl")


def _img_tag(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0;">'
    except Exception as e:
        logger.warning(f"Failed to embed image {path}: {e}")
        return f'<p>Image not available: {path.name}</p>'


def generate_report(
    metrics_all: dict,
    db_stats: dict,
    physics_checks: dict,
    cfg: dict
):
    plots_dir  = Path(cfg["paths"]["plots_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Metrics table HTML ────────────────────────────────────────────────
    metric_rows = ""
    for target, m in metrics_all.items():
        metric_rows += (
            f"<tr><td>{target}</td>"
            f"<td>{m.get('R2','–')}</td>"
            f"<td>{m.get('MAE','–')}</td>"
            f"<td>{m.get('RMSE','–')}</td></tr>\n"
        )

    # ── Physics checks HTML ───────────────────────────────────────────────
    phys_rows = ""
    for check, result in physics_checks.items():
        icon = "✅" if result else ("❌" if result is False else "⚠️ N/A")
        phys_rows += f"<tr><td>{check}</td><td>{icon}</td></tr>\n"

    # ── Embed all PNG plots ───────────────────────────────────────────────
    plot_section = "<h2>Generated Plots</h2>\n"
    if plots_dir.exists():
        for png in sorted(plots_dir.glob("*.png")):
            plot_section += f"<h3>{png.stem.replace('_',' ').title()}</h3>\n"
            plot_section += _img_tag(png) + "\n"
    else:
        plot_section += "<p>No plots generated yet.</p>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Square Cylinder XGBoost — Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: auto; padding: 24px; }}
  h1 {{ color: #1565C0; }} h2 {{ color: #283593; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 14px; text-align: left; }}
  th {{ background: #E3F2FD; color: #0D47A1; }}
  tr:nth-child(even) {{ background: #F9F9F9; }}
  code {{ background: #EEE; padding: 2px 6px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>🔷 Square Cylinder XGBoost Hydrodynamic Database</h1>
<p><strong>Generated:</strong> {now}</p>

<h2>Model Performance Metrics</h2>
<table>
  <tr><th>Target</th><th>R²</th><th>MAE</th><th>RMSE</th></tr>
  {metric_rows if metric_rows else "<tr><td colspan='4'>No metrics available</td></tr>"}
</table>

<h2>Physics Validation Checks</h2>
<table>
  <tr><th>Check</th><th>Status</th></tr>
  {phys_rows if phys_rows else "<tr><td colspan='2'>No checks performed</td></tr>"}
</table>

<h2>Database Statistics</h2>
<pre>{json.dumps(db_stats, indent=2, default=str)}</pre>

{plot_section}

<hr>
<p style="color:#888;font-size:12px;">
  Project: Square Cylinder XGBoost Hydrodynamic Database |
  Physics-informed ML for Re = 10³–10⁶, L/D = 1–6
</p>
</body>
</html>"""

    out_path = report_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML report → {out_path}")
    return out_path
