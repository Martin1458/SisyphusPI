import json
import os
from pathlib import Path
import subprocess

from config import OUTPUT_DIR


def build_html(all_data: dict) -> str:
    projects = all_data.get("projects", [])
    combos = all_data.get("combinations", {})

    # Flatten combinations for easier tabular/graph display
    flat_rows = []
    for wd, lr_dict in combos.items():
        for lr, stats in lr_dict.items():
            num_sac = int(stats.get("num_of_sacrifices", 0))
            num_grok = int(stats.get("num_of_grokked", 0))
            avg_time = float(stats.get("avg_train_time", 0.0))
            grok_rate = (num_grok / num_sac * 100.0) if num_sac > 0 else 0.0
            flat_rows.append(
                {
                    "wd": wd,
                    "lr": lr,
                    "num_of_sacrifices": num_sac,
                    "num_of_grokked": num_grok,
                    "grok_rate": grok_rate,
                    "avg_train_time": avg_time,
                }
            )

    total_projects = len(projects)
    total_combos = len(flat_rows)
    total_sacrifices = sum(r["num_of_sacrifices"] for r in flat_rows)
    total_grokked = sum(r["num_of_grokked"] for r in flat_rows)

    embedded_json = json.dumps(all_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>SisyphusPI Project Overview</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #0b1021; color: #f5f5f5; }}
    h1, h2, h3 {{ color: #f5f5f5; }}
    .card {{ background: #161b33; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 0 0 1px #222a4d; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .metric {{ flex: 1 1 160px; padding: 0.75rem 1rem; border-radius: 6px; background: #1f2645; }}
    .metric-label {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: .08em; opacity: 0.7; }}
    .metric-value {{ font-size: 1.3rem; margin-top: 0.25rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.9rem; }}
    th, td {{ padding: 0.4rem 0.6rem; text-align: right; border-bottom: 1px solid #242a4a; }}
    th {{ text-align: right; background: #171d37; position: sticky; top: 0; z-index: 1; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: #14192f; }}
    a, a:visited {{ color: #7bd7ff; }}
    canvas {{ max-width: 100%; background: #0b1021; }}
    .table-wrapper {{ max-height: 400px; overflow: auto; border-radius: 6px; border: 1px solid #242a4a; }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body>
  <h1>SisyphusPI – All Projects Overview</h1>

  <div class=\"card\">
    <h2>Summary</h2>
    <div class=\"metrics\">
      <div class=\"metric\">
        <div class=\"metric-label\">Projects</div>
        <div class=\"metric-value\">{total_projects}</div>
      </div>
      <div class=\"metric\">
        <div class=\"metric-label\">(wd, lr) Combos</div>
        <div class=\"metric-value\">{total_combos}</div>
      </div>
      <div class=\"metric\">
        <div class=\"metric-label\">Total Sacrifices</div>
        <div class=\"metric-value\">{total_sacrifices}</div>
      </div>
      <div class=\"metric\">
        <div class=\"metric-label\">Total Grokked</div>
        <div class=\"metric-value\">{total_grokked}</div>
      </div>
    </div>
    <p>Projects seen: {', '.join(projects) or '—'}</p>
  </div>

  <div class=\"card\">
    <h2>Grokked Rate per (weight_decay, learning_rate)</h2>
    <canvas id=\"comboChart\" height=\"120\"></canvas>
  </div>

  <div class=\"card\">
    <h2>Raw Combination Table</h2>
    <div class=\"table-wrapper\">
      <table>
        <thead>
          <tr>
            <th>weight_decay</th>
            <th>learning_rate</th>
            <th>num_of_sacrifices</th>
            <th>num_of_grokked</th>
            <th>grok_rate %</th>
            <th>avg_train_time s</th>
          </tr>
        </thead>
        <tbody>
"""

    # Insert table rows
    for row in flat_rows:
        html += (
            f"          <tr>"
            f"<td>{row['wd']}</td>"
            f"<td>{row['lr']}</td>"
            f"<td>{row['num_of_sacrifices']}</td>"
            f"<td>{row['num_of_grokked']}</td>"
            f"<td>{row['grok_rate']:.1f}</td>"
            f"<td>{row['avg_train_time']:.2f}</td>"
            f"</tr>\n"
        )

    html += """        </tbody>
      </table>
    </div>
  </div>

  <script id=\"all-projects-data\" type=\"application/json\">""" + embedded_json + """</script>
  <script>
    (function() {
      const script = document.getElementById('all-projects-data');
      if (!script) return;
      const data = JSON.parse(script.textContent || '{}');
      const combos = data.combinations || {};

      const labels = [];
      const grokRates = [];

      for (const [wd, lrDict] of Object.entries(combos)) {
        for (const [lr, stats] of Object.entries(lrDict)) {
          const numSac = stats.num_of_sacrifices || 0;
          const numGrok = stats.num_of_grokked || 0;
          const rate = numSac > 0 ? (numGrok / numSac * 100.0) : 0.0;
          labels.push(`${wd} / ${lr}`);
          grokRates.push(rate);
        }
      }

      const ctx = document.getElementById('comboChart').getContext('2d');
      // eslint-disable-next-line no-undef
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Grokked %',
            data: grokRates,
            backgroundColor: 'rgba(123, 215, 255, 0.4)',
            borderColor: 'rgba(123, 215, 255, 1.0)',
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          animation: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Grokked %' },
            },
            x: {
              ticks: { autoSkip: true, maxTicksLimit: 12 },
            },
          },
        },
      });
    })();
  </script>
</body>
</html>
"""
    return html


def main() -> None:
  # OUTPUT_DIR is project-specific (output/project_name).
  # all_projects_data.json lives under the shared output/ folder,
  # and we write the website to SisyphusPI-website/index.html at
  # the repository root.
  output_dir = Path(OUTPUT_DIR).resolve()
  projects_root = output_dir.parent  # e.g. .../SisyphusPI/output
  repo_root = projects_root.parent   # e.g. .../SisyphusPI

  all_projects_data_path = projects_root / "all_projects_data.json"

  if not all_projects_data_path.is_file():
    print(f"No all_projects_data.json found at {all_projects_data_path}. Run training first.")
    return

  with all_projects_data_path.open("r", encoding="utf-8") as f:
    all_data = json.load(f)

  html = build_html(all_data)
  index_path = repo_root / "SisyphusPI-website" / "index.html"
  index_path.parent.mkdir(parents=True, exist_ok=True)
  index_path.write_text(html, encoding="utf-8")
  print(f"Wrote {index_path}")



if __name__ == "__main__":
    main()

def git_auto_commit_website(model_no: int, total_models: int) -> None:
    """Git add/commit/push inside the SisyphusPI-website repo.

    Best-effort: errors (no changes, no remote, no repo) are printed
    but do not stop training.
    """

    repo_root = os.path.dirname(os.path.abspath(__file__))
    website_repo = os.path.join(repo_root, "SisyphusPI-website")
    msg = f"auto: updated site after {model_no}/{total_models} models"

    if not os.path.isdir(website_repo):
        print(f"Website repo not found at {website_repo}, skipping git push.")
        return

    try:
        subprocess.run(["git", "add", "."], cwd=website_repo, check=False)
        subprocess.run(["git", "commit", "-m", msg], cwd=website_repo, check=False)
        subprocess.run(["git", "push"], cwd=website_repo, check=False)
    except Exception as e:
        print(f"Git auto-commit (website) failed: {e}")
