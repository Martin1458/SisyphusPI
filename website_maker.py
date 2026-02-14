import json
import os
from pathlib import Path
import subprocess

from config import OUTPUT_DIR, AGGREGATE_DATA_PATH


def build_html(data: dict) -> str:
    combos = data.get("combinations", {})
    wd_stats = data.get("weight_decay", {})
    lr_stats = data.get("learning_rate", {})
    n_stats = data.get("N", {})

    # Optional system information (e.g. Raspberry Pi 5 CPU temperature)
    system_info = data.get("system", {}) or {}
    cpu_temp_c = system_info.get("cpu_temp_c")

    if cpu_temp_c is not None:
        cpu_temp_text = f"{cpu_temp_c:.1f} °C"
    else:
        cpu_temp_text = "—"

    # Flatten combinations for overview table/chart
    flat_rows: list[dict] = []
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

    total_combos = len(flat_rows)
    total_sacrifices = int(data.get("num_of_sacrifices", 0))
    total_grokked = int(data.get("num_of_grokked", 0))
    overall_grok_rate = (total_grokked / total_sacrifices * 100.0) if total_sacrifices > 0 else 0.0

    embedded_json = json.dumps(data, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <title>SisyphusPI</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <link rel=\"stylesheet\" href=\"styles.css\" />
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body>
    <div class=\"cpu-temp\">CPU: {cpu_temp_text}</div>
    <h1>SisyphusPI</h1>

    <div class=\"tabs\">
        <button class=\"tab-button active\" data-tab=\"overview\">Overview</button>
        <button class=\"tab-button\" data-tab=\"drilldown\">Drilldown</button>
    </div>

    <div id=\"tab-overview\" class=\"tab-panel active\">
        <div class=\"card\">
            <h2>Summary</h2>
            <div class=\"metrics\">
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
                <div class=\"metric\">
                    <div class=\"metric-label\">Grok Rate</div>
                    <div class=\"metric-value\">{overall_grok_rate:.1f}%</div>
                </div>
            </div>
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

    # Insert overview table rows
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
    </div>

    <div id=\"tab-drilldown\" class=\"tab-panel\">
        <div class=\"card\">
            <h2>Grokked % by weight_decay</h2>
            <canvas id=\"wdChart\" height=\"120\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Grokked % by learning_rate</h2>
            <canvas id=\"lrChart\" height=\"120\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Filter by weight_decay</h2>
            <label for=\"wd-select\">Choose weight_decay:</label>
            <select id=\"wd-select\"></select>
            <canvas id=\"wdDrillChart\" height=\"120\" style=\"margin-top:1rem\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Filter by learning_rate</h2>
            <label for=\"lr-select\">Choose learning_rate:</label>
            <select id=\"lr-select\"></select>
            <canvas id=\"lrDrillChart\" height=\"120\" style=\"margin-top:1rem\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Per-N Statistics</h2>
            <div class=\"table-wrapper\">
                <table id=\"n-stats-table\">
                    <thead>
                        <tr>
                            <th>N</th>
                            <th>Sacrifices</th>
                            <th>Grokked</th>
                            <th>Grok %</th>
                            <th>Avg Train Time (s)</th>
                            <th>Median Steps to Grok</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Insert per-N table rows
    sorted_n_keys = sorted(
        [k for k in n_stats.keys() if k not in ("1", "2") or n_stats[k].get("num_of_sacrifices", 0) > 0],
        key=lambda x: int(x),
    )
    for n_key in sorted_n_keys:
        n_entry = n_stats[n_key]
        n_sac = int(n_entry.get("num_of_sacrifices", 0))
        n_grok = int(n_entry.get("num_of_grokked", 0))
        n_rate = (n_grok / n_sac * 100.0) if n_sac > 0 else 0.0
        n_avg_time = float(n_entry.get("avg_train_time", 0.0))
        n_steps_grok = n_entry.get("avg_data", {}).get("steps_to_grok", 0)
        if n_sac > 0:
            html += (
                f"          <tr>"
                f"<td>{n_key}</td>"
                f"<td>{n_sac}</td>"
                f"<td>{n_grok}</td>"
                f"<td>{n_rate:.1f}</td>"
                f"<td>{n_avg_time:.2f}</td>"
                f"<td>{n_steps_grok}</td>"
                f"</tr>\n"
            )

    html += """        </tbody>
                </table>
            </div>
        </div>
    </div>

    <script id=\"sisyphus-data\" type=\"application/json\">""" + embedded_json + """</script>
    <script>
        (function() {
            const script = document.getElementById('sisyphus-data');
            if (!script) return;
            const data = JSON.parse(script.textContent || '{}');
            const combos = data.combinations || {};
            const wdStats = data.weight_decay || {};
            const lrStats = data.learning_rate || {};

            // ---- Helper: compute grok rate ----
            function grokRate(s) {
                const n = (s && s.num_of_sacrifices) || 0;
                const g = (s && s.num_of_grokked) || 0;
                return n > 0 ? (g / n * 100.0) : 0.0;
            }

            // ---- Chart defaults ----
            const barOpts = (yLabel) => ({
                responsive: true, maintainAspectRatio: true, animation: false,
                scales: {
                    y: { beginAtZero: true, max: 100, title: { display: true, text: yLabel } },
                    x: { ticks: { autoSkip: true, maxTicksLimit: 16 } },
                },
            });

            // ---- Overview: combo chart ----
            const overviewLabels = [];
            const overviewRates = [];
            for (const [wd, lrDict] of Object.entries(combos)) {
                for (const [lr, stats] of Object.entries(lrDict)) {
                    overviewLabels.push(`${wd} / ${lr}`);
                    overviewRates.push(grokRate(stats));
                }
            }
            const comboCanvas = document.getElementById('comboChart');
            if (comboCanvas) {
                new Chart(comboCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: overviewLabels,
                        datasets: [{ label: 'Grokked %', data: overviewRates,
                            backgroundColor: 'rgba(123, 215, 255, 0.4)',
                            borderColor: 'rgba(123, 215, 255, 1.0)', borderWidth: 1 }],
                    },
                    options: barOpts('Grokked %'),
                });
            }

            // ---- Tabs ----
            document.querySelectorAll('.tab-button').forEach((btn) => {
                btn.addEventListener('click', () => {
                    const target = btn.getAttribute('data-tab');
                    if (!target) return;
                    document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
                    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
                    btn.classList.add('active');
                    const panel = document.getElementById(`tab-${target}`);
                    if (panel) panel.classList.add('active');
                });
            });

            // ---- Drilldown: WD chart ----
            const wdCanvas = document.getElementById('wdChart');
            if (wdCanvas) {
                const wdLabels = Object.keys(wdStats).sort();
                const wdRates = wdLabels.map((k) => grokRate(wdStats[k]));
                new Chart(wdCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: wdLabels,
                        datasets: [{ label: 'Grokked %', data: wdRates,
                            backgroundColor: 'rgba(180, 123, 255, 0.4)',
                            borderColor: 'rgba(180, 123, 255, 1.0)', borderWidth: 1 }],
                    },
                    options: barOpts('Grokked %'),
                });
            }

            // ---- Drilldown: LR chart ----
            const lrCanvas = document.getElementById('lrChart');
            if (lrCanvas) {
                const lrLabels = Object.keys(lrStats).sort();
                const lrRates = lrLabels.map((k) => grokRate(lrStats[k]));
                new Chart(lrCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: lrLabels,
                        datasets: [{ label: 'Grokked %', data: lrRates,
                            backgroundColor: 'rgba(123, 255, 196, 0.4)',
                            borderColor: 'rgba(123, 255, 196, 1.0)', borderWidth: 1 }],
                    },
                    options: barOpts('Grokked %'),
                });
            }

            // ---- Drilldown: filter by WD → show LR breakdown ----
            const wdSelect = document.getElementById('wd-select');
            const wdDrillCanvas = document.getElementById('wdDrillChart');
            let wdDrillChart = null;

            function renderWdDrill(wd) {
                if (wdDrillChart) { wdDrillChart.destroy(); wdDrillChart = null; }
                const lrDict = combos[wd];
                if (!lrDict || !wdDrillCanvas) return;
                const labels = Object.keys(lrDict).sort();
                const rates = labels.map((lr) => grokRate(lrDict[lr]));
                wdDrillChart = new Chart(wdDrillCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{ label: `Grokked % (wd=${wd})`, data: rates,
                            backgroundColor: 'rgba(255, 180, 123, 0.4)',
                            borderColor: 'rgba(255, 180, 123, 1.0)', borderWidth: 1 }],
                    },
                    options: barOpts('Grokked %'),
                });
            }

            if (wdSelect) {
                const wdKeys = Object.keys(combos).sort();
                wdKeys.forEach((wd) => {
                    const opt = document.createElement('option');
                    opt.value = wd; opt.textContent = wd;
                    wdSelect.appendChild(opt);
                });
                wdSelect.addEventListener('change', () => renderWdDrill(wdSelect.value));
                if (wdKeys.length > 0) { wdSelect.value = wdKeys[0]; renderWdDrill(wdKeys[0]); }
            }

            // ---- Drilldown: filter by LR → show WD breakdown ----
            const lrSelect = document.getElementById('lr-select');
            const lrDrillCanvas = document.getElementById('lrDrillChart');
            let lrDrillChart = null;

            function renderLrDrill(lr) {
                if (lrDrillChart) { lrDrillChart.destroy(); lrDrillChart = null; }
                if (!lrDrillCanvas) return;
                // Collect all WDs that have this LR
                const wdKeys = Object.keys(combos).sort();
                const labels = [];
                const rates = [];
                for (const wd of wdKeys) {
                    const lrDict = combos[wd] || {};
                    if (lr in lrDict) {
                        labels.push(wd);
                        rates.push(grokRate(lrDict[lr]));
                    }
                }
                lrDrillChart = new Chart(lrDrillCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{ label: `Grokked % (lr=${lr})`, data: rates,
                            backgroundColor: 'rgba(255, 123, 180, 0.4)',
                            borderColor: 'rgba(255, 123, 180, 1.0)', borderWidth: 1 }],
                    },
                    options: barOpts('Grokked %'),
                });
            }

            if (lrSelect) {
                // Collect all unique LR keys across all WDs
                const allLrs = new Set();
                for (const lrDict of Object.values(combos)) {
                    for (const lr of Object.keys(lrDict)) allLrs.add(lr);
                }
                const lrKeys = [...allLrs].sort();
                lrKeys.forEach((lr) => {
                    const opt = document.createElement('option');
                    opt.value = lr; opt.textContent = lr;
                    lrSelect.appendChild(opt);
                });
                lrSelect.addEventListener('change', () => renderLrDrill(lrSelect.value));
                if (lrKeys.length > 0) { lrSelect.value = lrKeys[0]; renderLrDrill(lrKeys[0]); }
            }
        })();
    </script>
</body>
</html>
"""
    return html


def main() -> None:
    """Read the aggregate data.json and generate the website HTML."""
    output_dir = Path(OUTPUT_DIR).resolve()
    repo_root = output_dir.parent  # e.g. .../SisyphusPI

    data_path = Path(AGGREGATE_DATA_PATH).resolve()

    if not data_path.is_file():
        print(f"No data.json found at {data_path}. Run training first.")
        return

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    html = build_html(data)
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
        # Detect current branch. If we're in detached HEAD, explicitly
        # push HEAD to the remote's default branch (origin/HEAD), which
        # avoids the "You are not currently on a branch" fatal.
        head_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=website_repo,
            check=False,
            capture_output=True,
            text=True,
        )
        current_branch = (head_proc.stdout or "").strip()

        if head_proc.returncode != 0:
            print("Could not determine current branch; skipping git push for website.")
            return

        if current_branch == "HEAD":
            # We are in detached HEAD. Try to resolve origin/HEAD and
            # push the new commit there.
            origin_head_proc = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "origin/HEAD"],
                cwd=website_repo,
                check=False,
                capture_output=True,
                text=True,
            )
            origin_head = (origin_head_proc.stdout or "").strip()

            if origin_head_proc.returncode != 0 or not origin_head:
                print(
                    "Website repo is in detached HEAD and origin/HEAD is unknown; "
                    "skipping git push for SisyphusPI-website."
                )
                return

            if "/" in origin_head:
                remote, remote_branch = origin_head.split("/", 1)
            else:
                remote, remote_branch = "origin", origin_head

            subprocess.run(
                ["git", "push", remote, f"HEAD:{remote_branch}"],
                cwd=website_repo,
                check=False,
            )
        else:
            # Normal case: we're on a branch, so a plain push works.
            subprocess.run(["git", "push"], cwd=website_repo, check=False)
    except Exception as e:
        print(f"Git auto-commit (website) failed: {e}")
