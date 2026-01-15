import json
import os
from pathlib import Path
import subprocess

from config import OUTPUT_DIR


def build_html(all_data: dict) -> str:
    projects = all_data.get("projects", [])
    combos = all_data.get("combinations", {})

    # Optional system information (e.g. Raspberry Pi 5 CPU temperature)
    system_info = all_data.get("system", {}) or {}
    cpu_temp_c = system_info.get("cpu_temp_c")

    if cpu_temp_c is not None:
        cpu_temp_text = f"{cpu_temp_c:.1f} °C"
    else:
        cpu_temp_text = "—"

    # Flatten cross-project combinations for overview table/chart
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
    <link rel=\"stylesheet\" href=\"styles.css\" />
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body>
    <div class=\"cpu-temp\">CPU: {cpu_temp_text}</div>
    <h1>SisyphusPI – Projects</h1>

    <div class=\"tabs\">
        <button class=\"tab-button active\" data-tab=\"overview\">Overview</button>
        <button class=\"tab-button\" data-tab=\"project\">Project</button>
    </div>

    <div id=\"tab-overview\" class=\"tab-panel active\">
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
            <p>Projects: {', '.join(projects) or '—'}</p>
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

    <div id=\"tab-project\" class=\"tab-panel\">
        <div class=\"card\">
            <h2>Project selection</h2>
            <label for=\"project-select\">Choose project:</label>
            <select id=\"project-select\"></select>
        </div>

        <div id=\"project-summary-card\" class=\"card\">
            <h2>Project summary</h2>
            <p>Select a project to see details.</p>
        </div>

        <div class=\"card\">
            <h2>Grokked % by weight_decay</h2>
            <canvas id=\"projectWdChart\" height=\"120\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Grokked % by learning_rate</h2>
            <canvas id=\"projectLrChart\" height=\"120\"></canvas>
        </div>
    </div>

    <script id=\"all-projects-data\" type=\"application/json\">""" + embedded_json + """</script>
    <script>
        (function() {
            const script = document.getElementById('all-projects-data');
            if (!script) return;
            const data = JSON.parse(script.textContent || '{}');
            const combos = data.combinations || {};
            const projects = data.projects || [];
            const projectDetails = data.project_details || {};

            // ---------------- Overview chart ----------------
            const overviewLabels = [];
            const overviewRates = [];

            for (const [wd, lrDict] of Object.entries(combos)) {
                for (const [lr, stats] of Object.entries(lrDict)) {
                    const numSac = stats.num_of_sacrifices || 0;
                    const numGrok = stats.num_of_grokked || 0;
                    const rate = numSac > 0 ? (numGrok / numSac * 100.0) : 0.0;
                    overviewLabels.push(`${wd} / ${lr}`);
                    overviewRates.push(rate);
                }
            }

            const comboCanvas = document.getElementById('comboChart');
            if (comboCanvas) {
                const ctx = comboCanvas.getContext('2d');
                // eslint-disable-next-line no-undef
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: overviewLabels,
                        datasets: [{
                            label: 'Grokked %',
                            data: overviewRates,
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
            }

            // ---------------- Tabs behaviour ----------------
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabPanels = document.querySelectorAll('.tab-panel');

            tabButtons.forEach((btn) => {
                btn.addEventListener('click', () => {
                    const target = btn.getAttribute('data-tab');
                    if (!target) return;

                    tabButtons.forEach((b) => b.classList.remove('active'));
                    tabPanels.forEach((panel) => panel.classList.remove('active'));

                    btn.classList.add('active');
                    const activePanel = document.getElementById(`tab-${target}`);
                    if (activePanel) {
                        activePanel.classList.add('active');
                    }
                });
            });

            // ---------------- Project view ----------------
            const projectSelect = document.getElementById('project-select');
            const projectSummaryCard = document.getElementById('project-summary-card');
            const projectWdCanvas = document.getElementById('projectWdChart');
            const projectLrCanvas = document.getElementById('projectLrChart');

            let projectWdChart = null;
            let projectLrChart = null;

            function renderProjectSummary(name) {
                if (!projectSummaryCard) return;
                const details = projectDetails[name];
                if (!details) {
                    projectSummaryCard.innerHTML = '<h2>Project summary</h2><p>No data for this project yet.</p>';
                    return;
                }

                const numSac = details.num_of_sacrifices || 0;
                const numGrok = details.num_of_grokked || 0;
                const grokRate = numSac > 0 ? (numGrok / numSac * 100.0) : 0.0;

                projectSummaryCard.innerHTML = `
                    <h2>Project: ${name}</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Sacrifices</div>
                            <div class="metric-value">${numSac}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Grokked</div>
                            <div class="metric-value">${numGrok}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Grokked %</div>
                            <div class="metric-value">${grokRate.toFixed(1)}</div>
                        </div>
                    </div>
                `;
            }

            function renderProjectCharts(name) {
                const details = projectDetails[name];
                if (!details) return;

                const wdStats = details.weight_decay || {};
                const lrStats = details.learning_rate || {};

                if (projectWdChart) {
                    projectWdChart.destroy();
                    projectWdChart = null;
                }
                if (projectLrChart) {
                    projectLrChart.destroy();
                    projectLrChart = null;
                }

                if (projectWdCanvas) {
                    const wdLabels = Object.keys(wdStats).sort();
                    const wdRates = wdLabels.map((wd) => {
                        const s = wdStats[wd] || {};
                        const numSac = s.num_of_sacrifices || 0;
                        const numGrok = s.num_of_grokked || 0;
                        return numSac > 0 ? (numGrok / numSac * 100.0) : 0.0;
                    });

                    const ctxWd = projectWdCanvas.getContext('2d');
                    // eslint-disable-next-line no-undef
                    projectWdChart = new Chart(ctxWd, {
                        type: 'bar',
                        data: {
                            labels: wdLabels,
                            datasets: [{
                                label: 'Grokked %',
                                data: wdRates,
                                backgroundColor: 'rgba(180, 123, 255, 0.4)',
                                borderColor: 'rgba(180, 123, 255, 1.0)',
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
                            },
                        },
                    });
                }

                if (projectLrCanvas) {
                    const lrLabels = Object.keys(lrStats).sort();
                    const lrRates = lrLabels.map((lr) => {
                        const s = lrStats[lr] || {};
                        const numSac = s.num_of_sacrifices || 0;
                        const numGrok = s.num_of_grokked || 0;
                        return numSac > 0 ? (numGrok / numSac * 100.0) : 0.0;
                    });

                    const ctxLr = projectLrCanvas.getContext('2d');
                    // eslint-disable-next-line no-undef
                    projectLrChart = new Chart(ctxLr, {
                        type: 'bar',
                        data: {
                            labels: lrLabels,
                            datasets: [{
                                label: 'Grokked %',
                                data: lrRates,
                                backgroundColor: 'rgba(123, 255, 196, 0.4)',
                                borderColor: 'rgba(123, 255, 196, 1.0)',
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
                            },
                        },
                    });
                }
            }

            function handleProjectChange() {
                if (!projectSelect) return;
                const name = projectSelect.value;
                if (!name) return;
                renderProjectSummary(name);
                renderProjectCharts(name);
            }

            if (projectSelect && Array.isArray(projects)) {
                projectSelect.innerHTML = '';
                projects.slice().sort().forEach((name) => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    projectSelect.appendChild(opt);
                });

                projectSelect.addEventListener('change', handleProjectChange);

                if (projectSelect.options.length > 0) {
                    projectSelect.selectedIndex = 0;
                    handleProjectChange();
                }
            }
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

    # Enrich with per-project aggregates for the Project tab.
    # For each project, we read its project-level data.json (if present)
    # and expose total sacrifices/grokked plus per-weight-decay and
    # per-learning-rate stats.
    project_details: dict[str, dict] = {}
    for project_name in all_data.get("projects", []):
        proj_data_path = projects_root / project_name / "data.json"
        if not proj_data_path.is_file():
            continue
        try:
            with proj_data_path.open("r", encoding="utf-8") as pf:
                proj_data = json.load(pf)
        except json.JSONDecodeError:
            continue

        num_sac = int(proj_data.get("num_of_sacrifices", 0))
        num_grok = int(proj_data.get("num_of_grokked", 0))
        wd_stats = proj_data.get("weight_decay", {}) or {}
        lr_stats = proj_data.get("learning_rate", {}) or {}

        project_details[project_name] = {
            "num_of_sacrifices": num_sac,
            "num_of_grokked": num_grok,
            "weight_decay": wd_stats,
            "learning_rate": lr_stats,
        }

    all_data["project_details"] = project_details

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
