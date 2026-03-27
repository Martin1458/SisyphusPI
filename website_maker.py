import json
import os
from pathlib import Path
import subprocess

from datetime import timedelta

from config import OUTPUT_DIR, AGGREGATE_DATA_PATH, get_total_number_of_models


def build_html(data: dict, total_models: int = 0) -> str:
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

    pct_done = (total_sacrifices / total_models * 100.0) if total_models > 0 else 0.0
    pct_done_clamped = min(pct_done, 100.0)

    avg_time_per_model = float(data.get("avg_train_time", 0.0))
    remaining_models = max(total_models - total_sacrifices, 0)
    if avg_time_per_model > 0 and remaining_models > 0:
        eta_str = str(timedelta(seconds=int(avg_time_per_model * remaining_models)))
    else:
        eta_str = "—"
    avg_time_str = f"{avg_time_per_model:.1f}s/model" if avg_time_per_model > 0 else "—"

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

    <div class=\"project-hero card\">
        <div class=\"project-title\">Modular Addition</div>
        <div class=\"project-subtitle\">( a + b ) mod N &nbsp;&mdash;&nbsp; sweeping all hyperparameters</div>
        <div class=\"ai-counter\">{total_sacrifices:,}<span class=\"ai-counter-total\"> / {total_models:,}</span></div>
        <div class=\"ai-counter-label\">AIs trained</div>
        <div class=\"progress-bar-track\">
            <div class=\"progress-bar-fill\" style=\"width: {pct_done_clamped:.2f}%\"></div>
        </div>
        <div class=\"progress-bar-label\">{pct_done_clamped:.1f}% complete &nbsp;&bull;&nbsp; {total_grokked:,} grokked &nbsp;&bull;&nbsp; {overall_grok_rate:.1f}% grok rate</div>
        <div class=\"eta-row\">
            <div class=\"eta-block\">
                <div class=\"eta-label\">ETA</div>
                <div class=\"eta-value\">{eta_str}</div>
            </div>
            <div class=\"eta-block\">
                <div class=\"eta-label\">avg time</div>
                <div class=\"eta-value\">{avg_time_str}</div>
            </div>
            <div class=\"eta-block\">
                <div class=\"eta-label\">remaining</div>
                <div class=\"eta-value\">{remaining_models:,}</div>
            </div>
        </div>
    </div>

    <div class=\"tabs\">
        <button class=\"tab-button active\" data-tab=\"overview\">Overview</button>
        <button class=\"tab-button\" data-tab=\"by-n\">By N</button>
        <button class=\"tab-button\" data-tab=\"hyperparams\">Hyperparams</button>
    </div>

    <!-- OVERVIEW TAB -->
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
            <h2>Grok Rate — weight_decay &times; learning_rate</h2>
            <p class=\"card-hint\">Darker green = more grokking. Grey = no data yet.</p>
            <div id=\"heatmap-container\"></div>
        </div>
    </div>

    <!-- BY N TAB -->
    <div id=\"tab-by-n\" class=\"tab-panel\">
        <div class=\"card\">
            <h2>Grok Rate vs N</h2>
            <canvas id=\"nGrokChart\" height=\"100\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Steps to Grok vs N</h2>
            <p class=\"card-hint\">Median training steps until test accuracy &gt; 97%. Only shown for N values with at least one grokked model.</p>
            <canvas id=\"nStepsChart\" height=\"100\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Average Training Curve</h2>
            <label for=\"n-curve-select\">Choose N:</label>
            <select id=\"n-curve-select\"></select>
            <canvas id=\"nCurveChart\" height=\"100\" style=\"margin-top:1rem\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Per-N Statistics</h2>
            <div class=\"table-wrapper\">
                <table>
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
                f"<td>{n_steps_grok if n_steps_grok else '—'}</td>"
                f"</tr>\n"
            )

    html += """        </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- HYPERPARAMS TAB -->
    <div id=\"tab-hyperparams\" class=\"tab-panel\">
        <div class=\"card\">
            <h2>Grok Rate by weight_decay</h2>
            <canvas id=\"wdChart\" height=\"100\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Grok Rate by learning_rate</h2>
            <canvas id=\"lrChart\" height=\"100\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Fix weight_decay &rarr; see learning_rate breakdown</h2>
            <label for=\"wd-select\">weight_decay:</label>
            <select id=\"wd-select\"></select>
            <canvas id=\"wdDrillChart\" height=\"100\" style=\"margin-top:1rem\"></canvas>
        </div>

        <div class=\"card\">
            <h2>Fix learning_rate &rarr; see weight_decay breakdown</h2>
            <label for=\"lr-select\">learning_rate:</label>
            <select id=\"lr-select\"></select>
            <canvas id=\"lrDrillChart\" height=\"100\" style=\"margin-top:1rem\"></canvas>
        </div>
    </div>

    <script id=\"sisyphus-data\" type=\"application/json\">""" + embedded_json + """</script>
    <script>
    (function () {
        const data = JSON.parse(document.getElementById('sisyphus-data').textContent || '{}');
        const combos  = data.combinations  || {};
        const wdStats = data.weight_decay  || {};
        const lrStats = data.learning_rate || {};
        const nStats  = data.N             || {};

        // ── helpers ──────────────────────────────────────────────────────
        const fmtKey = k => k.replace(/_/g, '.');

        function grokRate(s) {
            const n = (s && s.num_of_sacrifices) || 0;
            const g = (s && s.num_of_grokked)    || 0;
            return n > 0 ? g / n * 100 : null;   // null = no data
        }

        function grokColor(rate) {
            if (rate === null) return '#1a1f36';  // no data
            const h = rate * 1.2;                 // 0 → red, 100 → green
            return `hsl(${h},70%,28%)`;
        }
        function grokBorder(rate) {
            if (rate === null) return '#2a3060';
            const h = rate * 1.2;
            return `hsl(${h},70%,45%)`;
        }

        const sortNum = arr => [...arr].sort((a, b) => parseFloat(a) - parseFloat(b));

        const CHART_DEFAULTS = {
            responsive: true,
            maintainAspectRatio: true,
            animation: false,
            plugins: { legend: { labels: { color: '#d0d4ff' } } },
            scales: {
                x: { ticks: { color: '#a0a8d0' }, grid: { color: '#1e2444' } },
                y: { beginAtZero: true, max: 100,
                     ticks: { color: '#a0a8d0' }, grid: { color: '#1e2444' },
                     title: { display: true, color: '#a0a8d0' } },
            },
        };

        function barChart(canvas, labels, datasets, yLabel) {
            if (!canvas) return null;
            const opts = JSON.parse(JSON.stringify(CHART_DEFAULTS));
            opts.scales.y.title.text = yLabel || '';
            return new Chart(canvas.getContext('2d'), { type: 'bar', data: { labels, datasets }, options: opts });
        }

        function lineChart(canvas, labels, datasets, yLabel) {
            if (!canvas) return null;
            const opts = JSON.parse(JSON.stringify(CHART_DEFAULTS));
            opts.scales.y.title.text = yLabel || '';
            return new Chart(canvas.getContext('2d'), { type: 'line', data: { labels, datasets }, options: opts });
        }

        // ── tabs ─────────────────────────────────────────────────────────
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.dataset.tab;
                document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
                btn.classList.add('active');
                const panel = document.getElementById(`tab-${target}`);
                if (panel) panel.classList.add('active');
            });
        });

        // ── OVERVIEW: heatmap ─────────────────────────────────────────────
        (function buildHeatmap() {
            const container = document.getElementById('heatmap-container');
            if (!container) return;

            const wdKeys = sortNum(Object.keys(combos));
            const lrSet  = new Set();
            wdKeys.forEach(wd => Object.keys(combos[wd] || {}).forEach(lr => lrSet.add(lr)));
            const lrKeys = sortNum([...lrSet]);

            if (!wdKeys.length || !lrKeys.length) {
                container.innerHTML = '<p style="opacity:.5">No data yet.</p>';
                return;
            }

            const table = document.createElement('table');
            table.className = 'heatmap-table';

            // header row
            const thead = table.createTHead();
            const hr = thead.insertRow();
            hr.insertCell().textContent = 'wd \\ lr';
            lrKeys.forEach(lr => {
                const th = document.createElement('th');
                th.textContent = fmtKey(lr);
                hr.appendChild(th);
            });

            const tbody = table.createTBody();
            wdKeys.forEach(wd => {
                const row = tbody.insertRow();
                const lh = row.insertCell();
                lh.textContent = fmtKey(wd);
                lh.style.fontWeight = '600';
                lrKeys.forEach(lr => {
                    const cell = row.insertCell();
                    const s = (combos[wd] || {})[lr];
                    const rate = grokRate(s);
                    cell.style.background = grokColor(rate);
                    cell.style.border = `1px solid ${grokBorder(rate)}`;
                    cell.style.textAlign = 'center';
                    cell.style.padding = '0.45rem 0.6rem';
                    cell.style.fontVariantNumeric = 'tabular-nums';
                    if (rate === null) {
                        cell.textContent = '—';
                        cell.style.opacity = '0.35';
                    } else {
                        const n = s.num_of_sacrifices;
                        cell.innerHTML = `<strong>${rate.toFixed(0)}%</strong><br><span style="font-size:.75rem;opacity:.6">${n} runs</span>`;
                    }
                });
            });

            container.appendChild(table);
        })();

        // ── BY-N: grok rate vs N ──────────────────────────────────────────
        (function buildNCharts() {
            const nKeys = sortNum(
                Object.keys(nStats).filter(k => (nStats[k].num_of_sacrifices || 0) > 0)
            );

            const nLabels     = nKeys.map(k => `N=${k}`);
            const nRates      = nKeys.map(k => grokRate(nStats[k]) ?? 0);
            const nStepsRaw   = nKeys.map(k => nStats[k]?.avg_data?.steps_to_grok || 0);
            const nStepsFiltered = nStepsRaw.map(v => v > 0 ? v : null);

            barChart(
                document.getElementById('nGrokChart'),
                nLabels,
                [{ label: 'Grok %', data: nRates,
                   backgroundColor: nRates.map(r => `hsla(${r*1.2},70%,45%,0.5)`),
                   borderColor:     nRates.map(r => `hsl(${r*1.2},70%,55%)`),
                   borderWidth: 1 }],
                'Grok %'
            );

            lineChart(
                document.getElementById('nStepsChart'),
                nLabels,
                [{ label: 'Median Steps to Grok', data: nStepsFiltered,
                   borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.15)',
                   pointBackgroundColor: '#a78bfa', tension: 0.3, spanGaps: true }],
                'Steps'
            );

            // ── average training curve picker ─────────────────────────────
            const curveSelect = document.getElementById('n-curve-select');
            let curveChart = null;

            function renderCurve(nKey) {
                if (curveChart) { curveChart.destroy(); curveChart = null; }
                const canvas = document.getElementById('nCurveChart');
                if (!canvas) return;
                const entry = nStats[nKey];
                if (!entry) return;
                const trainAcc = entry.avg_data?.train_acc || [];
                const testAcc  = entry.avg_data?.test_acc  || [];
                if (!trainAcc.length) {
                    canvas.style.display = 'none';
                    return;
                }
                canvas.style.display = '';
                const labels = trainAcc.map((_, i) => i);
                const opts = JSON.parse(JSON.stringify(CHART_DEFAULTS));
                opts.scales.y.title.text = 'Accuracy %';
                opts.plugins.legend.labels.color = '#d0d4ff';
                curveChart = new Chart(canvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [
                            { label: 'Train acc', data: trainAcc,
                              borderColor: '#7bd7ff', backgroundColor: 'rgba(123,215,255,0.1)',
                              tension: 0.2, pointRadius: 0 },
                            { label: 'Test acc',  data: testAcc,
                              borderColor: '#f9a8d4', backgroundColor: 'rgba(249,168,212,0.1)',
                              tension: 0.2, pointRadius: 0 },
                        ],
                    },
                    options: opts,
                });
            }

            if (curveSelect) {
                nKeys.forEach(k => {
                    const opt = document.createElement('option');
                    opt.value = k; opt.textContent = `N = ${k}`;
                    curveSelect.appendChild(opt);
                });
                curveSelect.addEventListener('change', () => renderCurve(curveSelect.value));
                if (nKeys.length) { curveSelect.value = nKeys[0]; renderCurve(nKeys[0]); }
            }
        })();

        // ── HYPERPARAMS: WD / LR bar charts ──────────────────────────────
        (function buildHyperparamCharts() {
            const wdKeys = sortNum(Object.keys(wdStats));
            const lrKeys = sortNum(Object.keys(lrStats));

            barChart(
                document.getElementById('wdChart'),
                wdKeys.map(fmtKey),
                [{ label: 'Grok %', data: wdKeys.map(k => grokRate(wdStats[k]) ?? 0),
                   backgroundColor: 'rgba(180,123,255,0.4)',
                   borderColor: 'rgba(180,123,255,1)', borderWidth: 1 }],
                'Grok %'
            );

            barChart(
                document.getElementById('lrChart'),
                lrKeys.map(fmtKey),
                [{ label: 'Grok %', data: lrKeys.map(k => grokRate(lrStats[k]) ?? 0),
                   backgroundColor: 'rgba(123,255,196,0.4)',
                   borderColor: 'rgba(123,255,196,1)', borderWidth: 1 }],
                'Grok %'
            );

            // WD drill
            const wdSelect = document.getElementById('wd-select');
            let wdDrillChart = null;
            function renderWdDrill(wd) {
                if (wdDrillChart) { wdDrillChart.destroy(); wdDrillChart = null; }
                const lrDict = combos[wd] || {};
                const keys = sortNum(Object.keys(lrDict));
                wdDrillChart = barChart(
                    document.getElementById('wdDrillChart'),
                    keys.map(fmtKey),
                    [{ label: `Grok % (wd=${fmtKey(wd)})`,
                       data: keys.map(lr => grokRate(lrDict[lr]) ?? 0),
                       backgroundColor: 'rgba(255,180,123,0.4)',
                       borderColor: 'rgba(255,180,123,1)', borderWidth: 1 }],
                    'Grok %'
                );
            }
            if (wdSelect) {
                const allWd = sortNum(Object.keys(combos));
                allWd.forEach(wd => {
                    const opt = document.createElement('option');
                    opt.value = wd; opt.textContent = fmtKey(wd);
                    wdSelect.appendChild(opt);
                });
                wdSelect.addEventListener('change', () => renderWdDrill(wdSelect.value));
                if (allWd.length) { wdSelect.value = allWd[0]; renderWdDrill(allWd[0]); }
            }

            // LR drill
            const lrSelect = document.getElementById('lr-select');
            let lrDrillChart = null;
            function renderLrDrill(lr) {
                if (lrDrillChart) { lrDrillChart.destroy(); lrDrillChart = null; }
                const allWd = sortNum(Object.keys(combos));
                const keys = allWd.filter(wd => lr in (combos[wd] || {}));
                lrDrillChart = barChart(
                    document.getElementById('lrDrillChart'),
                    keys.map(fmtKey),
                    [{ label: `Grok % (lr=${fmtKey(lr)})`,
                       data: keys.map(wd => grokRate((combos[wd] || {})[lr]) ?? 0),
                       backgroundColor: 'rgba(255,123,180,0.4)',
                       borderColor: 'rgba(255,123,180,1)', borderWidth: 1 }],
                    'Grok %'
                );
            }
            if (lrSelect) {
                const allLr = new Set();
                Object.values(combos).forEach(d => Object.keys(d).forEach(lr => allLr.add(lr)));
                const sortedLr = sortNum([...allLr]);
                sortedLr.forEach(lr => {
                    const opt = document.createElement('option');
                    opt.value = lr; opt.textContent = fmtKey(lr);
                    lrSelect.appendChild(opt);
                });
                lrSelect.addEventListener('change', () => renderLrDrill(lrSelect.value));
                if (sortedLr.length) { lrSelect.value = sortedLr[0]; renderLrDrill(sortedLr[0]); }
            }
        })();
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

    html = build_html(data, total_models=get_total_number_of_models())
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
