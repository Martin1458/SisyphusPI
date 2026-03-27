# SisyphusPI

A **grokking experiment runner** designed to run on a Raspberry Pi. It systematically trains tiny transformer models on modular arithmetic (`(a + b) % N`) and records whether each model "groks" — the phenomenon where a neural network suddenly generalizes after memorizing training data.

The name is a reference to Sisyphus: the experiment endlessly trains models, saves results, and repeats.

---

## Architecture

```
config.py          — Load all hyperparams from config.ini / secrets.ini
  |
m.py               — Main loop: iterates all (N, params) states, calls train_until_grok()
  |
stepping_utils.py  — State machine: encodes the sweep as a mixed-radix counter
  |
model_trainer.py   — NanoMath transformer + training loop
  |
data_gen_utils.py  — Generate all (a, b) pairs for modulus N, split train/test
  |
data_save_utils.py — ModelInfo: save per-model JSON + update aggregate data.json
  |
website_maker.py   — Read data.json → generate index.html, git push to website repo
```

---

## Data Flow

1. **Config** — `config.ini` defines the sweep: `D_MODELS`, `N_HEADS_LIST`, `TRAIN_PCTS`, `WEIGHT_DECAYS`, `LEARNING_RATES`, and the range of moduli N to sweep over.

2. **State machine** (`stepping_utils.py`) — treats all parameter combinations as a mixed-radix counter. `get_resume_state()` reads `data.json` to find how many models have run and resumes from there. `get_next_state()` advances like an odometer.

3. **Training** (`model_trainer.py`) — `NanoMath` is a 1-layer transformer encoder that takes `[a, b, N]` as tokens and predicts `(a+b)%N`. Trained with AdamW + high weight decay (key for grokking). Stops early if test accuracy >97%.

4. **Saving** (`data_save_utils.py`) — Each model writes a `<model_id>.json` with step-by-step train/test accuracy curves. The aggregate `output/data.json` is updated with running averages, grok rates per N, per weight_decay, per learning_rate, and per (wd, lr) combo. Curves are phase-aligned by their grok point before averaging.

5. **Website** (`website_maker.py`) — Every 10 models, reads `data.json` and generates `SisyphusPI-website/index.html` with Chart.js charts: grok rate by (wd, lr) combo, per-N stats table, drilldown filters. Then auto-commits and pushes to a separate `SisyphusPI-website` git repo (e.g. GitHub Pages).

---

## Running

```bash
python m.py
```

Training resumes automatically from where it left off if interrupted.

---

## Configuration

Edit `config.ini` to control the hyperparameter sweep:

```ini
[model]
D_MODELS = 64, 128
N_HEADS_LIST = 2, 4
TRAIN_PCTS = 0.3, 0.4
STEPS = 5000

[training]
NUM_OF_WAVES = 1
MIN_N = 10
MAX_N = 60
N_STEP = 10
WEIGHT_DECAYS = 0.01, 0.1
LEARNING_RATES = 0.001, 0.005

[plot]
PLOTTING = false
```

For local path overrides (TCL/TK, custom output directory), edit `secrets.ini` (gitignored, created automatically on first run).

---

## Key Design Notes

- **File locking** (`filelock`) prevents corruption of `data.json` when parallel workers are used.
- **Crash recovery** — progress is stored in `data.json`'s `num_of_sacrifices` count; kill and restart safely at any time.
- `main.py` is an old standalone prototype. `m.py` is the actual runner.
