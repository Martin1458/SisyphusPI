"""Configuration and secrets initialization for SisyphusPI.

This module is responsible for:
- Ensuring config.ini and secrets.ini exist (creating templates if missing).
- Loading configuration values.
- Applying optional TCL/TK and OUTPUT_DIR overrides from secrets.ini.

All values are computed once at import time and re-used by other modules.
"""

from __future__ import annotations

import os
import configparser

BASE_DIR = os.path.dirname(__file__)
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.ini")
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")


def _ensure_secrets_file(path: str) -> None:
	if os.path.exists(path):
		return
	with open(path, "w", encoding="utf-8") as f:
		f.write(
			"""[paths]
# Optional local overrides. This file is ignored by git.

# Absolute path to your TCL library (optional)
#TCL_PATH = C:\\path\\to\\python\\tcl\\tcl8.6

# Absolute path to your TK library (optional)
#TK_PATH = C:\\path\\to\\python\\tcl\\tk8.6

# Absolute path to your output directory (optional)
#OUTPUT_DIR = D:\\path\\to\\output\\SisyphusPI
"""
		)
	print("Created template secrets.ini; edit it with your local paths if needed.")


def _ensure_config_file(path: str) -> None:
	if os.path.exists(path):
		return
	with open(path, "w", encoding="utf-8") as f:
		f.write(
			"""[model]
# Hidden dimension
D_MODELS = 128
# Comma-separated list of attention head counts to sweep over
N_HEADS_LIST = 4
# Default weight decay (used if WEIGHT_DECAYS list is not set)
WEIGHT_DECAY = 1.0
# Comma-separated list of training data fractions to sweep over (0.0 - 1.0)
TRAIN_PCTS = 0.4
# Maximum training steps per model
STEPS = 10000

[training]
# Number of waves of sacrifices to perform
NUM_OF_WAVES = 5
# Minimum modulus N to try
MIN_N = 5
# Maximum modulus N to try
MAX_N = 60
# Step size when sweeping N (1 = every N, 2 = every second N, ...)
N_STEP = 5
# Comma-separated list of weight decays to sweep over
WEIGHT_DECAYS = 0.01, 0.05, 0.1, 0.5, 1.0, 2.0
# Comma-separated list of learning rates to sweep over
LEARNING_RATES = 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1

[plot]
# Whether to show live training plots
PLOTTING = false
"""
		)
	print("Created template config.ini with default training settings.")


_ensure_secrets_file(SECRETS_PATH)
_ensure_config_file(CONFIG_PATH)

_secrets = configparser.ConfigParser()
_secrets.read(SECRETS_PATH, encoding="utf-8")

_config = configparser.ConfigParser()
_config.read(CONFIG_PATH, encoding="utf-8")


# Optional TCL/TK paths from secrets.ini (no hard-coded defaults)
if _secrets.has_section("paths"):
	TCL_PATH: str | None = _secrets.get("paths", "TCL_PATH", fallback=None)
	TK_PATH: str | None = _secrets.get("paths", "TK_PATH", fallback=None)
else:
	TCL_PATH = None
	TK_PATH = None

if TCL_PATH:
	os.environ["TCL_LIBRARY"] = TCL_PATH
if TK_PATH:
	os.environ["TK_LIBRARY"] = TK_PATH


# Model and training hyperparameters from config.ini
_dm_raw = _config.get("model", "D_MODELS", fallback="128")
D_MODELS: list[int] = [int(x.strip()) for x in _dm_raw.split(",") if x.strip()]
D_MODEL: int = D_MODELS[0]  # default scalar for backward compat

_nh_raw = _config.get("model", "N_HEADS_LIST", fallback="4")
N_HEADS_LIST: list[int] = [int(x.strip()) for x in _nh_raw.split(",") if x.strip()]
N_HEADS: int = N_HEADS_LIST[0]  # default scalar for backward compat

WEIGHT_DECAY: float = _config.getfloat("model", "WEIGHT_DECAY", fallback=1.0)

_tp_raw = _config.get("model", "TRAIN_PCTS", fallback="0.4")
TRAIN_PCTS: list[float] = [float(x.strip()) for x in _tp_raw.split(",") if x.strip()]
TRAIN_PCT: float = TRAIN_PCTS[0]  # default scalar for backward compat

STEPS: int = _config.getint("model", "STEPS", fallback=10000)

NUM_OF_WAVES: int = _config.getint("training", "NUM_OF_WAVES", fallback=3)
MIN_N: int = _config.getint("training", "MIN_N", fallback=20)
MAX_N: int = _config.getint("training", "MAX_N", fallback=200)
N_STEP: int = _config.getint("training", "N_STEP", fallback=1)
_wd_raw = _config.get("training", "WEIGHT_DECAYS", fallback=str(WEIGHT_DECAY))
WEIGHT_DECAYS: list[float] = [float(x.strip()) for x in _wd_raw.split(",") if x.strip()]
_lr_raw = _config.get("training", "LEARNING_RATES", fallback="0.005")
LEARNING_RATES: list[float] = [float(x.strip()) for x in _lr_raw.split(",") if x.strip()]

# Output directory with optional override from secrets.ini
_base_output_dir = "output"
if _config.has_section("paths"):
	_base_output_dir = _config.get("paths", "OUTPUT_DIR", fallback=_base_output_dir)

_secrets_output_dir: str | None = None
if _secrets.has_section("paths"):
	_secrets_output_dir = _secrets.get("paths", "OUTPUT_DIR", fallback=None)

if _secrets_output_dir:
	_base_output_dir = _secrets_output_dir

# Final output directory (flat — no project subdirectory)
OUTPUT_DIR: str = _base_output_dir

PLOTTING: bool = _config.getboolean("plot", "PLOTTING", fallback=False)

# Aggregate JSON storing averaged stats across runs
AGGREGATE_DATA_PATH: str = os.path.join(OUTPUT_DIR, "data.json")


SMART_CONFIG = (
    (MIN_N, MAX_N, N_STEP), NUM_OF_WAVES,
    [D_MODELS, N_HEADS_LIST, TRAIN_PCTS, WEIGHT_DECAYS, LEARNING_RATES]
    )
SMART_LISTS_NAMES = ["D_MODELS", "N_HEADS_LIST", "TRAIN_PCTS", "WEIGHT_DECAYS", "LEARNING_RATES"]

