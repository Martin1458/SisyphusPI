import json
import os
from typing import Optional

# TCL/TK paths are set by config.py at import time via secrets.ini.
# Import config to ensure environment is configured before matplotlib.
import config as _config  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from stepping_utils import get_param_values, state_to_model_dir, _INFO_KEYS
from config import SMART_CONFIG, SMART_LISTS_NAMES

# Default paths derived from config
_DEFAULT_DATA_PATH: str = _config.AGGREGATE_DATA_PATH
_DEFAULT_BASE_DIR: str = _config.OUTPUT_DIR


# ---------------------------------------------------------------------------
# Helpers for discovering model JSON files in the nested directory tree.
# ---------------------------------------------------------------------------

def _find_model_files_for_N(N: int, base_dir: str | None = None) -> list[str]:
	"""Walk the output tree and return all .json model files whose
	immediate parent directory matches ``str(N)``."""
	base_path = os.path.join(os.path.dirname(__file__), base_dir or _DEFAULT_BASE_DIR)
	results: list[str] = []
	if not os.path.isdir(base_path):
		return results
	n_str = str(N)
	for root, _dirs, files in os.walk(base_path):
		if os.path.basename(root) != n_str:
			continue
		for fname in files:
			if fname.endswith(".json"):
				results.append(os.path.join(root, fname))
	return results


def _find_model_files_for_state(state: tuple, base_dir: str | None = None) -> list[str]:
	"""Return all .json model files in the exact directory for a state tuple."""
	model_dir = state_to_model_dir(state)
	if not os.path.isdir(model_dir):
		return []
	return [
		os.path.join(model_dir, f)
		for f in os.listdir(model_dir)
		if f.endswith(".json")
	]


def _find_all_model_files(base_dir: str | None = None) -> list[tuple[str, str]]:
	"""Walk the output tree and return (parent_N_str, full_path) for every
	model .json file, where parent_N_str is the leaf directory name (N)."""
	base_path = os.path.join(os.path.dirname(__file__), base_dir or _DEFAULT_BASE_DIR)
	results: list[tuple[str, str]] = []
	if not os.path.isdir(base_path):
		return results
	for root, _dirs, files in os.walk(base_path):
		json_files = [f for f in files if f.endswith(".json")]
		if not json_files:
			continue
		leaf = os.path.basename(root)
		# Only include if the leaf dir looks like an integer N
		if not leaf.isdigit():
			continue
		for fname in json_files:
			results.append((leaf, os.path.join(root, fname)))
	return results


def start_live_accuracy_plot() -> tuple[plt.Figure, plt.Axes]:
	"""Initialize an interactive matplotlib figure/axes for live accuracy plots.

	Returns (fig, ax) with interactive mode enabled.
	"""

	plt.ion()
	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	return fig, ax


def update_live_accuracy_plot(ax: plt.Axes, model_info, step: int, pause: float = 0.1) -> None:
	"""Update the live accuracy plot using data from ``model_info``.

	``model_info`` is expected to provide ``get_data_lists()``, returning
	(steps_list, train_accs, test_accs).
	"""

	steps_list, train_accs, test_accs = model_info.get_data_lists()
	ax.clear()
	ax.plot(steps_list, train_accs, label="Train Acc")
	ax.plot(steps_list, test_accs, label="Test Acc (Grokking)")
	ax.set_title(f"Accuracy Over Time (Step {step})")
	ax.set_xlabel("Step")
	ax.set_ylabel("Accuracy (%)")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.draw()
	plt.pause(pause)


def finalize_live_plot(block: bool = True) -> None:
	"""Turn off interactive mode and optionally block with ``plt.show()``."""

	plt.ioff()
	if block:
		plt.show()


def _resolve_base_paths(data_path: str) -> str:
	"""Return absolute path to a data file relative to this repo."""
	return os.path.join(os.path.dirname(__file__), data_path)


def plot_avg_for_N(
	N: int,
	data_path: str | None = None,
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot averaged train/test accuracy over steps for a specific N.

	Uses the aggregate file written by ModelInfo.save_data_info (data_path).
	Curves are indexed by step number (0, 1, 2, ...).
	"""

	abs_path = _resolve_base_paths(data_path or _DEFAULT_DATA_PATH)

	if not os.path.exists(abs_path):
		raise FileNotFoundError(f"Aggregate data file not found: {abs_path}")

	with open(abs_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	n_str = str(N)
	if "N" not in data or n_str not in data["N"]:
		raise KeyError(f"No aggregate entry for N={N} in {abs_path}")

	n_entry = data["N"][n_str]
	avg_data = n_entry.get("avg_data", {})
	train_acc = avg_data.get("train_acc", [])
	test_acc = avg_data.get("test_acc", [])

	if not train_acc or not test_acc:
		raise ValueError(f"No averaged accuracy data stored for N={N}")

	steps = list(range(len(train_acc)))

	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(8, 6))

	ax.plot(steps, train_acc, label="Avg Train Acc")
	ax.plot(steps, test_acc, label="Avg Test Acc")
	ax.set_xlabel("Step index")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title(f"Average Accuracy Over Time for N={N}")
	ax.legend()
	ax.grid(True, alpha=0.3)

	if show:
		plt.show()


def plot_model_for_N(
	N: int,
	model_id: int,
	state: tuple | None = None,
	base_dir: str | None = None,
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot train/test accuracy over steps for a specific model.

	Parameters
	----------
	N : int
		Modulus value.
	model_id : int
		JSON file name without extension (e.g. 0.json -> model_id=0).
	state : tuple, optional
		State tuple to pinpoint an exact parameter combination directory.
		If None, walks the tree and finds the first match.
	base_dir : str, optional
		Root directory where per-model JSON files are stored (default "output").
	"""

	if state is not None:
		model_dir = state_to_model_dir(state)
	else:
		# Fall back to scanning the tree for the first directory for this N
		files = _find_model_files_for_N(N, base_dir)
		if not files:
			raise FileNotFoundError(f"No model files found for N={N}")
		model_dir = os.path.dirname(files[0])

	model_path = os.path.join(model_dir, f"{model_id}.json")

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")

	with open(model_path, "r", encoding="utf-8") as f:
		model_data = json.load(f)

	steps = model_data.get("steps", [])
	train_acc = model_data.get("train_acc", [])
	test_acc = model_data.get("test_acc", [])

	if not steps or not train_acc or not test_acc:
		raise ValueError(
			f"Model file {model_path} does not contain steps/train_acc/test_acc data"
		)

	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(8, 6))

	ax.plot(steps, train_acc, label="Train Acc")
	ax.plot(steps, test_acc, label="Test Acc")
	ax.set_xlabel("Step")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title(f"Model {model_id} Accuracy Over Time (N={N})")
	ax.legend()
	ax.grid(True, alpha=0.3)

	if show:
		plt.show()


def plot_all_models_for_N(N: int, state: tuple | None = None, base_dir: str | None = None) -> None:
	"""Overlay all saved *test* curves for a given N on one graph.

	Walks the output tree to find all model JSON files for this N.
	If ``state`` is provided, only plots models from that exact parameter combo.
	"""

	if state is not None:
		model_files_paths = _find_model_files_for_state(state, base_dir)
	else:
		model_files_paths = _find_model_files_for_N(N, base_dir)

	if not model_files_paths:
		raise FileNotFoundError(f"No model JSON files found for N={N}")

	def _model_id_from_path(path: str) -> int:
		return int(os.path.splitext(os.path.basename(path))[0])

	_, ax = plt.subplots(1, 1, figsize=(8, 6))
	base_step = None
	step_size = None

	for model_path in sorted(model_files_paths, key=_model_id_from_path):
		mid = _model_id_from_path(model_path)
		with open(model_path, "r", encoding="utf-8") as f:
			model_data = json.load(f)

			steps = model_data.get("steps", [])
			test_acc = model_data.get("test_acc", [])
			if not steps or not test_acc:
				continue

			# Remember the step spacing from the first valid model
			if base_step is None and len(steps) >= 2:
				base_step = steps[0]
				step_size = steps[1] - steps[0]

			ax.plot(steps, test_acc, label=f"model {mid} test")

	# add avg curve if available
	agg_path = _resolve_base_paths(os.path.join(base_dir or _DEFAULT_BASE_DIR, "data.json"))
	if os.path.exists(agg_path):
		with open(agg_path, "r", encoding="utf-8") as f:
			agg_data = json.load(f)

		n_str = str(N)
		if "N" in agg_data and n_str in agg_data["N"]:
			n_entry = agg_data["N"][n_str]
			avg_data = n_entry.get("avg_data", {})
			avg_test = avg_data.get("test_acc", [])
			if avg_test:
				# Use the same step spacing as the individual models if known
				if step_size is not None and base_step is not None:
					avg_steps = [base_step + i * step_size for i in range(len(avg_test))]
				else:
					avg_steps = list(range(len(avg_test)))
				# Bold/thicker line for the average curve
				ax.plot(
					avg_steps,
					avg_test,
					label="Avg test",
					linewidth=3.0,
					color="black",
				)

	ax.set_xlabel("Step")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title(f"Test Accuracy Over Time for all models (N={N})")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.show()


def plot_all_models(base_dir: str | None = None) -> None:
	"""Overlay test curves of *all* individual models for all N on one graph.

	Walks the entire output tree to find model JSON files at any depth.
	"""

	all_files = _find_all_model_files(base_dir)
	if not all_files:
		raise ValueError("No model JSON files found in output directory")

	_, ax = plt.subplots(1, 1, figsize=(8, 6))

	for n_str, model_path in sorted(all_files, key=lambda p: (int(p[0]), p[1])):
		with open(model_path, "r", encoding="utf-8") as f:
			model_data = json.load(f)

		steps = model_data.get("steps", [])
		test_acc = model_data.get("test_acc", [])
		if not steps or not test_acc:
			continue

		model_id = os.path.splitext(os.path.basename(model_path))[0]
		# Include parent dirs for context since N alone is no longer unique
		rel_path = os.path.relpath(os.path.dirname(model_path),
			os.path.join(os.path.dirname(__file__), base_dir or _DEFAULT_BASE_DIR))
		ax.plot(steps, test_acc, label=f"{rel_path}/m{model_id}")

	ax.set_xlabel("Step")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title("Test accuracy over time for all models")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.show()


def plot_avg_train_time_by_N(
	data_path: str | None = None,
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot average training time per N.

	Reads model_training_times from the aggregate data file and plots
	N on the x-axis vs average training time (seconds) on the y-axis.
	"""

	abs_path = _resolve_base_paths(data_path or _DEFAULT_DATA_PATH)
	if not os.path.exists(abs_path):
		raise FileNotFoundError(f"Aggregate data file not found: {abs_path}")

	with open(abs_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if "N" not in data or not data["N"]:
		raise ValueError("No N entries found in aggregate data")

	Ns = []
	avg_times = []
	for n_str, n_entry in data["N"].items():
		avg_time = n_entry.get("avg_train_time")
		if avg_time is None:
			continue
		Ns.append(int(n_str))
		avg_times.append(float(avg_time))

	if not Ns:
		raise ValueError("No training time data found in aggregate file")

	sorted_pairs = sorted(zip(Ns, avg_times), key=lambda p: p[0])
	Ns_sorted, times_sorted = zip(*sorted_pairs)

	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(8, 6))

	ax.plot(Ns_sorted, times_sorted, marker="o")
	ax.set_xlabel("N (modulus)")
	ax.set_ylabel("Average training time (s)")
	ax.set_title("Average training time per N")
	ax.grid(True, alpha=0.3)

	if show:
		plt.show()


def plot_train_time_heatmap(
	n_per_row: int,
	data_path: str | None = None,
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot a heatmap of average training times.

	Each cell corresponds to one modulus N; the grid is filled
	row-wise with N values, with at most ``n_per_row`` Ns per row.
	Color encodes average training time (seconds), and each cell
	is annotated with its N value.
	"""

	if n_per_row <= 0:
		raise ValueError("n_per_row must be a positive integer")

	abs_path = _resolve_base_paths(data_path or _DEFAULT_DATA_PATH)
	if not os.path.exists(abs_path):
		raise FileNotFoundError(f"Aggregate data file not found: {abs_path}")

	with open(abs_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if "N" not in data or not data["N"]:
		raise ValueError("No N entries found in aggregate data")

	Ns: list[int] = []
	avg_times: list[float] = []
	for n_str, n_entry in data["N"].items():
		avg_time = n_entry.get("avg_train_time")
		if avg_time is None:
			continue
		Ns.append(int(n_str))
		avg_times.append(float(avg_time))

	if not Ns:
		raise ValueError("No training time data found in aggregate file")

	sorted_pairs = sorted(zip(Ns, avg_times), key=lambda p: p[0])
	Ns_sorted, times_sorted = zip(*sorted_pairs)
	Ns_sorted = list(Ns_sorted)
	times_sorted = list(times_sorted)

	n_vals = len(Ns_sorted)
	n_cols = n_per_row
	n_rows = (n_vals + n_cols - 1) // n_cols

	grid = np.full((n_rows, n_cols), np.nan, dtype=float)
	labels = [["" for _ in range(n_cols)] for _ in range(n_rows)]

	for idx, (n_val, t_val) in enumerate(zip(Ns_sorted, times_sorted)):
		row = idx // n_cols
		col = idx % n_cols
		grid[row, col] = t_val
		labels[row][col] = str(n_val)

	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(10, 6))

	cmap = plt.cm.viridis
	im = ax.imshow(grid, cmap=cmap, aspect="auto")
	plt.colorbar(im, ax=ax, label="Average training time (s)")

	ax.set_xticks(range(n_cols))
	ax.set_yticks(range(n_rows))
	ax.set_xlabel("Column within row")
	ax.set_ylabel("Row index")
	ax.set_title("Heatmap of average training time per N")

	# Annotate each cell with its N value
	for r in range(n_rows):
		for c in range(n_cols):
			label = labels[r][c]
			if not label:
				continue
			ax.text(c, r, label, ha="center", va="center", color="white")

	if show:
		plt.show()


if __name__ == "__main__":
	# Example usage — pass a state tuple to filter by exact parameter combo,
	# or omit it to aggregate across all combos for that N.
	print("all models for N=10 (all param combos):")
	plot_all_models_for_N(10)
	print("all Ns:")
	plot_all_models()
	print("heatmap:")
	plot_train_time_heatmap(5)