import json
import os
from typing import Optional

# Ensure Tcl/Tk paths are correctly set for matplotlib's Tk backend
tcl_path: str = r"C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
tk_path: str = r"C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

os.environ["TCL_LIBRARY"] = tcl_path
os.environ["TK_LIBRARY"] = tk_path

import matplotlib.pyplot as plt
import numpy as np


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
	data_path: str = r"output\data.json",
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot averaged train/test accuracy over steps for a specific N.

	Uses the aggregate file written by ModelInfo.save_data_info (data_path).
	Curves are indexed by step number (0, 1, 2, ...).
	"""

	abs_path = _resolve_base_paths(data_path)

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
	base_dir: str = "output",
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot train/test accuracy over steps for a specific model.

	Parameters
	----------
	N : int
		Modulus; used as the subdirectory name under base_dir.
	model_id : int
		JSON file name without extension (e.g. 0.json -> model_id=0).
	base_dir : str, optional
		Root directory where per-model JSON files are stored (default "output").
	"""

	models_dir = _resolve_base_paths(os.path.join(base_dir, str(N)))
	model_path = os.path.join(models_dir, f"{model_id}.json")

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


def plot_all_models_for_N(N: int, base_dir: str = "output") -> None:
	"""Overlay all saved *test* curves for a given N on one graph.

	Uses the JSON files in output/N/. Each model contributes one test curve.
	"""

	models_dir = _resolve_base_paths(os.path.join(base_dir, str(N)))
	if not os.path.isdir(models_dir):
		raise FileNotFoundError(f"No directory for N={N}: {models_dir}")

	model_files = [f for f in os.listdir(models_dir) if f.endswith(".json")]
	if not model_files:
		raise ValueError(f"No model JSON files found for N={N} in {models_dir}")

	def _model_id_from_name(name: str) -> int:
		return int(os.path.splitext(name)[0])

	_, ax = plt.subplots(1, 1, figsize=(8, 6))
	base_step = None
	step_size = None

	for fname in sorted(model_files, key=_model_id_from_name):
		mid = _model_id_from_name(fname)
		model_path = os.path.join(models_dir, fname)
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
	agg_path = _resolve_base_paths(os.path.join(base_dir, "data.json"))
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


def plot_all_models(base_dir: str = "output") -> None:
	"""Overlay test curves of *all* individual models for all N on one graph.

	Iterates over base_dir/<N>/<model_id>.json and plots each model's
	test accuracy vs its stored steps on a single axes.
	"""

	base_path = _resolve_base_paths(base_dir)
	if not os.path.isdir(base_path):
		raise FileNotFoundError(f"Base output directory not found: {base_path}")

	n_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
	if not n_dirs:
		raise ValueError(f"No N subdirectories found under {base_path}")

	_, ax = plt.subplots(1, 1, figsize=(8, 6))

	for n_str in sorted(n_dirs, key=int):
		models_dir = os.path.join(base_path, n_str)
		model_files = [f for f in os.listdir(models_dir) if f.endswith(".json")]
		for fname in sorted(model_files, key=lambda name: int(os.path.splitext(name)[0])):
			model_path = os.path.join(models_dir, fname)
			with open(model_path, "r", encoding="utf-8") as f:
				model_data = json.load(f)

			steps = model_data.get("steps", [])
			test_acc = model_data.get("test_acc", [])
			if not steps or not test_acc:
				continue

			model_id = os.path.splitext(fname)[0]
			ax.plot(steps, test_acc, label=f"N={n_str}, model={model_id}")

	ax.set_xlabel("Step")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title("Test accuracy over time for all models")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.show()


def plot_avg_train_time_by_N(
	data_path: str = r"output\data.json",
	show: bool = True,
	ax: Optional[plt.Axes] = None,
) -> None:
	"""Plot average training time per N.

	Reads model_training_times from the aggregate data file and plots
	N on the x-axis vs average training time (seconds) on the y-axis.
	"""

	abs_path = _resolve_base_paths(data_path)
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
	data_path: str = r"output\data.json",
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

	abs_path = _resolve_base_paths(data_path)
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
	# Example usage
	#plot_avg_for_N(67)
	#print("0:")
	#plot_model_for_N(67, 0)
	#print("1:")
	#plot_model_for_N(67, 1)
	#print("2:")
	#plot_model_for_N(67, 2)
	print("all 67:")
	plot_all_models_for_N(3)
	print("all Ns:")
	plot_all_models()
	print("heatmap:")
	plot_train_time_heatmap(5)