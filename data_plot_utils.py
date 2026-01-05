import json
import os
from typing import Optional

# Ensure Tcl/Tk paths are correctly set for matplotlib's Tk backend
tcl_path: str = r"C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
tk_path: str = r"C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

os.environ["TCL_LIBRARY"] = tcl_path
os.environ["TK_LIBRARY"] = tk_path

import matplotlib.pyplot as plt


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

	for fname in sorted(model_files, key=_model_id_from_name):
		mid = _model_id_from_name(fname)
		model_path = os.path.join(models_dir, fname)
		with open(model_path, "r", encoding="utf-8") as f:
			model_data = json.load(f)

			steps = model_data.get("steps", [])
			test_acc = model_data.get("test_acc", [])
			if not steps or not test_acc:
				continue

			ax.plot(steps, test_acc, label=f"model {mid} test")

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


if __name__ == "__main__":
    # Example usage
    plot_avg_for_N(67)
    print("0:")
    plot_model_for_N(67, 0)
    print("1:")
    plot_model_for_N(67, 1)
    print("2:")
    plot_model_for_N(67, 2)
    print("all 67:")
    plot_all_models_for_N(67)
    print("all Ns:")
    plot_all_models()