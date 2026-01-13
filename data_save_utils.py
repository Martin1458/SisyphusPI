import json
import os
import re
import math
from copy import deepcopy

EMPTY_N_ENTRY = {
    'num_of_models': 0,
    'num_of_grokked': 0,
    'models': []
}

EMPTY_MODEL_ENTRY = {
    'steps': [],
    'num_of_steps': 0,
    'train_acc': [],
    'test_acc': [],
    'grokked': 0
}
EMPTY_SMALL_DATA = {
    'num_of_sacrifices': 0,
    'num_of_grokked': 0,
    'N': {
        # 'specific_N': EMPTY_SMALL_DATA_ENTRY 
    }
}

EMPTY_SMALL_DATA_ENTRY = {
    'num_of_sacrifices': 0,
    'num_of_grokked': 0,
    'avg_data': {
        'train_acc': [], # per step
        'test_acc': [], # per step
        'steps_to_grok': 0
    },
    'avg_train_time': 0.0
}

class ModelInfo:
    data_path: str
    output_dir_path: str
    N: int

    def __init__(self, N: int, data_path: str, output_dir_path: str):
        base_dir = os.path.dirname(__file__)
        # Allow both relative (project-local) and absolute paths.
        self.data_path = data_path if os.path.isabs(data_path) else os.path.join(base_dir, data_path)
        self.output_dir_path = output_dir_path if os.path.isabs(output_dir_path) else os.path.join(base_dir, output_dir_path)
        self.N = N
        # Use a deep copy so each model gets its own independent lists
        self.model_data = deepcopy(EMPTY_MODEL_ENTRY)
        self._initialize_data_file()

    def add_data_point(self, step: int, train_acc: float, test_acc: float):
        self.model_data['steps'].append(step)
        self.model_data['train_acc'].append(train_acc)
        self.model_data['test_acc'].append(test_acc)
        self.model_data['num_of_steps'] += 1

    def save_model_info(self, grokked: bool):
        self.model_data['grokked'] = 1 if grokked else 0
        # Save per-model info in output/N/ folder
        base_dir = os.path.dirname(self.data_path)
        n_dir = self.output_dir_path
        os.makedirs(n_dir, exist_ok=True)
        # Use number of files in n_dir as model_id
        model_id = len([name for name in os.listdir(n_dir) if name.endswith('.json')])
        model_file_path = os.path.join(n_dir, f"{model_id}.json")
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_data, f, indent=4)
        format_json_number_lists(model_file_path)

    def save_data_info(self, grokked: bool, train_time: float):
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        data['num_of_sacrifices'] += 1
        if grokked:
            data['num_of_grokked'] += 1

        n_str = str(self.N)
        if n_str not in data['N']:
            # Deep copy so each N entry has its own avg_data lists
            data['N'][n_str] = deepcopy(EMPTY_SMALL_DATA_ENTRY)

        n_entry = data['N'][n_str]
        n_entry['num_of_sacrifices'] += 1
        if grokked:
            n_entry['num_of_grokked'] += 1

        # Maintain running average of training time (seconds) for this N.
        # Backward-compatible: if legacy "model_training_times" list exists
        # and no avg_train_time yet, seed from its mean once.
        if 'avg_train_time' not in n_entry:
            legacy_times = n_entry.get('model_training_times', [])
            if legacy_times:
                n_entry['avg_train_time'] = float(sum(legacy_times) / len(legacy_times))
            else:
                n_entry['avg_train_time'] = 0.0

        count = n_entry['num_of_sacrifices']
        prev_avg = float(n_entry.get('avg_train_time', 0.0))
        new_avg = prev_avg + (float(train_time) - prev_avg) / count
        n_entry['avg_train_time'] = new_avg

        # Recompute avg_data by aligning runs in "phase" (relative progress)
        # instead of raw training step index.

        base_dir = os.path.dirname(self.data_path)

        # Collect all model JSON files for this N, regardless of the
        # weight-decay / learning-rate directory structure. We look for
        # directories whose final component is the string form of N and
        # gather all .json files inside them.
        model_files: list[str] = []
        if os.path.isdir(base_dir):
            for root, _dirs, files in os.walk(base_dir):
                if os.path.basename(root) != str(self.N):
                    continue
                for fname in files:
                    if fname.endswith('.json'):
                        model_files.append(os.path.join(root, fname))

        # Collect only grokked models for phase-aligned averaging.
        all_steps: list[list[int]] = []
        all_train_curves: list[list[float]] = []
        all_test_curves: list[list[float]] = []
        step_counts: list[int] = []
        grok_indices: list[int] = []
        grok_steps_actual: list[int] = []

        GROK_THRESHOLD = 99.0

        for m_path in model_files:
            with open(m_path, 'r', encoding='utf-8') as mf:
                m_data = json.load(mf)

            steps = m_data.get('steps', [])
            train_acc = m_data.get('train_acc', [])
            test_acc = m_data.get('test_acc', [])
            steps_len = len(train_acc)
            if steps_len == 0 or steps_len != len(test_acc) or steps_len != len(steps):
                continue

            # Find the first index where test accuracy crosses the grok threshold.
            g_idx = None
            for idx, val in enumerate(test_acc):
                if val >= GROK_THRESHOLD:
                    g_idx = idx
                    break
            if g_idx is None:
                g_idx = steps_len - 1

            all_steps.append(steps)
            all_train_curves.append(train_acc)
            all_test_curves.append(test_acc)
            step_counts.append(steps_len)
            grok_indices.append(g_idx)
            grok_steps_actual.append(steps[g_idx])

        def _sample_curve(curve: list[float], pos: float) -> float | None:
            """Sample curve at fractional index pos using linear interpolation.

            Returns None if pos lies outside [0, len(curve) - 1].
            """
            if pos < 0.0 or pos > len(curve) - 1:
                return None
            i0 = int(math.floor(pos))
            i1 = int(math.ceil(pos))
            if i0 == i1:
                return curve[i0]
            alpha = pos - i0
            return (1.0 - alpha) * curve[i0] + alpha * curve[i1]

        if all_train_curves and grok_indices:
            # Target length is the longest run; we will horizontally shift each
            # curve so that its grok index lines up with the median grok index.
            target_len = max(step_counts)

            sorted_g = sorted(grok_indices)
            mid = len(sorted_g) // 2
            median_g_idx = sorted_g[mid]

            agg_train = [0.0] * target_len
            agg_test = [0.0] * target_len
            counts = [0] * target_len

            for tr_curve, te_curve, g_idx in zip(all_train_curves, all_test_curves, grok_indices):
                offset = g_idx - median_g_idx
                for t in range(target_len):
                    pos = t + offset
                    v_tr = _sample_curve(tr_curve, pos)
                    v_te = _sample_curve(te_curve, pos)
                    if v_tr is None or v_te is None:
                        continue
                    agg_train[t] += v_tr
                    agg_test[t] += v_te
                    counts[t] += 1

            for i in range(target_len):
                if counts[i] > 0:
                    agg_train[i] /= counts[i]
                    agg_test[i] /= counts[i]
                else:
                    agg_train[i] = 0.0
                    agg_test[i] = 0.0

            # After the avg test curve has grokked, keep it (and train) at 100.
            grok_idx_avg = None
            for i, v in enumerate(agg_test):
                if v >= GROK_THRESHOLD:
                    grok_idx_avg = i
                    break
            if grok_idx_avg is not None:
                for i in range(grok_idx_avg, target_len):
                    agg_train[i] = 100.0
                    agg_test[i] = 100.0

            n_entry['avg_data']['train_acc'] = agg_train
            n_entry['avg_data']['test_acc'] = agg_test
        else:
            n_entry['avg_data']['train_acc'] = []
            n_entry['avg_data']['test_acc'] = []

        # steps_to_grok = median of actual grok steps over grokked models.
        if grok_steps_actual:
            sorted_steps = sorted(grok_steps_actual)
            mid = len(sorted_steps) // 2
            if len(sorted_steps) % 2 == 1:
                median_step = float(sorted_steps[mid])
            else:
                median_step = 0.5 * (sorted_steps[mid - 1] + sorted_steps[mid])
            n_entry['avg_data']['steps_to_grok'] = median_step
        else:
            n_entry['avg_data']['steps_to_grok'] = 0

        # ------------------------------------------------------------------
        # Global aggregates grouped by weight decay and learning rate
        # ------------------------------------------------------------------
        # We scan all per-model JSON files under the aggregate data directory
        # and compute simple statistics per weight decay and per learning rate.

        weight_decay_stats: dict[str, dict[str, float | int]] = {}
        learning_rate_stats: dict[str, dict[str, float | int]] = {}

        if os.path.isdir(base_dir):
            for root, _dirs, files in os.walk(base_dir):
                for fname in files:
                    if not fname.endswith('.json'):
                        continue

                    full_path = os.path.join(root, fname)
                    # Skip the main aggregate data file itself.
                    if os.path.normpath(full_path) == os.path.normpath(self.data_path):
                        continue

                    rel = os.path.relpath(full_path, base_dir)
                    parts = rel.split(os.sep)
                    # Expect structure: weight_decay/learning_rate/N/model.json
                    if len(parts) < 4:
                        continue
                    wd_key, lr_key = parts[0], parts[1]

                    try:
                        with open(full_path, 'r', encoding='utf-8') as mf:
                            m_data = json.load(mf)
                    except (OSError, json.JSONDecodeError):
                        continue

                    train_time_val = float(m_data.get('train_time', 0.0))
                    grokked_flag = 1 if m_data.get('grokked', 0) else 0

                    # Update per-weight-decay stats (running average for train_time)
                    wd_entry = weight_decay_stats.setdefault(
                        wd_key,
                        {'num_of_sacrifices': 0, 'num_of_grokked': 0, 'avg_train_time': 0.0},
                    )
                    wd_count = int(wd_entry['num_of_sacrifices']) + 1
                    wd_prev_avg = float(wd_entry['avg_train_time'])
                    wd_new_avg = wd_prev_avg + (train_time_val - wd_prev_avg) / wd_count
                    wd_entry['num_of_sacrifices'] = wd_count
                    wd_entry['num_of_grokked'] = int(wd_entry['num_of_grokked']) + grokked_flag
                    wd_entry['avg_train_time'] = wd_new_avg

                    # Update per-learning-rate stats
                    lr_entry = learning_rate_stats.setdefault(
                        lr_key,
                        {'num_of_sacrifices': 0, 'num_of_grokked': 0, 'avg_train_time': 0.0},
                    )
                    lr_count = int(lr_entry['num_of_sacrifices']) + 1
                    lr_prev_avg = float(lr_entry['avg_train_time'])
                    lr_new_avg = lr_prev_avg + (train_time_val - lr_prev_avg) / lr_count
                    lr_entry['num_of_sacrifices'] = lr_count
                    lr_entry['num_of_grokked'] = int(lr_entry['num_of_grokked']) + grokked_flag
                    lr_entry['avg_train_time'] = lr_new_avg

        data['weight_decay'] = weight_decay_stats
        data['learning_rate'] = learning_rate_stats

        self._write_json(self.data_path, data)
        format_json_number_lists(self.data_path)

    def _initialize_data_file(self):
        """Ensure the aggregate data file exists, is structurally valid,
        and has zeroed entries for N=1 and N=2.

        If the file already exists and is non-empty, its JSON structure is
        validated. If it is missing or empty, a fresh EMPTY_SMALL_DATA
        structure is created instead.
        """

        if not os.path.exists(self.data_path):
            data = deepcopy(EMPTY_SMALL_DATA)
        else:
            with open(self.data_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                data = deepcopy(EMPTY_SMALL_DATA)
            else:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in data file: {self.data_path}") from e

                # Validate existing structure before using it.
                self._validate_data_file_schema(data)

        # Ensure base structure exists and pre-populate N=1 and N=2 with
        # zeroed entries so plots that expect them don't crash.
        if "N" not in data or not isinstance(data["N"], dict):
            data["N"] = {}

        for n_str in ("1", "2"):
            if n_str not in data["N"]:
                data["N"][n_str] = deepcopy(EMPTY_SMALL_DATA_ENTRY)

        self._write_json(self.data_path, data)
        format_json_number_lists(self.data_path)

    def _validate_data_file_schema(self, data: dict) -> None:
        """Basic structural validation for the aggregate JSON file.

        Raises ValueError if required keys or shapes are missing so that
        subtle schema corruption is caught at initialization time.
        """

        if not isinstance(data, dict):
            raise ValueError("data.json root must be a JSON object")

        for key in ("num_of_sacrifices", "num_of_grokked", "N"):
            if key not in data:
                raise ValueError(f"data.json missing required top-level key: {key}")

        if not isinstance(data["N"], dict):
            raise ValueError("data['N'] must be a JSON object mapping N to entries")

        for n_key, n_entry in data["N"].items():
            if not isinstance(n_entry, dict):
                raise ValueError(f"Entry for N={n_key} must be an object")

            for k in ("num_of_sacrifices", "num_of_grokked", "avg_data"):
                if k not in n_entry:
                    raise ValueError(f"Entry for N={n_key} missing key: {k}")

            if "avg_train_time" not in n_entry:
                raise ValueError(f"Entry for N={n_key} missing key: avg_train_time")

            avg_data = n_entry.get("avg_data")
            if not isinstance(avg_data, dict):
                raise ValueError(f"avg_data for N={n_key} must be an object")

            for k in ("train_acc", "test_acc", "steps_to_grok"):
                if k not in avg_data:
                    raise ValueError(f"avg_data for N={n_key} missing key: {k}")

    def _write_json(self, path: str, data: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    

# helper func for json formatting
def single_line_array(text, key):
    pattern = rf'("{key}": \[)([\s\S]*?)(\])'
    return re.sub(pattern, repl, text)

def repl(match):
    head, inner, tail = match.group(1), match.group(2), match.group(3)
    nums = ''.join(ch for ch in inner if ch not in ' \n\r\t')
    nums = nums.replace(',', ', ')
    return f"{head}{nums}{tail}"

def format_json_number_lists(path: str, keys: tuple[str, ...] = ("steps", "train_acc", "test_acc")) -> None:
    """Format a JSON file so that specified numeric lists are on a single line.

    The function reads the whole file as text, applies single-line formatting
    to each of the given keys using single_line_array, and writes the result
    back to the same file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    for key in keys:
        text = single_line_array(text, key)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)