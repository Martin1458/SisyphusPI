import json
import os
import re

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
    }
}

class ModelInfo:
    data_path: str
    N: int

    def __init__(self, data_path: str, N: int):
        self.data_path = os.path.join(os.path.dirname(__file__), data_path)
        self.N = N
        self.model_data = EMPTY_MODEL_ENTRY.copy()
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
        n_dir = os.path.join(base_dir, str(self.N))
        os.makedirs(n_dir, exist_ok=True)
        # Use number of files in n_dir as model_id
        model_id = len([name for name in os.listdir(n_dir) if name.endswith('.json')])
        model_file_path = os.path.join(n_dir, f"{model_id}.json")
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_data, f, indent=4)
        format_json_number_lists(model_file_path)

    def save_data_info(self, grokked: bool):
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        data['num_of_sacrifices'] += 1
        if grokked:
            data['num_of_grokked'] += 1

        n_str = str(self.N)
        if n_str not in data['N']:
            data['N'][n_str] = EMPTY_SMALL_DATA_ENTRY.copy()

        n_entry = data['N'][n_str]
        n_entry['num_of_sacrifices'] += 1
        if grokked:
            n_entry['num_of_grokked'] += 1

        num_steps = self.model_data['num_of_steps']
        avg_train_acc = n_entry['avg_data']['train_acc']
        avg_test_acc = n_entry['avg_data']['test_acc']

        while len(avg_train_acc) < num_steps:
            avg_train_acc.append(100.0)
            avg_test_acc.append(100.0)

        for i in range(num_steps):
            avg_train_acc[i] += (self.model_data['train_acc'][i] - avg_train_acc[i]) / n_entry['num_of_sacrifices']
            avg_test_acc[i] += (self.model_data['test_acc'][i] - avg_test_acc[i]) / n_entry['num_of_sacrifices']

        if grokked:
            n_entry['avg_data']['steps_to_grok'] += (self.model_data['num_of_steps'] - n_entry['avg_data']['steps_to_grok']) / n_entry['num_of_grokked']

        self._write_json(self.data_path, data)
        format_json_number_lists(self.data_path)

    def _initialize_data_file(self):
        if not os.path.exists(self.data_path):
            with open(self.data_path, "w") as f:
                json.dump(EMPTY_SMALL_DATA, f)
        else:
            with open(self.data_path, "r+") as f:
                content = f.read().strip()
                if not content:
                    f.seek(0)
                    json.dump(EMPTY_SMALL_DATA, f)
                    f.truncate()
        format_json_number_lists(self.data_path)

    def _write_json(self, path: str, data: dict) -> None:
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