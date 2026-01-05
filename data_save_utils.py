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
    path: str
    small_path: str
    N: int
    data: dict
    small_data: dict

    def __init__(self, path: str, small_path: str, N: int):
        self.path = os.path.join(os.path.dirname(__file__), path)
        self.small_path = os.path.join(os.path.dirname(__file__), small_path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.N = N
        self.data = EMPTY_MODEL_ENTRY.copy()
        self.small_data = EMPTY_SMALL_DATA.copy()
        
        # initialize data file if not exists
        self._initialize_data_files()

        # add N entry if not exists
        self._add_N_entry()

    def add_data_point(self, step: int, train_acc: float, test_acc: float):
        self.data['steps'].append(step)
        self.data['train_acc'].append(train_acc)
        self.data['test_acc'].append(test_acc)

        self.data['num_of_steps'] += 1

    def save_model_info(self, grokked: bool):
        self.data['grokked'] = 1 if grokked else 0

        # load existing data
        with open(self.path, 'r') as f:
            all_data = json.load(f)

        # append model data to N entry, including small_model_data
        n_str = str(self.N)
        model_id = all_data[n_str]['num_of_models']
        base_dir = os.path.dirname(self.path)
        n_dir = os.path.join(base_dir, n_str)
        os.makedirs(n_dir, exist_ok=True)


        # Save model data to its own file 
        model_file_path = os.path.join(n_dir, f"{model_id}.txt")
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)
        format_json_number_lists(model_file_path)

        # Add to models list in data.json
        all_data[n_str]['num_of_models'] += 1
        if grokked:
            all_data[n_str]['num_of_grokked'] += 1
        self._write_json(self.path, all_data)
        # Ensure pretty formatting for number lists in data.json
        format_json_number_lists(self.path)


    def save_small_info(self, grokked: bool):
        # load existing small data
        with open(self.small_path, 'r') as f:
            small_data = json.load(f)

        # update global small data
        small_data['num_of_sacrifices'] += 1
        if grokked:
            small_data['num_of_grokked'] += 1

        # initialize N entry if not exists
        n_str = str(self.N)
        if n_str not in small_data['N']:
            small_data['N'][n_str] = EMPTY_SMALL_DATA_ENTRY.copy()

        # update N-specific small data
        n_entry = small_data['N'][n_str]
        n_entry['num_of_sacrifices'] += 1
        if grokked: 
            n_entry['num_of_grokked'] += 1

        # accumulate average train/test accuracy per step
        num_steps = self.data['num_of_steps']
        avg_train_acc = n_entry['avg_data']['train_acc']
        avg_test_acc = n_entry['avg_data']['test_acc']

        # Extend avg lists if needed
        while len(avg_train_acc) < num_steps:
            avg_train_acc.append(0.0)
            avg_test_acc.append(0.0)

        for i in range(num_steps):
            avg_train_acc[i] += (self.data['train_acc'][i] - avg_train_acc[i]) / n_entry['num_of_sacrifices']
            avg_test_acc[i] += (self.data['test_acc'][i] - avg_test_acc[i]) / n_entry['num_of_sacrifices']

        self._write_json(self.small_path, small_data)
        format_json_number_lists(self.small_path)

    def _initialize_data_files(self):
        # Ensure output files exist with valid JSON

        # small data
        if not os.path.exists(self.small_path):
            with open(self.small_path, "w") as f:
                json.dump(EMPTY_SMALL_DATA, f)
        else:
            with open(self.small_path, "r+") as f:
                content = f.read().strip()
                if not content:
                    f.seek(0)
                    json.dump(EMPTY_SMALL_DATA, f)
                    f.truncate()

        # main data
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)
        else:
            with open(self.path, "r+") as f:
                content = f.read().strip()
                if not content:
                    f.seek(0)
                    json.dump({}, f)
                    f.truncate()

    def _add_N_entry(self):
        # At this point _initialize_data_files has guaranteed valid JSON
        with open(self.path, 'r') as f:
            all_data = json.load(f)

        if str(self.N) not in all_data:
            all_data[str(self.N)] = EMPTY_N_ENTRY.copy()
            self._write_json(self.path, all_data)

    def get_data_lists(self) -> tuple[list[int], list[float], list[float]]:
        return self.data['steps'], self.data['train_acc'], self.data['test_acc']
    
    def _write_json(self, path: str, data: dict) -> None:
        """Write JSON to self.path with the desired indentation and style."""
        # Pretty-print with indent=4 and ensure lists stay multi-line
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