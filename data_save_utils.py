import json
import os
import re

EMPTY_N_ENTRY = {
    'models': [],
    'num_of_models': 0,
    'num_of_grokked': 0
}

EMPTY_MODEL_ENTRY = {
    'steps': [],
    'num_of_steps': 0,
    'train_acc': [],
    'test_acc': [],
    'grokked': 0
}

class ModelInfo:
    path: str
    N: int
    data: dict

    def __init__(self, path: str, N: int):
        self.path = os.path.join(os.path.dirname(__file__), path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.N = N
        self.data = EMPTY_MODEL_ENTRY.copy()
        
        # initialize data file if not exists
        #self._initialize_data_file()

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

        # append model data to N entry
        n_str = str(self.N)
        all_data[n_str]['models'].append(self.data)
        #   update counts
        all_data[n_str]['num_of_models'] += 1
        if grokked:
            all_data[n_str]['num_of_grokked'] += 1

        # save back to file using the shared JSON writer
        self._write_json(all_data)

    def _initialize_data_file(self):
        # Ensure the file exists and contains valid JSON; if it's missing,
        # empty, or corrupted, reset it to an empty dict.
        if not os.path.exists(self.path):
            self._write_json({})
            return

        try:
            with open(self.path, 'r') as f:
                content = f.read().strip()

            # Treat empty content as invalid and reset
            if not content:
                raise json.JSONDecodeError("empty file", content, 0)

            # Validate JSON; we don't care about the value, only that it parses
            json.loads(content)
        except json.JSONDecodeError:
            self._write_json({})

    def _add_N_entry(self):
        # At this point _initialize_data_file has guaranteed valid JSON
        with open(self.path, 'r') as f:
            all_data = json.load(f)

        if str(self.N) not in all_data:
            all_data[str(self.N)] = EMPTY_N_ENTRY.copy()
            self._write_json(all_data)

    def get_data_lists(self) -> tuple[list[int], list[float], list[float]]:
        return self.data['steps'], self.data['train_acc'], self.data['test_acc']
    
    def _write_json(self, data: dict) -> None:
        """Write JSON to self.path with the desired indentation and style."""
        # Pretty-print with indent=4 and ensure lists stay multi-line
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=4)
    

# helper func for json formatting
def single_line_array(text, key):
    pattern = rf'("{key}": \[)([\s\S]*?)(\])'

    def repl(match):
        head, inner, tail = match.group(1), match.group(2), match.group(3)
        nums = ''.join(ch for ch in inner if ch not in ' \n\r\t')
        nums = nums.replace(',', ', ')
        return f"{head}{nums}{tail}"

    return re.sub(pattern, repl, text)


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