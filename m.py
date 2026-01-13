import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cpu")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import time

import os
import sys

import data_gen_utils
from data_save_utils import ModelInfo, format_json_number_lists
import configparser

from sklearn.metrics import confusion_matrix
import numpy as np

from sympy import prime

secrets_path = os.path.join(os.path.dirname(__file__), 'secrets.ini')

# Create a template secrets.ini on first run if it doesn't exist
if not os.path.exists(secrets_path):
    with open(secrets_path, 'w', encoding='utf-8') as f:
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

_secrets = configparser.ConfigParser()
_secrets.read(secrets_path, encoding='utf-8')

# TCL/TK paths can be provided via secrets.ini and are not hard-coded here
if _secrets.has_section('paths'):
    tcl_path = _secrets.get('paths', 'TCL_PATH', fallback=None)
    tk_path = _secrets.get('paths', 'TK_PATH', fallback=None)
else:
    tcl_path = None
    tk_path = None

if tcl_path:
    os.environ['TCL_LIBRARY'] = tcl_path
if tk_path:
    os.environ['TK_LIBRARY'] = tk_path


# Model
# no annotations cuz idk how this works in depth
class NanoMath(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x.long()) + self.pos_embed
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])

_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), 'config.ini'), encoding='utf-8')

D_MODEL: int = _config.getint('model', 'D_MODEL', fallback=128)
N_HEADS: int = _config.getint('model', 'N_HEADS', fallback=4)
WEIGHT_DECAY: float = _config.getfloat('model', 'WEIGHT_DECAY', fallback=1.0)
TRAIN_PCT: float = _config.getfloat('model', 'TRAIN_PCT', fallback=0.4)
STEPS: int = _config.getint('model', 'STEPS', fallback=10000)

NUM_OF_WAVES: int = _config.getint('training', 'NUM_OF_WAVES', fallback=3)
MIN_N: int = _config.getint('training', 'MIN_N', fallback=20)
MAX_N: int = _config.getint('training', 'MAX_N', fallback=200)
_wd_raw = _config.get('training', 'WEIGHT_DECAYS', fallback=str(WEIGHT_DECAY))
WEIGHT_DECAYS: list[float] = [float(x.strip()) for x in _wd_raw.split(',') if x.strip()]
_lr_raw = _config.get('training', 'LEARNING_RATES', fallback='0.005')
LEARNING_RATES: list[float] = [float(x.strip()) for x in _lr_raw.split(',') if x.strip()]

if _config.has_section('paths'):
    _output_dir = _config.get('paths', 'OUTPUT_DIR', fallback='output')
else:
    _output_dir = 'output'
OUTPUT_DIR: str = _output_dir

# Allow secrets.ini to override OUTPUT_DIR without committing machine-specific paths
_secrets_output_dir = None
if _secrets.has_section('paths'):
    _secrets_output_dir = _secrets.get('paths', 'OUTPUT_DIR', fallback=None)
if _secrets_output_dir:
    OUTPUT_DIR = _secrets_output_dir
PLOTTING: bool = _config.getboolean('plot', 'PLOTTING', fallback=False)

# Aggregate JSON storing averaged stats across runs
AGGREGATE_DATA_PATH: str = os.path.join(OUTPUT_DIR, 'data.json')

plotting: bool = PLOTTING

import json


def train_until_grok(
    N: int,
    learning_rate: float,
    weight_decay: float,
    train_pct: float,
    info: dict,
    model_dir: str,
) -> bool:
    """Train a single model for modulus N until it groks or hits max steps.

    Returns True if grokked, False otherwise.
    """

    if N < 2:
        return False

    # generate data
    (train_in, train_lab), (test_in, test_lab) = data_gen_utils.generate_data(N, train_pct)

    # setup model, optimizer, criterion
    model = NanoMath(N + 1, D_MODEL, N_HEADS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # setup Plots
    if plotting:
        plt.ion()
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # setup ModelInfo to save data
    # model_dir is computed by the caller (main loop) and should
    # follow: OUTPUT_DIR / WEIGHT_DECAY / LEARNING_RATE / N /
    print(f"Model directory: {model_dir}")
    print(f"Aggregate data file: {AGGREGATE_DATA_PATH}")

    model_info = ModelInfo(N=N, data_path=AGGREGATE_DATA_PATH, output_dir_path=model_dir)

    print(f"Training on {len(train_in)} samples. Testing on {len(test_in)} samples.")
    for key, value in info.items():
        print(f"  {key}: {value}")
    start_time = time.perf_counter()
    grokked = False

    # training loop
    for step in range(STEPS + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_in)
        loss = criterion(outputs, train_lab)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (torch.argmax(model(train_in), dim=1) == train_lab).float().mean() * 100
                test_acc = (torch.argmax(model(test_in), dim=1) == test_lab).float().mean() * 100

                # add step data to model info
                model_info.add_data_point(step, train_acc.item(), test_acc.item())

                # live plot
                if plotting:
                    ax1.clear()
                    (steps_list, train_accs, test_accs) = model_info.get_data_lists()
                    ax1.plot(steps_list, train_accs, label='Train Acc')
                    ax1.plot(steps_list, test_accs, label='Test Acc (Grokking)')
                    ax1.set_title(f"Accuracy Over Time (Step {step})")
                    ax1.legend()
                    plt.draw()
                    plt.pause(0.1)

                # print progress
                print(f"Step {step:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%")

                if test_acc > 97:
                    print("\nMODEL HAS GROKKED!")
                    if plotting:
                        plt.ioff()
                        plt.show()
                    grokked = True
                    break

    # measure training time for this model
    train_time = time.perf_counter() - start_time
    model_info.model_data['train_time'] = float(train_time)

    # save model info
    model_info.save_model_info(grokked)
    model_info.save_data_info(grokked, train_time)

    return grokked


# start main loop: choose N for each sacrifice and train
no_of_all_sacrifices = NUM_OF_WAVES * len(WEIGHT_DECAYS) * len(LEARNING_RATES) * (MAX_N - MIN_N + 1)
for wave_index in range(NUM_OF_WAVES):
    for weight_decay in WEIGHT_DECAYS:
        for learning_rate in LEARNING_RATES:
            for N in range(MIN_N, MAX_N + 1):
                """        
                with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                num_done = data.get('num_of_sacrifices', 0)
                N = num_done + 3
                """
                model_no = (
                    wave_index * len(WEIGHT_DECAYS) * len(LEARNING_RATES) * (MAX_N - MIN_N + 1)
                    + WEIGHT_DECAYS.index(weight_decay) * len(LEARNING_RATES) * (MAX_N - MIN_N + 1)
                    + LEARNING_RATES.index(learning_rate) * (MAX_N - MIN_N + 1)
                    + (N - MIN_N)
                    + 1
                )
                # Build per-model directory once and pass it down.
                wd_folder = str(weight_decay).replace('.', '_')
                lr_folder = str(learning_rate).replace('.', '_')
                n_folder = str(N)
                model_dir = os.path.join(OUTPUT_DIR, wd_folder, lr_folder, n_folder)

                info = {
                    'wave_index': wave_index,
                    'weight_decay': weight_decay,
                    'learning_rate': learning_rate,
                    'N': N,
                    'total_progress': f"{model_no/no_of_all_sacrifices:.2%}",
                }
                train_until_grok(N, learning_rate, weight_decay, TRAIN_PCT, info, model_dir)

print("All sacrifices complete.")