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

from sklearn.metrics import confusion_matrix
import numpy as np

from sympy import prime

# Change these paths to match current Python 3.13 installation folder
tcl_path: str = r'C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
tk_path: str = r'C:\Users\marti\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

os.environ['TCL_LIBRARY'] = tcl_path
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


# configuration
D_MODEL: int = 128       # Hidden dimension
N_HEADS: int = 4         # Number of attention heads
LEARNING_RATE: float = 0.005
WEIGHT_DECAY: float = 1.0  # High decay forces the model to find the "simple" circle rule
TRAIN_PCT: float = 0.4     # Hide 60% of the data from the model to test generalization
STEPS: int = 100       # Maximum training steps

DATA_FOLDER_PATH: str = os.path.join('output')
DATA_FILE_PATH: str = os.path.join(DATA_FOLDER_PATH, 'data.json')

plotting: bool = False

import json


def train_until_grok(
    N: int,
    learning_rate: float,
    weight_decay: float,
    train_pct: float,
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
    model_output_folder = os.path.join(DATA_FOLDER_PATH, os.path.join(str(N)))
    m = os.path.join(model_output_folder, os.path.join(str(learning_rate).replace('.', '_')))
    model_output_data_file = os.path.join(m, 'data.json')
    model_info = ModelInfo(N=N, data_path=DATA_FILE_PATH, output_dir_path=m)

    print(f"Training on {len(train_in)} samples. Testing on {len(test_in)} samples. Modulus N={N}.")
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
num_of_sacrifices: int = 5
max_N: int = 200
list_of_lr: list[float] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

for learning_rate in list_of_lr:
    LEARNING_RATE = learning_rate
    for N in range(1, max_N + 1):
        """        
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        num_done = data.get('num_of_sacrifices', 0)
        N = num_done + 3
        """
        train_until_grok(N, LEARNING_RATE, WEIGHT_DECAY, TRAIN_PCT)

print("All sacrifices complete.")