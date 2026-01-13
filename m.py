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

import website_maker
import subprocess

from config import (
    D_MODEL,
    N_HEADS,
    WEIGHT_DECAY,
    TRAIN_PCT,
    STEPS,
    NUM_OF_WAVES,
    MIN_N,
    MAX_N,
    N_STEP,
    WEIGHT_DECAYS,
    LEARNING_RATES,
    OUTPUT_DIR,
    PLOTTING,
    AGGREGATE_DATA_PATH,
)


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

import json


def git_auto_commit_website(model_no: int, total_models: int) -> None:
    """Git add/commit/push inside the SisyphusPI-website repo.

    Best-effort: errors (no changes, no remote, no repo) are printed
    but do not stop training.
    """

    repo_root = os.path.dirname(os.path.abspath(__file__))
    website_repo = os.path.join(repo_root, "SisyphusPI-website")
    msg = f"auto: updated site after {model_no}/{total_models} models"

    if not os.path.isdir(website_repo):
        print(f"Website repo not found at {website_repo}, skipping git push.")
        return

    try:
        subprocess.run(["git", "add", "."], cwd=website_repo, check=False)
        subprocess.run(["git", "commit", "-m", msg], cwd=website_repo, check=False)
        subprocess.run(["git", "push"], cwd=website_repo, check=False)
    except Exception as e:
        print(f"Git auto-commit (website) failed: {e}")


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
    if PLOTTING:
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
                if PLOTTING:
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
                    if PLOTTING:
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


# Determine resume point for this project based on aggregate data
start_from = 0
if os.path.exists(AGGREGATE_DATA_PATH):
    try:
        with open(AGGREGATE_DATA_PATH, "r", encoding="utf-8") as f:
            agg_data = json.load(f)
        start_from = int(agg_data.get("num_of_sacrifices", 0))
    except Exception:
        start_from = 0

# start main loop: choose N for each sacrifice and train
num_N_values = ((MAX_N - MIN_N) // N_STEP) + 1
no_of_all_sacrifices = NUM_OF_WAVES * len(WEIGHT_DECAYS) * len(LEARNING_RATES) * num_N_values

model_no = 0
for wave_index in range(NUM_OF_WAVES):
    for weight_decay in WEIGHT_DECAYS:
        for learning_rate in LEARNING_RATES:
            for N in range(MIN_N, MAX_N + 1, N_STEP):
                model_no += 1

                # Skip models that were already completed in this project
                if model_no <= start_from:
                    continue

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
                    'model_no': model_no,
                }
                train_until_grok(N, learning_rate, weight_decay, TRAIN_PCT, info, model_dir)

                # Periodically regenerate the global HTML report using
                # the website_maker main entrypoint.
                if model_no % 10 == 0:
                    print("Regenerating HTML report...")
                    website_maker.main()
                    git_auto_commit_website(model_no, no_of_all_sacrifices)

print("All sacrifices complete.")