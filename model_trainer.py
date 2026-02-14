
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

import data_gen_utils
from data_save_utils import ModelInfo
from data_plot_utils import (start_live_accuracy_plot, update_live_accuracy_plot, finalize_live_plot)
from stepping_utils import _INFO_KEYS, state_to_model_dir, state_to_param
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

def train_until_grok(current_state: tuple, quiet: bool, device: torch.device) -> bool:
    """Train a single model for modulus N until it groks or hits max steps.

    Args:
        current_state: Tuple representing the current state of the model parameters.
        quiet: If True, suppress all per-step and per-model print output.

    Returns True if grokked, False otherwise.
    """
    # device
    print(f"Using device: {device}")

    N = current_state[0]
    learning_rate = state_to_param(current_state, 'learning_rate')
    weight_decay = state_to_param(current_state, 'weight_decay')
    train_pct = state_to_param(current_state, 'train_pct')
    model_dir = state_to_model_dir(current_state)
    d_model = state_to_param(current_state, 'd_model')
    n_heads = state_to_param(current_state, 'n_heads')

    if N < 2:
        return False

    # generate data
    (train_in, train_lab), (test_in, test_lab) = data_gen_utils.generate_data(N, train_pct)
    train_in, train_lab = train_in.to(device), train_lab.to(device)
    test_in, test_lab = test_in.to(device), test_lab.to(device)

    # setup model, optimizer, criterion
    model = NanoMath(N + 1, d_model, n_heads).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # setup Plots
    if PLOTTING:
        fig, ax1 = start_live_accuracy_plot()

    # setup ModelInfo to save data
    if not quiet:
        print(f"Model directory: {model_dir}")
        print(f"Aggregate data file: {AGGREGATE_DATA_PATH}")

    model_info = ModelInfo(N=N, data_path=AGGREGATE_DATA_PATH, output_dir_path=model_dir)

    if not quiet:
        print(f"Training on {len(train_in)} samples. Testing on {len(test_in)} samples.")
        #for key, value in info.items():
        #    print(f"  {key}: {value}")
    start_time = time.perf_counter()
    grokked = False

    # Evaluate every EVAL_INTERVAL steps instead of every 10.
    # For 5000 steps, every 100 gives 50 eval passes instead of 500.
    EVAL_INTERVAL = 50

    # training loop
    for step in range(STEPS + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_in)
        loss = criterion(outputs, train_lab)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                train_preds = torch.argmax(model(train_in), dim=1)
                test_preds = torch.argmax(model(test_in), dim=1)
                train_acc = (train_preds == train_lab).float().mean() * 100
                test_acc = (test_preds == test_lab).float().mean() * 100

                # add step data to model info
                model_info.add_data_point(step, train_acc.item(), test_acc.item())

                # live plot
                if PLOTTING:
                    update_live_accuracy_plot(ax1, model_info, step, pause=0.1)

                # print progress
                if not quiet:
                    print(f"Step {step:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%")

                if test_acc > 97:
                    if not quiet:
                        print("\nMODEL HAS GROKKED!")
                    if PLOTTING:
                        finalize_live_plot(block=True)
                    grokked = True
                    break

    # measure training time for this model
    train_time = time.perf_counter() - start_time
    model_info.model_data['train_time'] = float(train_time)

    # save model info
    model_info.save_model_info(grokked)
    model_info.save_data_info(grokked, train_time)

    return grokked