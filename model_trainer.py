import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cpu")
import time

import data_gen_utils
from data_save_utils import ModelInfo
from data_plot_utils import (start_live_accuracy_plot, update_live_accuracy_plot, finalize_live_plot)

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

def train_until_grok(info: dict) -> bool:
    """Train a single model for modulus N until it groks or hits max steps.

    Returns True if grokked, False otherwise.
    """

    N = info['N']
    learning_rate = info['learning_rate']
    weight_decay = info['weight_decay']
    train_pct = info['train_pct']
    model_dir = info['model_dir']

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
        fig, ax1 = start_live_accuracy_plot()

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
                    update_live_accuracy_plot(ax1, model_info, step, pause=0.1)

                # print progress
                print(f"Step {step:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%")

                if test_acc > 97:
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