import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cpu")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

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
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 1.0  # High decay forces the model to find the "simple" circle rule
TRAIN_PCT: float = 0.4     # Hide 60% of the data from the model to test generalization
STEPS: int = 10000       # Maximum training steps

# start main loop
plotting: bool = False
num_of_sacrifices: int = 15

for _ in range(num_of_sacrifices):
    # configure per-sacrifice
    grooked: bool = False
    #N: int = prime(random.randint(10, 25)) # The modulus
    N = 67
    # generate data
    (train_in, train_lab), (test_in, test_lab) = data_gen_utils.generate_data(N, TRAIN_PCT)

    # setup model, optimizer, criterion
    model = NanoMath(N + 1, D_MODEL, N_HEADS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # setup Plots
    if plotting:
        plt.ion() 
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # setup ModelInfo to save data
    model_info = ModelInfo(data_path=r'output\data.json', N=N)


    print(f"Training on {len(train_in)} samples. Testing on {len(test_in)} samples. Modulus N={N}.")

    # training loop
    for step in range(STEPS + 1):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_in)
        loss = criterion(outputs, train_lab)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (torch.argmax(model(train_in), dim=1) == train_lab).float().mean() * 100
                test_acc = (torch.argmax(model(test_in), dim=1) == test_lab).float().mean() * 100
                
                # add step data to model info
                model_info.add_data_point(step, train_acc.item(), test_acc.item())

                # live plot
                if plotting:
                    ax1.clear()
                    (steps, train_accs, test_accs) = model_info.get_data_lists()
                    ax1.plot(steps, train_accs, label='Train Acc')
                    ax1.plot(steps, test_accs, label='Test Acc (Grokking)')
                    ax1.set_title(f"Accuracy Over Time (Step {step})")
                    ax1.legend()
                    plt.draw()
                    plt.pause(0.1)
                
                # print progress
                print(f"Step {step:4d} | Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%")
                
                if test_acc > 99.2:
                    print("\nMODEL HAS GROKKED!")
                    if plotting:
                        plt.ioff()
                        plt.show()
                    grooked = True
                    break

    # save model info 
    model_info.save_model_info(grooked)
    model_info.save_data_info(grooked)

print("All sacrifices complete.")