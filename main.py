import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- CONFIG ---
N = 59  # Modulus
D_MODEL = 128 # Slightly larger to help grokking happen faster
N_HEADS = 4
BATCH_SIZE = 512 # Larger batches help stabilize the 'click'
LEARNING_RATE = 0.001
TRAIN_PCT = 0.5  # Only show the model 50% of the possible sums

# --- DATA SPLIT ---
def generate_split_data(n, train_pct):
    all_pairs = [(a, b) for a in range(n) for b in range(n)]
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * train_pct)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    def to_tensor(pairs):
        inputs = torch.tensor([[a, b, n] for a, b in pairs])
        labels = torch.tensor([(a + b) % n for a, b in pairs])
        return inputs, labels

    return to_tensor(train_pairs), to_tensor(test_pairs)

# --- MODEL (Same NanoMath as before) ---
class NanoMath(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])

# --- SETUP ---
(train_in, train_lab), (test_in, test_lab) = generate_split_data(N, TRAIN_PCT)
model = NanoMath(N + 1, D_MODEL, N_HEADS)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0) # High weight decay is key for grokking!
criterion = nn.CrossEntropyLoss()

print(f"Training on {len(train_in)} samples. Testing on {len(test_in)} samples.")

# --- TRAINING LOOP ---
history = []

for step in range(5001):
    model.train()
    optimizer.zero_grad()
    
    # We use full-batch training here since the dataset is small enough for Pi RAM
    outputs = model(train_in)
    loss = criterion(outputs, train_lab)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = (torch.argmax(model(train_in), dim=1) == train_lab).float().mean() * 100
            test_acc = (torch.argmax(model(test_in), dim=1) == test_lab).float().mean() * 100
            
            history.append((step, train_acc.item(), test_acc.item()))
            print(f"Step {step:4d} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}%")
            
            if test_acc > 99:
                print("!!! GROKKED !!!")
                break


