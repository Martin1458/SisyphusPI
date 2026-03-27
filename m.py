import json as _json
import math
import os
import time
from collections import deque
from datetime import timedelta
import torch
device = torch.device("cpu")
print(f"Using device: {device}")
#torch.set_num_threads(2)
#torch.set_num_interop_threads(2)

import website_maker
from model_trainer import train_until_grok
from stepping_utils import get_resume_state, get_next_state, get_param_values
from config import SMART_CONFIG, MIN_N, MAX_N, N_STEP

# Calculate total number of models
n_values = (MAX_N - MIN_N) // N_STEP + 1
param_combinations = math.prod(len(lst) for lst in SMART_CONFIG[2])
total_models = n_values * param_combinations * SMART_CONFIG[1]

current_state = get_resume_state()

# Start model_no from saved progress so the counter reflects global position
_model_no_start = 0
if os.path.exists("output/data.json"):
    try:
        with open("output/data.json", "r") as _f:
            _model_no_start = int(_json.load(_f).get("num_of_sacrifices", 0))
    except Exception:
        pass
model_no = _model_no_start
recent_times = deque(maxlen=10)

while current_state is not None:
    (N, wave, param_indices) = current_state
    model_no += 1
    pct = model_no / total_models * 100
    remaining = total_models - model_no
    if recent_times:
        avg_time = sum(recent_times) / len(recent_times)
        eta = timedelta(seconds=int(avg_time * remaining))
        print(f"\n[{model_no}/{total_models}] {pct:.1f}% — ETA: {eta} (avg {avg_time:.1f}s/model) — N={N}, params={get_param_values(current_state)}")
    else:
        print(f"\n[{model_no}/{total_models}] {pct:.1f}% — N={N}, params={get_param_values(current_state)}")

    t0 = time.time()
    train_until_grok(current_state, quiet=False, device=device)
    recent_times.append(time.time() - t0)

    # Periodically regenerate the global HTML report
    if model_no % 10== 0:
        print("Regenerating HTML report...")
        website_maker.main()
        website_maker.git_auto_commit_website(model_no, -1)

    current_state = get_next_state(current_state)

print("All sacrifices complete.")