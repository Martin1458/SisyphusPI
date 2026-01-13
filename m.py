import torch
device = torch.device("cpu")
import os

import website_maker
import model_trainer

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



import json


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
                    'train_pct': TRAIN_PCT,
                    'model_no': model_no,
                    'model_dir': model_dir,
                }
                model_trainer.train_until_grok(info)

                # Periodically regenerate the global HTML report using
                # the website_maker main entrypoint.
                if model_no % 10 == 0:
                    print("Regenerating HTML report...")
                    website_maker.main()
                    website_maker.git_auto_commit_website(model_no, no_of_all_sacrifices)

print("All sacrifices complete.")