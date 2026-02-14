import os, json
import model_trainer, website_maker
from config import TRAIN_PCT, NUM_OF_WAVES, MIN_N, MAX_N, N_STEP, WEIGHT_DECAYS, LEARNING_RATES, OUTPUT_DIR, AGGREGATE_DATA_PATH

MAX_MODELS = 90
model_no = 0
done = False
for wave_index in range(NUM_OF_WAVES):
    if done: break
    for weight_decay in WEIGHT_DECAYS:
        if done: break
        for learning_rate in LEARNING_RATES:
            if done: break
            for N in range(MIN_N, MAX_N + 1, N_STEP):
                if model_no >= MAX_MODELS:
                    done = True
                    break
                model_no += 1
                wd_folder = str(weight_decay).replace('.', '_')
                lr_folder = str(learning_rate).replace('.', '_')
                n_folder = str(N)
                model_dir = os.path.join(OUTPUT_DIR, wd_folder, lr_folder, n_folder)
                info = {
                    'wave_index': wave_index,
                    'weight_decay': weight_decay,
                    'learning_rate': learning_rate,
                    'N': N,
                    'total_progress': f'{model_no}/{MAX_MODELS}',
                    'train_pct': TRAIN_PCT,
                    'model_no': model_no,
                    'model_dir': model_dir,
                }
                model_trainer.train_until_grok(info)

print(f'\nTrained {model_no} models. Generating website...')
website_maker.main()
print('Done!')
