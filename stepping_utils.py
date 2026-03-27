import json
import os
import math

from config import AGGREGATE_DATA_PATH, SMART_CONFIG, SMART_LISTS_NAMES, OUTPUT_DIR, MAX_N, MIN_N, N_STEP

# Canonical keys for the info dict, matching SMART_LISTS_NAMES order
_INFO_KEYS = ['d_model', 'n_heads', 'train_pct', 'weight_decay', 'learning_rate']


def get_resume_state():
    """Return state as (N, wave, [index_per_param_list]) from saved progress.

    State ordering (innermost → outermost):
        N  →  param_indices  →  wave
    """
    param_list_lengths = [len(lst) for lst in SMART_CONFIG[2]]
    n_values    = (MAX_N - MIN_N) // N_STEP + 1
    total_params = math.prod(param_list_lengths)
    num_of_waves = SMART_CONFIG[1]

    s = 0
    if os.path.exists(AGGREGATE_DATA_PATH):
        try:
            with open(AGGREGATE_DATA_PATH, "r", encoding="utf-8") as f:
                agg_data = json.load(f)
            s = int(agg_data.get("num_of_sacrifices", 0))
        except Exception:
            s = 0

    total_per_wave = n_values * total_params
    wave_0indexed  = s // total_per_wave
    if wave_0indexed >= num_of_waves:
        return None  # all done

    within_wave  = s % total_per_wave
    n_index      = within_wave % n_values
    N            = MIN_N + n_index * N_STEP
    param_linear = within_wave // n_values

    # Decompose param_linear into per-list indices (mixed-radix)
    param_indices = []
    remaining = param_linear
    for i in range(len(param_list_lengths)):
        right_side_product = math.prod(param_list_lengths[i + 1:])
        param_indices.append(remaining // right_side_product)
        remaining %= right_side_product

    wave = wave_0indexed + 1  # 1-indexed
    return (N, wave, param_indices)


def get_next_state(current_state):
    """Advance state by one step. Returns None when all combinations are exhausted.

    Order of advancement (innermost → outermost):
        N  →  param_indices  →  wave
    """
    (N_curr, wave_curr, param_indices) = current_state
    param_list_lengths = [len(lst) for lst in SMART_CONFIG[2]]

    # Advance N (innermost wheel)
    if N_curr + N_STEP <= MAX_N:
        return (N_curr + N_STEP, wave_curr, list(param_indices))

    # N wrapped — advance param_indices (odometer carry, right-to-left)
    N_next = MIN_N
    new_indices = list(param_indices)
    for i in reversed(range(len(new_indices))):
        if new_indices[i] + 1 < param_list_lengths[i]:
            new_indices[i] += 1
            return (N_next, wave_curr, new_indices)
        else:
            new_indices[i] = 0  # carry

    # All param indices wrapped — advance wave
    wave_next = wave_curr + 1
    if wave_next > SMART_CONFIG[1]:
        return None  # all done
    return (N_next, wave_next, new_indices)  # new_indices is all zeros


def get_param_values(state):
    """Resolve a state's index list into actual parameter values."""
    return [SMART_CONFIG[2][i][state[2][i]] for i in range(len(state[2]))]


def state_to_model_dir(state):
    """Build the info dict that model_trainer expects from a state tuple."""
    N = state[0]
    param_values = get_param_values(state)
    folder_parts = [str(v).replace('.', '_') for v in param_values] + [str(N)]
    model_dir = os.path.join(OUTPUT_DIR, *folder_parts)

    return model_dir

def state_to_param(state: tuple, param_name: str):
    """Get the value of a specific parameter from a state tuple."""
    if param_name == "N":
        return state[0]
    elif param_name == "wave_index":
        return state[1]
    else:
        if param_name in _INFO_KEYS:
            index = _INFO_KEYS.index(param_name)
            param_list = SMART_CONFIG[2][index]
            param_value = param_list[state[2][index]]
            return param_value
        else:
            raise ValueError(f"Unknown parameter name: {param_name}")

def get_first_state():
    return (MIN_N, 1, [0] * len(SMART_CONFIG[2]))


print("SMART_CONFIG: ", SMART_CONFIG[0])
state = get_resume_state()
print("resume state: ", state)
print("param values: ", get_param_values(state))