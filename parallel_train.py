"""Launch N independent worker processes that each train different models on the
same GPU.  Works because the models are tiny enough to coexist in GPU memory.

Usage:
    python parallel_train.py              # uses NUM_WORKERS workers (default 4)
    python parallel_train.py 6            # uses 6 workers
    python parallel_train.py 4 --max 90   # only train the first 90 models
"""

import multiprocessing as mp
import os
import sys
import io
import json
import time

# ---------------------------------------------------------------------------
# Parse CLI arguments BEFORE importing heavy modules so help is instant.
# ---------------------------------------------------------------------------
NUM_WORKERS = 40
MAX_MODELS = None  # None = train all

args = sys.argv[1:]
positional = [a for a in args if not a.startswith("--")]
if positional:
    NUM_WORKERS = int(positional[0])

if "--max" in args:
    idx = args.index("--max")
    MAX_MODELS = int(args[idx + 1])

# ---------------------------------------------------------------------------
# Build the full list of jobs as state tuples.
# ---------------------------------------------------------------------------
from stepping_utils import get_resume_state, get_next_state, get_param_values, state_to_param


def build_all_jobs():
    """Return a list of state tuples, one per model to train."""
    jobs = []
    state = get_resume_state()
    while state is not None:
        jobs.append(state)
        state = get_next_state(state)
    return jobs


# ---------------------------------------------------------------------------
# Worker function — runs in a child process.
# ---------------------------------------------------------------------------
def worker(worker_id: int, jobs: list[tuple], counter: mp.Value, total: int, lock: mp.Lock):
    """Train the assigned subset of models sequentially."""
    # Suppress all stdout from model_trainer / torch / config imports
    sys.stdout = io.StringIO()
    import torch
    import model_trainer
    sys.stdout = sys.__stdout__

    for state in jobs:
        t0 = time.perf_counter()
        grokked = model_trainer.train_until_grok(state, quiet=True, device=torch.device("cuda"))
        elapsed = time.perf_counter() - t0

        N = state[0]
        wd = state_to_param(state, 'weight_decay')
        lr = state_to_param(state, 'learning_rate')

        with lock:
            counter.value += 1
            done = counter.value
        status = "GROKKED" if grokked else "no grok"
        print(f"[{done}/{total}] N={N:>3d}  wd={wd:<6}  "
              f"lr={lr:<8}  {status}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Main — split work across workers and launch them.
# ---------------------------------------------------------------------------
def main():
    all_jobs = build_all_jobs()

    # Optionally cap the number of models (handy for testing).
    if MAX_MODELS is not None:
        all_jobs = all_jobs[:MAX_MODELS]

    print(f"Total models to train : {len(all_jobs)}")
    print(f"Parallel workers      : {NUM_WORKERS}")
    print()

    # Round-robin assignment so each worker gets a mix of hyperparams.
    worker_jobs: list[list[tuple]] = [[] for _ in range(NUM_WORKERS)]
    for i, job in enumerate(all_jobs):
        worker_jobs[i % NUM_WORKERS].append(job)

    t_start = time.perf_counter()

    # Shared counter so all workers can report global progress.
    counter = mp.Value("i", 0)
    lock = mp.Lock()
    total = len(all_jobs)

    processes: list[mp.Process] = []
    for wid in range(NUM_WORKERS):
        p = mp.Process(
            target=worker,
            args=(wid, worker_jobs[wid], counter, total, lock),
            name=f"worker-{wid}",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.perf_counter() - t_start
    print(f"\nAll {len(all_jobs)} models trained in {elapsed:.1f}s across {NUM_WORKERS} workers.")

    # Generate the website once at the end.
    import website_maker
    print("Regenerating HTML report...")
    website_maker.main()
    print("Done!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
