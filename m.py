import torch
device = torch.device("cpu")
print(f"Using device: {device}")
#torch.set_num_threads(2)
#torch.set_num_interop_threads(2)

import website_maker
from model_trainer import train_until_grok
from stepping_utils import get_resume_state, get_next_state

current_state = get_resume_state()
model_no = 0

while current_state is not None:
    (N, wave, param_indices) = current_state
    model_no += 1

    train_until_grok(current_state, quiet=False, device=device)

    # Periodically regenerate the global HTML report
    if model_no % 22545 == 0:
        print("Regenerating HTML report...")
        website_maker.main()
        website_maker.git_auto_commit_website(model_no, -1)

    current_state = get_next_state(current_state)

print("All sacrifices complete.")