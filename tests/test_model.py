import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import PPO

DEVICE = 'cuda'
INPUT_DIMS = (50, 50, 3)

randInput = torch.randint(0, 3, (8, 3, 5, 50, 50), device=DEVICE)
model = PPO(INPUT_DIMS, 9)
model.to(DEVICE)

for deterministic in [False, True]:
    with torch.no_grad():
        policy, value = model(randInput, deterministic)
        print(policy, value)
