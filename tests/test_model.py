import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import PPO

device = 'cpu'
INPUT_DIMS = (50, 50, 3)

randInput = torch.randint(0, 3, (16, 3, 5, 50, 50), device=device)
model = PPO(INPUT_DIMS, 9)
model.to(device)

policy, value = model(randInput)
policy, value = model.pi(randInput)
