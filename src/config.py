import torch

DEBUG = 0
GAMMA = 0.9
NUM_ENVS = 1
LAMBDA = 0.95
BATCH_SIZE = 1
NUM_FRAMES = 5
BUFFER_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
