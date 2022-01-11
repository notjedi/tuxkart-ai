def test_env():

    import numpy as np
    from src.utils import make_env
    from matplotlib import pyplot as plt

    env = make_env(0)()
    env.reset()

    action = [1, 0, 0, 0, 1, 0, 0]
    for _ in range(100):
        # action = env.action_space.sample()
        image, reward, _, info = env.step(action)
        plt.imshow(image.astype(np.uint8))
        plt.pause(0.1)

    env.close()
    print("src/env.py test successful")


def test_model():

    import torch
    from src.model import Net

    OBS_DIM = (60, 40, 3)
    ACT_DIM = (2, 2, 3, 2, 2, 2, 2)
    DEVICE, BATCH_SIZE, NUM_FRAMES = 'cuda', 4, 5

    randInput = torch.randint(0, 10, (BATCH_SIZE, OBS_DIM[-1], NUM_FRAMES, *OBS_DIM[:-1]),
            device=DEVICE, dtype=torch.float32)
    model = Net(OBS_DIM, ACT_DIM, NUM_FRAMES)
    model.to(DEVICE)

    policy, value = model(randInput)
    print(policy.sample(), value, sep='\n')
    print("src/model.py test successful")


def test_ppo():

    from torch import optim
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.ppo import PPO
    from src.model import Net
    from src.env import STKEnv
    from src.utils import STK, make_env

    DEVICE, BUFFER_SIZE, NUM_FRAMES, NUM_ENVS, LR = 'cuda', 8, 5, 1, 1e-3
    env = SubprocVecEnv([make_env(id) for id in range(NUM_ENVS)], start_method='spawn')
    obs_shape, act_shape = env.observation_space.shape, env.action_space.nvec

    model = Net(obs_shape, act_shape, NUM_FRAMES)
    model.to(DEVICE)

    buf_args = { 'buffer_size': BUFFER_SIZE, 'batch_size': NUM_ENVS, 'obs_dim': obs_shape,
            'act_dim': act_shape, 'num_frames': NUM_FRAMES, 'gamma': PPO.GAMMA, 'lam': PPO.LAMBDA }
    optimizer = optim.Adam(model.parameters(), lr=LR)
    writer = SummaryWriter('/tmp/tensorboard')

    ppo = PPO(env, model, optimizer, writer, DEVICE, **buf_args)
    ppo.rollout()
    ppo.train()
    env.close()
    print("src/ppo.py test successful")


if __name__ == "__main__":
    test_env()
    test_model()
    test_ppo()
