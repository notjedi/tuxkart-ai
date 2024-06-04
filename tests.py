def test_env():

    import numpy as np
    from matplotlib import pyplot as plt

    from src.utils import make_env

    env = make_env(0)()
    env.reset()

    action = [1, 0, 1, 0, 0, 0]
    for _ in range(10):
        # action = env.action_space.sample()
        image, reward, _, info = env.step(action)
        plt.imshow(np.array(image).astype(np.uint8), cmap="gray")
        plt.pause(0.1)

    env.close()
    plt.close()
    print("src/env.py test successful")


def test_model():

    import torch
    from torchinfo import summary

    from src.model import Net

    ACT_DIM = (2, 2, 3, 2, 2, 2)
    DEVICE, BATCH_SIZE, ZDIM, NUM_FRAMES = torch.device("cuda"), 8, 256, 5

    model = Net(ZDIM, ACT_DIM, BATCH_SIZE)
    model.to(DEVICE)
    rand_input = torch.rand(
        NUM_FRAMES, BATCH_SIZE, ZDIM, device=DEVICE, dtype=torch.float32
    )

    # summary(model, input_data=rand_input, verbose=1) # remove MultiCategorical while using summary
    policy, value = model(rand_input)
    print(policy.sample(), value, sep="\n")
    print("src/model.py test successful")


def test_ppo():

    from stable_baselines3.common.vec_env import SubprocVecEnv
    from torch import optim
    from torch.utils.tensorboard import SummaryWriter

    from src.model import Net
    from src.ppo import PPO
    from src.utils import Logger, make_env
    from src.vae.model import ConvVAE, Decoder, Encoder

    DEVICE, BUFFER_SIZE, NUM_FRAMES, NUM_ENVS, LR, ZDIM = (
        "cuda",
        8,
        5,
        1,
        1e-3,
        256,
    )
    env = SubprocVecEnv([make_env(id) for id in range(NUM_ENVS)], start_method="spawn")
    obs_shape, act_shape = env.observation_space.shape, env.action_space.nvec

    vae = ConvVAE(obs_shape, Encoder, Decoder, ZDIM)
    vae.to(DEVICE)
    lstm = Net(ZDIM + 4, act_shape, NUM_ENVS)
    lstm.reset(BUFFER_SIZE, NUM_ENVS)
    lstm.to(DEVICE)

    buf_args = {
        "buf_size": BUFFER_SIZE,
        "num_envs": NUM_ENVS,
        "zdim": ZDIM + 4,
        "act_dim": act_shape,
        "num_frames": NUM_FRAMES,
        "gamma": PPO.GAMMA,
        "lam": PPO.LAMBDA,
    }
    optimizer = optim.Adam(lstm.parameters(), lr=LR)
    writer = SummaryWriter("/tmp/tensorboard")
    logger = Logger(writer)

    ppo = PPO(env, vae, lstm, optimizer, logger, DEVICE, **buf_args)
    ppo.rollout()
    ppo.train()
    env.close()
    print("src/ppo.py test successful")


def test_vae_model():

    import torch
    from torchinfo import summary

    from src.vae.model import ConvVAE, Decoder, Encoder

    OBS_DIM = (600, 400, 1)
    DEVICE, BATCH_SIZE = "cuda", 8

    rand_input = torch.randint(
        0,
        255,
        (BATCH_SIZE, OBS_DIM[-1], *OBS_DIM[:-1]),
        device=DEVICE,
        dtype=torch.float32,
    )
    vae = ConvVAE(OBS_DIM, Encoder, Decoder, 128)
    vae.to(DEVICE)

    # summary(vae, input_data=rand_input, verbose=1)
    recons_image_sto, _, _ = vae(rand_input)
    recons_image_det = vae.reconstruct(rand_input)
    # print(recons_image_sto.shape, recons_image_det.shape)
    print("src/vae/model.py test successful")


if __name__ == "__main__":
    test_env()
    test_model()
    test_ppo()
    test_vae_model()
