import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import Net
from src.utils import Logger, get_encoder, make_env
from src.vae.model import ConvVAE, Decoder, Encoder


@torch.no_grad()
def eval(env, vae, lstm, logger, args, self_control=False, log=False, render=False):

    assert env.num_envs == 1, "eval is only supported for num_envs = 1"
    vae.eval()
    lstm.eval()
    obs = torch.from_numpy(np.array(env.reset())).unsqueeze(dim=1).to(args.device)

    info_encoder = get_encoder()
    prev_info = info_encoder(env.env_method("get_info"))
    latent_repr = deque(
        np.zeros((args.num_frames, env.num_envs, vae.zdim + 4), dtype=np.float32),
        maxlen=args.num_frames,
    )
    latent_repr.append(np.column_stack((vae.encode(obs)[0].cpu().numpy(), prev_info)))
    act = np.array([None])
    tot_reward = 0

    def to_numpy(x):
        return x.to(device="cpu").numpy()

    t = tqdm(range(args.eval_steps))
    for i in t:

        if self_control:
            obs, reward, done, info = env.step(act)
        else:
            dist, value = lstm(torch.from_numpy(np.array(latent_repr)).to(args.device))
            action = to_numpy(dist.mode())
            obs, reward, done, info = env.step(action)
            obs = torch.from_numpy(np.array(obs)).unsqueeze(dim=1).to(args.device)
            prev_info = info_encoder(info)
            latent_repr.append(
                np.column_stack((vae.encode(obs)[0].cpu().numpy(), prev_info))
            )

        sum_reward = reward.sum()
        tot_reward += sum_reward
        t.set_description(f"rewards: {sum_reward}")
        image = np.array(env.env_method("render")).squeeze().astype(np.uint8)

        if log:
            logger.log_eval(sum_reward, value.item(), tot_reward, image)

        if render:
            plt.imshow(image)
            plt.pause(0.1)

        if done.any():
            break

    return tot_reward


def main(args):

    writer = SummaryWriter(log_dir=args.log_dir)
    logger = Logger(writer)
    race_config_args = {
        "track": args.track,
        "kart": args.kart,
        "numKarts": args.num_karts,
        "laps": args.laps,
        "reverse": args.reverse,
        "vae": args.self_control,
        "difficulty": args.difficulty,
    }

    env = make_env(id)()
    obs_shape, act_shape = env.observation_space.shape, env.action_space.nvec
    env.close()

    vae = ConvVAE(obs_shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    lstm = Net(vae.zdim + 4, act_shape, 1)
    lstm.to(args.device)

    if args.vae_model_path is not None:
        vae.load_state_dict(torch.load(args.vae_model_path), strict=False)
    if args.lstm_model_path is not None:
        lstm.load_state_dict(torch.load(args.lstm_model_path))

    env = SubprocVecEnv(
        [make_env(id, args.graphic, race_config_args) for id in range(1)],
        start_method="spawn",
    )
    reward = eval(
        env,
        vae,
        lstm,
        logger,
        args,
        self_control=args.self_control,
        log=False,
        render=True,
    )
    print(f"Total rewards: {reward}")
    env.close()


if __name__ == "__main__":

    from os.path import join

    from src.utils import STK

    parser = argparse.ArgumentParser(
        "Implementation of the PPO algorithm for the SuperTuxKart game"
    )
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--num_karts", type=int, default=5)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--reverse", type=bool, default=False)
    parser.add_argument("--self_control", type=bool, default=False)
    parser.add_argument("--kart", type=str, choices=STK.KARTS, default=None)
    parser.add_argument("--track", type=str, choices=STK.TRACKS, default=None)
    parser.add_argument("--graphic", type=str, choices=["hd", "ld", "sd"], default="hd")

    parser.add_argument("--zdim", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=2500)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--vae_model_path",
        type=Path,
        default=None,
        help="Load VAE model from path.",
    )
    parser.add_argument(
        "--lstm_model_path",
        type=Path,
        default=None,
        help="Load LSTM model from path.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=join(Path(__file__).absolute().parent, "tensorboard"),
        help="Path to the directory in which the trained models are saved.",
    )
    args = parser.parse_args()

    main(args)
