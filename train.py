import os
import torch
import argparse
import numpy as np

from torch import optim
from tqdm import trange
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.ppo import PPO
from src.model import Net
from src.utils import STK, Logger, make_env
from src.vae.model import ConvVAE, Encoder, Decoder


def eval(vae, lstm, logger, args):

    from eval import eval

    try:
        env = SubprocVecEnv(
            [make_env(id) for id in range(1)], start_method='spawn'
        )
        tot_reward = (
            np.sum(eval(env, vae, lstm, logger, args, log=True)) / args.num_envs
        )
        env.close()
    except EOFError as e:
        print(e)
        print("EOFError while evaluvating the model")
    return tot_reward


def main(args):

    args.seed = np.random.rand()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    env = make_env(id)()
    obs_shape, act_shape = env.observation_space.shape, env.action_space.nvec
    buf_args = {
        'buf_size': args.buffer_size,
        'num_envs': args.num_envs,
        'zdim': args.zdim + 4,
        'act_dim': env.action_space.nvec,
        'num_frames': args.num_frames,
    }
    env.close()

    vae = ConvVAE(obs_shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    lstm = Net(vae.zdim + 4, act_shape, args.num_envs)
    lstm.to(args.device)

    if args.vae_model_path is not None:
        print(f"loading VAE model from {args.vae_model_path}")
        vae.load_state_dict(torch.load(args.vae_model_path), strict=False)

    if args.lstm_model_path is not None:
        print(f"loading LSTM model from {args.lstm_model_path}")
        lstm.load_state_dict(torch.load(args.lstm_model_path))

    prev_reward, curr_reward = -float('inf'), 0
    optimizer = optim.Adam(lstm.parameters(), lr=args.lr, eps=1e-5)
    writer = SummaryWriter(log_dir=args.log_dir)
    logger = Logger(writer)

    for i in trange(args.num_global_steps):
        torch.cuda.empty_cache()
        race_config_args = {
            'track': args.track,
            'kart': args.kart,
            'reverse': np.random.choice([True, False]),
        }
        env = SubprocVecEnv(
            [
                make_env(id, args.graphic, race_config_args)
                for id in range(args.num_envs)
            ],
            start_method='spawn',
        )
        ppo = PPO(env, vae, lstm, optimizer, logger, args.device, **buf_args)

        try:
            ppo.rollout()
            env.close()
            ppo.train()
            torch.save(lstm.state_dict(), f'{args.save_dir}/stacked-temp.pth')
        except EOFError as e:
            print(e)
            print(f"EOFError at timestep {i+1}")
        except KeyboardInterrupt:
            env.close()
            print("Exiting...")

        if i % args.eval_interval == 0 and i != 0:
            curr_reward = eval(vae, lstm, logger, args)
            print(curr_reward)
            if curr_reward > prev_reward:
                print(
                    f'{curr_reward} is better than {prev_reward}, \
                        saving model to path model/stacked-{i}.pth'
                )
                prev_reward = curr_reward
                torch.save(
                    lstm.state_dict(),
                    f'{args.save_dir}/stacked-{i}-{curr_reward:.2f}.pth',
                )


if __name__ == '__main__':
    from os.path import join

    parser = argparse.ArgumentParser(
        "Implementation of the PPO algorithm for the SuperTuxKart game"
    )
    # env arguments
    parser.add_argument('--kart', type=str, choices=STK.KARTS, default=None)
    parser.add_argument('--track', type=str, choices=STK.TRACKS, default=None)
    parser.add_argument(
        '--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd'
    )

    # model args
    parser.add_argument(
        '--vae_model_path',
        type=Path,
        default=None,
        help='Load VAE model from path.',
    )
    parser.add_argument(
        '--lstm_model_path',
        type=Path,
        default=None,
        help='Load LSTM model from path.',
    )
    parser.add_argument('--zdim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=512)

    # train args
    parser.add_argument('--num_envs', type=int, default=7)
    parser.add_argument('--eval_steps', type=int, default=512)
    parser.add_argument('--eval_interval', type=int, default=20)
    parser.add_argument('--num_global_steps', type=int, default=5000)
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda'], default='cuda'
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        default=join(Path(__file__).absolute().parent, 'tensorboard'),
        help='Path to the directory in which the tensorboard logs are saved.',
    )
    parser.add_argument(
        '--save_dir',
        type=Path,
        default=join(Path(__file__).absolute().parent, 'models'),
        help='Path to the directory in which the trained models are saved.',
    )
    args = parser.parse_args()

    main(args)
