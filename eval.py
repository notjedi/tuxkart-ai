import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import deque
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.model import Net
from src.utils import Logger, make_env, get_encoder, action_to_dict

def eval(env, model, logger, args, log=False, render=False):

    assert env.num_envs == 1, 'eval is only supported for num_envs = 1'
    prevInfo = [None for _ in range(env.num_envs)]
    images = env.get_images()
    images = deque([np.zeros_like(images) for _ in range(args.num_frames)], maxlen=args.num_frames)
    to_numpy = lambda x: x.to(device='cpu').numpy()
    encoder = get_encoder(env.observation_space.shape)
    tot_reward = 0

    with torch.no_grad():
        t = tqdm(range(args.eval_steps))
        for i in t:
            images.append(np.array(env.get_images()))
            encoded_infos = encoder(prevInfo)
            images[0] = encoded_infos
            obs = torch.from_numpy(np.transpose(np.array(images), (1, 2, 0, 3, 4))).to(args.device)

            dist, value = model(obs)
            action = to_numpy(dist.mode())
            _, reward, done, info = env.step(action)
            sum_reward = reward.sum()
            tot_reward += sum_reward
            prevInfo = info
            t.set_description(f"rewards: {sum_reward}")

            if log:
                logger.log_eval(sum_reward, value, tot_reward, images[-1].squeeze())

            if render:
                image = np.array(env.env_method('render')).squeeze()
                plt.imshow(image.astype(np.uint8))
                plt.pause(0.1)

            if done.any():
                break

    return tot_reward


def main(args):

    writer = SummaryWriter(log_dir=args.log_dir)
    logger = Logger(writer)
    race_config_args = { 'track': args.track, 'kart': args.kart, 'numKarts': args.num_karts,
            'laps': args.laps, 'reverse': args.reverse, 'difficulty': args.difficulty }
    env = SubprocVecEnv([make_env(id, args.graphic, race_config_args) for id in range(1)],
            start_method='spawn')

    model = Net(env.observation_space.shape, env.action_space.nvec, args.num_frames)
    model.to(args.device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    reward = eval(env, model, logger, args, log=True, render=True)
    print(f'Total rewards: {reward}')
    env.close()


if __name__ == '__main__':

    from os.path import join
    from src.utils import STK

    parser = argparse.ArgumentParser("Implementation of the PPO algorithm for the SuperTuxKart game")
    parser.add_argument('--laps', type=int, default=1)
    parser.add_argument('--num_karts', type=int, default=5)
    parser.add_argument('--difficulty', type=int, default=1)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--kart', type=str, choices=STK.KARTS, default=None)
    parser.add_argument('--track', type=str, choices=STK.TRACKS, default=None)
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')

    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=2500)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model_path', type=Path, default=None, help='Load model from path.')
    parser.add_argument('--log_dir', type=Path, default=join(Path(__file__).absolute().parent,
    'tensorboard'), help='Path to the directory in which the trained models are saved.')
    args = parser.parse_args()

    main(args)
