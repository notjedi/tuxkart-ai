import torch
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.model import Net
from src.utils import make_env

def eval(env, model, writer, args, log=False, render=False):

    if render:
        from gym.envs.classic_control.rendering import SimpleImageViewer
        viewer = SimpleImageViewer()
        assert env.num_envs == 1, 'Render mode is only supported for num_envs = 1'
    images = env.get_images()
    images = deque([np.zeros_like(images) for _ in range(args.num_frames)], maxlen=args.num_frames)
    to_numpy = lambda x: x.to(device='cpu').numpy()
    tot_reward = 0

    with torch.no_grad():
        for i in (t:=tqdm(range(args.eval_steps))):
            images.append(np.array(env.get_images()))
            obs = torch.from_numpy(np.transpose(np.array(images), (1, 2, 0, 3, 4))).to(args.device)

            dist, value = model(obs)
            action = dist.mode()
            log_prob = dist.log_prob(action)
            _, reward, done, _ = env.step(to_numpy(action))
            tot_reward += reward
            t.set_description(f"rewards: {sum(reward) / len(reward)}")

            if log:
                writer.add_scalar('eval/rewards', reward, i)
                writer.add_scalar('eval/values', value.item(), i)
                writer.add_scalar('eval/total_rewards', tot_reward, i)
                writer.add_image('eval/image', images[-1].squeeze(), i, dataformats='CWH')

            if render:
                image = env.env_method('render')
                viewer.imshow(image)

            if done.any():
                break

    writer.flush()
    return tot_reward


def main(args):

    writer = SummaryWriter(log_dir=args.log_dir)
    race_config_args = { 'track': args.track, 'kart': args.kart, 'numKarts': args.num_karts,
            'laps': args.laps, 'reverse': args.reverse, 'difficulty': args.difficulty }
    env = SubprocVecEnv([make_env(id, args.graphic, race_config_args) for id in range(1)],
            start_method='spawn')

    model = Net(env.observation_space.shape, env.action_space.nvec, args.num_frames)
    model.to(args.device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    reward = eval(env, model, writer, args, log=True)
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
