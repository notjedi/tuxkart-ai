import torch
import argparse
import numpy as np

from pathlib import Path
from tqdm import trange
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.ppo import PPO
from src.model import Net
from src.env import STKEnv
from src.utils import STK, make_env


def eval(model, args):

    from eval import eval
    env = SubprocVecEnv([make_env(id) for id in range(args.num_envs)], start_method='spawn')
    tot_reward = np.sum(eval(env, model, args)) / args.num_envs
    env.close()
    return tot_reward


def main(args):

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # TODO: load model from args
    prev_reward, curr_reward = 0, 0
    env = make_env(id)()
    model = Net(env.observation_space.shape, env.action_space.nvec, args.num_frames)
    model.to(args.device)
    env.close()
    print(eval(model, args))
    exit(0)

    for i in trange(args.num_global_steps):
        env = SubprocVecEnv([make_env(id) for id in range(args.num_envs)], start_method='spawn')
        buf_args = { 'buffer_size': args.buffer_size, 'batch_size': args.num_envs, 'obs_dim':
                env.observation_space.shape, 'act_dim': env.action_space.nvec, 'num_frames':
                args.num_frames, 'gamma': args.gamma, 'lam': args.lam }
        ppo = PPO(env, model, args.device, **buf_args)

        ppo.rollout()
        ppo.train()

        if (i % args.eval_interval == 0 and i != 0):
            curr_reward = eval(model, args)
            if (curr_reward > prev_reward):
                print(f'{curr_reward} is better than {prev_reward}, \
                        saving model to path model/model-{i}.pth')
                prev_reward = curr_reward
                torch.save(net.state_dict(), f'{args.save_dir}/model-{i}.pth')


if __name__ == '__main__':
    from os.path import join
    parser = argparse.ArgumentParser("Implementation of the PPO algorithm for the SuperTuxKart game")
    # TODO: parameterize all raceconfig options
    # env arguments
    parser.add_argument('--kart', type=str, default=None)
    parser.add_argument('--track', type=str, default=None)
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')

    # model arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=1024)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    # parser.add_argument('--epsilon', type=float, default=0.2, help='')
    # TODO: do i need to parameterize all these?
    parser.add_argument('--tau', type=float, default=1.0, help='')
    parser.add_argument('--beta', type=float, default=0.01, help='')
    parser.add_argument('--gamma', type=float, default=0.9, help='')
    parser.add_argument('--lam', type=float, default=0.9, help='')

    # train args
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=512)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_global_steps', type=int, default=5000)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--log_dir', type=Path, default=join(Path(__file__).absolute().parent,
    '/tensorboard'), help='Path to the directory in which the trained models are saved.')
    parser.add_argument('--save_dir', type=Path, default=join(Path(__file__).absolute().parent,
        '/models'), help='Path to the directory in which the trained models are saved.')
    args = parser.parse_args()

    main(args)
