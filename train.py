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
from src.utils import STK, make_env


def eval(model, writer, args):

    from eval import eval
    env = SubprocVecEnv([make_env(id) for id in range(1)], start_method='spawn')
    tot_reward = np.sum(eval(env, model, writer, args)) / args.num_envs
    env.close()
    return tot_reward


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    env = make_env(id)()
    model = Net(env.observation_space.shape, env.action_space.nvec, args.num_frames)
    model.to(args.device)
    env.close()
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    race_config_args = { 'track': args.track, 'kart': args.kart }
    prev_reward, curr_reward = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    for i in trange(args.num_global_steps):
        env = SubprocVecEnv([make_env(id, args.graphic, race_config_args) for id in
            range(args.num_envs)], start_method='spawn')
        buf_args = { 'buffer_size': args.buffer_size, 'batch_size': args.num_envs, 'obs_dim':
                env.observation_space.shape, 'act_dim': env.action_space.nvec, 'num_frames':
                args.num_frames }
        ppo = PPO(env, model, optimizer, writer, args.device, **buf_args)

        ppo.rollout()
        ppo.train()

        if (i % args.eval_interval == 0 and i != 0):
            curr_reward = eval(model, writer, args)
            if (curr_reward > prev_reward):
                print(f'{curr_reward} is better than {prev_reward}, \
                        saving model to path model/model-{i}.pth')
                prev_reward = curr_reward
                torch.save(net.state_dict(), f'{args.save_dir}/model-{i}.pth')


if __name__ == '__main__':
    from os.path import join
    parser = argparse.ArgumentParser("Implementation of the PPO algorithm for the SuperTuxKart game")
    # env arguments
    parser.add_argument('--kart', type=str, choices=STK.KARTS, default=None)
    parser.add_argument('--track', type=str, choices=STK.TRACKS, default=None)
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')

    # model args
    parser.add_argument('--model_path', type=Path, default=None, help='Load model from path.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=1024)

    # train args
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=512)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_global_steps', type=int, default=5000)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--log_dir', type=Path, default=join(Path(__file__).absolute().parent,
    'tensorboard'), help='Path to the directory in which the tensorboard logs are saved.')
    parser.add_argument('--save_dir', type=Path, default=join(Path(__file__).absolute().parent,
        '/models'), help='Path to the directory in which the trained models are saved.')
    args = parser.parse_args()

    main(args)
