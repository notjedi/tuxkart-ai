import torch
import numpy as np

from tqdm import tqdm, trange
from collections import deque
from scipy.signal import lfilter
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv


class PPOBuffer:
    """
    Buffer to save all the (s, a, r, s`) for each step taken.
    """

    def __init__(self, buffer_size, batch_size, obs_dim, act_dim, num_frames, gamma, lam):
        self.ptr = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.obs = np.zeros((self.buffer_size, self.batch_size, *reversed(self.obs_dim)), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.batch_size, len(self.act_dim)), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.batch_size), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.batch_size), dtype=np.float32)
        self.values = np.zeros((self.buffer_size+1, self.batch_size), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.batch_size), dtype=np.float32)
        self.advantage = np.zeros((self.buffer_size, self.batch_size), dtype=np.float32)

    def save(self, obs, act, reward, value, log_prob):
        assert self.ptr < self.buffer_size
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def discounted_sum(self, x, discount):
        """
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L29
        input:
            [[x0, x1, x2]
             [y0, y1, y2]]
        output:
            [[x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
             [y0 + discount * y1 + discount^2 * y2, y1 + discount * y2, y2]]
        """
        return np.flip(lfilter([1], [1, -discount], np.flip(x, axis=0), axis=0), axis=0)

    def compute_gae(self, next_value):
        # advantage = discounted sum of rewards - baseline estimate
        # meaning what is the reward i got for taking an action - the reward i was expecting for that action
        # delta = (reward + value of next state) - value of the current state
        # advantage = discounted sum of delta
        # advantage = gae = (current reward + (gamma * value of next state) - value of current state) + (discounted sum of gae)
        # advantage = gae = (current reward + essentially a measure of how better the next state is compared to the current state) + (discounted sum of gae)
        self.values[self.ptr] = next_value
        deltas = self.rewards + self.gamma * self.values[1:] - self.values[:-1]
        self.advantage = self.discounted_sum(deltas, self.gamma * self.lam)     # advantage estimate using GAE
        self.returns = self.discounted_sum(self.rewards, self.gamma)            # discounted sum of rewards
        self.advantage = (self.advantage - np.mean(self.advantage, axis=0)) / np.std(self.advantage,
                axis=0) # axis 0 because advantage is of shape (buffer_size, num_env)
        # self.returns = self.advantage - self.values[:-1]                      # some use this, some use the above
        del self.values

    def can_train(self):
        return (self.ptr - self.num_frames - 1) > 0

    def get_ptr(self):
        return self.ptr

    def get(self):
        idx = np.random.randint(low=0, high=self.ptr-self.num_frames-1)
        idx_range = slice(idx, idx+self.num_frames)
        return self.obs[idx_range], self.actions[idx], self.returns[idx], self.log_probs[idx], self.advantage[idx]


class PPO():

    EPOCHS = 3
    GAMMA = 0.9
    LAMBDA = 0.95
    EPSILON = 0.2
    ENTROPY_BETA = 0.2
    CRITIC_DISCOUNT = 0.5

    def __init__(self, env: SubprocVecEnv, model, optimizer, writer, device, **buffer_args):
        """
        :param env: list of STKEnv or vectorized envs?
        """

        self.env = env
        self.model = model
        self.opt = optimizer
        self.device = device
        self.writer = writer
        buffer_args['gamma'], buffer_args['lam'] = self.GAMMA, self.LAMBDA
        self.buffer = PPOBuffer(**buffer_args)
        self.num_frames = buffer_args['num_frames']
        self.buffer_size = buffer_args['buffer_size']

    def rollout(self):

        images = self.env.get_images()
        images = deque([np.zeros_like(images) for _ in range(self.num_frames)], maxlen=self.num_frames)
        to_numpy = lambda x: x.to(device='cpu').numpy()

        with torch.no_grad():
            for i in trange(self.buffer_size):

                images.append(np.array(self.env.get_images()))
                obs = torch.from_numpy(np.transpose(np.array(images), (1, 2, 0, 3, 4))).to(self.device)

                dist, value = self.model(obs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                _, reward, done, info = self.env.step(to_numpy(action))
                self.buffer.save(images[-1], to_numpy(action), reward,
                        to_numpy(value.squeeze(dim=-1)), to_numpy(log_prob))

                if done.any():
                    print('-------------------------------------------------------------')
                    print(f'Trajectory cut off at {i} time steps')
                    env_infos = np.array(self.env.env_method('get_env_info'))[done]
                    for env_info in env_infos:
                        for key, value in env_info.items():
                            print(f'{key}: {value}')
                    print('-------------------------------------------------------------\n')
                    break

            images.append(np.array(self.env.get_images()))
            obs = torch.from_numpy(np.transpose(np.array(images), (1, 2, 0, 3, 4))).to(self.device)
            _, next_value = self.model(obs)
            self.buffer.compute_gae(to_numpy(next_value.squeeze(dim=1)))
            self.env.close()

    def train(self):

        to_cuda = lambda x: torch.from_numpy(x).to(device=torch.device(self.device), dtype=torch.float32)
        if not self.buffer.can_train():
            print("Buffer size is too small")
            return

        for epoch in trange(self.EPOCHS):
            for timestep in (t:=tqdm((range(self.buffer.get_ptr())))):

                self.opt.zero_grad()
                obs, act, returns, logp_old, adv = map(to_cuda, self.buffer.get())
                dist, value_new = self.model(obs.permute(1, 2, 0, 3, 4)) # transpose axes because it is originally in shape (D, N, C, H, W)
                logp_new = dist.log_prob(act)

                ratio = (logp_new - logp_old).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 + self.EPSILON, 1 - self.EPSILON) * adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.CRITIC_DISCOUNT * ((value_new.squeeze() - returns)**2).mean()
                entropy_loss = self.ENTROPY_BETA * dist.entropy().mean()

                loss = actor_loss + critic_loss - entropy_loss
                loss.backward()
                self.opt.step()

                step = epoch * timestep
                t.set_description(f"loss: {loss}")
                self.writer.add_scalar("train/ratio", ratio.item(), step)
                self.writer.add_scalar("train/advantage", adv.item(), step)
                self.writer.add_scalar("train/returns", returns.item(), step)
                self.writer.add_scalar("train/log_old", logp_old.item(), step)
                self.writer.add_scalar("train/log_new", logp_new.item(), step)
                self.log(step, actor_loss, critic_loss, entropy_loss, loss)

    def log(self, step, actor_loss, critic_loss, entropy_loss, loss):

        self.writer.add_scalar("train/entropy_loss", entropy_loss.item(), step)
        self.writer.add_scalar("train/policy_loss", actor_loss.item(), step)
        self.writer.add_scalar("train/value_loss", critic_loss.item(), step)
        self.writer.add_scalar("train/loss", loss.item(), step)
        self.writer.flush()
