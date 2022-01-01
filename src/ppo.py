import torch
import gym
import numpy as np

from torch import optim
from collections import deque
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import DEVICE, BATCH_SIZE, BUFFER_SIZE, NUM_FRAMES, NUM_ENVS, GAMMA, LAMBDA
from utils import make_env

class PPOBuffer:
    """
    Buffer to save all the (s, a, r, s`) for each step taken.
    """

    def __init__(self, buffer_size, batch_size, obs_dim, act_dim, num_frames, gamma, lam):
        self.ptr = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.gamma = gamma
        self.lam = lam

        self.obs = np.zeros((buffer_size, batch_size, *obs_dim), dtype=np.float16)
        self.actions = np.zeros((buffer_size, batch_size, *act_dim), dtype=np.float16)
        self.rewards = np.zeros((buffer_size, batch_size), dtype=np.float16)
        self.returns = np.zeros((buffer_size, batch_size), dtype=np.float16)
        self.values = np.zeros((buffer_size+1, batch_size), dtype=np.float16)
        self.log_probs = np.zeros((buffer_size, batch_size), dtype=np.float16)
        self.advantage = np.zeros((buffer_size, batch_size), dtype=np.float16)

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
        # FIXME: consider batch_size
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


    def compute_gae(self, next_value):
        # advantage = discounted sum of rewards - baseline estimate
        # meaning what is the reward i got for taking an action - the reward i was expecting for that action
        # delta = (reward + value of next state) - value of the current state
        # advantage = discounted sum of delta
        # advantage = gae = (current reward + (gamma * value of next state) - value of current state) + (discounted sum of gae)
        # advantage = gae = (current reward + essentially a measure of how better the next state is compared to the current state) + (discounted sum of gae)
        self.values[self.ptr + 1] = next_value

        deltas = self.rewards + self.gamma * self.values[1:] - self.values[:-1]
        self.advantage = self.discounted_sum(deltas, self.gamma * self.lam)     # advantage estimate using GAE
        self.returns = self.discounted_sum(self.rewards, self.gamma)            # discounted sum of rewards

    def get(self):
        assert (self.ptr - self.num_frames - 1) > 0
        idx = np.random.randint(low=0, high=self.ptr-self.num_frames-1)
        idx_range = slice(rand_num, rand_num+self.num_frames)
        return self.obs[idx_range], self.actions[idx], self.returns[idx], self.values[idx], self.log_probs[idx], self.advantage[idx]

class PPO():

    EPOCHS = 3
    CRITIC_DISCOUNT = 0.5
    ENTROPY_DISCOUNT = 0.2
    CLIP_RATIO = 0.2

    def __init__(self, env: SubprocVecEnv, model, max_time_step, **buffer_args):
        """
        :param env: list of STKEnv or vectorized envs?
        """

        self.env = env
        self.model = model
        self.buffer = PPOBuffer(**buffer_args)
        self.max_time_step = max_time_step
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    def rollout(self):

        images = self.env.get_images()
        images = deque([np.zeros_like(images) for _ in range(NUM_FRAMES)], maxlen=NUM_FRAMES)
        for i in range(self.max_time_step):
            # GAE algorithm
            # TODO: consider only 2 frames
            # Collect data using CPU and train on GPU
            images.append(self.env.get_images())
            obs = torch.from_numpy(np.swapaxes(np.array(images), 0, 1)).to(DEVICE)
            dist, value = self.model(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            _, reward, done, info = self.env.step(action)
            self.buffer.save(action, reward, value.detach(), log_prob)

            if done:
                # can it finish the race in 1024 time steps?
                print('-------------------------------------------------------------')
                print(f'Trajectory cut off at {i} time steps')
                info = self.env.get_env_info()
                for key, value in info.items():
                    print(f'{key}: {value}')
                print('-------------------------------------------------------------\n')
                break

        _, next_value = self.model(self.env.image)
        self.buffer.compute_gae(next_value)
        self.env.close()

    def train(self):

        for _ in range(self.EPOCHS):
            for _ in range(self.max_time_step):

                obs, act, returns, values, logp_old, adv= self.buffer.get()
                dist, value_new = self.model(obs)
                logp_new = dist.log_prob(action)

                ratio = (logp_new - logp_old).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 + self.CLIP_RATIO, 1 - self.CLIP_RATIO) * adv

                actor_loss = torch.min(surr1, surr2).mean(dim=1)
                critic_loss = ((value_new - returns)**2).mean(dim=1)
                entropy = dist.entropy().mean(dim=1)

                loss = actor_loss + self.CRITIC_DISCOUNT * critic_loss + self.ENTROPY_DISCOUNT * entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':

    from env import STKEnv
    from model import Net
    env = SubprocVecEnv([make_env(id) for id in range(NUM_ENVS)], start_method='spawn')
    model = Net(env.observation_space.shape, np.sum(env.action_space.nvec), NUM_FRAMES)
    model.to(DEVICE)

    buf_args = { 'buffer_size': BUFFER_SIZE, 'batch_size': BATCH_SIZE, 'obs_dim':
            env.observation_space.shape, 'act_dim': env.action_space.nvec, 'num_frames': NUM_FRAMES,
            'gamma': GAMMA, 'lam': LAMBDA }
    ppo = PPO(env, model, buf_args['buffer_size'], **buf_args)
    ppo.rollout()
    ppo.train()
