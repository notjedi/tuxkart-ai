import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class ResNet(nn.Module):
    """
    A class implementing ResNet from the ImageNet paper.

    :param in_channels: Number of input channels
    :param in_channels: Number of output channels
    :param kernel_size: Size of the kernel/filter
    :param padding: The size of the zero-padding
    :param stride: Value of stride
    """

    def __init__(self, module):
        super(ResNet, self).__init__()
        self.module = module

    def forward(self, x):
        # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
        return self.module(x) + x


class MultiCategorical():

    def __init__(self, action_shape):
        self.action_shape = tuple(action_shape)
        self.dist = None

    def update_logits(self, logits):
        self.dist = [Categorical(logits=split) for split in torch.split(logits,
            self.action_shape, dim=1)]
        return self

    def get_actions(self, logits=None, deterministic=False) -> torch.Tensor:
        if logits is not None:
            self.update_logits(logits)
        assert self.dist is not None, "Distribution is not initialized, try passing in logits"
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob(self, actions) -> torch.Tensor:
        return torch.stack([dist.log_prob(action) for dist, action in zip(self.dist,
            torch.unbind(actions, dim=1))], dim=1).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.dist], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        assert self.dist is not None
        return torch.stack([dist.sample() for dist in self.dist], dim=1)

    def mode(self) -> torch.Tensor:
        assert self.dist is not None
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.dist], dim=1)


class Actor(nn.Module):
    """
    Actor head for the PPO class to determine the best policy and action distribution.
    References:
        1. https://discuss.pytorch.org/t/multidmensional-actions/88259

    :param obs_shape: The shape of the input image (W, H, C)
    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, latent_shape, action_shape):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(np.prod((1,) + latent_shape[-2:]), np.sum(action_shape))
        )
        self.dist = MultiCategorical(action_shape)

    def forward(self, inputs):
        inputs = self.actor(inputs)
        return inputs
        # return self.dist.update_logits(logits=inputs)


class Critic(nn.Module):
    """
    Critic head for the PPO class to determine the expected rewards.

    :param obs_shape: The shape of the input image (W, H, C)
    """

    def __init__(self, latent_shape):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(np.prod((1,) + latent_shape[-2:]), 1)
        )

    def forward(self, inputs):
        return self.critic(inputs)


class Net(nn.Module):
    """
    Proximal Policy Optimization algorithm (PPO)

    Paper: https://arxiv.org/abs/1707.06347
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Implementation: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    :param obs_shape: The shape of the input image (W, H, C)
    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, obs_shape: tuple, action_shape: tuple, num_frames: int):
        super(Net, self).__init__()

        torch.set_default_dtype(torch.float32)

        self.shared = nn.Sequential(
            nn.Conv2d(num_frames, 128, kernel_size=10, padding=1, stride=4),
            nn.Tanh(),
            nn.BatchNorm2d(128),
            ResNet(nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=4, padding=2, stride=1),
                nn.Tanh(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=1),
                nn.Tanh(),
                nn.BatchNorm2d(128)
            )),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(128),
            ResNet(nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=4, padding=2, stride=1),
                nn.Tanh(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=1),
                nn.Tanh(),
                nn.BatchNorm2d(128)
            )),
        )

        with torch.no_grad():
            latent_shape = tuple(self.shared(torch.randn((8, num_frames) + obs_shape[:-1])).shape)

        self.actor = Actor(latent_shape, action_shape)
        self.critic = Critic(latent_shape)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: torch.Tensor):
        inputs = self.shared(inputs)
        policy = self.actor(inputs)
        value = self.critic(inputs)
        return policy, value
