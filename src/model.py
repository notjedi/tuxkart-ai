import torch
import numpy as np

from torch import nn
from torch.distributions import Categorical


# https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
class MultiCategorical:
    def __init__(self, action_shape):
        self.action_shape = tuple(action_shape)
        self.dist = None

    def update_logits(self, logits):
        self.dist = [
            Categorical(logits=split) for split in torch.split(logits, self.action_shape, dim=1)
        ]
        return self

    def log_prob(self, actions) -> torch.Tensor:
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self.dist, torch.unbind(actions, dim=1))
            ],
            dim=1,
        ).sum(dim=1)

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

    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, latent_shape, action_shape):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(latent_shape, latent_shape // 2),
            nn.Tanh(),
            nn.Linear(latent_shape // 2, np.sum(action_shape)),
        )
        self.dist = MultiCategorical(action_shape)

    def forward(self, inputs):
        inputs = self.actor(inputs)
        return self.dist.update_logits(logits=inputs)


class Critic(nn.Module):
    """
    Critic head for the PPO class to determine the expected rewards.

    """

    def __init__(self, latent_shape):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(latent_shape, latent_shape // 2), nn.Tanh(), nn.Linear(latent_shape // 2, 1)
        )

    def forward(self, inputs):
        return self.critic(inputs)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, buffer_size=1, batch_size=8):
        super(LSTM, self).__init__()
        self.device = 'cuda'
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.reset(buffer_size, batch_size)

    def reset(self, buffer_size, batch_size):
        self.ptr = 0
        self.buffer_size = buffer_size
        device = 'cpu' if batch_size > 1 else 'cuda'
        self.h0 = torch.zeros(
            (buffer_size, self.lstm.num_layers, batch_size, self.lstm.hidden_size),
            dtype=torch.float32,
            device=device,
        )
        self.c0 = torch.zeros(
            (buffer_size, self.lstm.num_layers, batch_size, self.lstm.hidden_size),
            dtype=torch.float32,
            device=device,
        )

    def forward(self, input, idx):
        out, (h0, c0) = self.lstm(
            input, (self.h0[idx].to(self.device), self.c0[idx].to(self.device))
        )
        if not self.training:
            self.h0[self.ptr], self.c0[self.ptr] = (h0.detach().cpu(), c0.detach().cpu())
            self.ptr = min(self.ptr + 1, self.buffer_size - 1)
        return out


class Net(nn.Module):
    """
    Proximal Policy Optimization algorithm (PPO)

    Paper: https://arxiv.org/abs/1707.06347
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Implementation: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, zdim, action_shape: tuple, batch_size: int):
        super(Net, self).__init__()

        # https://discuss.pytorch.org/t/lstm-network-inside-a-sequential-container/19304/2
        torch.set_default_dtype(torch.float32)
        hidden_size, num_layers = 256, 2

        self.lstm = LSTM(zdim, hidden_size, num_layers, batch_size)
        self.actor = Actor(hidden_size, action_shape)
        self.critic = Critic(hidden_size)
        self._initialize_weights()

    def reset(self, buffer_size, batch_size):
        self.lstm.reset(buffer_size, batch_size)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input: torch.Tensor, idx: int = -1):
        input = self.lstm(input, idx)[-1]
        return self.actor(input), self.critic(input)
