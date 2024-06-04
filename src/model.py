import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


# https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
class MultiCategorical:
    def __init__(self, action_shape):
        self.action_shape = tuple(action_shape)
        self.dist = None

    def update_logits(self, logits):
        self.dist = [
            Categorical(logits=split)
            for split in torch.split(logits, self.action_shape, dim=1)
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
        return torch.stack(
            [torch.argmax(dist.probs, dim=1) for dist in self.dist], dim=1
        )


class Actor(nn.Module):
    """
    Actor head for the PPO class to determine the best policy and action distribution.
    References:
        1. https://discuss.pytorch.org/t/multidmensional-actions/88259

    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, zdim, action_shape):
        super(Actor, self).__init__()

        self.shared = StackedLinear(zdim)
        latent_shape = zdim // 2
        self.actor = nn.Sequential(
            nn.Linear(latent_shape, latent_shape // 2),
            nn.ReLU(),
            nn.Linear(latent_shape // 2, np.sum(action_shape)),
        )
        self.dist = MultiCategorical(action_shape)

    def forward(self, inputs):
        inputs = self.shared(inputs)[-1]
        inputs = self.actor(inputs)
        return self.dist.update_logits(logits=inputs)


class Critic(nn.Module):
    """
    Critic head for the PPO class to determine the expected rewards.

    """

    def __init__(self, zdim):
        super(Critic, self).__init__()

        self.shared = StackedLinear(zdim)
        latent_shape = zdim // 2
        self.critic = nn.Sequential(
            nn.Linear(latent_shape, latent_shape // 2),
            nn.ReLU(),
            nn.Linear(latent_shape // 2, 1),
        )

    def forward(self, inputs):
        inputs = self.shared(inputs)[-1]
        return self.critic(inputs)


class StackedLinear(nn.Module):
    def __init__(self, latent_shape):
        super(StackedLinear, self).__init__()
        self.latent_shape = latent_shape
        self.model = nn.Sequential(
            nn.Linear(latent_shape, latent_shape // 2),
            nn.Tanh(),
            nn.Linear(latent_shape // 2, latent_shape // 2),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.model(input)


class Net(nn.Module):
    """
    Proximal Policy Optimization algorithm (PPO)

    Paper: https://arxiv.org/abs/1707.06347
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Implementation: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, zdim, action_shape: tuple, batch_size: int, isLSTM=False):
        super(Net, self).__init__()

        # https://discuss.pytorch.org/t/lstm-network-inside-a-sequential-container/19304/2
        torch.set_default_dtype(torch.float32)
        hidden_size, num_layers = 256, 2
        self.isLSTM = isLSTM

        self.actor = Actor(zdim, action_shape)
        self.critic = Critic(zdim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input: torch.Tensor, idx: int = -1):
        return self.actor(input), self.critic(input)
