import numpy as np
import torch

from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F


class FCView(nn.Module):
    """
    Flatten conv layer.
    """

    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        """
        :param x: of shape(batch_size, channels, H, W)
        :return: Tensor of shape (batch_size, channels * H * W)
        """
        shape = x.data.size(0)
        x = x.view(shape, -1)
        return x


class ConvBlock(nn.Module):
    """
    Wrapper class for a Convolution layer.
    References:
        1. https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        2. https://discuss.pytorch.org/t/how-to-do-convolution-on-tube-tensors-3d-conv/21446
        3. https://discuss.pytorch.org/t/feeding-3d-volumes-to-conv3d/32378
    W_out = ((W_in + 2P - (F - 1) - 1) / S) + 1 (assuming dilation=1)

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param kernel_size: Size of the kernel/filter
    :param padding: The size of the zero-padding
    :param stride: Value of stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        """
        Forward pass for the conv layer

        :param x: of shape(batch_size, in_channels, H, W)
        :return: Tensor of shape (batch_size, out_channels, H, W)
        """
        x = F.relu(self.conv(x))
        return self.batch_norm(x)


class ResBlock(nn.Module):
    """
    A class implementing ResNet from the ImageNet paper.

    :param in_channels: Number of input channels
    :param in_channels: Number of output channels
    :param kernel_size: Size of the kernel/filter
    :param padding: The size of the zero-padding
    :param stride: Value of stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(F.relu(out))
        out = F.relu(self.conv2(out))
        # skip connection
        # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
        out = out + x
        return self.bn2(out)


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
        return torch.stack([torch.argmax(dist.probs) for dist in self.dist], dim=1)


class Actor(nn.Module):
    """
    Actor head for the PPO class to determine the best policy and action distribution.
    References:
        1. https://discuss.pytorch.org/t/multidmensional-actions/88259

    :param obs_shape: The shape of the input image (W, H, C)
    :param action_shape: The shape of the action space Eg: `MultiDiscrete().nvec`
    """

    def __init__(self, obs_shape, action_shape):
        super(Actor, self).__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reshape = FCView()
        self.actor1 = ConvBlock(in_channels=256, out_channels=64,
                kernel_size=3, padding=1, stride=1)
        self.actor2 = ConvBlock(in_channels=64, out_channels=1,
                kernel_size=3, padding=1, stride=1)
        self.fc_actor = nn.Linear(np.prod((1,) + self.obs_shape), np.sum(self.action_shape))
        self.dist = MultiCategorical(action_shape)

    def forward(self, x):
        x = F.relu(self.actor1(x))
        x = self.actor2(x)
        x = self.fc_actor(self.reshape(x))
        return self.dist.update_logits(logits=x)


class Critic(nn.Module):
    """
    Critic head for the PPO class to determine the expected rewards.

    :param obs_shape: The shape of the input image (W, H, C)
    """

    def __init__(self, obs_shape):
        super(Critic, self).__init__()

        self.obs_shape = obs_shape
        self.reshape = FCView()
        self.critic= ConvBlock(in_channels=256, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.fc_critic = nn.Linear(np.prod((1,) + self.obs_shape), 1)

    def forward(self, x):
        x = F.relu(self.critic(x))
        x = self.fc_critic(self.reshape(x))
        return x


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
        self.num_res_blocks = 6
        self.downSampleLayer = (self.num_res_blocks // 2)
        self.obs_shape = obs_shape

        self.conv1 = ConvBlock(in_channels=self.obs_shape[-1], out_channels=256,
                kernel_size=3, padding=1, stride=2)
        for block in range(0, self.downSampleLayer):
            setattr(self, f"res-block-{block+1}", ResBlock(in_channels=256,
                out_channels=256, kernel_size=3, padding=1, stride=1))

        self.conv2 = ConvBlock(in_channels=256, out_channels=256,
                kernel_size=3, padding=0, stride=3)
        for block in range(self.downSampleLayer, self.num_res_blocks):
            setattr(self, f"res-block-{block+1}", ResBlock(in_channels=256,
                out_channels=256, kernel_size=3, padding=1, stride=1))

        # change input dimensions for the Actor and Critic block
        # as it is downscaled by previous resnet layers
        # W_out = ((W_in + 2P - (F - 1) - 1) / S) + 1 (assuming dilation=1)
        # Refer: https://pytorch.org/docs/1.9.1/generated/torch.nn.Conv3d.html
        # TODO: make this dynamic
        obs_shape = (num_frames, ) + self.obs_shape[:-1]
        # for the 1st conv - (3, 200, 300)
        obs_shape = tuple(map(lambda x: (x + 2 - 2 - 1)//2 + 1, obs_shape))
        # for the 2nd conv - (1, 66, 100)
        obs_shape = tuple(map(lambda x: (x - 2 - 1)//3 + 1, obs_shape))

        self.actor = Actor(obs_shape, action_shape)
        self.critic = Critic(obs_shape)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward_base(self, obs: torch.Tensor):
        obs = self.conv1(obs)
        for block in range(0, self.downSampleLayer):
            obs = getattr(self, f"res-block-{block+1}")(obs)

        obs = self.conv2(obs)
        for block in range(self.downSampleLayer, self.num_res_blocks):
            obs = getattr(self, f"res-block-{block+1}")(obs)
        return obs

    def forward(self, obs: torch.Tensor):
        obs = self._forward_base(obs)
        policy = self.actor(obs)
        value = self.critic(obs)
        return policy, value

if __name__ == '__main__':

    DEVICE = 'cuda'
    BATCH_SIZE = 8
    NUM_FRAMES = 5
    OBS_DIM = (60, 40, 3)
    ACT_DIM = (2, 2, 3, 2, 2, 2, 2)

    randInput = torch.randint(0, 10, (BATCH_SIZE, OBS_DIM[-1], NUM_FRAMES, *OBS_DIM[:-1]),
            device=DEVICE, dtype=torch.float16)
    model = Net(OBS_DIM, ACT_DIM, NUM_FRAMES)
    model.to(DEVICE)

    policy, value = model(randInput)
    print(policy.sample(), value, sep='\n')
