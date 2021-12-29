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
    :param in_channels: Number of output channels
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
        x = F.relu(self.conv(x.float()))
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


class Actor(nn.Module):
    """
    Actor head for the PPO class to determine the best policy and action distribution.
    References:
        1. https://discuss.pytorch.org/t/multidmensional-actions/88259

    :param inputDims: The shape of the input image (W, H, C)
    :param numLogits: The number of logits in the last layer of the network
    """

    def __init__(self, inputDims, numLogits):
        super(Actor, self).__init__()

        self.inputDims = inputDims
        self.numLogits = numLogits
        self.numActions = numLogits - 2
        self.reshape = FCView()
        self.actor1 = ConvBlock(in_channels=256, out_channels=128,
                kernel_size=3, padding=1, stride=1)
        self.actor2 = ConvBlock(in_channels=128, out_channels=32,
                kernel_size=3, padding=1, stride=1)
        self.fc_actor = nn.Linear(np.prod((32,) + self.inputDims), self.numLogits)

    def __call__(self, x, deterministic=False):
        return self.pi(x, deterministic)

    def forward(self, x):
        x = F.relu(self.actor1(x))
        x = self.actor2(x)
        x = self.fc_actor(self.reshape(x))
        return x

    def pi(self, x, deterministic=False):
        x = self.forward(x)
        steerProb = F.softmax(x[:, :3], dim=1)
        actionProb = torch.sigmoid(x[:, 3:])
        if deterministic:
            steer = torch.argmax(steerProb, dim=1)
            actions = torch.round(actionProb)
            pi = torch.cat((steer.view(-1, 1), actions), dim=1)
            return pi
        else:
            # https://mathworld.wolfram.com/BernoulliDistribution.html
            # https://stats.stackexchange.com/a/113381
            steerDist = Categorical(steerProb)
            actionDist = Bernoulli(actionProb)
            return steerDist, actionDist


class Critic(nn.Module):
    """
    Critic head for the PPO class to determine the expected rewards.

    :param inputDims: The shape of the input image (W, H, C)
    """

    def __init__(self, inputDims):
        super(Critic, self).__init__()

        self.inputDims = inputDims
        self.reshape = FCView()
        self.critic= ConvBlock(in_channels=256, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.fc_critic = nn.Linear(np.prod((32,) + self.inputDims), 1)

    def forward(self, x):
        x = F.relu(self.critic(x))
        x = self.fc_critic(self.reshape(x))
        return x


class PPO(nn.Module):
    """
    Proximal Policy Optimization algorithm (PPO)

    Paper: https://arxiv.org/abs/1707.06347
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Implementation: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    :param inputDims: The shape of the input image (W, H, C)
    :param numLogits: The number of logits in the last layer of the network
    """

    def __init__(self, inputDims, numLogits):
        super(PPO, self).__init__()

        self.timeDim = 5
        self.numResBlocks = 10
        self.downSampleLayer = (self.numResBlocks // 2)
        # channel-first layout
        self.inputDims = tuple(reversed(inputDims))

        self.conv1 = ConvBlock(in_channels=self.inputDims[0], out_channels=256,
                kernel_size=3, padding=1, stride=1)
        for block in range(0, self.downSampleLayer):
            setattr(self, f"res-block-{block+1}", ResBlock(in_channels=256,
                out_channels=256, kernel_size=3, padding=1, stride=1))

        self.conv2 = ConvBlock(in_channels=256, out_channels=256,
                kernel_size=5, padding=1, stride=1)

        for block in range(self.downSampleLayer, self.numResBlocks):
            setattr(self, f"res-block-{block+1}", ResBlock(in_channels=256,
                out_channels=256, kernel_size=3, padding=1, stride=1))

        inputDims = tuple(map(lambda x: (x + 2 - 4 - 1)//1 + 1,
            (self.timeDim,) + self.inputDims[1:]))
        self.actor = Actor(inputDims, numLogits)
        self.critic = Critic(inputDims)
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

    def forward(self, x, deterministic=False):
        x = self.conv1(x)
        for block in range(0, self.downSampleLayer):
            x = getattr(self, f"res-block-{block+1}")(x)

        x = self.conv2(x)
        for block in range(self.downSampleLayer, self.numResBlocks):
            x = getattr(self, f"res-block-{block+1}")(x)
        policy = self.actor(x, deterministic)
        value = self.critic(x)
        return policy, value
