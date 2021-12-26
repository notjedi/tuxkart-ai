import torch
import numpy as np

from torch.distributions import Categorical, Bernoulli
from torch.nn import functional as F
from torch import nn


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
    ((W - F + 2P) / S) + 1

    :param in_channels: Number of input channels
    :param in_channels: Number of output channels
    :param kernel_size: Size of the kernel/filter
    :param padding: The size of the zero-padding
    :param stride: Value of stride
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
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

    :param inputDims: The shape of the input image (W, H, C)
    :param numLogits: The number of logits in the last layer of the network
    """

    def __init__(self, inputDims, numLogits):
        super(Actor, self).__init__()

        self.inputDims = inputDims
        self.numLogits = numLogits
        self.numActions = numLogits - 2
        self.reshape = FCView()
        self.actor1 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.actor2 = ConvBlock(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.fc_actor = nn.Linear(np.prod((32,) + self.inputDims[:2]), self.numLogits)

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
        self.fc_critic = nn.Linear(np.prod((32,) + self.inputDims[:2]), 1)

    def forward(self, x):
        x = F.relu(self.critic(x))
        x = self.fc_critic(self.reshape(x))
        return x


class PPO(nn.Module):
    """
    Proximal Policy Optimization algorithm (PPO)

    Paper: https://arxiv.org/abs/1707.06347
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param inputDims: The shape of the input image (W, H, C)
    :param numLogits: The number of logits in the last layer of the network
    """

    def __init__(self, inputDims, numLogits):
        super(PPO, self).__init__()

        self.numResBlocks = 10
        self.inputDims = inputDims

        self.conv = ConvBlock(in_channels=self.inputDims[2], out_channels=256, kernel_size=3,
                padding=1, stride=1)
        for block in range(0, self.numResBlocks):
            setattr(self, "res-block-{}".format(block+1), ResBlock(in_channels=256,
                out_channels=256, kernel_size=3, padding=1, stride=1))

        self.actor = Actor(self.inputDims, numLogits)
        self.critic = Critic(self.inputDims)
        self._initialize_weights()

    def _initialize_weights(self):
        print("Modules: ", self.modules)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        for block in range(0, self.numResBlocks):
            x = getattr(self, "res-block-{}".format(block+1))(x)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

    def pi(self, x):
        x = self.conv(x)
        for block in range(0, self.numResBlocks):
            x = getattr(self, "res-block-{}".format(block+1))(x)
        policy = self.actor(x, True)
        value = self.critic(x)
        return policy, value
