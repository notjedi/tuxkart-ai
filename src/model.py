from torch.nn import functional as F
from torch import nn


class FCView(nn.Module):

    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        shape = x.data.size(0)
        x = x.view(shape, -1)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv(x.float()))
        return self.batch_norm(x)


class ResBlock(nn.Module):

    def __init__(self, channels,  kernel_size, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
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


class OutBlock(nn.Module):

    def __init__(self, numActions):
        super(OutBlock, self).__init__()

        self.policy_depth = 3
        self.numActions = numActions
        self.reshape = FCView()

        for block in range(0, self.policy_depth):
            setattr(self, "conv-block-policy-{}".format(block+1), ConvBlock(256, 2, 1, 0, 1))
        self.fc_policy = nn.Linear(2 * 8 * 8, self.numActions)

        self.conv_block_value = ConvBlock(256, 2, 3, 1, 1)
        self.fc_value = nn.Linear(2 * 8 * 8, 1)

    def forward(self, x):
        for block in range(0, self.policy_depth):
            x = getattr(self, "conv-block-policy-{}".format(block+1))(x)
        policy = self.reshape(x)
        policy = self.fc_policy(policy)

        value = self.reshape(self.conv_block_value(x))
        value = self.fc_value(F.relu(value))
        return policy, value.view(-1)


class PPO(nn.Module):

    def __init__(self, numInputChannels, numActions):
        super(PPO, self).__init__()

        self.numResBlocks = 10
        self.numInputs = numInputChannels
        self.conv = ConvBlock(self.numInputs, 256, 3, 1, 1)
        for block in range(0, self.numResBlocks):
            setattr(self, "res-block-{}".format(block+1), ResBlock(256, 3, 1, 1))
        self.out_block = OutBlock(numActions)
        self._initialize_weights()

    def _initialize_weights(self):
        # TODO: test orthogonal init
        # TODO: is the resblock modules included here?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
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
        policy, value = self.out_block(x)
        return policy, value
