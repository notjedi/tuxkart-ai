import torch

from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    """
    https://dylandjian.github.io/world-models/
    https://github.com/pytorch/examples/blob/master/vae/main.py

    :param build_encoder:
    :param build_decoder:
    """

    def __init__(self, obs_shape: tuple, encoder_class, decoder_class, zdim: int):
        super(ConvVAE, self).__init__()
        self.zdim = zdim
        obs_shape += (1,) if len(obs_shape) == 2 else ()
        self.encoder = encoder_class(obs_shape, zdim)
        self.decoder = decoder_class(zdim, self.encoder.latent_shape, obs_shape)
        assert torch.all(torch.tensor(obs_shape) == self.decoder.recons_shape)
        self._init_weights()

    def sample(self, image):
        mu, logvar = self.encoder(image)
        return self.reparameterize(mu, logvar)

    def reconstruct(self, image):
        mu, _ = self.encoder(image)
        return self.decoder(mu)

    def encode(self, image):
        return self.encoder(image)

    def reparameterize(self, mu, logvar):
        """
        for newbies this explains how 0.5 comes in while calculating the `std`:
            std = sqrt(var)
            std = var ^ 1/2
            std = exp(log(var ^ 1/2))       # using: log(a ^ b) = b * log(a)
            std = exp(0.5 * log(var))
        """
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std)
        return mu + (z * logvar)

    def forward(self, image):
        mu, logvar = self.encoder(image)
        latent_repr = self.reparameterize(mu, logvar)
        return self.decoder(latent_repr), mu, logvar

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.uniform_(module.weight, -0.08, 0.08)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss
        # out of all the suggestions, this is the only thread that worked for me
        nn.init.constant_(self.encoder.fc2.weight, 0)


class Encoder(nn.Module):
    def __init__(self, obs_shape, zdim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 128, kernel_size=10, padding=1, stride=4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, padding=2, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=4, padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=3, padding=1, stride=2, bias=False),
        )

        with torch.no_grad():
            x = torch.rand(1, obs_shape[-1], *obs_shape[:-1])
            self.latent_shape = torch.tensor(self.encoder(x).shape[1:])

        self.fc1 = nn.Linear(torch.prod(self.latent_shape), zdim)  # mean
        self.fc2 = nn.Linear(torch.prod(self.latent_shape), zdim)  # logvar

    def forward(self, x):
        x = torch.flatten(self.encoder(x), start_dim=1)
        return self.fc1(x), self.fc2(x)


class Decoder(nn.Module):
    def __init__(self, zdim, latent_shape, obs_shape):
        super(Decoder, self).__init__()
        self.latent_shape = latent_shape
        self.fc1 = nn.Linear(zdim, torch.prod(latent_shape))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_shape[0], 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, obs_shape[-1], kernel_size=10, padding=0, stride=2),
        )
        with torch.no_grad():
            x = torch.rand(1, *latent_shape)
            recons_shape = self.decoder(x).shape[1:]
            self.recons_shape = torch.tensor([recons_shape[1], recons_shape[2], recons_shape[0]])

    def forward(self, x):
        x = self.fc1(x).view(-1, *self.latent_shape)
        return torch.sigmoid(self.decoder(x))
