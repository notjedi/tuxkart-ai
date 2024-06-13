import argparse
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pystk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from pystk_gym.common.race import RaceConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

SAMPLE_RATE = 0.9


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=2048):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 4, stride=2, padding=1),
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z_e = self.encoder(x)
        return z_e

    def decode(self, z_q):
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x):
        z_e = self.encode(x)
        z_e_flattened = (
            z_e.view(z_e.shape[0], z_e.shape[1], -1).permute(0, 2, 1).contiguous()
        )
        distances = (
            (z_e_flattened**2).sum(dim=2, keepdim=True)
            + (self.codebook.weight**2).sum(dim=1)
            - 2 * (z_e_flattened @ self.codebook.weight.t())
        )
        encoding_indices = distances.argmin(dim=2)
        z_q = (
            self.codebook(encoding_indices)
            .permute(0, 2, 1)
            .contiguous()
            .view(z_e.shape)
        )
        x_recon = self.decode(z_q)

        commitment_loss = 0.25 * torch.mean((z_e.detach() - z_q) ** 2)
        quantization_loss = torch.mean((z_e - z_q.detach()) ** 2)
        loss = commitment_loss + quantization_loss

        return x_recon, loss


class CustomImageDataset(Dataset):
    def __init__(self, image_datas, transform=None):
        self.image_datas = image_datas
        self.transform = transform

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        # TODO: process and split the data here
        image = self.image_datas[idx]
        if self.transform:
            image = self.transform(image)
        return image


def generate_images(
    graphic_config: pystk.GraphicsConfig,
    race_config: pystk.RaceConfig,
    sample_rate: float,
) -> npt.NDArray[np.float32]:
    datas = []
    total_samples = 64
    race, state, steps, t0 = None, None, 0, 0
    pbar = tqdm(total=total_samples)

    while pbar.n < total_samples:
        if (race is None or state is None) or any(
            kart.finish_time > 0 for kart in state.karts
        ):
            if race is not None:
                race.stop()
                del race
                pystk.clean()

            pystk.init(graphic_config)
            race = pystk.Race(race_config)
            race.start()
            race.step()

            state = pystk.WorldState()
            state.update()
            t0 = time.time()
            steps = 0

        race.step()
        state.update()

        for kart_render_data in race.render_data:
            if random.random() < sample_rate:
                img = np.array(
                    Image.fromarray(kart_render_data.image).convert("L")
                ) / np.float32(255.0)
                depth = kart_render_data.depth
                instance = (kart_render_data.instance & 0xFFFFFF).astype(np.float32)
                # semantic = (kart_render_data.instance >> 24) & 0xFF
                data = np.dstack((img, depth, instance))
                datas.append(data)
                pbar.update(1)

        steps += 1
        delta_d = steps * race_config.step_size - (time.time() - t0)
        if delta_d > 0:
            time.sleep(delta_d)

    if race is not None:
        race.stop()
        del race
        pystk.clean()
    return np.array(datas)


def get_pystk_configs() -> Tuple[pystk.GraphicsConfig, pystk.RaceConfig]:
    graphic_config = pystk.GraphicsConfig.hd()
    graphic_config.screen_width = 960
    graphic_config.screen_height = 540

    num_players = 5
    race_config = pystk.RaceConfig()
    race_config.laps = 1
    race_config.num_kart = num_players
    race_config.players[0].kart = np.random.choice(RaceConfig.KARTS)
    race_config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL

    for _ in range(1, num_players):
        race_config.players.append(
            pystk.PlayerConfig(
                np.random.choice(RaceConfig.KARTS),
                pystk.PlayerConfig.Controller.AI_CONTROL,
                0,
            )
        )
    race_config.track = np.random.choice(RaceConfig.TRACKS)
    race_config.step_size = 0.345

    return (graphic_config, race_config)


def main(args):
    graphic_config, race_config = get_pystk_configs()
    datas = generate_images(graphic_config, race_config, SAMPLE_RATE)
    print(datas.shape)

    device = args.device
    if device == "cuda":
        assert torch.cuda.is_available(), "cuda is not is_available"

    # TODO: should i mmap the data and read it later?
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomImageDataset(datas, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    tensorboard_file_name = args.log_dir.joinpath(f"vae/{args.zdim}-{args.loss_fn}/")
    logger = SummaryWriter(tensorboard_file_name, flush_secs=60)

    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for images in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, loss = model(images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)
        # model.eval()
        # decoded = model.decode(model.encode(images))
        # Image.fromarray(decoded.cpu().detach().numpy()[0].astype(np.uint8).transpose(1, 2, 0) * 255).show()
        # torchvision.transforms.functional.to_pil_image(decoded.cpu().detach().numpy()[0].astype(np.uint8)).show()
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_players", type=int, default=5)

    # model args
    parser.add_argument(
        "--model_path", type=Path, default=None, help="Load model from path."
    )
    parser.add_argument("--zdim", type=float, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)

    # train args
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--eval_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--per_env_sample", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--beta_anneal_interval", type=int, default=15)
    parser.add_argument("--loss_fn", type=str, choices=["mse", "bce"], default="bce")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=os.path.join(Path(__file__).absolute().parent, "tensorboard"),
        help="Path to the directory in which the tensorboard logs are saved.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=os.path.join(Path(__file__).absolute().parent, "models"),
        help="Path to the directory in which the trained models are saved.",
    )
    args = parser.parse_args()

    main(args)
