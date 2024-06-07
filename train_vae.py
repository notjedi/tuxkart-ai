import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
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
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
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


# Custom dataset to handle loading and transforming images
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def generate_images():
    import time

    import pystk
    import numpy as np
    from pystk_gym.common.race import RaceConfig
    from PIL import Image

    config = pystk.GraphicsConfig.hd()
    config.screen_width = 800
    config.screen_height = 600
    pystk.init(config)

    num_players = 2
    config = pystk.RaceConfig()
    config.laps = 1
    config.num_kart = num_players
    config.players[0].kart = np.random.choice(RaceConfig.KARTS)
    config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL

    for _ in range(1, num_players):
        config.players.append(
            pystk.PlayerConfig(
                np.random.choice(RaceConfig.KARTS),
                pystk.PlayerConfig.Controller.AI_CONTROL,
                0,
            )
        )
    config.track = np.random.choice(RaceConfig.TRACKS)
    config.step_size = 0.345

    race = pystk.Race(config)
    race.start()
    race.step()

    state = pystk.WorldState()
    state.update()
    t0 = time.time()
    n = 0

    datas = []
    while all(kart.finish_time <= 0 for kart in state.karts):
        race.step()
        state.update()

        SAMPLE_RATE = 0.5
        for kart_render_data in race.render_data:
            if random.random() < SAMPLE_RATE:
                img = np.array(
                    Image.fromarray(kart_render_data.image).convert("L")
                ) / np.float32(255.0)
                depth = kart_render_data.depth
                instance = kart_render_data.instance.astype(np.float32)
                data = np.dstack([img, depth, instance])
                datas.append(data)
                print(len(datas))
                # instance = kart_render_data.instance & 0xFFFFFF
                # semantic = (kart_render_data.instance >> 24) & 0xFF

        # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time.time() - t0)
        if delta_d > 0:
            time.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
    return datas


def main():
    datas = generate_images()
    print(np.array(datas).shape)
    exit(0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    transform = transforms.Compose(
        [transforms.Resize((540, 960)), transforms.ToTensor()]
    )

    # Assuming you have a list of image file paths
    image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]
    dataset = CustomImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize model, optimizer and loss function
    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, loss = model(images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")


if __name__ == "__main__":
    main()
