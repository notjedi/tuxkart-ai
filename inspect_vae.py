import torch
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.utils import make_env
from src.vae.model import ConvVAE, Encoder, Decoder


@torch.no_grad()
def inspect(vae, device, sample_size, quality):

    env = SubprocVecEnv(
        [
            make_env(
                i,
                quality,
                {'difficulty': 3, 'reverse': np.random.choice([True, False]), 'vae': True},
            )
            for i in range(1)
        ],
        start_method='spawn',
    )
    env.reset()

    step, obs_shape = 0, env.observation_space.shape
    acts, images = np.array([None]), np.empty((sample_size, 1) + obs_shape, dtype=np.float32)

    pbar = tqdm(total=sample_size, position=0)
    while step < sample_size:
        obs, _, done, _ = env.step(acts)
        if np.random.rand() < 0.1:
            images[step] = np.array(obs)
            pbar.update(1)
            step += 1
        if done.any():
            break

    env.close()
    recon_images = vae.reconstruct(torch.from_numpy(images).to(device)).cpu().numpy()
    return images.reshape(-1, *obs_shape), recon_images.reshape(-1, *obs_shape)


def plot_continuous(images):
    img_shape = images.shape
    images = images.reshape(img_shape[0] // 2, img_shape[1] * 2, img_shape[2])
    images = np.concatenate(images, axis=1)
    plt.imshow(images, cmap='gray')
    plt.axis('off')
    plt.show()


def plot(images):
    subplot_size, i = images.shape[0] // 2, 0
    fig, axs = plt.subplots(2, subplot_size, figsize=(40, 40))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.show()


def main(args):

    if args.device == 'cuda':
        assert torch.cuda.is_available(), "Cuda backend not available."

    env = make_env(id)()
    obs_shape = env.observation_space.shape
    obs_shape += (1,) if len(obs_shape) == 2 else ()
    env.close()

    vae = ConvVAE(obs_shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    if args.model_path is not None:
        print(f"loading model from {args.model_path}")
        vae.load_state_dict(torch.load(args.model_path))

    images, recon_images = inspect(vae, args.device, args.sample_size, args.graphic)
    plot_continuous(images)
    plot_continuous(recon_images)
    # plot(images)
    # plot(recon_images)


if __name__ == '__main__':
    import argparse
    from os.path import join

    parser = argparse.ArgumentParser()

    parser.add_argument('--zdim', type=float, default=256)
    parser.add_argument('--sample_size', type=int, default=8)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')
    parser.add_argument('--model_path', type=Path, default=None, help='Load model from path.')
    args = parser.parse_args()

    main(args)
