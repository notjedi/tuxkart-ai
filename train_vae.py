from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import Logger, make_env
from src.vae.model import ConvVAE, Decoder, Encoder


def preprocess_grayscale_images(images):
    return images / 255.0


def collect_data(num_envs, per_env_sample, quality):
    acts = np.array([None for _ in range(num_envs)])
    env = SubprocVecEnv(
        [
            make_env(
                i,
                quality,
                {
                    "difficulty": 3,
                    "reverse": np.random.choice([True, False]),
                    "vae": True,
                },
            )
            for i in range(num_envs)
        ],
        start_method="spawn",
    )
    env.reset()
    obs_shape, sample_prob = env.observation_space.shape, np.clip(
        np.random.rand(), 0.2, 0.5
    )
    data = np.zeros((per_env_sample, num_envs) + obs_shape, dtype=np.float32)

    pbar = tqdm(total=per_env_sample, position=1, leave=False)
    while pbar.n < per_env_sample:
        obs, _, done, _ = env.step(acts)
        if np.random.rand() < sample_prob:
            data[pbar.n] = np.array(obs)
            pbar.update(1)
        if done.any():
            break

    env.close()
    return data[: pbar.n].reshape(-1, 1, *obs_shape)


@torch.no_grad()
def eval(vae, loss_fn, logger, beta, device, eval_size, quality):
    eval_images = torch.from_numpy(
        preprocess_grayscale_images(collect_data(1, eval_size, quality))
    ).to(device)
    recon_images, mu, logvar = vae(eval_images)

    recon_loss = loss_fn(eval_images, recon_images, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    tot_loss = recon_loss + beta * kl_loss
    logger.log_vae_eval(
        recon_loss.item(),
        kl_loss.item(),
        tot_loss.item(),
        eval_images.cpu().numpy(),
        recon_images.cpu().numpy(),
        beta,
    )


def main(args):

    if args.device == "cuda":
        assert torch.cuda.is_available(), "Cuda backend not available."
    # torch.autograd.set_detect_anomaly(True) # uncomment this line if you get NaN's
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    loss_fn_dict = {"mse": F.mse_loss, "bce": F.binary_cross_entropy}
    loss_fn = loss_fn_dict[args.loss_fn]
    args.save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(id)()
    obs_shape = env.observation_space.shape
    obs_shape += (1,) if len(obs_shape) == 2 else ()
    env.close()

    # https://arxiv.org/abs/2004.14990
    # https://pytorch.org/vision/master/transforms.html
    transform = T.RandomApply(
        transforms=[
            T.RandomRotation(degrees=(0, 360)),
            T.RandomHorizontalFlip(),
            T.RandomPerspective(),
            T.RandomErasing(),
        ]
    )
    transform = T.Compose([transform, T.Resize(size=obs_shape[:-1])])

    vae = ConvVAE(obs_shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    if args.model_path is not None:
        print(f"loading model from {args.model_path}")
        vae.load_state_dict(torch.load(args.model_path))

    optim = Adam(vae.parameters(), lr=args.lr)
    tensorboard_file_name = args.log_dir.joinpath(f"vae/{args.zdim}-{args.loss_fn}/")
    logger = Logger(SummaryWriter(tensorboard_file_name, flush_secs=60))
    beta = 0
    min_loss = float("inf")

    g_bar = tqdm(total=1e3, position=0)
    while g_bar.n < g_bar.total:
        try:
            train_data = collect_data(args.num_envs, args.per_env_sample, args.graphic)
            train_data = torch.from_numpy(preprocess_grayscale_images(train_data))
            train_data.requires_grad = False
            data_len = len(train_data)
        except (EOFError, ConnectionResetError):
            continue

        mini_batch_size = min(args.mini_batch_size, data_len)
        epoch_size = data_len // (args.num_envs * 2)
        epoch_loss = 0

        t = tqdm((range(epoch_size)), position=1, leave=False)
        for _ in t:

            mini_batch_idx = np.random.randint(0, data_len, mini_batch_size)
            images = train_data[mini_batch_idx].to(args.device)
            recon_images, mu, logvar = vae(images)

            # KL-div = cross entropy - entropy
            # https://chrisorm.github.io/VAE-pyt.html
            recon_loss = loss_fn(images, recon_images, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            if torch.isnan(kl_loss) or torch.isnan(recon_loss):
                print("Loss is NaN, stopping...")
                print(f"Recon-loss: {recon_loss.item()}, KL-loss: {kl_loss.item()}")
                return
            # https://stats.stackexchange.com/questions/267924/explanation-of-the-free-bits-technique-for-variational-autoencoders
            # https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py#L76
            # https://github.com/hardmaru/WorldModelsExperiments/issues/8
            # https://arxiv.org/pdf/1606.04934.pdf - C.8 in page 14

            optim.zero_grad()
            tot_loss = recon_loss + beta * kl_loss
            tot_loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), args.clip)
            optim.step()

            epoch_loss += tot_loss.detach().cpu().numpy()
            logger.log_vae_train(
                recon_loss.item(), kl_loss.item(), tot_loss.item(), beta
            )
            t.set_description(
                f"Recon-loss: {recon_loss:.4f}, KL-loss: {beta*kl_loss:.4f}"
            )
            del tot_loss, recon_loss, kl_loss, images, recon_images, mu, logvar

        del train_data
        torch.cuda.empty_cache()
        if g_bar.n % args.beta_anneal_interval == 0:
            beta = min(beta + 0.03, 4)

        if g_bar.n % args.eval_interval == 0:
            try:
                eval(
                    vae,
                    loss_fn,
                    logger,
                    beta,
                    args.device,
                    args.eval_size,
                    args.graphic,
                )
                model_name = (
                    f"vae-{args.zdim}-{args.loss_fn}-{g_bar.n}-beta-{beta:.2f}-regular"
                )
                torch.save(vae.state_dict(), f"{args.save_dir}/{model_name}.pth")
            except (EOFError, ConnectionResetError):
                pass

        g_bar.update(1)
        if (epoch_loss / epoch_size) < min_loss:
            min_loss = epoch_loss / epoch_size
            model_name = f"vae-{args.zdim}-{args.loss_fn}-{g_bar.n}-beta-{beta:.2f}-min"
            torch.save(vae.state_dict(), f"{args.save_dir}/{model_name}.pth")


if __name__ == "__main__":
    import argparse
    from os.path import join

    parser = argparse.ArgumentParser()

    parser.add_argument("--graphic", type=str, choices=["hd", "ld", "sd"], default="hd")

    # model args
    parser.add_argument(
        "--model_path", type=Path, default=None, help="Load model from path."
    )
    parser.add_argument("--zdim", type=float, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)

    # train args
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--num_envs", type=int, default=4)
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
        default=join(Path(__file__).absolute().parent, "tensorboard"),
        help="Path to the directory in which the tensorboard logs are saved.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=join(Path(__file__).absolute().parent, "models"),
        help="Path to the directory in which the trained models are saved.",
    )
    args = parser.parse_args()

    main(args)
