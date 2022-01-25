import os
import torch
import numpy as np

from tqdm import tqdm, trange
from pathlib import Path
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.utils import STK, make_env, Logger
from src.vae.model import ConvVAE, Encoder, Decoder


def preprocess_grayscale_images(images):
    return images/255.0


def collect_data(num_envs, per_env_sample):
    # TODO: only sample at random steps
    # TODO: use np array instead of list?
    data = []
    acts = np.array([None for _ in range(num_envs)])
    env = SubprocVecEnv([make_env(i, 'hd', { 'difficulty': 3, 'reverse': np.random.choice([True, False]),
        'vae': True }) for i in range(num_envs)], start_method='spawn')
    obs_shape = env.observation_space.shape

    for _ in trange(per_env_sample):
        obs, _, done, _ = env.step(acts)
        data.append(np.array(obs))
        if done.any():
            break

    env.close()
    return np.array(data, dtype=np.float32).reshape(-1, 1, *obs_shape)


@torch.no_grad()
def eval(vae, loss_fn, logger, device, eval_size):
    eval_images = torch.from_numpy(preprocess_grayscale_images(collect_data(1,
        eval_size))).to(device)
    recon_images, mu, logvar = vae(eval_images)

    recon_loss = loss_fn(eval_images, recon_images, reduction='none').sum(tuple(range(1,
        eval_images.dim()))).mean()
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    tot_loss = recon_loss + kl_loss
    logger.log_vae_eval(recon_loss.item(), kl_loss.item(), tot_loss.item())


def main(args):

    if args.device == 'cuda':
        assert torch.cuda.is_available(), "Cuda backend not available."
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    loss_fn_dict = { 'mse': F.mse_loss, 'bce': F.binary_cross_entropy }
    loss_fn = loss_fn_dict[args.loss_fn]
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    env = make_env(id)()
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 2:
        obs_shape += (1, )
    env.close()

    vae = ConvVAE(obs_shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    if args.model_path is not None:
        print(f"loading model from {args.model_path}")
        vae.load_state_dict(torch.load(args.model_path))

    optim = Adam(vae.parameters(), lr=args.lr)
    logger = Logger(SummaryWriter(log_dir=args.log_dir))
    step = 1
    beta = 0
    min_loss = 1e6
    stop_counter = 0

    while True:
        train_data = torch.from_numpy(collect_data(args.num_envs, args.per_env_sample))
        train_data = preprocess_grayscale_images(train_data)
        data_len = len(train_data)

        # TODO: wrap these in functions?
        mini_batch_size = min(args.mini_batch_size, data_len)
        epoch_size = 1 if mini_batch_size == data_len else data_len

        for _ in trange(args.epoch):
            t = tqdm((range(epoch_size)))
            for _ in t:

                optim.zero_grad()
                mini_batch_idx = np.random.randint(0, data_len, mini_batch_size)
                images = train_data[mini_batch_idx].to(args.device)
                recon_images, mu, logvar = vae(images)

                # KL-div = cross entropy - entropy
                # https://www.geeksforgeeks.org/role-of-kl-divergence-in-variational-autoencoders/
                recon_loss = loss_fn(images, recon_images,
                        reduction='none').sum(tuple(range(1, images.dim()))).mean()
                kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
                # https://stats.stackexchange.com/questions/267924/explanation-of-the-free-bits-technique-for-variational-autoencoders
                # https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/vae/vae.py#L76
                # https://github.com/hardmaru/WorldModelsExperiments/issues/8
                # https://arxiv.org/pdf/1606.04934.pdf - C.8 in page 14

                tot_loss = recon_loss + beta * kl_loss
                tot_loss.backward()
                optim.step()
                t.set_description("Loss: {:.2f}".format(tot_loss))

                if step % args.beta_anneal_interval == 0:
                    beta = min(beta + 0.03, 2)
                logger.log_vae_train(recon_loss.item(), kl_loss.item(), tot_loss.item())

        if step % args.eval_interval == 0:
            # i only have 8 gigs of memory
            del train_data
            eval(vae, loss_fn, logger, args.device, args.eval_size)

        step += 1
        if tot_loss < min_loss:
            # TODO: should i add kl tolerance?
            stop_counter = 0
            min_loss = tot_loss
            model_name = f'vae-{args.zdim}-{args.loss_fn}-{step}'
            torch.save(vae.state_dict(), f'{args.save_dir}/{model_name}.pth')
        else:
            stop_counter += 1
            if stop_counter > 5:
                print("No improvement in the last 5 epochs, stopping")
                break


if __name__ == '__main__':
    import argparse
    from os.path import join
    parser = argparse.ArgumentParser()

    parser.add_argument('--kart', type=str, choices=STK.KARTS, default=None)
    parser.add_argument('--track', type=str, choices=STK.TRACKS, default=None)
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')

    # model args
    parser.add_argument('--model_path', type=Path, default=None, help='Load model from path.')
    parser.add_argument('--zdim', type=float, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1337)

    # train args
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--eval_size', type=int, default=512)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--per_env_sample', type=int, default=256)
    parser.add_argument('--mini_batch_size', type=int, default=16)
    parser.add_argument('--beta_anneal_interval', type=int, default=100)
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'bce'], default='bce')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--log_dir', type=Path, default=join(Path(__file__).absolute().parent,
    'tensorboard'), help='Path to the directory in which the tensorboard logs are saved.')
    parser.add_argument('--save_dir', type=Path, default=join(Path(__file__).absolute().parent,
        'models'), help='Path to the directory in which the trained models are saved.')
    args = parser.parse_args()

    main(args)
