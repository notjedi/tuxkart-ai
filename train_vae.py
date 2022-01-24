import torch
import numpy as np

from tqdm import trange
from pathlib import Path
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from src.utils import STK, make_env, Logger
from src.vae.model import ConvVAE, Encoder, Decoder


def preprocess_grayscale_image(images):
    return images/255.0


def collect_data(num_envs, buffer_size):
    data = []
    env = SubprocVecEnv([make_env(i, {'difficulty': 3}) for i in
        range(num_envs)], start_method='spawn')

    for _ in range(buffer_size):
        obs, _, done, _ = env.step()
        data.append(np.array(obs))

        if done.any():
            break

    return np.array(data, dtype=np.float32)


@torch.no_grad
def validate(num_envs, eval_size):
    eval_data = collect_data(num_envs, eval_size)
    # TODO
    pass


def main(args):

    # TODO: seed
    # TODO: lr scheduler?
    # TODO: beta annealing?
    # TODO: batch size args
    # TODO: kl div args
    # TODO: add reverse in env for training
    # TODO: load from model path
    if device == 'cuda':
        assert torch.cuda.is_available()
    loss_fn_dict = { 'mse': F.mse_loss, 'bce': F.binary_cross_entropy }
    loss_fn = loss_fn_dict[args.loss_fn]

    vae = ConvVAE(env.observation_space.shape, Encoder, Decoder, args.zdim)
    vae.to(args.device)
    optim = Adam(vae.parameters(), lr=args.lr)
    # TODO: init writer properly
    writer = SummaryWriter()
    logger = Logger()

    step = 1
    min_loss = 1e6
    stop_counter = 0

    while True:
        train_data = collect_data(args.num_envs, args.buffer_size)

        data_len = len(train_data)
        mini_batch_size = min(args.mini_batch_size, data_len)
        epoch_size = 2 if mini_batch_size == data_len else data_len

        for _ in range(args.epoch):
            for _ in range(epoch_size):

                mini_batch_idx = np.random.randint(0, data_len, mini_batch_size)
                images = train_data[mini_batch_idx]
                recons_images = vae(images)

                # TODO: check how to calc kl_div
                # TODO: add beta value in args
                loss = loss_fn(images, recons_images, reduction='sum') / args.batch_size
                kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / args.batch_size
                tot_loss = loss + args.beta * kl_div
                tot_loss.backward()

                self.logger.log_vae_train('train_vae/loss', loss)
                self.logger.log_vae_train('train_vae/kl_div', kl_div)
                self.logger.log_vae_train('train_vae/tot_loss', tot_loss)

        if step % eval_interval == 0:
            validate(args.num_envs, args.eval_size)

        step += 1
        if tot_loss < min_loss:
            # TODO: save_model and give valid name for model
            stop_counter = 0
            min_loss = tot_loss
        else:
            stop_counter += 1
            if stop_counter > 5:
                print("No improvement in the last 5 epochs, stopping")
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--kart', type=str, choices=STK.KARTS, default=None)
    parser.add_argument('--track', type=str, choices=STK.TRACKS, default=None)
    parser.add_argument('--graphic', type=str, choices=['hd', 'ld', 'sd'], default='hd')

    # model args
    parser.add_argument('--model_path', type=Path, default=None, help='Load model from path.')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=1337)

    # train args
    parser.add_argument('--num_envs', type=int, default=2)
    parser.add_argument('--eval_size', type=int, default=512)
    parser.add_argument('--buffer_size', type=int, default=1024)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--mini_batch_size', type=int, default=64)
    # TODO: mse or bce?
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'bce'], default='bce')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--log_dir', type=Path, default=join(Path(__file__).absolute().parent,
    'tensorboard'), help='Path to the directory in which the tensorboard logs are saved.')
    parser.add_argument('--save_dir', type=Path, default=join(Path(__file__).absolute().parent,
        'models'), help='Path to the directory in which the trained models are saved.')
    args = parser.parse_args()

    main(args)
