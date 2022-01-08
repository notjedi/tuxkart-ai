import torch
import numpy as np

from collections import deque

def eval(env, model, args):

    images = env.get_images()
    images = deque([np.zeros_like(images) for _ in range(args.num_frames)], maxlen=args.num_frames)
    to_numpy = lambda x: x.to(device='cpu').numpy()
    tot_reward = 0

    with torch.no_grad():
        for i in range(args.eval_steps):
            images.append(np.array(env.get_images()))
            obs = torch.from_numpy(np.transpose(np.array(images), (1, 2, 0, 3, 4))).to(args.device)

            dist, value = model(obs)
            action = dist.mode()
            log_prob = dist.log_prob(action)
            _, reward, done, _ = env.step(to_numpy(action))
            tot_reward += reward

            if done.any():
                break

    return tot_reward


if __name__ == '__main__':
    # TOOD: standalone testing
    pass
