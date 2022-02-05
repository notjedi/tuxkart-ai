# tuxkart-ai


LSTM model arch: (for RL agent)

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      --                        --
├─LSTM: 1-1                              [5, 8, 256]               --
│    └─LSTM: 2-1                         [5, 8, 256]               1,052,672
├─Actor: 1-2                             [8, 13]                   --
│    └─Sequential: 2-2                   [8, 13]                   --
│    │    └─Linear: 3-1                  [8, 128]                  32,896
│    │    └─Tanh: 3-2                    [8, 128]                  --
│    │    └─Linear: 3-3                  [8, 13]                   1,677
├─Critic: 1-3                            [8, 1]                    --
│    └─Sequential: 2-3                   [8, 1]                    --
│    │    └─Linear: 3-4                  [8, 128]                  32,896
│    │    └─Tanh: 3-5                    [8, 128]                  --
│    │    └─Linear: 3-6                  [8, 1]                    129
==========================================================================================
Total params: 1,120,270
Trainable params: 1,120,270
Non-trainable params: 0
Total mult-adds (M): 42.65
==========================================================================================
Input size (MB): 0.04
Forward/backward pass size (MB): 0.10
Params size (MB): 4.48
Estimated Total Size (MB): 4.62
==========================================================================================
```

### VAE model arch: (for representation learning)

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ConvVAE                                  --                        --
├─Encoder: 1-1                           [8, 128]                  --
│    └─Sequential: 2-1                   [8, 1, 75, 50]            --
│    │    └─Conv2d: 3-1                  [8, 128, 149, 99]         12,800
│    │    └─ReLU: 3-2                    [8, 128, 149, 99]         --
│    │    └─BatchNorm2d: 3-3             [8, 128, 149, 99]         256
│    │    └─Conv2d: 3-4                  [8, 256, 150, 100]        524,288
│    │    └─ReLU: 3-5                    [8, 256, 150, 100]        --
│    │    └─BatchNorm2d: 3-6             [8, 256, 150, 100]        512
│    │    └─Conv2d: 3-7                  [8, 128, 149, 99]         524,288
│    │    └─ReLU: 3-8                    [8, 128, 149, 99]         --
│    │    └─BatchNorm2d: 3-9             [8, 128, 149, 99]         256
│    │    └─Conv2d: 3-10                 [8, 1, 75, 50]            1,152
│    └─Linear: 2-2                       [8, 128]                  480,128
│    └─Linear: 2-3                       [8, 128]                  480,128
├─Decoder: 1-2                           [8, 1, 600, 400]          --
│    └─Linear: 2-4                       [8, 3750]                 483,750
│    └─Sequential: 2-5                   [8, 1, 600, 400]          --
│    │    └─ConvTranspose2d: 3-11        [8, 128, 149, 99]         1,280
│    │    └─ReLU: 3-12                   [8, 128, 149, 99]         --
│    │    └─ConvTranspose2d: 3-13        [8, 256, 149, 99]         295,168
│    │    └─ReLU: 3-14                   [8, 256, 149, 99]         --
│    │    └─ConvTranspose2d: 3-15        [8, 128, 296, 196]        524,416
│    │    └─ReLU: 3-16                   [8, 128, 296, 196]        --
│    │    └─ConvTranspose2d: 3-17        [8, 1, 600, 400]          12,801
==========================================================================================
Total params: 3,341,223
Trainable params: 3,341,223
Non-trainable params: 0
Total mult-adds (G): 429.30
==========================================================================================
Input size (MB): 7.68
Forward/backward pass size (MB): 1828.52
Params size (MB): 13.36
Estimated Total Size (MB): 1849.57
==========================================================================================
```

### TODO:

- [ ] keep a moving finish line?
- [ ] clean up encoding infos
- [ ] use stackedvec - stable_baselines_3
- [ ] train VAE on RGB images instead of using grayscale image as model input
- [ ] another regularization step (maybe) would be to take `x` number of actions randomly, like not
      sample from the dist but totally random steps and have the model recover from that - so as to see
      how the model recovers from that state?

### Observations:

* Half Precision doesn't work
* Training on RGB image absolutely does nothing

### References

- https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
- https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
- https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- https://youtu.be/WxQfQW48A4A
- https://youtu.be/5P7I-xPq8u8
- https://github.com/colinskow/move37/blob/master/ppo/ppo_train.py
- https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
- https://math.stackexchange.com/questions/1905533/find-perpendicular-distance-from-point-to-line-in-3d
- https://onlinemschool.com/math/library/analytic_geometry/p_line/
- https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
- https://github.com/philkr/pystk/blob/master/pystk_cpp/state.cpp
- https://deeplearning.lipingyang.org/2016/12/29/model-free-vs-model-based-methods/
- https://youtu.be/V8f8ueBc9sY
- https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
- https://ai.stackexchange.com/questions/7685/why-is-the-log-probability-replaced-with-the-importance-sampling-in-the-loss-fun
- https://costa.sh/blog-understanding-why-there-isn't-a-log-probability-in-trpo-and-ppo's-objective.html
- https://www.reddit.com/r/reinforcementlearning/comments/s18hjr/help_understanding_ppo_algorithm/
- https://github.com/dalmia/David-Silver-Reinforcement-learning


- https://stats.stackexchange.com/questions/394582/why-is-binary-cross-entropy-or-log-loss-used-in-autoencoders-for-non-binary-da
- https://stats.stackexchange.com/questions/370179/why-binary-crossentropy-can-be-used-as-the-loss-function-in-autoencoders
- https://stackoverflow.com/questions/52441877/how-does-binary-cross-entropy-loss-work-on-autoencoders/52443301#52443301
- https://towardsdatascience.com/what-a-disentangled-net-we-weave-representation-learning-in-vaes-pt-1-9e5dbc205bd1
- https://towardsdatascience.com/with-great-power-comes-poor-latent-codes-representation-learning-in-vaes-pt-2-57403690e92b
