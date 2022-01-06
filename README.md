# tuxkart-ai

### TODO:

- [ ] change all dtypes to float16 or int8
- [ ] not satisfied with the reward function change it
- [ ] consider FPS

instead of creating a mini-batch i'm using vectorized envs and i sample from them

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      --                        --
├─ConvBlock: 1-1                         [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-1                       [1, 256, 3, 200, 300]     20,992
│    └─BatchNorm3d: 2-2                  [1, 256, 3, 200, 300]     512
├─ResBlock: 1-2                          [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-3                       [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-4                  [1, 256, 3, 200, 300]     512
│    └─Conv3d: 2-5                       [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-6                  [1, 256, 3, 200, 300]     512
├─ResBlock: 1-3                          [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-7                       [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-8                  [1, 256, 3, 200, 300]     512
│    └─Conv3d: 2-9                       [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-10                 [1, 256, 3, 200, 300]     512
├─ResBlock: 1-4                          [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-11                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-12                 [1, 256, 3, 200, 300]     512
│    └─Conv3d: 2-13                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-14                 [1, 256, 3, 200, 300]     512
├─ResBlock: 1-5                          [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-15                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-16                 [1, 256, 3, 200, 300]     512
│    └─Conv3d: 2-17                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-18                 [1, 256, 3, 200, 300]     512
├─ResBlock: 1-6                          [1, 256, 3, 200, 300]     --
│    └─Conv3d: 2-19                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-20                 [1, 256, 3, 200, 300]     512
│    └─Conv3d: 2-21                      [1, 256, 3, 200, 300]     1,769,472
│    └─BatchNorm3d: 2-22                 [1, 256, 3, 200, 300]     512
├─ConvBlock: 1-7                         [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-23                      [1, 256, 1, 66, 100]      1,769,728
│    └─BatchNorm3d: 2-24                 [1, 256, 1, 66, 100]      512
├─ResBlock: 1-8                          [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-25                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-26                 [1, 256, 1, 66, 100]      512
│    └─Conv3d: 2-27                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-28                 [1, 256, 1, 66, 100]      512
├─ResBlock: 1-9                          [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-29                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-30                 [1, 256, 1, 66, 100]      512
│    └─Conv3d: 2-31                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-32                 [1, 256, 1, 66, 100]      512
├─ResBlock: 1-10                         [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-33                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-34                 [1, 256, 1, 66, 100]      512
│    └─Conv3d: 2-35                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-36                 [1, 256, 1, 66, 100]      512
├─ResBlock: 1-11                         [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-37                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-38                 [1, 256, 1, 66, 100]      512
│    └─Conv3d: 2-39                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-40                 [1, 256, 1, 66, 100]      512
├─ResBlock: 1-12                         [1, 256, 1, 66, 100]      --
│    └─Conv3d: 2-41                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-42                 [1, 256, 1, 66, 100]      512
│    └─Conv3d: 2-43                      [1, 256, 1, 66, 100]      1,769,472
│    └─BatchNorm3d: 2-44                 [1, 256, 1, 66, 100]      512
├─Actor: 1-13                            [1, 15]                   --
│    └─ConvBlock: 2-45                   [1, 64, 1, 66, 100]       --
│    │    └─Conv3d: 3-1                  [1, 64, 1, 66, 100]       442,432
│    │    └─BatchNorm3d: 3-2             [1, 64, 1, 66, 100]       128
│    └─ConvBlock: 2-46                   [1, 1, 1, 66, 100]        --
│    │    └─Conv3d: 3-3                  [1, 1, 1, 66, 100]        1,729
│    │    └─BatchNorm3d: 3-4             [1, 1, 1, 66, 100]        2
│    └─FCView: 2-47                      [1, 6600]                 --
│    └─Linear: 2-48                      [1, 15]                   99,015
├─Critic: 1-14                           [1, 1]                    --
│    └─ConvBlock: 2-49                   [1, 1, 1, 66, 100]        --
│    │    └─Conv3d: 3-5                  [1, 1, 1, 66, 100]        6,913
│    │    └─BatchNorm3d: 3-6             [1, 1, 1, 66, 100]        2
│    └─FCView: 2-50                      [1, 6600]                 --
│    └─Linear: 2-51                      [1, 1]                    6,601
==========================================================================================
Total params: 37,748,246
Trainable params: 37,748,246
Non-trainable params: 0
Total mult-adds (T): 3.32
==========================================================================================
Input size (MB): 14.40
Forward/backward pass size (MB): 8414.42
Params size (MB): 150.99
Estimated Total Size (MB): 8579.81
==========================================================================================
```

### References

- https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
- https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
- https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- https://youtu.be/WxQfQW48A4A
- https://youtu.be/5P7I-xPq8u8
- https://github.com/colinskow/move37/blob/master/ppo/ppo_train.py
