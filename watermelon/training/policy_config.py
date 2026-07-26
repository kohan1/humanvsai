"""
Shared policy architecture for Watermelon.

Imported by BOTH pretrain.py and train.py, for the same reason as Snake's
equivalent: two separate NET_ARCH declarations silently drift, and a BC
checkpoint whose architecture disagrees with the RL run simply fails to load.

The board is a 22x30 raster with 2 channels, so the same reasoning as Snake
applies — a flat MLP would have to relearn "these two cells are adjacent" for
every position, while a convolution gets it for free. Scalars (held tier, next
tier, stack height, fruit count) are not spatial and bypass the conv.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from watermelon_env import GRID_CHANNELS, GRID_H, GRID_W, N_SCALARS

FEATURES_DIM = 512
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])


class WatermelonGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)

        self.grid_size = GRID_W * GRID_H * GRID_CHANNELS
        self.n_scalars = N_SCALARS

        expected = self.grid_size + self.n_scalars
        actual = int(observation_space.shape[0])
        if actual != expected:
            raise ValueError(
                f"Observation is {actual} floats but this extractor expects "
                f"{expected} ({GRID_H}x{GRID_W}x{GRID_CHANNELS} grid + "
                f"{N_SCALARS} scalars). watermelon_env._get_obs and "
                "policy_config have drifted apart."
            )

        # One stride-2 layer here (unlike Snake): the board is 22x30 rather than
        # 16x16 and fruit span many cells each, so some downsampling is useful
        # and keeps the flattened size sane.
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, GRID_CHANNELS, GRID_H, GRID_W)).shape[1]

        self.cnn_head = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU())
        self.combine = nn.Sequential(
            nn.Linear(512 + self.n_scalars, features_dim), nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        grid_flat = obs[:, : self.grid_size]
        scalars = obs[:, self.grid_size:]

        # _get_obs builds (H, W, C) then flattens, so unflatten in that order
        # before moving channels first.
        grid = grid_flat.view(-1, GRID_H, GRID_W, GRID_CHANNELS)
        grid = grid.permute(0, 3, 1, 2).contiguous()

        return self.combine(torch.cat([self.cnn_head(self.cnn(grid)), scalars], dim=1))


def policy_kwargs():
    return dict(
        features_extractor_class=WatermelonGridExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=NET_ARCH,
        # SEPARATE extractors for policy and value.
        #
        # SB3 shares one extractor by default. Measured with sharing on:
        # value_loss ~82 against policy_gradient_loss ~1e-4, so the value term
        # dominated the gradient by ~5 orders of magnitude and flowed straight
        # through the shared CNN — rewriting the policy's features. Every PPO
        # update then blew past target_kl and aborted at step 0, and halving
        # the learning rate changed the per-epoch KL not at all, which is what
        # ruled the LR out and pointed here.
        #
        # Cost is a second copy of the conv stack; the benefit is that fitting
        # the critic can no longer damage the policy.
        share_features_extractor=False,
    )
