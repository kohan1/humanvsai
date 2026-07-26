"""
Shared policy architecture for Snake.

Imported by BOTH pretrain.py and train.py. They previously each declared their
own NET_ARCH, which is exactly the kind of duplication that silently drifts —
and a BC checkpoint whose architecture disagrees with the RL run simply fails
to load. One definition, two importers.

Why a CNN
---------
v1 flattened a 16x16x3 one-hot grid straight into an MLP and plateaued at 37.5
average, losing to the 51.96 of the flood-fill heuristic it was cloned from. A
flat MLP has no notion that cell (3,4) is adjacent to (3,5); every spatial
relationship has to be memorised per-position from scratch. A convolution gets
that for free and shares weights across the board, so "food is two cells left
of my head" generalises everywhere instead of being relearned 256 times.

The observation stays a flat vector (simpler ONNX export and JS mirroring);
this extractor splits it back apart: the leading cells*GRID_CHANNELS entries
reshape to (C, H, W) for the conv stack, and the trailing scalars — which are
not spatial and would be meaningless to convolve — bypass it and are
concatenated to the conv output.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snake_env import GRID_CHANNELS, N_SCALARS, TILE_COUNT

FEATURES_DIM = 512

# Sits on top of the extractor output, so it can be modest — the extractor is
# doing the representational work.
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])


class SnakeGridExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = FEATURES_DIM,
        tile_count: int = TILE_COUNT,
    ):
        super().__init__(observation_space, features_dim)

        self.tile_count = tile_count
        self.grid_channels = GRID_CHANNELS
        self.grid_size = tile_count * tile_count * GRID_CHANNELS
        self.n_scalars = N_SCALARS

        expected = self.grid_size + self.n_scalars
        actual = int(observation_space.shape[0])
        if actual != expected:
            raise ValueError(
                f"Observation is {actual} floats but this extractor expects "
                f"{expected} ({tile_count}x{tile_count}x{GRID_CHANNELS} grid + "
                f"{N_SCALARS} scalars). snake_env._get_obs and policy_config "
                "have drifted apart."
            )

        # padding=1 keeps the board at full resolution through the stack: at
        # 16x16 there is no spatial redundancy worth pooling away, and the
        # exact cell a wall or tail sits in is the whole point.
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, GRID_CHANNELS, tile_count, tile_count)
            cnn_out = self.cnn(dummy).shape[1]

        self.cnn_head = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU())
        self.combine = nn.Sequential(
            nn.Linear(512 + self.n_scalars, features_dim), nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        grid_flat = obs[:, : self.grid_size]
        scalars = obs[:, self.grid_size:]

        # _get_obs builds the grid as (H, W, C) and flattens, so unflatten in
        # that order and then move channels first for the conv.
        grid = grid_flat.view(-1, self.tile_count, self.tile_count, self.grid_channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()

        return self.combine(torch.cat([self.cnn_head(self.cnn(grid)), scalars], dim=1))


def policy_kwargs():
    """Identical policy_kwargs for pretrain.py and train.py."""
    return dict(
        features_extractor_class=SnakeGridExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=NET_ARCH,
    )
