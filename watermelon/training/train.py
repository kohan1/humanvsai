"""
PPO training for the Watermelon AI.

Run pretrain.py first for a behavioural-cloning warm start; this picks up
watermelon_pretrained.zip automatically if present, otherwise starts fresh.

Hyperparameter choices carry over from the Snake and Tetris builds:

  - GPU by default. The "small MLP belongs on CPU" rule does not hold for a
    conv net this size; measured on the 4060 Ti a batch-2048 update is ~19x
    faster on CUDA.
  - ent_coef > 0. Snake v1 ran at SB3's default 0.0 and, combined with a small
    LR and target_kl, pinned the policy so tightly it could not explore out of
    its starting basin.
  - Linear LR anneal on fresh runs; a small fixed LR plus target_kl when
    resuming, because that phase is protecting weights rather than exploring.
  - target_kl is NOT set on fresh runs. It exists to stop a good policy
    drifting; imposed from step zero it just throttles learning.

IF YOU CHANGE THE OBSERVATION OR REWARD IN watermelon_env.py, DELETE the
existing .zip files and retrain from scratch. Resuming a checkpoint across a
spec change silently trains on a different problem.
"""

import os
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from watermelon_env import WatermelonEnv
from policy_config import policy_kwargs

# Overridable so two games can train at once without editing files:
#   TRAIN_DEVICE=cpu  N_ENVS=8  python train.py
# Env stepping is CPU-bound in both games, so running two jobs means splitting
# the cores between them - oversubscribing just makes both slower.
DEVICE = os.environ.get("TRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

N_ENVS = int(os.environ.get("N_ENVS", 16))                  # physics is CPU-bound; leave headroom on 16 cores
TOTAL_TIMESTEPS = 3_000_000  # one "step" is a whole drop-and-settle, not a frame

# Entropy bonus, split by how good the starting policy is.
#
# An entropy bonus rewards being random. On a FRESH run that stops the policy
# committing early to a mediocre habit. On top of a good BC clone it is
# destructive — it drags a confident policy back toward uniform.
#
# Measured here: BC scored 749.50 and fine-tuning it with ent_coef=0.01 drove
# it to 535.37. The same thing had already happened on Snake (47.43 -> 17.50);
# this file was written before that was diagnosed and did not get the fix.
ENT_COEF_FRESH = 0.01
ENT_COEF_RESUME = 0.0
LR_START = 3e-4
LR_END = 1e-5
# Measured, not guessed. At 5e-5 a single epoch moved the policy by KL
# 0.05-0.16 — above target_kl — so EVERY update aborted at step 0. PPO was
# configured for n_epochs=10 and effectively ran one, making big noisy jumps
# and never completing a proper averaged update. Score drifted 716 -> 634.
# Scaling the LR down by ~0.4 should land one epoch under the 0.03 limit so
# updates actually finish.
LR_RESUME = 2e-5

PRETRAINED_PATH = "watermelon_pretrained.zip"
RESUME_PATH = "watermelon_final.zip"
FINAL_PATH = "watermelon_final.zip"   # always this exact name — never rely on
                                      # sorting numbered checkpoint filenames
TENSORBOARD_LOG = "tb_logs/"


def linear_schedule(start: float, end: float):
    """SB3 passes progress_remaining: 1.0 at the start, 0.0 at the end."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def make_env():
    return WatermelonEnv()


class ScoreLoggingCallback(BaseCallback):
    """
    SB3 logs episode reward and length, but nothing from the env's info dict.
    Reward here is a shaped quantity (merge points scaled, minus height
    penalties) and does not map 1:1 to game score, so log the real score
    separately — otherwise you cannot tell whether the agent is actually
    getting better at the game.
    """

    def __init__(self, buffer_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.score_buffer = deque(maxlen=buffer_size)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "score" in info:
                self.score_buffer.append(info["score"])
        return True

    def _on_rollout_end(self) -> None:
        if self.score_buffer:
            self.logger.record("rollout/ep_score_mean", float(np.mean(self.score_buffer)))


def main():
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    start_path = RESUME_PATH if os.path.exists(RESUME_PATH) else PRETRAINED_PATH

    if os.path.exists(start_path):
        print(f"Loading weights from {start_path}")
        model = PPO.load(
            start_path,
            env=vec_env,
            device=DEVICE,
            tensorboard_log=TENSORBOARD_LOG,
            learning_rate=LR_RESUME,
            target_kl=0.03,
            ent_coef=ENT_COEF_RESUME,
        )
        # pretrain.py sets verbose=0 and that is serialised into the .zip,
        # which would otherwise make training look frozen when it is fine.
        model.verbose = 1
    else:
        print("No pretrained weights found — starting from scratch")
        model = PPO(
            "MlpPolicy",   # overridden by features_extractor_class
            vec_env,
            policy_kwargs=policy_kwargs(),
            n_steps=512,
            batch_size=1024,
            n_epochs=10,
            learning_rate=linear_schedule(LR_START, LR_END),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ENT_COEF_FRESH,
            device=DEVICE,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
        )

    callback = CallbackList([
        CheckpointCallback(
            save_freq=max(1, 250_000 // N_ENVS),
            save_path="checkpoints/",
            name_prefix="watermelon_ckpt",
        ),
        ScoreLoggingCallback(),
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    finally:
        # Numbered checkpoints are for resuming/inspection only. This fixed
        # filename is what export.py and everything downstream should load.
        model.save(FINAL_PATH)
        print(f"Saved final model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
