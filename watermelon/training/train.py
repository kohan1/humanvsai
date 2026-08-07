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
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 4_000_000))  # one "step" is a whole drop-and-settle, not a frame

# How often BestScoreCallback checks the policy, and over how many fixed-seed
# episodes. 15 episodes costs ~1500 drops against 250k of training, so well
# under 1% overhead, and it is the only thing standing between a mid-run peak
# and losing it.
EVAL_FREQ = int(os.environ.get("EVAL_FREQ", 250_000))
# 40, not 15. Measured across 316 evaluations of the last run, the 15-episode
# score ranged 751 to 1136 with a standard deviation of 68 — so the callback
# was ranking models on differences far smaller than its own noise, and twice
# saved a "new best" that was actually WORSE than the model it started from
# (1015.80 and 996.78 over 60 seeds, against 1042.38). Selection is worthless
# until the measurement is tighter than the effect.
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", 40))

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
LR_RESUME = float(os.environ.get("LR_RESUME", 4e-5))

# Trust region and update shape on the resume path.
#
# The 2e-5 / 0.03 pairing above was tuned while share_features_extractor was
# still True — the critic's gradient was flowing through the policy's CNN and
# blowing the KL up, and the LR was cut to compensate. That bug is fixed, so
# the compensation is now just a brake: the 4M-step run that produced 936.70
# early-stopped on 49 of 49 iterations, completing roughly one epoch of ten.
#
# Raising both lets updates actually finish. What makes this safe to try is
# BestScoreCallback: a run that overshoots and regresses no longer costs
# anything, because the best checkpoint along the way is kept. Without that
# safety net these numbers would be reckless; with it they are a cheap
# experiment.
# The discount, and the single most important number for "survive a long time".
#
# gamma=0.99 gives an effective horizon of about 1/(1-gamma) = 100 drops. The
# old policy died at 113. So it could not value surviving past roughly the
# point it already died — the objective was invisible beyond its own lifetime.
# 0.997 moved the horizon to ~330 drops.
#
# Then the diameter ladder was compressed and the untrained heuristic teacher
# started living ~370 drops, which puts the horizon BEHIND the episode again —
# the same failure as before, just at a larger scale. 0.999 gives ~1000 drops,
# keeping the horizon roughly 3x the current episode length, which is the ratio
# 0.997 had when episodes were ~110.
GAMMA = float(os.environ.get("GAMMA", 0.999))

TARGET_KL = float(os.environ.get("TARGET_KL", 0.05))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2048))   # bigger batch, GPU-friendly

PRETRAINED_PATH = os.environ.get("PRETRAINED_PATH", "watermelon_pretrained.zip")
RESUME_PATH = os.environ.get("RESUME_PATH", "watermelon_final.zip")

# Overridable, because the default OVERWRITES the model being resumed from.
# watermelon_final.zip is both RESUME_PATH and FINAL_PATH, so a run that ends
# worse than it started silently replaces the better model with the worse one —
# which is exactly what the reward-v2 run did, dropping the shipped 1032.43 to
# its own 931.97 ending. Point a long run somewhere else and copy it in
# deliberately once it has been evaluated.
FINAL_PATH = os.environ.get("FINAL_PATH", "watermelon_final.zip")

# One directory per run. CheckpointCallback names its saves
# watermelon_ckpt_<steps>_steps.zip, which collides across runs: the 100M
# attempt and the reward-v2 run both wrote into checkpoints/ with overlapping
# step ranges, so the later silently overwrote the earlier one's files.
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/")

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


class BestScoreCallback(BaseCallback):
    """
    Periodically score the policy the way install_model.sh will, and keep a
    copy of the best one seen.

    Why this exists: train.py used to save only the FINAL model, so a run that
    peaked in the middle and drifted down threw the peak away. That is not
    hypothetical here — three consecutive Watermelon runs (614.40, 872.00,
    893.03) were rejected by the install guard for ending worse than the model
    already shipped, and any peak they passed through was lost with them.

    Evaluation uses deterministic=True and a FIXED set of seeds, so every
    check faces exactly the same episodes. Scores swing hugely between seeds
    (433 to 1426 in one 30-episode run), and without fixed seeds "best" would
    mostly select for a lucky draw rather than a better policy.
    """

    def __init__(self, eval_freq: int, n_episodes: int = 15,
                 save_path: str = "watermelon_best.zip", verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.save_path = save_path
        self.best_score = float("-inf")
        # Evaluate immediately rather than after the first eval_freq steps, so
        # the STARTING model's score becomes the bar. Otherwise a resume that
        # degrades early would save the first thing it measured — something
        # worse than the model it began from — and call it "best".
        self._next_eval = 0

    def _evaluate(self):
        """Returns (drops survived, score), both averaged over the fixed seeds.

        SELECTION IS ON DROPS, NOT SCORE. The objective changed: the env now
        pays per drop survived rather than per point merged, so ranking
        checkpoints by score would optimise one thing and select for another.
        Score is still measured and logged, because it is what the site
        displays and what install_model.sh gates on.
        """
        env = WatermelonEnv()
        drops = 0
        score = 0
        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=ep)
            done, info = False, {"score": 0}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            drops += env.drops
            score += info["score"]
        return drops / self.n_episodes, score / self.n_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval = self.num_timesteps + self.eval_freq

        drops, score = self._evaluate()
        self.logger.record("eval/score", score)
        self.logger.record("eval/drops", drops)

        if drops > self.best_score:
            self.best_score = drops
            self.model.save(self.save_path)
            if self.verbose:
                print(f"\n[best] new best {drops:.1f} drops (score {score:.0f}) "
                      f"at {self.num_timesteps} steps -> {self.save_path}",
                      flush=True)
        elif self.verbose:
            print(f"\n[best] {drops:.1f} drops (score {score:.0f}) at "
                  f"{self.num_timesteps} steps "
                  f"(best is still {self.best_score:.1f} drops)", flush=True)
        return True


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
            target_kl=TARGET_KL,
            ent_coef=ENT_COEF_RESUME,
            batch_size=BATCH_SIZE,
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
            gamma=GAMMA,
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
            save_path=CHECKPOINT_DIR,
            name_prefix="watermelon_ckpt",
        ),
        ScoreLoggingCallback(),
        BestScoreCallback(eval_freq=EVAL_FREQ, n_episodes=EVAL_EPISODES),
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
