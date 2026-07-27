"""
Main RL training loop for the Snake AI.

Plain PPO — no masking needed, since the Discrete(3) relative action space
makes every action always legal (see snake_env.py docstring).

Run pretrain.py first for a behavioural-cloning warm start; this script
picks up snake_pretrained.zip automatically if it exists, otherwise starts
from a fresh random-init policy.

Reminder from the Tetris build: if you change the reward function in
snake_env.py, restart training from scratch — don't resume a checkpoint
trained under the old reward. Changing rewards mid-training collapses
performance.
"""

import os
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from snake_env import SnakeEnv
from policy_config import policy_kwargs


def linear_schedule(start: float, end: float):
    """SB3 passes progress_remaining: 1.0 at the start, 0.0 at the end."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule

# Use the GPU when there is one. The old "CPU beats GPU for small MLPs" rule
# does not hold for this net: measured on the RTX 4060 Ti with the [512,512,256]
# architecture below, a batch-2048 forward+backward is ~19x faster on CUDA
# (59.1ms -> 3.1ms), and even the batch-32 rollout pass is slightly faster.
# The rule of thumb is real, but it applies to genuinely tiny nets — check
# before assuming it, don't inherit the assumption.
# Overridable so two games can train at once without editing files:
#   TRAIN_DEVICE=cpu  N_ENVS=8  python train.py
# Env stepping is CPU-bound in both games, so running two jobs means splitting
# the cores between them - oversubscribing just makes both slower.
DEVICE = os.environ.get("TRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

N_ENVS = int(os.environ.get("N_ENVS", 32))

# How often BestScoreCallback checks, and over how many fixed-seed games.
# 15 episodes is a fraction of a percent of a 1M-step window and is the
# only thing standing between a mid-run peak and losing it.
EVAL_FREQ = int(os.environ.get("EVAL_FREQ", 1_000_000))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", 15))
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", 25_000_000))

# Entropy bonus, and it depends entirely on how good the starting policy is.
#
# An entropy bonus rewards being random. On a FRESH run that is what stops the
# policy committing early to a mediocre habit. On top of a near-deterministic
# BC clone it is actively destructive: Snake has 3 actions and usually 2 of
# them are fatal, so pushing toward uniform is pushing toward death.
#
# Measured: BC scored 47.43, and fine-tuning it with ENT_COEF=0.01 drove it to
# 17.50 within 2.5M steps — a 63% loss, the same catastrophic-forgetting shape
# as the old learning-rate bug but with entropy as the cause. v1's ent_coef=0
# was not the mistake it looked like; it was correct for that phase.
#
# So: exploration pressure only when there is nothing good to protect.
ENT_COEF_FRESH = 0.01
ENT_COEF_RESUME = 0.0

# Fresh runs anneal the LR the way Tetris does — high enough early to actually
# move, low enough late to settle. Resumed runs use the small fixed LR instead
# (see below), because that phase is protecting existing weights, not
# exploring.
LR_START = 3e-4
LR_END = 1e-5
LR_RESUME = float(os.environ.get("LR_RESUME", 5e-5))

# Snake's last run early-stopped on 728 of 728 iterations, 716 of them at
# epoch 0 of 10 — it was collecting rollouts and then barely using them.
# Widening the trust region is what fixed exactly that on Watermelon, and
# BestScoreCallback now makes it safe to try: an overshoot costs nothing
# because the peak is kept.
TARGET_KL = float(os.environ.get("TARGET_KL", 0.05))

PRETRAINED_PATH = "snake_pretrained.zip"
RESUME_PATH = "snake_final.zip"  # a finished RL run, if there is one — prefer
                                   # continuing it over restarting from the BC
                                   # policy. Only valid while the observation
                                   # and reward spec are unchanged: an obs/reward
                                   # change means DELETING these and starting
                                   # over, since the old weights encode a
                                   # different input layout entirely.
FINAL_PATH = "snake_final.zip"   # always this exact name — never rely on
                                   # sorting numbered checkpoint filenames,
                                   # see the Tetris NOTES.md on that bug
TENSORBOARD_LOG = "tb_logs/"


def make_env():
    return SnakeEnv()


class ScoreLoggingCallback(BaseCallback):
    """
    SB3 only logs episode reward and length to TensorBoard by default — not
    anything from the env's `info` dict. This pulls `info["score"]` (actual
    food eaten, set in snake_env.py) out at the end of every episode and
    logs a rolling average as `rollout/ep_score_mean`, so you can watch the
    real score climb instead of inferring it from the reward number, which
    also includes the small distance-shaping bonus and doesn't map 1:1 to
    food eaten.
    """

    def __init__(self, buffer_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.score_buffer = deque(maxlen=buffer_size)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Monitor (auto-applied by make_vec_env) adds an "episode" key
            # to `info` on the exact step an episode ends — that's also the
            # step where our own info["score"] holds the episode's final
            # food-eaten count.
            if "episode" in info and "score" in info:
                self.score_buffer.append(info["score"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.score_buffer) > 0:
            self.logger.record("rollout/ep_score_mean", float(np.mean(self.score_buffer)))


class BestScoreCallback(BaseCallback):
    """
    Keep the best model seen, not the last one. Ported from Watermelon's
    train.py, where it immediately proved its worth: that game's 20M-step run
    peaked at 1032.43 and ENDED on 959.93, so without this the run would have
    shipped a model 72 points worse than the one it had already found.

    Snake has never had this. Its last run early-stopped on 728 of 728
    iterations and drifted for hours; whatever peak it passed through was lost.
    At 100M steps that risk is far larger, not smaller.

    Fixed seeds, so every check faces identical games — Snake's scores range
    41 to 177 on one unchanged model, and without fixing them "best" would
    mostly select a lucky draw. Evaluates at step 0 too, so the STARTING model
    sets the bar and a run that degrades early cannot save something worse than
    what it began with.
    """

    def __init__(self, eval_freq: int, n_episodes: int = 15,
                 save_path: str = "snake_best.zip", verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.save_path = save_path
        self.best_score = float("-inf")
        self._next_eval = 0

    def _evaluate(self) -> float:
        env = SnakeEnv()
        total = 0
        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=ep)
            done, info = False, {"score": 0}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            total += info["score"]
        return total / self.n_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval = self.num_timesteps + self.eval_freq

        score = self._evaluate()
        self.logger.record("eval/score", score)

        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path)
            if self.verbose:
                print(f"\n[best] new best {score:.2f} at {self.num_timesteps} "
                      f"steps -> {self.save_path}", flush=True)
        elif self.verbose:
            print(f"\n[best] {score:.2f} at {self.num_timesteps} steps "
                  f"(best is still {self.best_score:.2f})", flush=True)
        return True


def main():
    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    start_path = RESUME_PATH if os.path.exists(RESUME_PATH) else PRETRAINED_PATH

    if os.path.exists(start_path):
        print(f"Loading weights from {start_path}")
        # The pretrained policy already scores ~4x better than a fresh RL
        # run at 3M steps did (confirmed via evaluate.py) — PPO's default
        # learning_rate (3e-4) is tuned for training from scratch, and was
        # aggressive enough to bulldoze that good starting policy within
        # the first few million steps rather than build on it. Dropping it
        # ~6x, plus target_kl as a hard safety net that stops an update
        # early if any single batch drifts the policy too far, keeps this
        # phase a genuine fine-tune instead of a partial do-over.
        model = PPO.load(
            start_path,
            env=vec_env,
            device=DEVICE,
            tensorboard_log=TENSORBOARD_LOG,
            learning_rate=LR_RESUME,
            target_kl=TARGET_KL,
            ent_coef=ENT_COEF_RESUME,
        )
        # pretrain.py sets verbose=0 to keep the BC loop quiet — that
        # setting is saved into the .zip and would otherwise silently
        # carry over here, making training look frozen even though it's
        # running fine. Force it back on.
        model.verbose = 1
    else:
        print("No pretrained weights found — starting from scratch")
        model = PPO(
            "MlpPolicy",   # overridden by features_extractor_class below
            vec_env,
            policy_kwargs=policy_kwargs(),
            n_steps=2048,
            batch_size=2048,
            n_epochs=10,
            learning_rate=linear_schedule(LR_START, LR_END),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ENT_COEF_FRESH,
            # No target_kl on a fresh run. It exists to stop a good policy
            # drifting; imposed from step zero it just throttles learning.
            device=DEVICE,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 500_000 // N_ENVS),
        save_path="checkpoints/",
        name_prefix="snake_ckpt_r2",   # r2 = second fine-tune, resumed from the
                                       # 36.58 policy. Distinct prefix so this
                                       # run's checkpoints sit alongside run 1's
                                       # instead of overwriting them step-for-step.
    )
    score_cb = ScoreLoggingCallback()
    best_cb = BestScoreCallback(eval_freq=EVAL_FREQ, n_episodes=EVAL_EPISODES)
    callback = CallbackList([checkpoint_cb, score_cb, best_cb])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    finally:
        # Numbered checkpoints above are for resuming/inspection only.
        # This fixed filename is what export.py and everything downstream
        # should always load.
        model.save(FINAL_PATH)
        print(f"Saved final model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
