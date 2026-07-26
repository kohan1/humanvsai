"""
train.py
--------
Trains a Tetris AI using PPO (Proximal Policy Optimization) via Stable-Baselines3.
Designed to run on Apple M4 MacBook Pro (MPS backend via PyTorch).

Usage:
    python train.py

Requirements (install once):
    pip install stable-baselines3 torch gymnasium numpy tensorboard

Training output:
    models/tetris_ppo_final.zip   — the trained model (use this in export.py)
    models/tetris_ppo_XXXX.zip    — checkpoints saved every 50,000 steps
    logs/                         — TensorBoard logs (watch live progress)

Monitor training live (in a separate terminal):
    tensorboard --logdir logs

Then open http://localhost:6006 in your browser.
The key chart to watch is "rollout/ep_rew_mean" — episode reward mean.
It will be very negative early on and slowly climb toward positive values.
"""

import os
import torch
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from tetris_env import TetrisEnv

# ─── Device setup ─────────────────────────────────────────────────────────────
# SB3 doesn't use MPS directly, but PyTorch will use it under the hood.
# On M4, this gives a meaningful speedup over CPU for the neural net updates.

def get_device():
    # Environments always run on CPU (SB3 handles this automatically)
    # Network updates go to GPU — worth it with the larger 1024x1024x512 network
    torch.set_num_threads(16)  # max CPU threads for environment simulation
    if torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CPU threads for environments: {torch.get_num_threads()}")
        return "cuda"
    else:
        print(f"Using CPU ({torch.get_num_threads()} threads)")
        return "cpu"


# ─── Learning rate schedule ───────────────────────────────────────────────────
def linear_lr_schedule(initial_lr: float, final_lr: float):
    """
    Returns a function that linearly decays the learning rate
    from initial_lr to final_lr over the course of training.
    SB3 passes progress_remaining (1.0 → 0.0) into this function.
    """
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Total environment steps to train for.
    # ~500k  : AI starts to survive longer, occasional line clears
    # ~2M    : Consistent line clears, decent play
    # ~5M    : Strong play
    # ~10M+  : Competitive
    # Start with 2_000_000 — you can resume training later by loading a checkpoint.
    "total_timesteps": 1_000_000_000,

    # Number of parallel environments.
    # More = faster data collection. M4 with 16GB handles 8 comfortably.
    "n_envs": 32,

    # How many steps each env collects before a PPO update.
    # n_steps * n_envs = total steps per update batch.
    # 512 * 8 = 4096 steps per batch — good balance for Tetris episode lengths.
    "n_steps": 1024,  # larger rollouts = better gradient estimates for bigger network

    # PPO batch size for gradient updates. Must divide n_steps * n_envs evenly.
    "batch_size": 1024,  # larger batch for GPU efficiency

    # Number of epochs to run over each batch of data.
    "n_epochs": 10,

    # Learning rate — slightly lower than default for stability.
    "learning_rate": 2.5e-4,   # starting LR — decays to 1e-5 over training

    # PPO clip range — how much the policy is allowed to change per update.
    "clip_range": 0.2,

    # Discount factor — how much future rewards matter.
    # 0.99 = agent thinks ~100 steps ahead, good for Tetris.
    "gamma": 0.99,

    # GAE lambda — smooths advantage estimates.
    "gae_lambda": 0.95,

    # Entropy coefficient — encourages exploration.
    # Higher = more random early on. Will decay naturally via training.
    "ent_coef": 0.02,   # stable entropy

    # Value function loss coefficient.
    "vf_coef": 0.5,

    # Max gradient norm — prevents exploding gradients.
    "max_grad_norm": 0.5,

    # Neural network architecture.
    # Two hidden layers of 128 neurons each — small enough to run fast in browser via ONNX.
    "net_arch": [1024, 1024, 512],  # large network — GPU handles updates, CPU handles envs

    # Save a checkpoint every this many steps.
    "checkpoint_freq": 5_000_000,  # save every 5M steps instead of 50k

    # Evaluate the model every this many steps (uses a separate eval env).
    "eval_freq": 100_000,  # increased — 500 piece games need more time to evaluate

    # Number of episodes to average for evaluation.
    "n_eval_episodes": 20,  # more episodes for reliable average at high skill level

    # Output directories.
    "model_dir": "models",
    "log_dir":   "logs",
}

# ─── Progress callback ────────────────────────────────────────────────────────

class CleanupCallback(BaseCallback):
    """
    Deletes old numbered checkpoints after each save, keeping only the
    most recent N. The final model and best_model are never deleted.
    """
    def __init__(self, keep_last=3, verbose=0):
        super().__init__(verbose)
        self.keep_last = keep_last
        self._last_cleanup = 0

    def _on_step(self):
        # Only run cleanup at same frequency as checkpoints
        if self.num_timesteps - self._last_cleanup >= CONFIG["checkpoint_freq"]:
            self._last_cleanup = self.num_timesteps
            model_dir = CONFIG["model_dir"]
            if not os.path.exists(model_dir):
                return True
            checkpoints = sorted([
                f for f in os.listdir(model_dir)
                if f.startswith("tetris_ppo_") and f.endswith(".zip")
                and "final" not in f and "best" not in f
            ])
            # Delete all but the last N
            to_delete = checkpoints[:-self.keep_last]
            for f in to_delete:
                path = os.path.join(model_dir, f)
                os.remove(path)
                if self.verbose:
                    print(f"  Deleted old checkpoint: {f}")
        return True


class VisualiserCallback(BaseCallback):
    """
    Writes all environment states to vis_state.json every N steps.
    The visualiser_server.py reads this and broadcasts to the browser.
    Completely decoupled from training — if the file write fails, training continues.
    """
    def __init__(self, write_every=5, verbose=0):
        super().__init__(verbose)
        self.write_every = write_every
        self._counter    = 0

    def _on_step(self):
        self._counter += 1
        if self._counter % self.write_every != 0:
            return True
        try:
            import numpy as np
            import json

            ep_info = list(self.model.ep_info_buffer)
            mean_r  = float(np.mean([e["r"] for e in ep_info])) if ep_info else 0.0
            mean_l  = float(np.mean([e["l"] for e in ep_info])) if ep_info else 0.0

            # Get actual arena state with colour values from each environment
            # using get_attr to access the underlying env attribute directly
            try:
                arenas = self.training_env.get_attr("arena")
                pieces = self.training_env.get_attr("piece")
                envs_data = []
                for i, (arena, piece) in enumerate(zip(arenas, pieces)):
                    # arena is 18x10 with values 0-7
                    board = [row[:] for row in arena]
                    envs_data.append(board)
                n_envs = len(envs_data)
            except Exception:
                # Fallback to binary obs if get_attr fails
                obs_np = self.model._last_obs
                if obs_np is None:
                    return True
                envs_data = []
                for i in range(min(obs_np.shape[0], 32)):
                    board_flat = obs_np[i, :180]
                    board = [[int(board_flat[r*10+c]) for c in range(10)] for r in range(18)]
                    envs_data.append(board)
                n_envs = obs_np.shape[0]

            state = {
                "envs":     envs_data,
                "steps":    self.num_timesteps,
                "mean_r":   round(mean_r, 2),
                "mean_len": round(mean_l, 1),
                "n_envs":   n_envs,
            }

            with open("vis_state.json", "w") as f:
                json.dump(state, f)
        except Exception as e:
            pass
        return True


class ProgressCallback(BaseCallback):
    """
    Prints a human-readable progress line every N steps.
    Shows timesteps, episode reward mean, and episode length mean.
    """
    def __init__(self, print_every=10_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every
        self._last_print  = 0

    def _on_step(self):
        if self.num_timesteps - self._last_print >= self.print_every:
            self._last_print = self.num_timesteps

            # Pull stats from the monitor buffer
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mean_r = np.mean(rewards)
                mean_l = np.mean(lengths)
                print(
                    f"  Steps: {self.num_timesteps:>9,} / {CONFIG['total_timesteps']:,}"
                    f"  |  Ep reward: {mean_r:+.2f}"
                    f"  |  Ep length (pieces): {mean_l:.0f}"
                )
            else:
                print(f"  Steps: {self.num_timesteps:>9,} / {CONFIG['total_timesteps']:,}  |  (collecting...)")

        return True

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Tetris PPO Training")
    print("=" * 60)

    device = get_device()

    # Create output directories
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"],   exist_ok=True)

    # ── Environments ──────────────────────────────────────────────────────────
    # Training envs: run in parallel subprocesses for speed
    print(f"\nSpawning {CONFIG['n_envs']} parallel training environments...")

    def make_env():
        def _init():
            env = TetrisEnv()
            env = Monitor(env)
            return env
        return _init

    train_env = SubprocVecEnv([make_env() for _ in range(CONFIG["n_envs"])])
    train_env = VecMonitor(train_env)

    # Eval env: single env, used to measure real performance during training
    eval_env = Monitor(TetrisEnv())

    # ── Check for existing checkpoint ─────────────────────────────────────────
    # Priority: final model first (most steps), then latest numbered checkpoint
    checkpoint_path = None

    final_path_zip = os.path.join(CONFIG["model_dir"], "tetris_ppo_final.zip")
    if os.path.exists(final_path_zip):
        checkpoint_path = final_path_zip
    elif os.path.exists(CONFIG["model_dir"]):
        checkpoints = sorted([
            f for f in os.listdir(CONFIG["model_dir"])
            if f.startswith("tetris_ppo_") and f.endswith(".zip") and "final" not in f
        ])
        if checkpoints:
            checkpoint_path = os.path.join(CONFIG["model_dir"], checkpoints[-1])

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nResuming from: {checkpoint_path}")
        model = MaskablePPO.load(
            checkpoint_path,
            env=train_env,
            device=device,
        )
        # Update total timesteps to train for more
        remaining = CONFIG["total_timesteps"] - model.num_timesteps
        if remaining <= 0:
            print(f"\nModel already trained for {model.num_timesteps:,} steps.")
            print("Increase total_timesteps in CONFIG to train further.")
            train_env.close()
            return
        print(f"Already trained: {model.num_timesteps:,} steps. Training {remaining:,} more.")
        total_timesteps = remaining
    else:
        print("\nStarting fresh training run.")
        total_timesteps = CONFIG["total_timesteps"]

        # ── Model definition ──────────────────────────────────────────────────
        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_env,
            device=device,

            # Core PPO hyperparameters
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            learning_rate=linear_lr_schedule(CONFIG["learning_rate"], 1e-5),
            clip_range=CONFIG["clip_range"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            ent_coef=CONFIG["ent_coef"],
            vf_coef=CONFIG["vf_coef"],
            max_grad_norm=CONFIG["max_grad_norm"],

            # Network architecture
            policy_kwargs=dict(
                net_arch=CONFIG["net_arch"],
                activation_fn=torch.nn.ReLU,
            ),

            # TensorBoard logging
            tensorboard_log=CONFIG["log_dir"],

            verbose=0,
        )

    print(f"\nNetwork architecture: {CONFIG['net_arch']}")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"\nTraining for {total_timesteps:,} steps across {CONFIG['n_envs']} envs.")
    print("─" * 60)
    print("TIP: In a separate terminal, run:")
    print("     tensorboard --logdir logs")
    print("     Then open http://localhost:6006")
    print("     Watch 'rollout/ep_rew_mean' — it should slowly climb.")
    print("─" * 60)
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    # Save checkpoint every N steps, keep only last 3 to save disk space
    checkpoint_cb = CheckpointCallback(
        save_freq=max(CONFIG["checkpoint_freq"] // CONFIG["n_envs"], 1),
        save_path=CONFIG["model_dir"],
        name_prefix="tetris_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    # Evaluate model every N steps and save best
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CONFIG["model_dir"],
        log_path=CONFIG["log_dir"],
        eval_freq=max(CONFIG["eval_freq"] // CONFIG["n_envs"], 1),
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # Human-readable progress printer
    progress_cb = ProgressCallback(print_every=10_000)

    # Auto-delete old checkpoints, keep last 3
    cleanup_cb = CleanupCallback(keep_last=3, verbose=1)

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, progress_cb, cleanup_cb, VisualiserCallback(write_every=10)],
            reset_num_timesteps=False,
            tb_log_name="ppo_tetris",
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(CONFIG["model_dir"], "tetris_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")
    print("\nNext step: run export.py to convert to ONNX for the browser.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
