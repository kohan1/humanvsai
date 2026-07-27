"""
Measure real PPO throughput at different N_ENVS / device combinations.

The point is to pick N_ENVS by measurement rather than by "8 cores so use 8".
Watermelon's env is pymunk physics — CPU-bound and pure-Python-driven — so
hyperthreading may or may not help, and the only way to know is to time it.

Run from watermelon/training.
"""
import os
import sys
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from watermelon_env import WatermelonEnv

MODEL = "watermelon_final.zip"
WARMUP_STEPS = 4096       # first rollout pays process spawn + CUDA init
MEASURE_STEPS = 40_960


def make_env():
    return WatermelonEnv()


def bench(n_envs: int, device: str) -> float:
    vec = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    try:
        model = PPO.load(MODEL, env=vec, device=device, learning_rate=2e-5,
                         target_kl=0.03, ent_coef=0.0)
        model.verbose = 0
        model.learn(total_timesteps=WARMUP_STEPS, progress_bar=False)

        t0 = time.perf_counter()
        model.learn(total_timesteps=MEASURE_STEPS, progress_bar=False,
                    reset_num_timesteps=False)
        dt = time.perf_counter() - t0
        return MEASURE_STEPS / dt
    finally:
        vec.close()


if __name__ == "__main__":
    combos = []
    if torch.cuda.is_available():
        for n in (8, 12, 16, 20, 24):
            combos.append((n, "cuda"))
    combos.append((16, "cpu"))

    print(f"{'n_envs':>7} {'device':>7} {'fps':>9}")
    results = []
    for n, dev in combos:
        try:
            fps = bench(n, dev)
            results.append((n, dev, fps))
            print(f"{n:>7} {dev:>7} {fps:>9.0f}", flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"{n:>7} {dev:>7}   FAILED: {exc}", flush=True)

    if results:
        best = max(results, key=lambda r: r[2])
        print(f"\nbest: n_envs={best[0]} device={best[1]} at {best[2]:.0f} fps")
