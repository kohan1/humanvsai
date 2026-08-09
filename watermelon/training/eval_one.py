"""
Evaluate ONE model on a held-out seed range, so several can run in parallel.

    python eval_one.py <model.zip|heuristic> <n_episodes> <seed0>

Writes the per-episode drops to _drops_<tag>.npy so runs can be pooled or
re-analysed without replaying the games.
"""

import math
import os
import sys

import numpy as np

PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

from watermelon_env import WatermelonEnv  # noqa: E402

if PATH == "heuristic":
    from heuristic import heuristic_action_from_env
    act = lambda env, obs: heuristic_action_from_env(env)  # noqa: E731
else:
    from stable_baselines3 import PPO
    _m = PPO.load(PATH, device="cpu")
    act = lambda env, obs: int(_m.predict(obs, deterministic=True)[0])  # noqa: E731

drops, scores = [], []
for i in range(N):
    env = WatermelonEnv()
    obs, _ = env.reset(seed=SEED0 + i)
    done, info = False, {"score": 0}
    while not done:
        obs, _, term, trunc, info = env.step(act(env, obs))
        done = term or trunc
    drops.append(env.drops)
    scores.append(info["score"])
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{N}  running mean {np.mean(drops):.1f} drops", flush=True)

d = np.array(drops, dtype=float)
s = np.array(scores, dtype=float)
tag = os.path.basename(PATH).replace(".zip", "")
np.save(f"_drops_{tag}_{SEED0}.npy", d)
print(f"RESULT {tag}: drops {d.mean():.1f} +- {d.std(ddof=1) / math.sqrt(N):.1f}  "
      f"score {s.mean():.0f} +- {s.std(ddof=1) / math.sqrt(N):.0f}  "
      f"max {d.max():.0f}  n={N} seeds {SEED0}..{SEED0 + N - 1}")
