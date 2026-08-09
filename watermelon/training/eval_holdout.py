"""
Compare models on seeds NOBODY selected on.

BestScoreCallback evaluates seeds 0-39 and keeps whichever checkpoint scores
best on them, so its numbers are in-sample twice over: the seeds are fixed, and
the model was chosen for doing well on exactly those. evaluate.py then reuses
seeds 0..n-1, which means the obvious way to "check" a selected model re-runs
the selection set and confirms what selection already guaranteed.

This project has already been burned by that: a model that led by +3.1 SE on the
callback's numbers lost 104.6 to 106.8 when re-measured on 60 unseen seeds.

Usage:
    python eval_holdout.py 60 1000        # 60 episodes, seeds 1000+
"""

import math
import os
import sys

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

MODELS = [
    ("heuristic teacher", None),
    ("BC clone", "watermelon_pretrained.zip"),
    ("PPO best (750k)", "watermelon_best.geometry.zip"),
    ("PPO final (30M)", "watermelon_final.geometry.zip"),
]


def run_heuristic(seed):
    from heuristic import heuristic_action_from_env
    from watermelon_env import WatermelonEnv
    env = WatermelonEnv()
    env.reset(seed=seed)
    done, info = False, {"score": 0}
    while not done:
        _, _, term, trunc, info = env.step(heuristic_action_from_env(env))
        done = term or trunc
    return env.drops, info["score"]


def run_model(model, seed):
    from watermelon_env import WatermelonEnv
    env = WatermelonEnv()
    obs, _ = env.reset(seed=seed)
    done, info = False, {"score": 0}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(int(action))
        done = term or trunc
    return env.drops, info["score"]


print(f"{N} episodes, seeds {SEED0}..{SEED0 + N - 1} (held out from selection)\n")
print(f"{'model':<20} {'drops':>18}  {'score':>18}  {'max drops':>9}")
print("-" * 74)

for label, path in MODELS:
    if path is not None:
        if not os.path.exists(path):
            print(f"{label:<20} -- missing: {path}")
            continue
        from stable_baselines3 import PPO
        model = PPO.load(path, device="cpu")

    rows = [run_heuristic(SEED0 + i) if path is None else run_model(model, SEED0 + i)
            for i in range(N)]
    d = np.array([r[0] for r in rows], dtype=float)
    s = np.array([r[1] for r in rows], dtype=float)
    dse = d.std(ddof=1) / math.sqrt(N)
    sse = s.std(ddof=1) / math.sqrt(N)
    print(f"{label:<20} {d.mean():8.1f} +- {dse:4.1f}  {s.mean():9.0f} +- {sse:5.0f}  {d.max():9.0f}")
    np.save(f"_holdout_{label.split()[0].lower()}_{SEED0}.npy", d)

print("\nDifferences smaller than ~2x the combined standard error are noise.")
