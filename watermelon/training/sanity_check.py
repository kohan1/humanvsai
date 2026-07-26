"""
Pre-flight checks for Watermelon. Run BEFORE spending hours on training.

Modelled on snake/training/sanity_check.py, which exists because the Snake v2
rebuild burned hours on three failures that were each detectable in under a
minute:

  - BC collapsed onto the majority action while reporting a healthy-looking
    aggregate accuracy that was really just the class prior.
  - The reward was inverted by a blocked-set bug, so dying instantly paid
    better than playing a full game. PPO was not broken; it was obeying.
  - An entropy bonus dragged a good cloned policy back toward random.

The common thread was that nothing verified the *problem definition* before
optimising it. These checks do, for this game.

Usage:
    python sanity_check.py
    python sanity_check.py --quick   # skip the slower rollouts
"""

import sys

import numpy as np

import watermelon_env as WE
from heuristic import heuristic_action_from_env
from watermelon_env import WatermelonEnv

QUICK = "--quick" in sys.argv
failures = []
warnings = []


def check(name, ok, detail="", fatal=True):
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (failures if fatal else warnings).append(name)


def rollout(policy, seed):
    """Run one episode. policy is 'heuristic' or 'random'."""
    env = WatermelonEnv()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    done = False
    info = {"score": 0}
    while not done:
        a = (heuristic_action_from_env(env) if policy == "heuristic"
             else int(rng.integers(0, env.action_space.n)))
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    return info["score"], total, env.drops


print("=" * 68)
print("WATERMELON PRE-FLIGHT")
print("=" * 68)

# ── 1. Observation plumbing ──────────────────────────────────────────────
print("\n[1] observation")
env = WatermelonEnv()
obs, _ = env.reset(seed=0)
check("length matches declaration", obs.shape[0] == env.obs_size,
      f"{obs.shape[0]} vs {env.obs_size}")
check("inside observation_space", bool(env.observation_space.contains(obs)))
expected = WE.GRID_W * WE.GRID_H * WE.GRID_CHANNELS + WE.N_SCALARS
check("grid + scalars add up", env.obs_size == expected,
      f"{WE.GRID_H}x{WE.GRID_W}x{WE.GRID_CHANNELS} + {WE.N_SCALARS} = {expected}")

# The grid must actually contain something once fruit are on the board,
# otherwise the CNN is being fed a blank image and cannot possibly learn.
env.step(env.action_space.n // 2)
obs2 = env._get_obs()
occupied = obs2[: WE.GRID_W * WE.GRID_H * WE.GRID_CHANNELS].reshape(-1, WE.GRID_CHANNELS)
check("dropped fruit appears in the grid", occupied[:, 0].sum() > 0,
      f"{int(occupied[:, 0].sum())} occupied cells after one drop")
check("tier channel is populated", occupied[:, 1].sum() > 0, fatal=False)

# ── 2. Reward ordering ───────────────────────────────────────────────────
print("\n[2] reward")
if QUICK:
    print("  (skipped — needs rollouts)")
else:
    n = 8
    good = [rollout("heuristic", s) for s in range(n)]
    mean_score = np.mean([g[0] for g in good])
    mean_ret = np.mean([g[1] for g in good])
    mean_drops = np.mean([g[2] for g in good])

    print(f"       heuristic scores {mean_score:.0f} over {mean_drops:.0f} drops, "
          f"return {mean_ret:+.2f}")

    # THE check. If a full good game does not out-earn an instant loss, PPO
    # will correctly learn to lose on purpose and no hyperparameter fixes it.
    check("playing well beats losing immediately", mean_ret > WE.REWARD_LOSS,
          f"good game {mean_ret:+.2f} vs instant loss {WE.REWARD_LOSS:+.2f}")

    # The height penalty applies per drop. If it dominates the merge reward it
    # becomes a per-step tax, which is exactly how Snake's reward inverted.
    merge_upper = mean_score * WE.REWARD_MERGE_SCALE
    check("merge reward dominates the height penalty", mean_ret > merge_upper * 0.5,
          f"return {mean_ret:+.2f} vs merge component ~{merge_upper:+.2f}",
          fatal=False)

# ── 3. Teacher quality ───────────────────────────────────────────────────
print("\n[3] teacher")
if QUICK:
    print("  (skipped — needs rollouts)")
else:
    rnd = [rollout("random", s) for s in range(6)]
    mean_rand = np.mean([r[0] for r in rnd])
    print(f"       heuristic {mean_score:.0f}  vs  random {mean_rand:.0f}")
    check("teacher clearly beats random", mean_score > mean_rand * 1.3,
          f"{mean_score:.0f} vs {mean_rand:.0f} — a teacher barely above random "
          "is not worth cloning")

    # Action balance. Snake's BC collapsed because one action was 79% of the
    # data; with 24 columns the same risk exists if the teacher favours a few.
    env = WatermelonEnv()
    env.reset(seed=7)
    acts = []
    for _ in range(1500):
        a = heuristic_action_from_env(env)
        acts.append(a)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset(seed=int(np.random.randint(10000)))
    counts = np.bincount(acts, minlength=env.action_space.n).astype(float)
    mix = counts / counts.sum()
    print(f"       uses {int((counts > 0).sum())}/{env.action_space.n} columns; "
          f"most common {mix.max():.1%}")
    check("teacher does not fixate on one column", mix.max() < 0.6,
          f"most common column is {mix.max():.1%} of actions", fatal=False)
    check("class imbalance is handled in pretrain.py",
          getattr(__import__("pretrain"), "USE_CLASS_WEIGHTS", False) or mix.max() < 0.3,
          "unweighted BC will collapse onto the dominant column")

print("\n" + "=" * 68)
if failures:
    print(f"FAILED: {len(failures)} check(s) — do not start training")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
if warnings:
    print(f"PASSED with {len(warnings)} warning(s)")
    for w in warnings:
        print(f"  - {w}")
else:
    print("ALL CHECKS PASSED — safe to train")
sys.exit(0)
