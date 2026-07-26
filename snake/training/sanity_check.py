"""
Pre-flight checks. Run this BEFORE spending hours on training.

Why this exists
---------------
The Snake v2 rebuild burned several hours of compute across three failures, and
every one of them was detectable in under a minute:

  1. BC collapsed onto the majority class (teacher plays 79% STRAIGHT). It
     reported train_acc=0.787 — exactly the class prior — and scored 0.00.
     Aggregate accuracy cannot tell a classifier from a constant.

  2. The reward was INVERTED. A blocked-set bug (`body[:-1]` includes the head,
     so the flood fill started inside an obstacle and always returned 0) turned
     a graded trap penalty into a flat -0.1 per step. A full 50-food episode
     paid -18.1; dying instantly paid -1.0. PPO was not failing — it was
     correctly learning to die.

  3. An entropy bonus on top of a good BC clone dragged it toward uniform, and
     in a 3-action game where 2 actions are usually fatal, that is fatal.

The common thread is not any one bug. It is that nothing verified the *problem
definition* before optimising it. These checks do.

Usage:
    python sanity_check.py            # all checks
    python sanity_check.py --quick    # skip the slower teacher rollouts
"""

import sys

import numpy as np

import snake_env as SE
from heuristic import heuristic_action
from snake_env import SnakeEnv

QUICK = "--quick" in sys.argv
failures = []
warnings = []


def check(name, ok, detail="", fatal=True):
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        (failures if fatal else warnings).append(name)


def run_teacher(seed):
    """One heuristic episode, with the reward decomposed."""
    env = SnakeEnv()
    env.reset(seed=seed)
    parts = {"food": 0.0, "shaping": 0.0, "trap": 0.0, "death": 0.0}
    trap_steps = steps = 0
    done = False
    info = {"score": 0}

    while not done:
        a = heuristic_action(env.body, env.dir_idx, env.food, env.tile_count)
        prev_score = env.score
        prev_dist = env._food_distance(*env.body[0])

        _, _, term, trunc, info = env.step(int(a))
        steps += 1

        if term:
            parts["death"] += SE.DEATH_PENALTY
        else:
            if env.score > prev_score:
                parts["food"] += SE.FOOD_REWARD
            else:
                fade = max(0.0, 1.0 - len(env.body) / SE.SHAPING_FADE_LENGTH)
                d = env._food_distance(*env.body[0])
                parts["shaping"] += SE.SHAPING_SCALE * fade * (prev_dist - d)

            free = env._free_space(env.body[0], env.body[1:-1])
            if free < len(env.body):
                parts["trap"] -= SE.TRAP_PENALTY * (1.0 - free / max(1, len(env.body)))
                trap_steps += 1

        done = term or trunc

    return info["score"], steps, trap_steps, parts


print("=" * 68)
print("SNAKE PRE-FLIGHT")
print("=" * 68)

# ── 1. Observation plumbing ──────────────────────────────────────────────
print("\n[1] observation")
env = SnakeEnv()
obs, _ = env.reset(seed=0)
check("length matches declaration", obs.shape[0] == env.obs_size,
      f"{obs.shape[0]} vs {env.obs_size}")
check("inside observation_space", bool(env.observation_space.contains(obs)))
expected = env.cells * SE.GRID_CHANNELS + SE.N_SCALARS
check("grid + scalars add up", env.obs_size == expected,
      f"{env.cells}x{SE.GRID_CHANNELS} + {SE.N_SCALARS} = {expected}")

# The flood fill must not start inside its own blocked set — the bug that
# silently zeroed free space and inverted the reward.
free_ok = env._free_space(env.body[0], env.body[1:-1])
free_bad = env._free_space(env.body[0], env.body[:-1])
check("flood fill excludes the head from its own blocked set", free_ok > 0,
      f"free={free_ok} (passing body[:-1] instead gives {free_bad})")

# ── 2. Reward ordering ───────────────────────────────────────────────────
print("\n[2] reward")
if QUICK:
    print("  (skipped — needs teacher rollouts)")
else:
    n = 8
    rows = [run_teacher(s) for s in range(n)]
    score = np.mean([r[0] for r in rows])
    steps = np.mean([r[1] for r in rows])
    trap_frac = np.mean([r[2] / max(1, r[1]) for r in rows])
    tot = np.mean([sum(r[3].values()) for r in rows])
    parts = {k: np.mean([r[3][k] for r in rows]) for k in rows[0][3]}

    print(f"       teacher scores {score:.1f} over {steps:.0f} steps")
    print(f"       food {parts['food']:+.2f}  shaping {parts['shaping']:+.2f}  "
          f"trap {parts['trap']:+.2f}  death {parts['death']:+.2f}  =  {tot:+.2f}")

    # THE check. If good play does not out-earn instant death, PPO will
    # correctly learn to die and no hyperparameter will save it.
    check("playing well beats dying instantly", tot > SE.DEATH_PENALTY,
          f"good episode {tot:+.2f} vs death {SE.DEATH_PENALTY:+.2f}")

    # A penalty that fires almost every step is a flat per-step tax wearing a
    # structural penalty's clothes — the exact Tetris mistake.
    check("trap penalty is occasional, not a per-step tax", trap_frac < 0.5,
          f"fires on {trap_frac*100:.0f}% of steps", fatal=False)

    check("food dominates the return", parts["food"] > abs(parts["trap"]),
          f"food {parts['food']:+.2f} vs trap {parts['trap']:+.2f}")

# ── 3. Teacher quality and action balance ────────────────────────────────
print("\n[3] teacher")
env = SnakeEnv()
obs, _ = env.reset(seed=123)
acts = []
for _ in range(3000):
    a = heuristic_action(env.body, env.dir_idx, env.food, env.tile_count)
    acts.append(a)
    _, _, term, trunc, _ = env.step(a)
    if term or trunc:
        env.reset(seed=np.random.randint(100000))

counts = np.bincount(acts, minlength=3).astype(float)
mix = counts / counts.sum()
print(f"       action mix: " + ", ".join(f"a{i}={m:.3f}" for i, m in enumerate(mix)))
check("every action is used", (counts > 0).all())
check("class imbalance is handled in pretrain.py",
      getattr(__import__("pretrain"), "USE_CLASS_WEIGHTS", False) or mix.max() < 0.5,
      f"majority class is {mix.max():.1%} — unweighted BC will collapse onto it")

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
