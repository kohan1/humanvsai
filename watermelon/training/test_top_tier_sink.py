"""
Does the top-tier merge physically happen?

This is the one check the whole geometry change exists to satisfy. A vanishing
MAX_TIER pair is the only way area ever leaves the well; if two of them cannot
touch below the loss line, `_resolve_merges` never reaches its
`if tier < MAX_TIER` branch, area accumulates monotonically, and no policy can
survive indefinitely no matter how it is trained.

The old ladder failed this. Two 290px fruit needed 580px to sit side by side
against a 448px well, and 580px to stack against 484px of playable height.

The failure was subtler than "the code is wrong": force two of them into
contact and the handler fires correctly and both vanish. The problem is that
every position where they touch is already a loss. Stack them and the upper
fruit's top edge lands at y=-21, off the board entirely and 136px past the
loss line at y=115; approach diagonally and the walls cap their horizontal
separation, forcing the same verdict. The sink worked and was unreachable, so
area could only accumulate — and five training runs spent their compute
against a wall that no reward design could see, because it was not in the
reward.

Run this before training. It takes a couple of seconds.

    python test_top_tier_sink.py
"""

import math
import sys

import watermelon_env as WE
from watermelon_env import DIAMETERS, MAX_TIER, WatermelonEnv

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def area(env):
    return sum(math.pi * (DIAMETERS[t] / 2.0) ** 2 for t in env.fruits.values())


print("=" * 68)
print("TOP-TIER SINK")
print("=" * 68)

W = WE.CANVAS_W
H = WE.CANVAS_H - WE.LOSS_LINE_Y
top_d = DIAMETERS[MAX_TIER]

# ── 1. Geometry, before touching the physics engine ──────────────────────
# If these fail, no amount of simulation will help.
print("\n[1] can two top-tier fruit fit at all")
print(f"      well {W} wide x {H} tall (loss line to floor), top fruit {top_d}px")
check("two fit side by side", 2 * top_d <= W,
      f"need {2 * top_d}px of width, have {W}px")

# Diagonal contact is the general case: their centres must be exactly top_d
# apart, and each centre is at least a radius from either wall, which caps how
# far apart they can be horizontally and therefore forces a vertical offset.
r = top_d / 2.0
max_dx = W - top_d
min_dy = math.sqrt(max(0.0, top_d ** 2 - max_dx ** 2))
lower_cy = WE.CANVAS_H - r          # resting on the floor
upper_top_edge = (lower_cy - min_dy) - r
check("they can touch without either crossing the loss line",
      upper_top_edge >= WE.LOSS_LINE_Y,
      f"closest legal contact puts the upper fruit's top edge at "
      f"y={upper_top_edge:.0f}, loss line is y={WE.LOSS_LINE_Y}")

# ── 2. The merge actually fires in the simulator ─────────────────────────
print("\n[2] the pair merges and vanishes")
env = WatermelonEnv()
env.reset(seed=0)
for body in list(env.fruits):
    for s in list(body.shapes):
        env.space.remove(s)
    env.space.remove(body)
    del env.fruits[body]

# One resting on the floor, one released above it. Fruit placed side by side
# with a gap never touch — nothing pushes them together — so the honest test is
# the one the game actually performs: a drop.
env._spawn_fruit(W / 2.0, WE.CANVAS_H - r, MAX_TIER)
env._spawn_fruit(W / 2.0, WE.CANVAS_H - r - top_d - 40.0, MAX_TIER)
before_n, before_area = len(env.fruits), area(env)
gained, merges = env._settle()
after_n, after_area = len(env.fruits), area(env)

print(f"      {before_n} fruit ({before_area:,.0f} px2) -> "
      f"{after_n} fruit ({after_area:,.0f} px2), {merges} merge(s)")
check("the merge happened", merges >= 1)
check("both fruit are gone, nothing replaced them", after_n == 0,
      f"{after_n} left — a replacement was spawned, so the sink does not sink")
check("area actually left the well", after_area < before_area,
      f"{before_area:,.0f} -> {after_area:,.0f} px2")

# ── 3. The tier below still merges, or the ladder cannot be climbed ──────
print("\n[3] tier below still reaches the top")
env2 = WatermelonEnv()
env2.reset(seed=0)
for body in list(env2.fruits):
    for s in list(body.shapes):
        env2.space.remove(s)
    env2.space.remove(body)
    del env2.fruits[body]
r9 = DIAMETERS[MAX_TIER - 1] / 2.0
env2._spawn_fruit(W / 2.0, WE.CANVAS_H - r9, MAX_TIER - 1)
env2._spawn_fruit(W / 2.0, WE.CANVAS_H - r9 - DIAMETERS[MAX_TIER - 1] - 40.0,
                  MAX_TIER - 1)
env2._settle()
tiers = list(env2.fruits.values())
check("two tier-(max-1) fruit produce a top-tier fruit",
      tiers == [MAX_TIER], f"left with tiers {tiers}")

print("\n" + "=" * 68)
if failures:
    print(f"FAILED: {len(failures)} check(s) — the sink is not reachable, "
          "so the game is still finite by construction")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("SINK IS REACHABLE — area can leave the well")
sys.exit(0)
