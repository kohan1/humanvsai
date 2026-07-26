"""
Expert policy for Watermelon, used to warm-start training via behavioural
cloning (pretrain.py).

The Snake lesson applies directly: a BC teacher is only useful if the student
can *see* what the teacher reasons about. Everything this heuristic uses —
surface height per column, the tier of the fruit exposed at the surface — is
derivable from the observation grid, so the student is not being asked to
imitate decisions it has no basis for.

Strategy, in priority order:

  1. Drop onto an exposed fruit of the same tier. That is an immediate merge,
     and merges are the only source of points.
  2. Failing that, drop onto an exposed fruit one tier below the held one,
     since that sets up a merge next turn.
  3. Otherwise drop where the stack is lowest, which buys time.

Every candidate is penalised by the resulting surface height, so the agent
prefers merges that do not also build a tower.
"""

import math

from watermelon_env import (
    CANVAS_H,
    CANVAS_W,
    DIAMETERS,
    LOSS_LINE_Y,
    N_DROP_COLUMNS,
)

# Weights, tuned by hand against evaluate.py.
W_SAME_TIER = 100.0
W_SETUP = 25.0
W_HEIGHT = 60.0
W_EDGE = 4.0


def _column_x(action):
    return (action + 0.5) / N_DROP_COLUMNS * CANVAS_W


def surface_scan(fruits):
    """
    For each drop column, the topmost fruit whose disc spans that column.

    Returns (heights, tiers): `heights[i]` is the y of the highest surface in
    column i (CANVAS_H when empty, so lower y means a taller stack), and
    `tiers[i]` is that fruit's tier or None.
    """
    heights = [float(CANVAS_H)] * N_DROP_COLUMNS
    tiers = [None] * N_DROP_COLUMNS

    for (cx, cy, tier) in fruits:
        r = DIAMETERS[tier] / 2.0
        top = cy - r
        for i in range(N_DROP_COLUMNS):
            x = _column_x(i)
            if abs(x - cx) <= r and top < heights[i]:
                heights[i] = top
                tiers[i] = tier
    return heights, tiers


def heuristic_action(fruits, held_tier, next_tier=None):
    """
    `fruits` is an iterable of (x, y, tier). Returns a drop column index.

    Kept independent of the env object so pretrain.py can call it from a
    rollout and the same logic can be ported to JS for the browser opponent.
    """
    heights, tiers = surface_scan(fruits)
    held_r = DIAMETERS[held_tier] / 2.0

    best_action, best_score = 0, -math.inf
    for i in range(N_DROP_COLUMNS):
        x = _column_x(i)

        # Reject columns where the fruit would not fit between the walls.
        if x - held_r < 0 or x + held_r > CANVAS_W:
            continue

        score = 0.0

        if tiers[i] == held_tier:
            score += W_SAME_TIER
        elif tiers[i] is not None and tiers[i] == held_tier - 1:
            score += W_SETUP

        # Prefer low stacks. heights[] is a y coordinate, so larger is lower.
        headroom = (heights[i] - LOSS_LINE_Y) / max(1.0, CANVAS_H - LOSS_LINE_Y)
        score += W_HEIGHT * max(0.0, min(1.0, headroom))

        # Mild pull away from the walls: fruit wedged in a corner is hard to
        # merge later, since only one side is reachable.
        edge = min(x, CANVAS_W - x) / (CANVAS_W / 2.0)
        score += W_EDGE * edge

        if score > best_score:
            best_score, best_action = score, i

    return best_action


def heuristic_action_from_env(env):
    """Convenience wrapper for rollouts."""
    fruits = [
        (b.position.x, b.position.y, t) for b, t in env.fruits.items()
    ]
    return heuristic_action(fruits, env.held_tier, env.next_tier)
