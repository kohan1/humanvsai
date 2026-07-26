"""
Watermelon (Suika-style) merge game environment.

WHAT THIS IS
------------
A headless physics reimplementation of watermelon/game.js, for training an AI
opponent. Constants (board size, diameters, points, spawn pools, loss line,
gravity) are copied from the browser game so behaviour lines up.

PHYSICS FIDELITY — read this before trusting a trained policy
------------------------------------------------------------
The browser runs planck.js (a JS port of Box2D) via p5play. This env runs
pymunk (Chipmunk2D). They are different engines: contact solvers, resting
thresholds and restitution handling all differ in detail, so an identical
sequence of drops will NOT produce pixel-identical stacks in both.

That is a deliberate trade. A faithful Box2D binding on Windows/Python 3.14
means compiling pybox2d from source, and even then planck has its own
divergences. What matters for transfer is that the *decision problem* is the
same shape: same board, same fruit sizes, same merge rule, same failure
condition. A policy that learns "drop same-tier fruit near each other and keep
the stack low" transfers; one that exploits exact bounce trajectories will not.

Expect the browser score to differ from evaluate.py. If it differs *wildly*,
suspect physics divergence before suspecting the policy — and check the
settle-detection thresholds first, since those decide when a drop is "done".

ACTION SPACE
------------
Discrete(N_DROP_COLUMNS): which x-column to release from. One step = one drop
plus the settle that follows, so an episode is a sequence of placements rather
than of physics frames. The browser's AI hook takes a 0-1 fraction of board
width, so action -> fraction is (action + 0.5) / N_DROP_COLUMNS.

OBSERVATION
-----------
Flat, so ONNX export and the JS mirror stay simple — same approach as Snake:

    grid  GRID_W * GRID_H * 2, row-major (row, col, channel)
      0 occupancy         1 if any fruit covers the cell centre
      1 tier / MAX_TIER   normalised tier of that fruit
    scalars (12)
      5  held fruit tier one-hot   (spawn pool is tiers 0-4)
      5  next fruit tier one-hot
      1  highest stack point, as a fraction of board height
      1  fruit count / MAX_FRUIT_NORM

The JS encoder in ../game.js must mirror this field-for-field. See CLAUDE.md
for the cscript cross-check procedure used for Snake.
"""

import math

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium import spaces

# ── Constants mirrored from watermelon/game.js ───────────────────────────────
CANVAS_W = 448
CANVAS_H = 599
LOSS_LINE_Y = 115

POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]
DIAMETERS = [30, 46, 70, 80, 100, 125, 150, 177, 200, 230, 290]
MAX_TIER = len(DIAMETERS) - 1

WEIGHTED = {
    "initGame": [0, 1, 2, 3, 4],
    "midGame": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "endGame": [0, 1, 2, 2, 2, 3, 3, 3, 4, 4],
}

DROP_Y = 100          # cloud.y (50) + 50, as in the browser
GRAVITY = 20 * 60.0   # p5play applies gravity per-frame at 60fps; pymunk is per-second

# ── Env configuration ────────────────────────────────────────────────────────
N_DROP_COLUMNS = 24
GRID_W = 22
GRID_H = 30
GRID_CHANNELS = 2
N_SCALARS = 12
SPAWN_TIERS = 5        # spawn pool only ever yields tiers 0-4

MAX_FRUIT_NORM = 60.0  # normalising constant for the fruit-count scalar
MAX_DROPS = 300        # episode cap, so a very good policy still terminates

# Settle detection: after a drop, run physics until everything is nearly still.
PHYSICS_DT = 1.0 / 60.0
SETTLE_MAX_STEPS = 600         # 10s of sim; hard ceiling per drop
SETTLE_VELOCITY = 12.0         # px/s below which a body counts as at rest
SETTLE_QUIET_STEPS = 6         # consecutive quiet frames required

# Rewards
REWARD_MERGE_SCALE = 0.1       # POINTS[t] * this
REWARD_LOSS = -1.0
REWARD_HEIGHT_PENALTY = 0.05   # scaled by how far the stack rises past the line

COLLISION_TYPE_FRUIT = 1


class WatermelonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.obs_size = GRID_W * GRID_H * GRID_CHANNELS + N_SCALARS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_DROP_COLUMNS)
        self._rng = np.random.default_rng()

    # ── Gymnasium API ────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, GRAVITY)
        # Chipmunk sleeps bodies that stop moving; that both speeds up settling
        # and stops a tall stack jittering forever.
        self.space.sleep_time_threshold = 0.3
        self.space.idle_speed_threshold = SETTLE_VELOCITY

        self._build_walls()

        self.fruits = {}          # body -> tier
        self._pending_merges = []
        self._install_merge_handler()

        self.drops = 0
        self.score = 0
        self.done = False

        self.held_tier = self._weighted_tier()
        self.next_tier = self._weighted_tier()

        return self._get_obs(), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode — call reset() first")

        action = int(np.clip(action, 0, N_DROP_COLUMNS - 1))
        tier = self.held_tier
        radius = DIAMETERS[tier] / 2.0

        # Keep the whole fruit inside the walls, the same clamp the browser's
        # cloud does when the pointer runs past the edge.
        frac = (action + 0.5) / N_DROP_COLUMNS
        x = min(CANVAS_W - radius - 1, max(radius + 1, frac * CANVAS_W))

        self._spawn_fruit(x, DROP_Y, tier)
        self.drops += 1

        gained = self._settle()
        self.score += gained

        reward = gained * REWARD_MERGE_SCALE

        # Structural penalty: punish a stack creeping over the loss line rather
        # than waiting for the death it eventually causes. Same reasoning as
        # Tetris's per-hole penalty.
        top = self._stack_top_y()
        if top < LOSS_LINE_Y:
            overshoot = (LOSS_LINE_Y - top) / max(1.0, LOSS_LINE_Y)
            reward -= REWARD_HEIGHT_PENALTY * min(1.0, overshoot)

        self.held_tier = self.next_tier
        self.next_tier = self._weighted_tier()

        terminated = self._is_lost()
        if terminated:
            reward += REWARD_LOSS
            self.done = True

        truncated = (not terminated) and self.drops >= MAX_DROPS
        if truncated:
            self.done = True

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    # ── Physics setup ────────────────────────────────────────────────────
    def _build_walls(self):
        static = self.space.static_body
        # Left, right, floor. No ceiling: fruit must be free to poke above the
        # loss line, because that is exactly what ends the game.
        segments = [
            ((0, -CANVAS_H), (0, CANVAS_H)),
            ((CANVAS_W, -CANVAS_H), (CANVAS_W, CANVAS_H)),
            ((0, CANVAS_H), (CANVAS_W, CANVAS_H)),
        ]
        for a, b in segments:
            seg = pymunk.Segment(static, a, b, 1.0)
            seg.friction = 0.5
            seg.elasticity = 0.0      # browser sets bounciness = 0
            self.space.add(seg)

    def _install_merge_handler(self):
        # pymunk 7 replaced Space.add_collision_handler() with Space.on_collision(),
        # and the callbacks now return None rather than a bool.
        def on_begin(arbiter, space, data):
            a, b = arbiter.shapes
            ba, bb = a.body, b.body
            ta, tb = self.fruits.get(ba), self.fruits.get(bb)
            if ta is None or tb is None or ta != tb or ta >= MAX_TIER:
                return
            # Defer: mutating the space inside a callback is not allowed.
            self._pending_merges.append((ba, bb, ta))

        self.space.on_collision(
            COLLISION_TYPE_FRUIT, COLLISION_TYPE_FRUIT, begin=on_begin
        )

    def _spawn_fruit(self, x, y, tier):
        radius = DIAMETERS[tier] / 2.0
        # Mass scaled by area, so bigger fruit behave heavier as in the browser
        # (which calls resetMass() after setting the diameter).
        mass = math.pi * radius * radius * 0.001
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        shape.elasticity = 0.0
        shape.collision_type = COLLISION_TYPE_FRUIT
        self.space.add(body, shape)
        self.fruits[body] = tier
        return body

    def _resolve_merges(self):
        """Apply queued merges. Returns points gained."""
        gained = 0
        seen = set()
        for ba, bb, tier in self._pending_merges:
            if ba in seen or bb in seen:
                continue
            if ba not in self.fruits or bb not in self.fruits:
                continue
            if self.fruits[ba] != tier or self.fruits[bb] != tier:
                continue

            seen.add(ba)
            seen.add(bb)
            mid = ((ba.position.x + bb.position.x) / 2.0,
                   (ba.position.y + bb.position.y) / 2.0)

            for body in (ba, bb):
                for s in list(body.shapes):
                    self.space.remove(s)
                self.space.remove(body)
                del self.fruits[body]

            gained += POINTS[tier]
            if tier < MAX_TIER:
                self._spawn_fruit(mid[0], mid[1], tier + 1)

        self._pending_merges.clear()
        return gained

    def _settle(self):
        """Run physics until motion dies down. Returns points gained."""
        gained = 0
        quiet = 0
        for _ in range(SETTLE_MAX_STEPS):
            self.space.step(PHYSICS_DT)
            if self._pending_merges:
                gained += self._resolve_merges()
                quiet = 0
                continue
            if self._max_speed() < SETTLE_VELOCITY:
                quiet += 1
                if quiet >= SETTLE_QUIET_STEPS:
                    break
            else:
                quiet = 0
        # A merge on the final frame can leave something still moving; drain it.
        gained += self._resolve_merges()
        return gained

    def _max_speed(self):
        fastest = 0.0
        for body in self.fruits:
            v = body.velocity
            speed = v.x * v.x + v.y * v.y
            if speed > fastest:
                fastest = speed
        return math.sqrt(fastest)

    def _stack_top_y(self):
        """Smallest y (highest point) of any resting fruit; CANVAS_H if empty."""
        top = float(CANVAS_H)
        for body, tier in self.fruits.items():
            top = min(top, body.position.y - DIAMETERS[tier] / 2.0)
        return top

    def _is_lost(self):
        """Lost when a settled fruit sits above the loss line."""
        for body, tier in self.fruits.items():
            if body.position.y - DIAMETERS[tier] / 2.0 < LOSS_LINE_Y:
                # Ignore one still falling — the browser only ends the game on
                # a fruit that is actually resting up there.
                if abs(body.velocity.y) < SETTLE_VELOCITY:
                    return True
        return False

    def _weighted_tier(self):
        if self.drops > 50:
            pool = WEIGHTED["endGame"]
        elif self.drops > 25:
            pool = WEIGHTED["midGame"]
        else:
            pool = WEIGHTED["initGame"]
        return int(pool[int(self._rng.integers(0, len(pool)))])

    # ── Observation ──────────────────────────────────────────────────────
    def _get_obs(self):
        grid = np.zeros((GRID_H, GRID_W, GRID_CHANNELS), dtype=np.float32)
        cell_w = CANVAS_W / GRID_W
        cell_h = CANVAS_H / GRID_H

        for body, tier in self.fruits.items():
            cx, cy = body.position.x, body.position.y
            r = DIAMETERS[tier] / 2.0
            # Mark every cell whose centre falls inside the circle.
            col_lo = max(0, int((cx - r) / cell_w))
            col_hi = min(GRID_W - 1, int((cx + r) / cell_w))
            row_lo = max(0, int((cy - r) / cell_h))
            row_hi = min(GRID_H - 1, int((cy + r) / cell_h))
            for row in range(row_lo, row_hi + 1):
                py = (row + 0.5) * cell_h
                for col in range(col_lo, col_hi + 1):
                    px = (col + 0.5) * cell_w
                    if (px - cx) ** 2 + (py - cy) ** 2 <= r * r:
                        grid[row, col, 0] = 1.0
                        grid[row, col, 1] = tier / MAX_TIER

        held = np.zeros(SPAWN_TIERS, dtype=np.float32)
        held[min(self.held_tier, SPAWN_TIERS - 1)] = 1.0
        nxt = np.zeros(SPAWN_TIERS, dtype=np.float32)
        nxt[min(self.next_tier, SPAWN_TIERS - 1)] = 1.0

        top_frac = np.float32(
            max(0.0, min(1.0, 1.0 - self._stack_top_y() / CANVAS_H))
        )
        count_frac = np.float32(min(1.0, len(self.fruits) / MAX_FRUIT_NORM))

        return np.concatenate([
            grid.flatten(),
            held,
            nxt,
            np.array([top_frac, count_frac], dtype=np.float32),
        ])
