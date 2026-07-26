"""
Snake environment for training the AI opponent.

Design notes:

- Action space is Discrete(3): turn left / straight / turn right, relative to
  the current heading. This structurally rules out reversing into yourself, so
  unlike Tetris there's no need for MaskablePPO — plain PPO is enough since
  every action is always legal.

- OBSERVATION (v2). The previous version was a 16x16x3 one-hot grid flattened
  into an MLP, and it plateaued at 37.5 average — well below the 51.96 scored
  by the BFS + flood-fill heuristic it was cloned from. The cause was an
  information asymmetry: the teacher decides using reachable-space flood fill,
  but the student's observation contained nothing from which a flat MLP could
  practically compute connectivity. v2 closes that gap by handing the agent
  the same signal the teacher uses:

      grid, tile_count^2 x 5 channels, row-major (y, x, c)
        0  body excluding head
        1  head
        2  food
        3  tail          - the cell that vacates next move, so entering it is safe
        4  reachable     - flood fill from the head over free cells
      scalars (14)
        4  direction one-hot
        1  normalised length
        3  "this move kills me" flag, per relative action
        3  reachable free space after that move, as a fraction of the board
        2  signed food delta (dx, dy), normalised
        1  normalised steps since last food

  Layout stays FLAT so ONNX export and the JS mirror stay simple; train.py's
  feature extractor reshapes the grid part back to (5, H, W) for the CNN.

- REWARD (v2). Distance shaping now decays as the snake grows. Beelining at
  food is right when short and is exactly what traps you when long, so the
  shaping term fades out and a graded trap penalty fades in — the same idea as
  Tetris's per-hole penalty, punishing the structural mistake rather than the
  outcome. Death stays a hard terminal penalty; there is still no flat
  per-step penalty, which on Tetris taught the agent to end episodes early.

CHANGING THE OBSERVATION OR REWARD MEANS RETRAINING FROM SCRATCH. Never resume
a checkpoint trained under a different spec. The JS encoder in ../game.js
(`buildObservation`) must mirror `_get_obs` field-for-field, in the same order.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

TILE_COUNT = 16
START_LENGTH = 3
MAX_STEPS_WITHOUT_FOOD = TILE_COUNT * TILE_COUNT * 2

# Cardinal directions in clockwise order, so turning right is index+1 and
# turning left is index-1 (mod 4). 0=up, 1=right, 2=down, 3=left.
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

TURN_LEFT, STRAIGHT, TURN_RIGHT = 0, 1, 2

GRID_CHANNELS = 5
N_SCALARS = 14

# Reward weights.
FOOD_REWARD = 1.0
DEATH_PENALTY = -1.0
SHAPING_SCALE = 0.01
SHAPING_FADE_LENGTH = 40.0   # shaping reaches zero around this body length
TRAP_PENALTY = 0.1           # max per-step penalty for boxing yourself in


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tile_count: int = TILE_COUNT):
        super().__init__()
        self.tile_count = tile_count
        self.cells = tile_count * tile_count
        self.obs_size = self.cells * GRID_CHANNELS + N_SCALARS
        # low is -1: the food delta scalars are signed.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng()

    # ── Gymnasium API ───────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.dir_idx = int(self._rng.integers(0, 4))
        margin = START_LENGTH + 1
        cx = int(self._rng.integers(margin, self.tile_count - margin))
        cy = int(self._rng.integers(margin, self.tile_count - margin))

        dx, dy = DIRS[self.dir_idx]
        # body[0] is the head; segments trail behind the direction of travel
        self.body = [(cx - i * dx, cy - i * dy) for i in range(START_LENGTH)]

        self._place_food()
        self.steps_since_food = 0
        self.score = 0
        self.done = False

        return self._get_obs(), {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode — call reset() first")

        if action == TURN_LEFT:
            self.dir_idx = (self.dir_idx - 1) % 4
        elif action == TURN_RIGHT:
            self.dir_idx = (self.dir_idx + 1) % 4
        # STRAIGHT: dir_idx unchanged

        dx, dy = DIRS[self.dir_idx]
        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)
        prev_dist = self._food_distance(head_x, head_y)

        ate = new_head == self.food
        # If we're not eating, the tail vacates its cell this move, so moving
        # into it is safe. If we are eating, the tail stays put.
        blocking_body = self.body if ate else self.body[:-1]

        if self._out_of_bounds(new_head) or new_head in blocking_body:
            self.done = True
            return self._get_obs(), DEATH_PENALTY, True, False, {"score": self.score}

        self.body.insert(0, new_head)
        if ate:
            self._place_food()
            self.score += 1
            self.steps_since_food = 0
        else:
            self.body.pop()
            self.steps_since_food += 1

        if ate:
            reward = FOOD_REWARD
        else:
            # Fade the distance bonus out as the body grows: chasing food in a
            # straight line is correct early and suicidal late.
            new_dist = self._food_distance(*new_head)
            fade = max(0.0, 1.0 - len(self.body) / SHAPING_FADE_LENGTH)
            reward = SHAPING_SCALE * fade * (prev_dist - new_dist)

        # Graded penalty for cutting yourself off from space. Analogous to
        # Tetris's per-hole penalty: punish the structural error directly
        # rather than waiting for the death it eventually causes.
        #
        # body[1:-1], NOT body[:-1]: the head is the flood-fill's starting
        # cell, so including it in the blocked set makes _free_space return 0
        # every single time. That turned this graded penalty into a flat
        # -0.1 per step, which inverted the whole reward — a full 50-food
        # episode paid -18.1 while dying instantly paid -1.0, so the optimal
        # policy was to die immediately. The tail is excluded because it
        # vacates as we move, matching _get_obs's reachable mask.
        free = self._free_space(self.body[0], self.body[1:-1])
        if free < len(self.body):
            reward -= TRAP_PENALTY * (1.0 - free / max(1, len(self.body)))

        truncated = self.steps_since_food >= MAX_STEPS_WITHOUT_FOOD
        if truncated:
            self.done = True

        return self._get_obs(), reward, False, truncated, {"score": self.score}

    # ── Helpers ──────────────────────────────────────────────────────────
    def _out_of_bounds(self, pos):
        x, y = pos
        return x < 0 or x >= self.tile_count or y < 0 or y >= self.tile_count

    def _place_food(self):
        occupied = set(self.body)
        while True:
            fx = int(self._rng.integers(0, self.tile_count))
            fy = int(self._rng.integers(0, self.tile_count))
            if (fx, fy) not in occupied:
                self.food = (fx, fy)
                return

    def _food_distance(self, x, y):
        return abs(x - self.food[0]) + abs(y - self.food[1])

    # Flood fill runs up to five times per step, so it is the hot path of the
    # whole environment. Tuple-keyed sets cost ~750us/step; flat integer cell
    # indices into a reusable bytearray cut that by roughly 5x. Semantics are
    # unchanged, and the integer keying now matches the JS mirror's cellKey().
    def _blocked_mask(self, cells_iter):
        mask = bytearray(self.cells)
        n = self.tile_count
        for x, y in cells_iter:
            if 0 <= x < n and 0 <= y < n:
                mask[y * n + x] = 1
        return mask

    def _reachable_idx(self, start, blocked):
        """
        Indices 4-connected to `start`, avoiding `blocked` (a bytearray mask).

        Runs the component to completion rather than stopping at a cap, so the
        result is independent of traversal order — which is what lets the JS
        mirror produce identical numbers.
        """
        n = self.tile_count
        sx, sy = start
        if not (0 <= sx < n and 0 <= sy < n):
            return []
        seen = bytearray(self.cells)
        s0 = sy * n + sx
        seen[s0] = 1
        stack = [s0]
        out = [s0]
        while stack:
            cur = stack.pop()
            cy, cx = divmod(cur, n)
            if cx > 0:
                k = cur - 1
                if not seen[k] and not blocked[k]:
                    seen[k] = 1; stack.append(k); out.append(k)
            if cx < n - 1:
                k = cur + 1
                if not seen[k] and not blocked[k]:
                    seen[k] = 1; stack.append(k); out.append(k)
            if cy > 0:
                k = cur - n
                if not seen[k] and not blocked[k]:
                    seen[k] = 1; stack.append(k); out.append(k)
            if cy < n - 1:
                k = cur + n
                if not seen[k] and not blocked[k]:
                    seen[k] = 1; stack.append(k); out.append(k)
        return out

    def _free_space(self, start, blocked_cells):
        """`blocked_cells` is an iterable of (x, y) — kept for readability at
        call sites; converted to a mask internally."""
        n = self.tile_count
        sx, sy = start
        mask = self._blocked_mask(blocked_cells)
        if not (0 <= sx < n and 0 <= sy < n) or mask[sy * n + sx]:
            return 0
        return len(self._reachable_idx(start, mask))

    def _candidate_moves(self):
        """(is_fatal, free_space_fraction) for each relative action."""
        head_x, head_y = self.body[0]
        n = self.tile_count
        mask = self._blocked_mask(self.body[:-1])
        out = []
        for action in (TURN_LEFT, STRAIGHT, TURN_RIGHT):
            if action == TURN_LEFT:
                d = (self.dir_idx - 1) % 4
            elif action == TURN_RIGHT:
                d = (self.dir_idx + 1) % 4
            else:
                d = self.dir_idx
            dx, dy = DIRS[d]
            nx, ny = head_x + dx, head_y + dy
            if not (0 <= nx < n and 0 <= ny < n) or mask[ny * n + nx]:
                out.append((1.0, 0.0))
            else:
                out.append((0.0, len(self._reachable_idx((nx, ny), mask)) / self.cells))
        return out

    def _get_obs(self):
        n = self.tile_count
        grid = np.zeros((n, n, GRID_CHANNELS), dtype=np.float32)

        for x, y in self.body[1:]:
            grid[y, x, 0] = 1.0                      # body excluding head
        hx, hy = self.body[0]
        grid[hy, hx, 1] = 1.0                        # head
        fx, fy = self.food
        grid[fy, fx, 2] = 1.0                        # food
        tx, ty = self.body[-1]
        grid[ty, tx, 3] = 1.0                        # tail

        # Reachable space from the head. The tail is treated as free because
        # it vacates as we move.
        mask = self._blocked_mask(self.body[1:-1])
        flat4 = grid.reshape(-1, GRID_CHANNELS)
        for idx in self._reachable_idx((hx, hy), mask):
            flat4[idx, 4] = 1.0

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.dir_idx] = 1.0

        length_norm = np.float32(len(self.body) / self.cells)

        moves = self._candidate_moves()
        fatal = np.array([m[0] for m in moves], dtype=np.float32)
        space = np.array([m[1] for m in moves], dtype=np.float32)

        food_delta = np.array(
            [(fx - hx) / n, (fy - hy) / n], dtype=np.float32
        )
        hunger = np.float32(self.steps_since_food / MAX_STEPS_WITHOUT_FOOD)

        return np.concatenate([
            grid.flatten(),
            dir_onehot,
            np.array([length_norm], dtype=np.float32),
            fatal,
            space,
            food_delta,
            np.array([hunger], dtype=np.float32),
        ])
