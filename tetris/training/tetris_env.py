"""
tetris_env.py — v3 Minimalist Reward Design
--------------------------------------------
Philosophy: fewer, stronger signals instead of many competing ones.

Only 3 reward signals:
  1. Contact density — piece must be well-connected to get any reward
  2. Line clears — heavily weighted toward Tetris (4-line) clears
  3. Holes — hard penalty, episode ends if holes exceed threshold

No survival bonus. No height rewards. No bumpiness. No flatness.
Just: stack cleanly, clear lines, don't make holes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ARENA_WIDTH   = 10
ARENA_HEIGHT  = 18
MAX_ROTATIONS = 4
OBS_SIZE      = 238
MAX_HOLES     = 8   # episode ends immediately if holes exceed this

SHAPES = [
    [[1,1,1],[0,1,0],[0,0,0]],
    [[2,2],[2,2]],
    [[0,0,3,0],[0,0,3,0],[0,0,3,0],[0,0,3,0]],
    [[0,4,0],[0,4,0],[0,4,4]],
    [[0,5,0],[0,5,0],[5,5,0]],
    [[0,6,6],[6,6,0],[0,0,0]],
    [[7,7,0],[0,7,7],[0,0,0]],
]

def clone_matrix(m): return [row[:] for row in m]

def rotate_matrix(matrix, times=1):
    m = clone_matrix(matrix)
    for _ in range(times % 4):
        n = len(m)
        for r in range(n):
            for c in range(r): m[r][c], m[c][r] = m[c][r], m[r][c]
        for row in m: row.reverse()
    return m

def piece_cells(shape, px, py):
    return [(py+r, px+c) for r,row in enumerate(shape) for c,v in enumerate(row) if v]

def get_piece_id(shape):
    for row in shape:
        for v in row:
            if v: return v-1
    return 0

def new_bag(rng):
    bag = [clone_matrix(s) for s in SHAPES]
    rng.shuffle(bag)
    return bag

def has_collision(arena, shape, px, py):
    for r,row in enumerate(shape):
        for c,v in enumerate(row):
            if not v: continue
            ri,ci = py+r, px+c
            if ci<0 or ci>=ARENA_WIDTH: return True
            if ri>=ARENA_HEIGHT: return True
            if ri>=0 and arena[ri][ci]: return True
    return False

def spawn_y(shape): return -len(shape)

def lock_and_clear(arena, shape, px, py):
    a = clone_matrix(arena)
    for r,row in enumerate(shape):
        for c,v in enumerate(row):
            if not v: continue
            ri,ci = py+r, px+c
            if 0<=ri<ARENA_HEIGHT and 0<=ci<ARENA_WIDTH: a[ri][ci] = v
    lines = 0
    r = ARENA_HEIGHT-1
    while r>=0:
        if all(a[r]): a.pop(r); a.insert(0,[0]*ARENA_WIDTH); lines+=1
        else: r-=1
    return a, lines

def hard_drop_y(arena, shape, px, py):
    while not has_collision(arena, shape, px, py+1): py+=1
    return py

def col_heights(arena):
    h = []
    for c in range(ARENA_WIDTH):
        ht = 0
        for r in range(ARENA_HEIGHT):
            if arena[r][c]: ht = ARENA_HEIGHT-r; break
        h.append(ht)
    return h

def count_holes(arena):
    holes = 0
    for c in range(ARENA_WIDTH):
        filled = False
        for r in range(ARENA_HEIGHT):
            if arena[r][c]: filled = True
            elif filled: holes += 1
    return holes

def row_completeness_reward(arena):
    """
    For each row, reward the fraction of cells filled weighted by depth.
    A nearly-complete row near the bottom is very valuable.
    Returns a float reward.
    """
    reward = 0.0
    for r in range(ARENA_HEIGHT):
        filled = sum(1 for c in range(ARENA_WIDTH) if arena[r][c])
        if filled == 0:
            continue
        fraction     = filled / ARENA_WIDTH
        depth_weight = (r + 1) / ARENA_HEIGHT  # 0.06 at top, 1.0 at bottom
        # Exponential — nearly full rows are way more valuable
        reward += (fraction ** 2) * depth_weight * 2.0
    return reward

def contact_density(arena, shape, col, drop_y):
    """
    For each cell of the placed piece, count how many of its 4 sides
    touch a filled cell or wall. Weight by depth.
    Returns (score 0-1, has_floating bool).
    """
    total_score  = 0.0
    has_floating = False
    n_cells      = 0

    for r,row in enumerate(shape):
        for c,v in enumerate(row):
            if not v: continue
            ri,ci = drop_y+r, col+c
            if not (0<=ri<ARENA_HEIGHT and 0<=ci<ARENA_WIDTH): continue
            n_cells += 1
            depth_weight = (ri+1) / ARENA_HEIGHT

            contacts = 0
            if ci==0 or (ci>0 and arena[ri][ci-1]):             contacts+=1  # left
            if ci==ARENA_WIDTH-1 or (ci<ARENA_WIDTH-1 and arena[ri][ci+1]): contacts+=1  # right
            if ri>0 and arena[ri-1][ci]:                         contacts+=1  # above
            if ri==ARENA_HEIGHT-1 or (ri+1<ARENA_HEIGHT and arena[ri+1][ci]): contacts+=1  # below

            if contacts == 0:
                has_floating = True
            total_score += contacts * depth_weight

    max_possible = 4.0 * n_cells
    return (total_score / max(max_possible, 1)), has_floating

def get_valid_placements(arena, shape):
    seen, placements = [], []
    for rot in range(MAX_ROTATIONS):
        rotated = rotate_matrix(shape, rot)
        if any(rotated==s for s in seen): continue
        seen.append(rotated)
        pw = len(rotated[0])
        for col in range(-1, ARENA_WIDTH-pw+2):
            sy = spawn_y(rotated)
            if has_collision(arena, rotated, col, sy): continue
            dy = hard_drop_y(arena, rotated, col, sy)
            cells = piece_cells(rotated, col, dy)
            if any(0<=r<ARENA_HEIGHT and 0<=c<ARENA_WIDTH for r,c in cells):
                placements.append((rotated, col, dy))
    return placements

def well_col_index(heights): return heights.index(min(heights))

def build_obs(arena, cur_piece, next_pieces, combo):
    board = np.array(
        [1.0 if arena[r][c] else 0.0 for r in range(ARENA_HEIGHT) for c in range(ARENA_WIDTH)],
        dtype=np.float32
    )
    h = col_heights(arena)
    heights_norm = np.array(h, dtype=np.float32) / ARENA_HEIGHT
    cur_oh = np.zeros(7, dtype=np.float32); cur_oh[get_piece_id(cur_piece)] = 1.0
    next_oh = np.zeros(35, dtype=np.float32)
    for i,piece in enumerate(next_pieces[:5]):
        if piece is not None: next_oh[i*7+get_piece_id(piece)] = 1.0
    holes  = count_holes(arena)
    bumps  = sum(abs(h[i]-h[i+1]) for i in range(ARENA_WIDTH-1))
    agg_h  = sum(h)
    max_h  = max(h)
    scalars = np.array([
        min(holes, ARENA_WIDTH*ARENA_HEIGHT)/(ARENA_WIDTH*ARENA_HEIGHT),
        min(bumps, ARENA_HEIGHT*ARENA_WIDTH)/(ARENA_HEIGHT*ARENA_WIDTH),
        min(agg_h, ARENA_HEIGHT*ARENA_WIDTH)/(ARENA_HEIGHT*ARENA_WIDTH),
        max_h/ARENA_HEIGHT,
    ], dtype=np.float32)
    combo_norm = np.array([min(combo,20)/20.0], dtype=np.float32)
    s = sorted(h)
    wd_norm = np.array([min(s[1]-s[0],ARENA_HEIGHT)/ARENA_HEIGHT], dtype=np.float32)
    return np.concatenate([board, heights_norm, cur_oh, next_oh, scalars, combo_norm, wd_norm])


class TetrisEnv(gym.Env):
    """
    Minimalist 3-signal reward Tetris environment.

    Reward signals:
      + Contact density  — piece must touch neighbours to earn reward
      + Row completeness — reward nearly-full rows near the bottom
      + Line clears      — huge reward, heavily weighted toward Tetris
      - Holes            — hard penalty per hole, episode ends at MAX_HOLES
      - Game over        — large penalty
      - Floating cells   — piece with no contacts gets penalised
    """
    metadata = {"render_modes": []}
    MAX_PLACEMENTS = 40

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(self.MAX_PLACEMENTS)
        self._episode_count    = 0
        self._reset_state()

    def _reset_state(self):
        self.arena         = [[0]*ARENA_WIDTH for _ in range(ARENA_HEIGHT)]
        self.bag           = new_bag(self.rng)
        self.piece         = self._next_piece()
        self.next_pieces   = [self._next_piece() for _ in range(5)]
        self.score         = 0
        self.pieces_placed = 0
        self.combo         = 0
        self.total_lines   = 0
        self._placements   = get_valid_placements(self.arena, self.piece)

    def _next_piece(self):
        if not self.bag: self.bag = new_bag(self.rng)
        return self.bag.pop()

    def _obs(self):
        return build_obs(self.arena, self.piece, self.next_pieces, self.combo)

    def action_masks(self):
        mask = np.zeros(self.MAX_PLACEMENTS, dtype=bool)
        for i in range(min(len(self._placements), self.MAX_PLACEMENTS)):
            mask[i] = True
        return mask

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self._episode_count += 1
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        placements = self._placements
        action = int(action) % max(len(placements), 1)
        if not placements:
            return self._obs(), -20.0, True, False, {"score": self.score}

        rotated_shape, col, drop_y = placements[action]

        # Apply placement
        new_arena, lines = lock_and_clear(self.arena, rotated_shape, col, drop_y)
        self.arena          = new_arena
        self.pieces_placed += 1
        self.total_lines   += lines

        # ══════════════════════════════════════════════════════════════════════
        # REWARD — 3 signals only
        # ══════════════════════════════════════════════════════════════════════
        reward = 0.0

        # ── 1. Contact density ────────────────────────────────────────────────
        # Piece must be well-connected to earn any reward.
        # Floating/isolated placements earn nothing or get penalised.
        cd, has_floating = contact_density(self.arena, rotated_shape, col, drop_y)
        if has_floating:
            reward -= 2.0        # hard penalty for any floating cell
        else:
            reward += cd * 4.0   # 0-4 reward based on how well-connected

        # ── 2. Row completeness ───────────────────────────────────────────────
        # Reward having nearly-full rows near the bottom.
        # This incentivises systematic filling from the bottom up.
        reward += row_completeness_reward(self.arena) * 0.5

        # ── 3. Line clears ────────────────────────────────────────────────────
        # Heavily weighted toward Tetris (4 lines).
        # Single line clears barely worth it — forces planning for bigger clears.
        LINE_REWARDS = {0: 0.0, 1: 2.0, 2: 10.0, 3: 30.0, 4: 200.0}
        line_reward  = LINE_REWARDS.get(lines, 200.0)
        if lines > 0:
            combo_bonus  = min(self.combo * 5.0, 25.0)
            reward      += line_reward + combo_bonus
            self.combo  += 1
        else:
            self.combo   = 0

        # Perfect clear
        if all(self.arena[r][c]==0 for r in range(ARENA_HEIGHT) for c in range(ARENA_WIDTH)):
            reward += 500.0

        # ── Holes — hard penalty, episode ends if too many ────────────────────
        holes = count_holes(self.arena)
        reward -= holes * 3.0   # strong per-hole penalty

        if holes > MAX_HOLES:
            # Too many holes — episode ends immediately
            # This teaches hole avoidance as a hard constraint
            reward -= 50.0
            return self._obs(), reward, True, False, {
                "score": self.score, "lines": lines,
                "combo": self.combo, "total_lines": self.total_lines,
                "reason": "too_many_holes"
            }

        # ── Loss condition ─────────────────────────────────────────────────────
        done = any(self.arena[r][c] for r in range(3) for c in range(ARENA_WIDTH))
        if done:
            reward -= 20.0

        self.score += lines * 75 + 10

        if not done:
            self.piece       = self.next_pieces.pop(0)
            self.next_pieces.append(self._next_piece())
            self._placements = get_valid_placements(self.arena, self.piece)
            if not self._placements:
                done    = True
                reward -= 20.0

        return self._obs(), reward, done, False, {
            "score":       self.score,
            "lines":       lines,
            "combo":       self.combo,
            "total_lines": self.total_lines,
        }

    def render(self):
        holes = count_holes(self.arena)
        print(f"Score:{self.score} Pieces:{self.pieces_placed} Lines:{self.total_lines} Holes:{holes}/{MAX_HOLES}")
        for r,row in enumerate(self.arena):
            line = "|"+"".join("#" if v else "." for v in row)+"|"
            if r==2: line+=" <- danger"
            print(line)
        print("+"+"-"*ARENA_WIDTH+"+\n")


if __name__ == "__main__":
    print("Running sanity check...")
    env = TetrisEnv()
    obs,_ = env.reset()
    print(f"Observation shape: {obs.shape}  (expected {OBS_SIZE})")
    print(f"Action space: {env.action_space}")
    print(f"Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    done = False; steps = 0; total_r = 0
    while not done:
        obs,r,done,_,info = env.step(env.action_space.sample())
        total_r += r; steps += 1
        if steps % 10 == 0: env.render()
    print(f"\nGame over after {steps} pieces.")
    print(f"Score:{info['score']} Lines:{info['total_lines']}")
    print(f"Total reward: {total_r:.2f}")
    print("Sanity check passed.")
