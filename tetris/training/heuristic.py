"""
heuristic.py
------------
Pierre Dellacherie heuristic Tetris player.
Plays games using a hand-tuned scoring function and records
every (observation, action) pair to heuristic_data.json.

This data is used by pretrain.py to teach the neural network
expert-level play before RL fine-tuning.

Usage:
    python3 heuristic.py              # generate 1000 games
    python3 heuristic.py --games 5000 # generate more

The heuristic evaluates every valid placement by scoring:
  - Lines cleared (positive)
  - Holes created (very negative)
  - Bumpiness (negative)
  - Aggregate height (negative)
  - Height of tallest column (very negative)
  - Wells (slightly negative — one well is okay for I-pieces)

Weights from Pierre Dellacherie's original paper, tuned for
near-perfect play.
"""

import json
import os
import sys
import numpy as np
from tetris_env import (
    TetrisEnv, get_valid_placements, lock_and_clear,
    col_heights, count_holes, build_obs, rotate_matrix,
    hard_drop_y, has_collision, spawn_y, piece_cells,
    ARENA_WIDTH, ARENA_HEIGHT, clone_matrix
)

# ─── Dellacherie weights ──────────────────────────────────────────────────────
# These weights produce near-perfect play
WEIGHTS = {
    "lines_cleared":    3.4758,   # reward line clears
    "holes":           -7.8990,   # punish holes heavily
    "bumpiness":       -3.3836,   # punish uneven surface
    "agg_height":      -0.5100,   # punish overall height
    "max_height":      -0.3500,   # extra punish for tall stacks
    "wells":           -0.1840,   # slight punish for wells (one is okay)
}

def count_wells(heights):
    """Count total well depth — sum of differences where a col is lower than both neighbours."""
    wells = 0
    for i in range(len(heights)):
        left  = heights[i-1] if i > 0 else ARENA_HEIGHT
        right = heights[i+1] if i < len(heights)-1 else ARENA_HEIGHT
        if heights[i] < left and heights[i] < right:
            wells += min(left, right) - heights[i]
    return wells

def score_placement(arena, shape, col, drop_y):
    """
    Score a placement using the Dellacherie heuristic.
    Higher score = better placement.
    """
    # Simulate placement
    sim_arena, lines = lock_and_clear(arena, shape, col, drop_y)

    h      = col_heights(sim_arena)
    holes  = count_holes(sim_arena)
    bumps  = sum(abs(h[i]-h[i+1]) for i in range(ARENA_WIDTH-1))
    agg_h  = sum(h)
    max_h  = max(h)
    wells  = count_wells(h)

    score = (
        WEIGHTS["lines_cleared"] * lines +
        WEIGHTS["holes"]         * holes +
        WEIGHTS["bumpiness"]     * bumps +
        WEIGHTS["agg_height"]    * agg_h +
        WEIGHTS["max_height"]    * max_h +
        WEIGHTS["wells"]         * wells
    )
    return score

def heuristic_choose(arena, shape):
    """
    Evaluate all valid placements and return the index of the best one.
    Returns (best_idx, placements).
    """
    placements = get_valid_placements(arena, shape)
    if not placements:
        return None, []

    best_score = -float("inf")
    best_idx   = 0

    for i, (rotated, col, dy) in enumerate(placements):
        s = score_placement(arena, rotated, col, dy)
        if s > best_score:
            best_score = s
            best_idx   = i

    return best_idx, placements

def play_game(env, rng):
    """
    Play one game using the heuristic.
    Returns list of (obs, action_idx) pairs.
    """
    obs, _ = env.reset()
    records = []
    done    = False

    while not done:
        # Get current state
        arena       = env.arena
        piece       = env.piece
        next_pieces = env.next_pieces

        # Build observation
        observation = build_obs(arena, piece, next_pieces, env.combo)

        # Heuristic chooses best placement
        best_idx, placements = heuristic_choose(arena, piece)
        if best_idx is None:
            break

        # Record this (obs, action) pair
        records.append({
            "obs":    observation.tolist(),
            "action": best_idx,
        })

        # Step the environment
        obs, reward, done, _, info = env.step(best_idx)

    return records, info.get("score", 0), info.get("total_lines", 0)

def main():
    n_games = 1000
    if "--games" in sys.argv:
        idx = sys.argv.index("--games")
        n_games = int(sys.argv[idx+1])

    output_file = "heuristic_data.json"

    print("=" * 55)
    print("  Dellacherie Heuristic — Generating Training Data")
    print("=" * 55)
    print(f"\nGenerating {n_games} games...")
    print(f"Output: {output_file}\n")

    env = TetrisEnv()
    rng = np.random.default_rng()

    all_records  = []
    total_score  = 0
    total_lines  = 0
    best_score   = 0

    for game in range(n_games):
        records, score, lines = play_game(env, rng)
        all_records.extend(records)
        total_score += score
        total_lines += lines
        best_score   = max(best_score, score)

        if (game+1) % 100 == 0:
            avg_score = total_score / (game+1)
            avg_lines = total_lines / (game+1)
            print(f"  Game {game+1:4d}/{n_games}  |  "
                  f"Avg score: {avg_score:.0f}  |  "
                  f"Avg lines: {avg_lines:.1f}  |  "
                  f"Best: {best_score}  |  "
                  f"Records: {len(all_records):,}")

    print(f"\nSaving {len(all_records):,} records to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_records, f)

    print(f"\nDone!")
    print(f"Total records: {len(all_records):,}")
    print(f"Average score: {total_score/n_games:.0f}")
    print(f"Average lines: {total_lines/n_games:.1f}")
    print(f"Best score:    {best_score}")
    print(f"\nNow run: python3 pretrain.py --data heuristic_data.json")

if __name__ == "__main__":
    main()
