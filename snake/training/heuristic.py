"""
Hand-coded Snake heuristic — this build's equivalent of the Dellacherie
heuristic used to pretrain the Tetris AI. Used to generate expert
(observation, action) pairs for behavioural cloning in pretrain.py.

Two-tier strategy:
  1. If there's a safe shortest path to the food (BFS avoiding the snake's
     own body), take the first step of it.
  2. Otherwise there's no safe route to food right now — survive instead,
     by picking whichever of the 3 legal relative moves leaves the most
     reachable open space (flood fill), so the snake doesn't wall itself
     into a shrinking pocket.
"""

from collections import deque

from snake_env import DIRS, TURN_LEFT, STRAIGHT, TURN_RIGHT


def heuristic_action(body, dir_idx, food, tile_count):
    head = body[0]
    # The tail vacates its cell this move (unless we eat), so it's safe to
    # path through for planning purposes.
    blocked = set(body[:-1])

    path = _bfs_path(head, food, blocked, tile_count)
    if path:
        next_dir = _step_to_dir(head, path[0])
        return _dir_to_relative_action(dir_idx, next_dir)

    return _safest_move(head, dir_idx, blocked, tile_count)


# ── Pathfinding ──────────────────────────────────────────────────────────
def _bfs_path(start, target, blocked, tile_count):
    if start == target:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in DIRS:
            npos = (x + dx, y + dy)
            nx, ny = npos
            if not (0 <= nx < tile_count and 0 <= ny < tile_count):
                continue
            if npos in visited or npos in blocked:
                continue
            new_path = path + [npos]
            if npos == target:
                return new_path
            visited.add(npos)
            queue.append((npos, new_path))
    return None


def _flood_fill_size(start, blocked, tile_count, cap):
    if start in blocked:
        return 0
    seen = {start}
    queue = deque([start])
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        if count >= cap:
            return count
        for dx, dy in DIRS:
            npos = (x + dx, y + dy)
            nx, ny = npos
            if not (0 <= nx < tile_count and 0 <= ny < tile_count):
                continue
            if npos in seen or npos in blocked:
                continue
            seen.add(npos)
            queue.append(npos)
    return count


def _safest_move(head, dir_idx, blocked, tile_count):
    best_action, best_score = None, -1
    for action in (TURN_LEFT, STRAIGHT, TURN_RIGHT):
        nd = _relative_to_dir(dir_idx, action)
        dx, dy = DIRS[nd]
        npos = (head[0] + dx, head[1] + dy)
        nx, ny = npos
        if not (0 <= nx < tile_count and 0 <= ny < tile_count):
            continue
        if npos in blocked:
            continue
        space = _flood_fill_size(npos, blocked, tile_count, cap=tile_count * tile_count)
        if space > best_score:
            best_score = space
            best_action = action
    return best_action if best_action is not None else STRAIGHT


# ── Direction <-> relative-action conversions ───────────────────────────
def _step_to_dir(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    return DIRS.index(delta)


def _relative_to_dir(cur_dir_idx, action):
    if action == TURN_LEFT:
        return (cur_dir_idx - 1) % 4
    if action == TURN_RIGHT:
        return (cur_dir_idx + 1) % 4
    return cur_dir_idx


def _dir_to_relative_action(cur_dir_idx, next_dir_idx):
    diff = (next_dir_idx - cur_dir_idx) % 4
    if diff == 0:
        return STRAIGHT
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    # diff == 2 is a direct U-turn — shouldn't happen from a legal BFS step
    # against the snake's own body, but fall back safely just in case.
    return STRAIGHT
