"""Report the observation/action shape of every archived checkpoint.

The checkpoint switcher can only offer a checkpoint whose observation layout
still matches the encoder in game.js. That layout HAS changed during this
project — snake/training/archive/models/snake_final.v1_773obs.zip is named
after exactly this problem — and a mismatch does not fail loudly: onnxruntime
would happily run a 773-wide model on a 1294-wide tensor if the graph allowed
it, or throw at a point far from the cause.

So probe first, build the ladder from what survives.

    python tools/probe_checkpoints.py
"""

import glob
import os
import sys

sys.stderr = open(os.devnull, "w")  # SB3 prints load warnings we do not need

from stable_baselines3 import PPO  # noqa: E402

# What each game's game.js actually builds. Keep in step with the identical
# table in tools/install_model.sh.
ENCODER_WIDTH = {
    "snake": 16 * 16 * 5 + 14,
    "watermelon": 22 * 30 * 2 + 12,
    "tetris": None,  # tetris/game.js builds its own; probed but not checked
}


def algo_for(game):
    """Tetris trains with action masking, so its checkpoints are MaskablePPO.
    Loading one with plain PPO fails with an unexpected-keyword TypeError deep
    inside the policy constructor, which reads like a version incompatibility
    rather than the wrong class."""
    if game == "tetris":
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    return PPO


def probe(path, game):
    # Each game defines its own policy class in <game>/training/policy_config.py,
    # and the checkpoint stores it by import path. Without that directory on
    # sys.path every load fails with ModuleNotFoundError, which looks like a
    # corrupt checkpoint but is only a missing import root.
    #
    # All three files share the module NAME, so the first one imported wins and
    # stays in sys.modules for the rest of the process — which silently gave
    # Watermelon's checkpoints Snake's policy class. Drop the cached module and
    # the previous game's path before each game.
    root = os.path.abspath(f"{game}/training")
    for stale in [p for p in sys.path if p.endswith(os.path.join("training"))]:
        if stale != root:
            sys.path.remove(stale)
    sys.modules.pop("policy_config", None)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        m = algo_for(game).load(path, device="cpu")
        obs = m.observation_space.shape
        acts = int(m.action_space.n)
        return (obs[0] if len(obs) == 1 else obs), acts, None
    except Exception as e:  # a checkpoint from an older SB3/gym can simply refuse
        return None, None, type(e).__name__ + ": " + str(e)[:160]


def main():
    for game in ("snake", "watermelon", "tetris"):
        want = ENCODER_WIDTH[game]
        print(f"\n===== {game}  (game.js encoder: {want}) =====")
        paths = sorted(glob.glob(f"{game}/training/archive/models/*.zip"))
        paths += sorted(glob.glob(f"{game}/training/backups/*.zip"))
        paths += [p for p in (f"{game}/training/{game}_final.zip",) if os.path.exists(p)]
        if not paths:
            print("  (no checkpoints)")
            continue
        for p in paths:
            obs, acts, err = probe(p, game)
            name = os.path.basename(p)
            mb = os.path.getsize(p) / 1048576
            if err:
                print(f"  {name:<46} {mb:6.1f} MB  UNREADABLE  {err}")
                continue
            ok = "-" if want is None else ("OK " if obs == want else "MISMATCH")
            print(f"  {name:<46} {mb:6.1f} MB  obs={obs}  actions={acts}  {ok}")


if __name__ == "__main__":
    main()
