"""Build the checkpoint ladders the in-game switcher offers.

    python tools/build_checkpoints.py                # everything
    python tools/build_checkpoints.py snake          # one game
    python tools/build_checkpoints.py --episodes 10  # quicker, rougher scores

For each rung it exports the checkpoint's policy to <game>/checkpoints/<id>.onnx,
measures how strong it actually plays, and writes the manifest
shared/checkpoints.js
that the switcher reads.

WHY THE SCORES ARE MEASURED HERE RATHER THAN TAKEN FROM THE FILENAMES

The archive names carry the score each checkpoint got when it was installed
(snake_final.129pt34.zip), and reusing those would save an hour of evaluation.
But those numbers came from evaluate.py driving the SB3 model in Python, over
varying episode counts, across a month of changes to the environment. The
switcher is a claim about what the visitor is about to play against in the
BROWSER. So every rung is evaluated the same way, the same number of episodes,
through onnxruntime on the exact file the browser downloads — which also means
the export itself is exercised before it ships.

WHY EACH ONE IS A SEPARATE PROCESS

All three games define a module named policy_config, and the first one imported
wins for the life of the process. Loading Snake and then Watermelon in one
process silently gives Watermelon's checkpoints Snake's policy class. The env
modules (snake_env, tetris_env, watermelon_env) collide the same way. So each
rung's export and evaluation runs in a fresh interpreter.

WHY THE TOP RUNG IS NOT EXPORTED

The strongest rung of each ladder is the model the site already ships. Pointing
the manifest at that existing file means selecting "full strength" costs no
download at all, and there is only ever one copy of the shipped weights.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# What each game's game.js builds. A rung whose observation width disagrees is
# refused rather than shipped: onnxruntime would run it and return confident
# nonsense. Kept in step with the same table in install_model.sh.
ENCODER_WIDTH = {"snake": 1294, "watermelon": 1332, "tetris": 238}

# The rungs, weakest first. `src` is relative to <game>/training.
# `shipped` marks the rung that reuses the already-deployed model.
# WHAT `steps` MEANS, AND WHY IT IS NOT ALWAYS COMPARABLE
#
# num_timesteps is the count for the RUN that produced a checkpoint, and it
# resets whenever a run starts fresh instead of resuming. Measured on this
# ladder: snake s1 (score 70) reports 30.0M while the much stronger s3 (score
# 129) reports 6.0M, because s3 came from a short run that resumed from an
# already-trained model. Presenting those side by side as "training so far"
# would tell the visitor the opposite of the truth.
#
# So each ladder declares whether its step counts form one comparable lineage.
# Tetris does — 70M, 261M and 1B are one continuous run. Snake and Watermelon
# do not, and their switcher shows measured strength only.
LADDERS = {
    "snake": {
        "shipped_onnx": "snake/snake_ai.onnx",
        "stepsMeaning": "run",
        "episodes": 30,
        "rungs": [
            {"id": "s1", "src": "archive/models/snake_final.70pt54.zip"},
            {"id": "s2", "src": "archive/models/snake_final.105pt72.zip"},
            {"id": "s3", "src": "archive/models/snake_final.129pt34.zip"},
            {"id": "s4", "shipped": True, "src": "snake_final.zip"},
        ],
    },
    "watermelon": {
        "shipped_onnx": "watermelon/watermelon_ai.onnx",
        "stepsMeaning": "run",
        "episodes": 30,
        "rungs": [
            {"id": "w1", "src": "archive/models/watermelon_ppo_614.zip"},
            {"id": "w2", "src": "archive/models/watermelon_final.841pt87.zip"},
            {"id": "w3", "src": "archive/models/watermelon_final.936pt70.zip"},
            {"id": "w4", "shipped": True, "src": "watermelon_final.zip"},
        ],
    },
    "tetris": {
        "shipped_onnx": "tetris/training/tetris_ai.onnx",
        # One continuous run, so these three step counts are directly
        # comparable and the switcher shows them.
        "stepsMeaning": "lineage",
        # Tetris episodes run to six figures and take minutes each, so it gets
        # fewer of them. Its scores are also wildly skewed (mean 119k against a
        # median of 57k), which means a small sample is rough for Tetris in a
        # way it is not for the other two — the manifest records the count so
        # the page can say so.
        "episodes": 5,
        "rungs": [
            {"id": "t1", "src": "backups/70M BACKUP.zip"},
            {"id": "t2", "src": "backups/260M BACKUP.zip"},
            {"id": "t3", "shipped": True, "src": None},
        ],
    },
}

# Tetris' 1B-step checkpoint is gone — only its .onnx survives — so the top
# rung has no zip to read a step count from. This is the figure from its
# TensorBoard logs, which inside/data.js also uses.
TETRIS_SHIPPED_STEPS = 1000012896


def steps_of(zip_path):
    """num_timesteps out of the SB3 archive's JSON entry — no torch load."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        return int(json.loads(z.read("data")).get("num_timesteps", 0))


# ── worker modes, each run in a fresh interpreter ────────────────────────────

def do_export(game, src, out):
    """Export one checkpoint by calling the GAME'S OWN export.py.

    Reusing each game's exporter rather than reimplementing the wrapper here
    matters most for Tetris, whose export also promotes per-layer activations to
    graph outputs — the thing the brain visual on the Inside page reads. An
    export written fresh in this file would quietly drop them.
    """
    training = ROOT / game / "training"
    cmd = [sys.executable, "-u", "export.py", src, os.path.relpath(out, training)]
    if game in ("snake", "watermelon"):
        cmd.append("--no-critic")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=training, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        return False
    return True


def do_eval(game, onnx_path, episodes):
    """Play `episodes` games with this .onnx, always taking the best move.

    Argmax, not sampling: this measures the rung at full strength, so the
    difficulty temperature stays an independent dial on top of whichever rung
    is selected.

    TETRIS MUST BE MASKED, and getting this wrong is silent.

    Its 40 action logits are candidate PLACEMENTS, and only the first
    len(placements) of them are legal for the current piece. tetris/game.js caps
    the argmax there (Settings.chooseAction("tetris", logits, placements.length)).
    tetris_env.step() does NOT reject an out-of-range action — line 256 wraps it
    with `action % len(placements)` — so an unmasked argmax silently becomes an
    arbitrary legal placement instead of raising.

    Measured with no mask: the 70M checkpoint scored 559 and the 260M scored 211,
    i.e. the 260M model looked WORSE than the 70M one, and both looked ~100x
    worse than the shipped model. None of that was real; it was this bug.
    """
    import numpy as np
    import onnxruntime as ort

    training = ROOT / game / "training"
    sys.path.insert(0, str(training))
    mod, cls = {
        "snake": ("snake_env", "SnakeEnv"),
        "watermelon": ("watermelon_env", "WatermelonEnv"),
        "tetris": ("tetris_env", "TetrisEnv"),
    }[game]
    env = getattr(__import__(mod), cls)()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    # Tetris can run indefinitely once a policy is good enough to keep clearing
    # lines, so a cap keeps the build bounded. Recorded in the manifest when it
    # bites, because a capped episode understates the model.
    MOVE_CAP = 60000 if game == "tetris" else 100000

    # How many of the logits are legal right now. None means "all of them",
    # which is true for Snake (3 directions) and Watermelon (24 columns).
    def allowed(e):
        return len(e._placements) if game == "tetris" else None

    scores, capped = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, info, moves = False, {"score": 0}, 0
        while not done:
            lg = sess.run(None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            n = allowed(env)
            if n is not None:
                if n <= 0:
                    break          # no legal placement: the episode is over
                lg = lg[:min(n, len(lg))]
            obs, _, term, trunc, info = env.step(int(np.argmax(lg)))
            done = term or trunc
            moves += 1
            if moves >= MOVE_CAP:
                capped += 1
                break
        scores.append(float(info.get("score", 0)))

    scores.sort()
    n = len(scores)
    return {
        "episodes": n,
        "mean": round(sum(scores) / n, 2),
        "median": round(scores[n // 2], 2),
        "min": round(scores[0], 2),
        "max": round(scores[-1], 2),
        "capped": capped,
    }


# ── orchestration ────────────────────────────────────────────────────────────

def build_game(game, episodes_override):
    spec = LADDERS[game]
    training = ROOT / game / "training"
    outdir = ROOT / game / "checkpoints"
    outdir.mkdir(exist_ok=True)
    episodes = episodes_override or spec["episodes"]
    want = ENCODER_WIDTH[game]

    print(f"\n===== {game} =====")
    built = []
    for rung in spec["rungs"]:
        rid = rung["id"]
        shipped = rung.get("shipped", False)

        if shipped:
            onnx_path = ROOT / spec["shipped_onnx"]
            # The path the BROWSER requests, which is not where the file sits in
            # the repo. Tetris keeps its .onnx under training/, but deploy_pages.sh
            # copies it next to game.html — so the basename is what a page can
            # actually fetch, matching the fallback already in tetris/game.js.
            #
            # In practice the switcher never fetches this rung: the page hands it
            # the session it already built, via `initial`. It is recorded so the
            # deploy guard can check it, and so the manifest describes something
            # real rather than a path that happens never to be used.
            web = onnx_path.name
            if rung["src"]:
                steps = steps_of(training / rung["src"])
            else:
                steps = TETRIS_SHIPPED_STEPS
        else:
            src = rung["src"]
            src_path = training / src
            if not src_path.exists():
                print(f"  {rid}: SKIP — {src} not found")
                continue
            steps = steps_of(src_path)
            onnx_path = outdir / f"{rid}.onnx"
            web = f"checkpoints/{rid}.onnx"
            if onnx_path.exists():
                print(f"  {rid}: reusing existing {onnx_path.name}")
            else:
                print(f"  {rid}: exporting {src} ({steps:,} steps) …", flush=True)
                t0 = time.time()
                if not do_export(game, src, onnx_path):
                    print(f"  {rid}: EXPORT FAILED — omitted from the ladder")
                    continue
                print(f"      -> {onnx_path.stat().st_size / 1048576:.1f} MB "
                      f"in {time.time() - t0:.0f}s")

        # Verify the width before spending minutes evaluating it.
        import onnx as onnx_mod
        m = onnx_mod.load(str(onnx_path))
        dims = [d.dim_param or d.dim_value
                for d in m.graph.input[0].type.tensor_type.shape.dim]
        if dims[1] != want:
            print(f"  {rid}: WIDTH MISMATCH {dims[1]} vs encoder {want} — omitted")
            continue

        print(f"  {rid}: evaluating {episodes} episodes …", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-u", __file__, "--eval", game,
             str(onnx_path), str(episodes)],
            cwd=ROOT, capture_output=True, text=True,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not line:
            sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:])
            print(f"  {rid}: EVAL FAILED — omitted from the ladder")
            continue
        stats = json.loads(line[-1])
        print(f"      mean {stats['mean']}  median {stats['median']}  "
              f"({time.time() - t0:.0f}s)")

        built.append({
            "id": rid,
            "file": web,
            "steps": steps,
            "bytes": onnx_path.stat().st_size,
            "shipped": shipped,
            "score": stats["mean"],
            "median": stats["median"],
            "min": stats["min"],
            "max": stats["max"],
            "episodes": stats["episodes"],
            "capped": stats["capped"],
        })

    built.sort(key=lambda r: r["score"])
    return {"stepsMeaning": spec["stepsMeaning"], "rungs": built}


def write_manifest(data):
    out = ROOT / "shared" / "checkpoints.js"
    body = json.dumps(data, indent=2)
    out.write_text(
        "/* GENERATED by tools/build_checkpoints.py — do not edit by hand.\n"
        " *\n"
        " * The checkpoint ladders the in-game switcher offers. Each rung's score\n"
        " * is MEASURED by playing the exact .onnx the browser downloads, at\n"
        " * argmax, over `episodes` fixed-seed games — not copied from the\n"
        " * archive filenames. `steps` is num_timesteps out of the checkpoint\n"
        " * itself: how much training the run that produced it had done.\n"
        " *\n"
        " * stepsMeaning is 'lineage' when a game's step counts form one\n"
        " * continuous run and can be compared, and 'run' when they cannot —\n"
        " * see the long note in the builder. The switcher only shows steps for\n"
        " * a 'lineage' ladder.\n"
        " *\n"
        " * Rungs are ordered weakest to strongest by measured score, which is\n"
        " * not always the order they were trained in — that is the point of\n"
        " * measuring rather than assuming.\n"
        " */\n"
        "const CHECKPOINTS = " + body + ";\n"
        "if (typeof window !== 'undefined') window.CHECKPOINTS = CHECKPOINTS;\n",
        encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--eval":
        print(json.dumps(do_eval(argv[1], pathlib.Path(argv[2]), int(argv[3]))))
        return 0

    episodes = None
    if "--episodes" in argv:
        i = argv.index("--episodes")
        episodes = int(argv[i + 1])
        del argv[i:i + 2]

    games = argv or list(LADDERS)
    for g in games:
        if g not in LADDERS:
            print(f"unknown game '{g}'")
            return 1

    # Merge into any existing manifest so building one game does not erase the
    # other two.
    existing = {}
    mf = ROOT / "shared" / "checkpoints.js"
    if mf.exists():
        txt = mf.read_text(encoding="utf-8")
        try:
            existing = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception:
            existing = {}

    for g in games:
        existing[g] = build_game(g, episodes)

    write_manifest(existing)
    total = sum(r["bytes"] for g in existing.values()
                for r in g.get("rungs", []) if not r["shipped"])
    print(f"extra payload across all ladders: {total / 1048576:.0f} MB "
          f"(fetched only when a rung is selected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
