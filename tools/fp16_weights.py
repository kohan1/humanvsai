"""Halve a shipped .onnx by storing its weights as float16 — and prove it plays
the same before keeping it.

    python tools/fp16_weights.py snake      [--episodes 20]
    python tools/fp16_weights.py tetris
    python tools/fp16_weights.py watermelon
    python tools/fp16_weights.py snake --in some.onnx --out other.onnx

WHY
Model size is now almost all of the wait for the AI: at 10 Mbps the 34 MB
Snake model takes ~38 s to download and ~0.2 s to start (measured 2026-10-07).
fp32 weights barely compress, so gzip does not help; fp16 halves them.

WHAT IT CHANGES — AND WHAT IT DOES NOT
Every large float32 initializer becomes a float16 initializer followed by a
Cast back to float32. The graph's inputs, outputs and every operator stay
float32, so game.js, the encoders and onnxruntime-web's wasm kernels are
untouched: only the stored numbers lose precision (~3 significant digits).

THE GATE
Rounding the weights can flip a near-tie between two actions. So this plays
real games with the ORIGINAL model on fixed seeds and, at every decision, asks
the fp16 model what it would have done. It also plays the fp16 model on its
own and scores it the way build_checkpoints.py does (argmax, Tetris masked).
The fp16 file is written only if decisions agree on >= 99.5% of steps AND its
mean score is not more than 1% below the original's. Otherwise nothing is
replaced and the exit code is 1.

It REPLACES the shipped .onnx in place by default (keeping <name>.fp32.onnx
beside it), so re-run embed_model.py / embed_assets-equivalents afterwards for
the file:// copy, and rebuild the checkpoint ladder if you want its rungs
halved too (each rung can be passed with --in/--out).

One process per game: the three games' env modules collide (see CLAUDE.md,
"One process per rung").
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent

SHIPPED = {
    "snake": ROOT / "snake" / "snake_ai.onnx",
    "watermelon": ROOT / "watermelon" / "watermelon_ai.onnx",
    "tetris": ROOT / "tetris" / "training" / "tetris_ai.onnx",
}
ENVS = {
    "snake": ("snake_env", "SnakeEnv"),
    "watermelon": ("watermelon_env", "WatermelonEnv"),
    "tetris": ("tetris_env", "TetrisEnv"),
}
# Biases and other small tensors cost nothing to keep at full precision.
MIN_ELEMENTS = 1024
MIN_AGREEMENT = 0.995
MAX_SCORE_DROP = 0.01


def convert(src, dst):
    model = onnx.load(str(src))
    g = model.graph
    keep, casts, n_conv = [], [], 0
    for init in g.initializer:
        arr_size = int(np.prod(init.dims)) if init.dims else 1
        if init.data_type != TensorProto.FLOAT or arr_size < MIN_ELEMENTS:
            keep.append(init)
            continue
        half = numpy_helper.from_array(
            numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node(
            "Cast", [half.name], [init.name], to=TensorProto.FLOAT,
            name=init.name + "__cast"))
        n_conv += 1
    del g.initializer[:]
    g.initializer.extend(keep)
    # Casts first, so every consumer still finds its float32 tensor by name.
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return n_conv


def play(game, ref_path, new_path, episodes):
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / game / "training"))
    mod, cls = ENVS[game]
    env = getattr(__import__(mod), cls)()
    ref = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(str(new_path), providers=["CPUExecutionProvider"])
    name = ref.get_inputs()[0].name
    cap = 60000 if game == "tetris" else 100000

    def act(sess, e, obs):
        lg = sess.run(None, {name: obs.reshape(1, -1).astype(np.float32)})[0][0]
        if game == "tetris":           # masked, exactly as game.js does
            n = len(e._placements)
            if n <= 0:
                return None
            lg = lg[:min(n, len(lg))]
        return int(np.argmax(lg))

    def episode(sess, seed, shadow=None):
        obs, _ = env.reset(seed=seed)
        info, steps, same = {"score": 0}, 0, 0
        while steps < cap:
            a = act(sess, env, obs)
            if a is None:
                break
            if shadow is not None and act(shadow, env, obs) == a:
                same += 1
            obs, _, term, trunc, info = env.step(a)
            steps += 1
            if term or trunc:
                break
        return float(info.get("score", 0)), steps, same

    ref_scores, new_scores, steps, same = [], [], 0, 0
    for ep in range(episodes):
        s, n, k = episode(ref, ep, shadow=new)
        ref_scores.append(s)
        steps += n
        same += k
        new_scores.append(episode(new, ep)[0])
    return ref_scores, new_scores, same / max(steps, 1), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=sorted(SHIPPED))
    ap.add_argument("--in", dest="src")
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()

    src = Path(a.src) if a.src else SHIPPED[a.game]
    dst = Path(a.dst) if a.dst else src
    if not src.exists():
        sys.exit(f"{src} not found")
    tmp = dst.with_name(dst.stem + ".fp16-candidate.onnx")

    n = convert(src, tmp)
    mb = lambda p: p.stat().st_size / 1048576
    print(f"{src.name}: {mb(src):.1f} MB -> {mb(tmp):.1f} MB ({n} tensors halved)")

    ref, new, agree, steps = play(a.game, src, tmp, a.episodes)
    mr, mn = sum(ref) / len(ref), sum(new) / len(new)
    print(f"decisions identical on {agree:.2%} of {steps} steps")
    print(f"mean score over {a.episodes} seeds: fp32 {mr:.2f}  fp16 {mn:.2f}")

    ok = agree >= MIN_AGREEMENT and mn >= mr * (1 - MAX_SCORE_DROP)
    if not ok:
        tmp.unlink()
        sys.exit("REFUSED: fp16 weights change how it plays; nothing replaced")
    if dst == src:
        backup = src.with_name(src.stem + ".fp32.onnx")
        shutil.copy2(src, backup)
        print(f"kept the original as {backup.name}")
    tmp.replace(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
