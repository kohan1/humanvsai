"""
Record what the Tetris network actually does while it plays.

    python tools/capture_brain_activity.py [frames]

The Inside page animates its neurons firing. The activations are REAL: this
plays a genuine game with the shipped model and records, at each move, the
output of every layer for the specific neurons the diagram draws.

HOW THE ACTIVATIONS ARE OBTAINED
tetris_ai.onnx only declares action_logits as an output, so the hidden layers
are invisible from outside. ONNX lets any internal tensor be promoted to a
graph output without touching the weights, so this script copies the model,
adds the Relu outputs to graph.output, and runs THAT. Same network, same
numbers — just more of them visible. The copy is temporary and never shipped.

WHAT IS SHIPPED
Only the sampled neurons (18/18/18/14/40 = 108 per frame), quantised to one
byte each. A few hundred frames is then ~30 KB, against ~7 MB for shipping the
model and running it in the browser. The page replays a real game rather than
playing a live one, which is a distinction the section states plainly.
"""

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "tetris" / "training" / "tetris_ai.onnx"
DATA = ROOT / "inside" / "data.js"
OUT = ROOT / "inside" / "activity.json"


def build_instrumented(src: pathlib.Path, dst: pathlib.Path):
    """Promote every Relu output to a graph output. Weights are untouched."""
    import onnx

    model = onnx.load(str(src))
    existing = {o.name for o in model.graph.output}
    added = []
    for node in model.graph.node:
        if node.op_type != "Relu":
            continue
        name = node.output[0]
        if name in existing:
            continue
        model.graph.output.append(onnx.ValueInfoProto(name=name))
        added.append(name)
    onnx.save(model, str(dst))
    return added


def main():
    frames_wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 260

    if not DATA.exists():
        print("inside/data.js missing — run build_training_data.py first")
        return 1
    text = DATA.read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{"):text.rindex(";")])
    brain = (payload.get("brain") or {}).get("tetris")
    if not brain or "picks" not in brain:
        print("no brain picks in data.js — rebuild it first")
        return 1
    picks = brain["picks"]

    import numpy as np
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "tetris" / "training"))
    from tetris_env import TetrisEnv

    tmp = pathlib.Path(tempfile.gettempdir()) / "tetris_instrumented.onnx"
    relus = build_instrumented(MODEL, tmp)
    print(f"exposed {len(relus)} hidden activations: {', '.join(relus)}")

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    logits_idx = out_names.index("action_logits")
    relu_idx = [out_names.index(r) for r in relus]

    env = TetrisEnv()
    obs, _ = env.reset(seed=7)

    frames, scores = [], []
    episode = 0
    for _ in range(frames_wanted):
        res = sess.run(None, {"observation": obs.reshape(1, -1).astype(np.float32)})
        logits = res[logits_idx][0]
        hidden = [res[i][0] for i in relu_idx]

        # Column 0 is the observation itself, then one column per hidden
        # layer, then the outputs as probabilities.
        columns = [obs] + hidden
        exp = np.exp(logits - logits.max())
        columns.append(exp / exp.sum())

        frame = []
        for col, idx in enumerate(picks):
            vals = columns[min(col, len(columns) - 1)]
            take = np.asarray(vals)[list(idx)]
            peak = float(np.abs(take).max()) or 1.0
            frame.extend(int(round(min(1.0, abs(v) / peak) * 255)) for v in take)
        frames.append(frame)

        action = int(np.argmax(logits))
        obs, _, terminated, truncated, info = env.step(action)
        scores.append(int(info.get("score", 0)))
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=7 + episode)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "columns": [len(p) for p in picks],
        "frames": frames,
        "finalScore": scores[-1] if scores else 0,
        "episodes": episode + 1,
    }, separators=(",", ":")), encoding="utf-8")

    print(f"captured {len(frames)} frames over {episode + 1} game(s)")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
