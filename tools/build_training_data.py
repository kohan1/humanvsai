"""
Turn every training log into one JS file for the Inside page.

    python tools/build_training_data.py

Re-run this after any training run finishes; inside/data.js is generated and
should never be hand-edited.

WHY A .js AND NOT A .json
Same reason the models are embedded: fetch() of a local file is blocked under
file://, and the site has to keep working when game.html is opened directly.
A <script> defining a const works under both file:// and http://.

WHAT IS MEASURED AND WHAT IS ANNOTATED
Everything in `series` is parsed straight out of the SB3 log tables — steps,
reward, score, fps, elapsed seconds. Those are facts.

`outcome` and `evalScore` are NOT in the logs. They come from evaluate.py runs
recorded in RUN_NOTES below. A run with no entry there is emitted with its
real series and no outcome label, rather than being given a guessed one.
"""

import json
import pathlib
import re
import sys
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "inside" / "data.js"

# Down-sample long runs. A 25M-step run logs ~760 rows; ~400 points is more
# than a chart 900px wide can show.
MAX_POINTS = 400

# Evaluation scores and outcomes, measured with evaluate.py. Keyed by log
# filename. Anything absent is reported without an outcome — see the module
# docstring. Keep this honest: only add a number that was actually measured.
RUN_NOTES = {
    # ── Snake ────────────────────────────────────────────────────────────
    "train_day1.log": dict(
        game="snake", label="resume · GPU", evalScore=135.78,
        outcome="improved",
        checkpoint="snake/training/archive/models/snake_final.135pt78.zip",
        note="Reward curve was flat from iteration 50 onward, but evaluation "
             "still came out ahead of the shipped 129.34. Early-stopped on "
             "728 of 728 iterations, 716 of them at epoch 0 of 10.",
    ),
    # ── Watermelon ───────────────────────────────────────────────────────
    "pipeline_entropy_regression.log": dict(
        game="watermelon", label="BC + PPO with entropy", evalScore=614.40,
        outcome="rejected",
        checkpoint="watermelon/training/archive/models/watermelon_ppo_614.zip",
        note="ent_coef=0.01 on top of a good behavioural-cloning start "
             "dragged a confident policy back toward uniform.",
    ),
    "train_v2_bad_critic.log": dict(
        game="watermelon", label="PPO · untrained critic", outcome="rejected",
        note="The critic reached PPO at random initialisation — "
             "explained_variance -0.016. Fixed by fitting the value function "
             "during pretraining.",
    ),
    "train_v3_kl_throttled.log": dict(
        game="watermelon", label="PPO · KL throttled", outcome="rejected",
        note="Shared feature extractor let the value loss (~82) dominate the "
             "policy gradient (~1e-4) and rewrite the policy's own features.",
    ),
    "pipeline_v2_done.log": dict(
        game="watermelon", label="BC + PPO · separate extractors",
        evalScore=841.87, outcome="improved",
        checkpoint="watermelon/training/archive/models/watermelon_final.841pt87.zip",
        note="share_features_extractor=False. First run where fitting the "
             "critic could no longer damage the policy.",
    ),
    "train_v3_resume_done.log": dict(
        game="watermelon", label="resume", evalScore=902.10,
        outcome="shipped",
        parent="pipeline_v2_done.log",
        checkpoint="watermelon/training/archive/models/watermelon_final.902pt10.zip",
        note="The model currently live on the site.",
    ),
    "train_v4_done.log": dict(
        game="watermelon", label="resume", evalScore=872.00,
        outcome="rejected",
        parent="train_v3_resume_done.log",
        checkpoint="watermelon/training/archive/models/watermelon_final.872_rejected.zip",
        note="Below the shipped 902.10, so the install guard refused it.",
    ),
    "train_day1_893.log": dict(
        game="watermelon", label="resume", evalScore=893.03,
        outcome="rejected",
        parent="train_v3_resume_done.log",
        checkpoint="watermelon/training/archive/models/watermelon_final.893_rejected.zip",
        note="Third consecutive rejection. Preserved rather than installed.",
    ),
    "train_day2.log": dict(
        game="watermelon", label="resume · CPU", evalScore=936.70,
        outcome="improved",
        parent="train_v3_resume_done.log",
        checkpoint="watermelon/training/archive/models/watermelon_final.936pt70.zip",
        note="Beat the shipped model at last, but took 3h45m on CPU for "
             "+3.8%, and early-stopped on 49 of 49 iterations.",
    ),
    "train_day3_gpu.log": dict(
        game="watermelon", label="resume · GPU, wider trust region",
        evalScore=1032.43, outcome="improved",
        parent="train_day2.log",
        checkpoint="watermelon/training/archive/models/watermelon_final.1032pt43.zip",
        note="20M steps on the GPU in 6h51m, with 20 envs, target_kl 0.05 and "
             "best-checkpoint tracking — only 3 early stops in 489 iterations, "
             "against 49 of 49 on the previous run. It peaked at 18M steps and "
             "then declined: the best checkpoint scores 1032.43 but the model "
             "it ENDED on scores 959.93. Keeping the best rather than the last "
             "is worth 72 points here, and this is the run that proved it.",
    ),
    "train_100m.log": dict(
        game="snake", label="100M resume · GPU", evalScore=145.70,
        outcome="improved",
        parent="train_day1.log",
        checkpoint="snake/training/archive/models/snake_final.145pt70_best.zip",
        note="100 million steps in 12h58m. Almost all of the gain arrived "
             "early: 133 to 151 in the first 7M steps, then only +3.7 across "
             "the next 48M, and nothing at all after 55M. It early-stopped on "
             "1,526 of 1,526 iterations, so unlike Watermelon the wider trust "
             "region did not loosen Snake's updates. The best checkpoint and "
             "the final model tie on average (145.70 against 145.46) but not "
             "on their worst game — 80 against 29 — so the best was kept for "
             "being far less prone to collapse.",
    ),
}

FIELDS = {
    "total_timesteps": "t",
    "ep_rew_mean": "r",
    "ep_score_mean": "s",
    "fps": "f",
    "time_elapsed": "e",
}

ROW = re.compile(r"^\|\s+(\w+)\s+\|\s+([-\d.e+]+)\s*\|$")
BEST = re.compile(r"\[best\]\s+(?:new best\s+)?([\d.]+)\s+at\s+(\d+)\s+steps")


def parse_log(path: pathlib.Path):
    """Pull the SB3 table rows and any [best] evaluation lines out of a log."""
    series, evals, current = [], [], {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BEST.search(line)
        if m:
            evals.append({
                "score": round(float(m.group(1)), 2),
                "t": int(m.group(2)),
                "best": "new best" in line,
            })
            continue

        m = ROW.match(line.rstrip())
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        if key not in FIELDS:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue

        short = FIELDS[key]
        # total_timesteps closes a block — SB3 prints it once per table.
        current[short] = value
        if short == "t" and "r" in current:
            series.append(current)
            current = {}

    return series, evals


def describe_onnx(path: pathlib.Path):
    """
    Describe a model from its .onnx rather than its training checkpoint.

    All three games have a .onnx; only two still have the SB3 .zip that made
    it (Tetris's 1B-step checkpoint is gone). Reading the exported graph is
    therefore the only way to describe all three the same way — and it
    describes what the site actually runs, not what training once held.

    Conv and Gemm are the layers with weights; everything between them is
    activations and reshapes, which would pad the diagram without adding
    anything.
    """
    try:
        import numpy as np
        import onnx
    except ImportError:
        return None
    if not path.exists():
        return None

    model = onnx.load(str(path))
    graph = model.graph
    init = {i.name: i for i in graph.initializer}
    total_params = sum(int(np.prod(i.dims)) for i in init.values())

    def shape_of(t):
        return [d.dim_param or d.dim_value
                for d in t.type.tensor_type.shape.dim]

    layers = []
    for node in graph.node:
        if node.op_type not in ("Conv", "Gemm", "MatMul"):
            continue
        weight = None
        for name in node.input:
            if name in init and len(init[name].dims) >= 2:
                weight = init[name]
                break
        if weight is None:
            continue
        dims = [int(d) for d in weight.dims]
        layers.append({
            "kind": "conv" if node.op_type == "Conv" else "dense",
            "shape": dims,
            "params": int(np.prod(dims)),
        })

    return {
        "file": path.name,
        "sizeMB": round(path.stat().st_size / 1048576, 1),
        "params": total_params,
        "inputs": [{"name": i.name, "shape": shape_of(i)} for i in graph.input],
        "outputs": [{"name": o.name, "shape": shape_of(o)} for o in graph.output],
        "layers": layers,
    }


def extract_brain(path: pathlib.Path, sample=(18, 18, 18, 14)):
    """
    Pull the REAL weights out of a model so the page can draw the actual
    network rather than a stylised picture of one.

    Three things come out, all measured, none invented:

      heat    each weight matrix reduced to at most 64x64 by averaging blocks,
              then quantised to 0-255. A 238x1024 matrix is 244k floats; the
              reduction is what makes it shippable, and block averaging keeps
              the structure that block sampling would alias away.
      nodes   a handful of real neurons per layer, and the true weight on every
              connection between the sampled ones. Drawing all 1.84M edges is
              meaningless; drawing a sample at true strength is not.
      stats   min / max / mean / std and a histogram per layer.

    Only dense layers. A conv kernel is a different shape of thing and would
    need its own treatment, so Snake and Watermelon get nothing here rather
    than something misleading.
    """
    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return None
    if not path.exists():
        return None

    model = onnx.load(str(path))
    init = {i.name: i for i in model.graph.initializer}

    mats = []
    for node in model.graph.node:
        if node.op_type not in ("Gemm", "MatMul"):
            continue
        for name in node.input:
            if name in init and len(init[name].dims) == 2:
                mats.append(numpy_helper.to_array(init[name]))
                break

    if not mats:
        return None

    def reduce_to(m, cap=64):
        """Average m down to at most cap x cap. Averaging, not sampling —
        sampling a sparse matrix mostly returns the gaps."""
        rows, cols = m.shape
        rs, cs = max(1, rows // cap), max(1, cols // cap)
        r_end, c_end = (rows // rs) * rs, (cols // cs) * cs
        block = m[:r_end, :c_end].reshape(rows // rs, rs, cols // cs, cs)
        return block.mean(axis=(1, 3))

    layers = []
    for i, m in enumerate(mats):
        small = reduce_to(m)
        peak = float(np.abs(small).max()) or 1.0
        # Signed, centred on 128: 0 is the most negative, 255 the most
        # positive, so the page can colour sign as well as strength.
        quant = np.clip(np.rint(small / peak * 127) + 128, 0, 255).astype(int)
        hist, edges = np.histogram(m, bins=24)

        layers.append({
            "shape": [int(x) for x in m.shape],
            "heat": {
                "w": int(quant.shape[1]),
                "h": int(quant.shape[0]),
                "peak": round(peak, 5),
                "data": quant.flatten().tolist(),
            },
            "stats": {
                "min": round(float(m.min()), 4),
                "max": round(float(m.max()), 4),
                "mean": round(float(m.mean()), 6),
                "std": round(float(m.std()), 5),
                "zeroish": round(float((np.abs(m) < 0.01).mean()), 4),
            },
            "hist": {
                "counts": [int(c) for c in hist],
                "from": round(float(edges[0]), 4),
                "to": round(float(edges[-1]), 4),
            },
        })

    # A sampled node-link view. Neurons are taken evenly across each layer so
    # the sample is not biased toward one end of the matrix.
    counts = list(sample[:len(mats)]) + [int(mats[-1].shape[0])]
    picks = []
    picks.append(np.linspace(0, mats[0].shape[1] - 1, counts[0], dtype=int))
    for i, m in enumerate(mats):
        n = counts[i + 1] if i + 1 < len(counts) else m.shape[0]
        picks.append(np.linspace(0, m.shape[0] - 1, min(n, m.shape[0]), dtype=int))

    edges_out = []
    for i, m in enumerate(mats):
        src, dst = picks[i], picks[i + 1]
        w = m[np.ix_(dst, src)]
        edges_out.append({
            "layer": i,
            "from": len(src),
            "to": len(dst),
            "peak": round(float(np.abs(w).max()) or 1.0, 5),
            "w": [round(float(v), 4) for v in w.flatten()],
        })

    return {
        "file": path.name,
        "layers": layers,
        "nodes": [len(p) for p in picks],
        # The exact neurons sampled, so capture_brain_activity.py can record
        # activations for THESE and no others — the animation has to light up
        # the same neurons the edges were drawn between.
        "picks": [[int(v) for v in p] for p in picks],
        "edges": edges_out,
    }


def parse_tetris_tensorboard():
    """
    Tetris predates the text-log convention and only left TensorBoard events.

    The 26 event files are 26 SESSIONS of one continuous run — step numbers
    carry straight on across them (6.9M in the first, 1,000,012,896 in the
    last) — so they are merged into a single series rather than reported as 26
    separate runs.

    Elapsed time is the sum of each session's own wall-clock span, NOT last
    minus first across everything. The gaps between sessions are days when the
    machine was doing something else, and counting them would claim weeks of
    compute that never happened.

    Event filenames embed the machine's hostname, so nothing derived from them
    reaches the output — this data ships in a public repo.
    """
    log_dir = ROOT / "tetris" / "training" / "logs"
    files = sorted(log_dir.rglob("events.out.tfevents*"))
    if not files:
        return None

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("  ! tensorboard not installed — skipping Tetris")
        return None

    by_step, evals, elapsed, wall_first, wall_last = {}, [], 0.0, None, None

    for path in files:
        ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags()["scalars"])
        if not available:
            continue

        session_walls = []
        for tag, short in (("rollout/ep_rew_mean", "r"),
                           ("time/fps", "f")):
            if tag not in available:
                continue
            for ev in ea.Scalars(tag):
                by_step.setdefault(ev.step, {})[short] = ev.value
                session_walls.append(ev.wall_time)

        if "eval/mean_reward" in available:
            for ev in ea.Scalars("eval/mean_reward"):
                evals.append({"t": int(ev.step), "score": round(ev.value, 2)})

        if session_walls:
            lo, hi = min(session_walls), max(session_walls)
            elapsed += hi - lo
            wall_first = lo if wall_first is None else min(wall_first, lo)
            wall_last = hi if wall_last is None else max(wall_last, hi)

    if not by_step:
        return None

    series = []
    running = 0.0
    for step in sorted(by_step):
        row = dict(by_step[step])
        row["t"] = float(step)
        series.append(row)

    # Spread the measured compute time across the series so the tooltip's
    # "elapsed" reads sensibly. It is an even spread, not a per-point
    # measurement — TensorBoard gives wall clock, not cumulative compute.
    if len(series) > 1:
        for i, row in enumerate(series):
            row["e"] = elapsed * i / (len(series) - 1)

    # The `best` flag has to be computed over EVERY point — a running maximum
    # taken after thinning would mark points that were not actually records.
    evals.sort(key=lambda e: e["t"])
    best = float("-inf")
    for e in evals:
        e["best"] = e["score"] > best
        best = max(best, e["score"])

    # Tetris logged ~15,000 evaluations, which is most of the payload for a
    # series the page does not even plot. Thin it, keeping each point's
    # already-correct flag.
    if len(evals) > MAX_POINTS:
        stride = len(evals) / MAX_POINTS
        thinned = [evals[int(i * stride)] for i in range(MAX_POINTS)]
        thinned[-1] = evals[-1]
        evals = thinned

    return {
        "game": "tetris",
        "file": "tensorboard",
        "label": "PPO · 26 sessions",
        "note": "Trained long before the other two and by far the longest run "
                "on the project — a billion steps across 26 sessions. It left "
                "only TensorBoard events, so there is no game-score curve "
                "here, just reward. Its evaluation figures are in reward "
                "units and are not comparable to Snake's or Watermelon's "
                "scores.",
        "outcome": "shipped",
        "evalScore": None,
        "steps": int(series[-1]["t"]),
        "elapsed": int(elapsed),
        "endedAt": datetime.fromtimestamp(wall_last).isoformat(timespec="seconds"),
        "startedAt": datetime.fromtimestamp(wall_first).isoformat(timespec="seconds"),
        "finalScore": None,
        "avgFps": round(series[-1]["t"] / elapsed, 1) if elapsed else None,
        "series": tidy(downsample(series)),
        "evals": evals,
    }


def downsample(series, limit=MAX_POINTS):
    """Keep the first and last point exactly; thin the middle evenly."""
    if len(series) <= limit:
        return series
    step = len(series) / limit
    picked = [series[int(i * step)] for i in range(limit)]
    if picked[-1] is not series[-1]:
        picked[-1] = series[-1]
    return picked


def tidy(series):
    out = []
    for row in series:
        rec = {"t": int(row["t"])}
        for k in ("r", "s", "f", "e"):
            if k in row:
                rec[k] = round(row[k], 2)
        out.append(rec)
    return out


def main():
    runs = []
    for game in ("snake", "tetris", "watermelon"):
        # Finished runs are filed under training/logs/; a run still in progress
        # is written at the top level and moved once it completes.
        base = ROOT / game / "training"
        found = sorted(base.glob("*.log")) + sorted((base / "logs").glob("*.log"))
        for path in found:
            series, evals = parse_log(path)
            if len(series) < 3:
                continue  # pretrain logs and failed starts have no curve

            note = RUN_NOTES.get(path.name, {})
            if note.get("game") and note["game"] != game:
                print(f"  ! {path.name}: RUN_NOTES says {note['game']}, "
                      f"found under {game} — check build_training_data.py")

            last = series[-1]
            ended = datetime.fromtimestamp(path.stat().st_mtime)
            elapsed = last.get("e", 0)

            runs.append({
                "game": game,
                "file": path.name,
                "label": note.get("label", path.stem.replace("_", " ")),
                "note": note.get("note"),
                "outcome": note.get("outcome"),
                "evalScore": note.get("evalScore"),
                "parent": note.get("parent"),
                # Only advertise a checkpoint that is actually still on disk,
                # or the progression track would offer a model it cannot load.
                "checkpoint": note.get("checkpoint")
                              if note.get("checkpoint")
                              and (ROOT / note["checkpoint"]).exists() else None,
                "steps": int(last["t"]),
                "elapsed": int(elapsed),
                "endedAt": ended.isoformat(timespec="seconds"),
                "startedAt": (ended - timedelta(seconds=elapsed)).isoformat(timespec="seconds"),
                "finalScore": last.get("s"),
                "avgFps": round(last["t"] / elapsed, 1) if elapsed else None,
                "series": tidy(downsample(series)),
                "evals": evals,
            })

    tetris = parse_tetris_tensorboard()
    if tetris:
        runs.append(tetris)

    runs.sort(key=lambda r: (r["game"], r["endedAt"]))

    shipped = {}
    for game in ("snake", "tetris", "watermelon"):
        f = ROOT / game / "training" / ".shipped_score"
        if f.exists():
            shipped[game] = float(f.read_text().strip())

    # Cumulative steps along each lineage. A resume inherits everything its
    # parent had already trained, so the progression track can plot skill
    # against total experience rather than against one run's own counter.
    by_file = {r["file"]: r for r in runs}
    def cumulative(run, seen=None):
        seen = seen or set()
        if run["file"] in seen:          # a mis-typed parent must not hang here
            return run["steps"]
        seen.add(run["file"])
        parent = by_file.get(run.get("parent"))
        return run["steps"] + (cumulative(parent, seen) if parent else 0)
    for r in runs:
        r["cumulativeSteps"] = cumulative(r)
        # Score gained per million steps of THIS run — the sample-efficiency
        # figure. Only meaningful when both this run and its parent were
        # measured the same way.
        parent = by_file.get(r.get("parent"))
        if r.get("evalScore") and parent and parent.get("evalScore") and r["steps"]:
            r["gainPerMStep"] = round(
                (r["evalScore"] - parent["evalScore"]) / (r["steps"] / 1e6), 3)

    architecture = {}
    for game, onnx_path in (
        ("snake", ROOT / "snake" / "snake_ai.onnx"),
        ("tetris", ROOT / "tetris" / "training" / "tetris_ai.onnx"),
        ("watermelon", ROOT / "watermelon" / "watermelon_ai.onnx"),
    ):
        desc = describe_onnx(onnx_path)
        if desc:
            architecture[game] = desc

    # The real weights, for the network diagram. Tetris only: it is a pure MLP,
    # so a dense-layer view describes it completely. Snake and Watermelon are
    # convolutional, and drawing their kernels as a node-link graph would show
    # something that is not how they work.
    brain = extract_brain(ROOT / "tetris" / "training" / "tetris_ai.onnx")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "shipped": shipped,
        "runs": runs,
        "architecture": architecture,
        "brain": {"tetris": brain} if brain else {},
        # Measured with watermelon/training/bench_envs.py on 2026-07-27:
        # Ryzen 7 3700X (8 physical / 16 logical), RTX 4060 Ti 16 GB.
        "benchmark": {
            "machine": "Ryzen 7 3700X · RTX 4060 Ti 16 GB",
            "game": "watermelon",
            "unit": "steps / second",
            "results": [
                {"envs": 8,  "device": "cuda", "fps": 534},
                {"envs": 12, "device": "cuda", "fps": 685},
                {"envs": 16, "device": "cuda", "fps": 593},
                {"envs": 20, "device": "cuda", "fps": 1079},
                {"envs": 24, "device": "cuda", "fps": 1028},
                {"envs": 16, "device": "cpu",  "fps": 217},
            ],
        },
    }

    dist_file = ROOT / "inside" / "distributions.json"
    if dist_file.exists():
        payload["distributions"] = json.loads(dist_file.read_text(encoding="utf-8"))

    # Recorded activations, so the diagram can show the network actually
    # firing. Written by capture_brain_activity.py, which must run AFTER this
    # script has emitted brain.picks — hence a separate file rather than being
    # generated inline here.
    act_file = ROOT / "inside" / "activity.json"
    if act_file.exists() and payload.get("brain"):
        payload["activity"] = json.loads(act_file.read_text(encoding="utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// GENERATED by tools/build_training_data.py — do not edit by hand.\n"
        "const TRAINING_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )

    total_steps = sum(r["steps"] for r in runs)
    total_hours = sum(r["elapsed"] for r in runs) / 3600
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(runs)} runs, {total_steps/1e6:.1f}M steps, {total_hours:.1f} hours")
    for r in runs:
        mark = {"shipped": "*", "rejected": "x", "improved": "+",
                "running": ">"}.get(r["outcome"], " ")
        score = f"{r['evalScore']:.2f}" if r["evalScore"] else "     -"
        print(f"  {mark} {r['game']:<11}{r['file']:<38}"
              f"{r['steps']/1e6:>6.2f}M {r['elapsed']/3600:>5.1f}h  {score}")


if __name__ == "__main__":
    sys.exit(main())
