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
        note="Reward curve was flat from iteration 50 onward, but evaluation "
             "still came out ahead of the shipped 129.34. Early-stopped on "
             "728 of 728 iterations, 716 of them at epoch 0 of 10.",
    ),
    # ── Watermelon ───────────────────────────────────────────────────────
    "pipeline_entropy_regression.log": dict(
        game="watermelon", label="BC + PPO with entropy", evalScore=614.40,
        outcome="rejected",
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
        note="share_features_extractor=False. First run where fitting the "
             "critic could no longer damage the policy.",
    ),
    "train_v3_resume_done.log": dict(
        game="watermelon", label="resume", evalScore=902.10,
        outcome="shipped",
        note="The model currently live on the site.",
    ),
    "train_v4_done.log": dict(
        game="watermelon", label="resume", evalScore=872.00,
        outcome="rejected",
        note="Below the shipped 902.10, so the install guard refused it.",
    ),
    "train_day1_893.log": dict(
        game="watermelon", label="resume", evalScore=893.03,
        outcome="rejected",
        note="Third consecutive rejection. Preserved rather than installed.",
    ),
    "train_day2.log": dict(
        game="watermelon", label="resume · CPU", evalScore=936.70,
        outcome="improved",
        note="Beat the shipped model at last, but took 3h45m on CPU for "
             "+3.8%, and early-stopped on 49 of 49 iterations.",
    ),
    "train_day3_gpu.log": dict(
        game="watermelon", label="resume · GPU, wider trust region",
        outcome="running",
        note="First run on the GPU with 20 envs, target_kl 0.05 and "
             "best-checkpoint tracking. No early stopping at all.",
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

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "shipped": shipped,
        "runs": runs,
    }

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
