"""
Record EVERY episode score from an evaluation, not just the summary.

    python tools/eval_distribution.py snake 60
    python tools/eval_distribution.py watermelon 60

evaluate.py prints average / median / max / min, which is what the install
guard needs but throws away the shape of the result. Watermelon's episodes
range from 433 to 1426 on one unchanged model, and that spread is the whole
argument for keeping the best checkpoint rather than the last — so the Inside
page draws the distribution, and this is where the numbers come from.

Seeds are fixed (0..n-1), the same convention BestScoreCallback uses, so two
models can be compared on identical episodes.

Results are merged into inside/distributions.json; build_training_data.py
folds that into inside/data.js.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "inside" / "distributions.json"

GAMES = {
    "snake": dict(module="snake_env", cls="SnakeEnv", model="snake_final.zip"),
    "watermelon": dict(module="watermelon_env", cls="WatermelonEnv",
                       model="watermelon_final.zip"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in GAMES:
        print(f"usage: eval_distribution.py <{'|'.join(GAMES)}> [episodes] [model.zip]")
        return 1

    game = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    cfg = GAMES[game]
    model_name = sys.argv[3] if len(sys.argv) > 3 else cfg["model"]

    training = ROOT / game / "training"
    sys.path.insert(0, str(training))

    from stable_baselines3 import PPO
    env_module = __import__(cfg["module"])
    env = getattr(env_module, cfg["cls"])()

    model_path = training / model_name
    if not model_path.exists():
        print(f"no such model: {model_path}")
        return 1

    # CPU on purpose: this runs one env at a time, so the GPU would only add
    # transfer overhead — and it may well be busy training something.
    model = PPO.load(str(model_path), device="cpu")

    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done, info = False, {"score": 0}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        scores.append(int(info["score"]))
        print(f"  episode {ep + 1:>3}/{episodes}  score {scores[-1]}", flush=True)

    ordered = sorted(scores)
    payload = {}
    if OUT.exists():
        payload = json.loads(OUT.read_text(encoding="utf-8"))

    payload[game] = {
        "model": model_name,
        "episodes": episodes,
        "scores": scores,
        "mean": round(sum(scores) / len(scores), 2),
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"\n{game}: mean {payload[game]['mean']}, "
          f"median {payload[game]['median']}, "
          f"range {payload[game]['min']}-{payload[game]['max']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
