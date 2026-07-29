"""
Measure what sampling temperature does to a trained policy's strength.

    python tools/temperature_sweep.py snake 12

The site currently takes argmax of the action logits, so the AI always plays
its single best move. Sampling from softmax(logits / T) instead turns one
model into a difficulty dial:

    T -> 0   approaches argmax — today's behaviour, full strength
    T = 1    the policy's own trained distribution
    T > 1    flattened toward uniform — makes mistakes

The question this answers is not whether that works in principle but whether
it produces a USABLE curve: difficulty has to fall smoothly, without a cliff
where the AI goes from unbeatable to useless.

Fixed seeds per temperature, so every setting faces identical games and the
differences are the temperature rather than the draw.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

GAMES = {
    "snake": dict(module="snake_env", cls="SnakeEnv", model="snake_final.zip"),
    "watermelon": dict(module="watermelon_env", cls="WatermelonEnv",
                       model="watermelon_final.zip"),
}


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "snake"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    cfg = GAMES[game]

    training = ROOT / game / "training"
    sys.path.insert(0, str(training))
    from stable_baselines3 import PPO
    env_mod = __import__(cfg["module"])
    env = getattr(env_mod, cfg["cls"])()
    model = PPO.load(str(training / cfg["model"]), device="cpu")

    # Pull the raw logits rather than letting SB3 sample, so temperature is
    # applied exactly the way game.js would have to apply it.
    import torch

    def logits_for(obs):
        with torch.no_grad():
            t = torch.as_tensor(obs).float().unsqueeze(0)
            dist = model.policy.get_distribution(t)
            return dist.distribution.logits.numpy()[0]

    print(f"{game}, {episodes} fixed-seed games per temperature\n")
    print(f"{'T':>6} {'mean':>9} {'median':>8} {'min':>7} {'max':>7} "
          f"{'vs argmax':>10} {'differs':>9}")

    rng = np.random.default_rng(0)
    baseline = None
    for T in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        scores, changed, total = [], 0, 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            done, info = False, {"score": 0}
            while not done:
                lg = logits_for(obs)
                greedy = int(np.argmax(lg))
                if T <= 1e-6:
                    action = greedy
                else:
                    z = lg / T
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    action = int(rng.choice(len(p), p=p))
                total += 1
                if action != greedy:
                    changed += 1
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            scores.append(info["score"])

        m = float(np.mean(scores))
        if baseline is None:
            baseline = m
        rel = 100.0 * m / baseline if baseline else 0.0
        print(f"{T:>6.2f} {m:>9.2f} {np.median(scores):>8.0f} "
              f"{min(scores):>7} {max(scores):>7} {rel:>9.0f}% "
              f"{100.0 * changed / max(1, total):>8.1f}%")

    print("\n'differs' is how often the sampled move was not the greedy one —")
    print("the mechanism — and 'vs argmax' is what that did to the score.")


if __name__ == "__main__":
    sys.exit(main())
