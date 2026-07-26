"""
Headless evaluation of a trained Watermelon model — no browser, no ONNX.

Isolates where a problem lives. If this shows the AI doing fine but the browser
does not, the bug is in game.js (observation encoder or ONNX wiring). If this
also shows it doing poorly, the problem is the model or the training setup.

Also accepts "heuristic" instead of a path, to score the BC teacher. Always
compare a trained model against that number: on Snake, 20M steps of PPO
produced a policy that was still 28% WORSE than its own teacher, and nothing
in the training curves made that obvious.

Usage:
    python evaluate.py watermelon_final.zip
    python evaluate.py watermelon_pretrained.zip 30
    python evaluate.py heuristic 50
"""

import sys

from stable_baselines3 import PPO

from watermelon_env import WatermelonEnv
from heuristic import heuristic_action_from_env


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "watermelon_final.zip"
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    use_heuristic = path == "heuristic"
    model = None if use_heuristic else PPO.load(path, device="cpu")

    env = WatermelonEnv()
    scores, drops = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        info = {"score": 0}
        while not done:
            if use_heuristic:
                action = heuristic_action_from_env(env)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        scores.append(info["score"])
        drops.append(env.drops)

    scores.sort()
    print(f"Model: {path}")
    print(f"Episodes: {n_episodes}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Median score: {scores[len(scores) // 2]}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")
    print(f"Average drops: {sum(drops) / len(drops):.1f}")


if __name__ == "__main__":
    main()
