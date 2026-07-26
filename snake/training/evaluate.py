"""
Quick headless evaluation of a trained Snake model — runs N episodes
directly in the Python environment (no browser, no ONNX involved) and
reports average/max/min score.

Useful for isolating whether a performance problem lives in the model
itself, or somewhere in the browser-side observation encoding / ONNX
inference. If this script shows the AI doing fine but the browser doesn't,
the bug is in game.js. If this script also shows it doing poorly, the bug
(or just insufficient training) is on the Python side.

Usage:
    python evaluate.py snake_final.zip
    python evaluate.py snake_pretrained.zip      # check BC-only performance
    python evaluate.py snake_final.zip 200       # run 200 episodes instead of 100
"""

import sys

from stable_baselines3 import PPO

from snake_env import SnakeEnv


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "snake_final.zip"
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    model = PPO.load(path, device="cpu")
    env = SnakeEnv()

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        scores.append(info["score"])

    scores.sort()
    print(f"Model: {path}")
    print(f"Episodes: {n_episodes}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Median score: {scores[len(scores) // 2]}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")


if __name__ == "__main__":
    main()
