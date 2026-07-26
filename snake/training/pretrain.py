"""
Behavioural-cloning pretraining for the Snake AI.

Generates expert (observation, action) pairs from the hand-coded heuristic
in heuristic.py, then trains the PPO policy network to imitate it via
plain supervised cross-entropy.

NOTE ON A FIX: an earlier version of this script anchored training with a
KL-divergence penalty against the policy's own pre-update snapshot, using a
very low learning rate — mirroring the "protect existing weights" lesson
from the Tetris build. That's the right move when you're gently nudging an
*already-competent* policy (e.g. a later corrective BC pass on top of a
partially-RL-trained model). Applied here, on this very first pass from a
freshly-initialized random policy, it was actively counterproductive: there
was nothing worth protecting yet, and the KL term just pulled training back
toward random behaviour the whole time. Combined with a learning rate ~200x
smaller than a normal supervised setup, the network never had a real chance
to learn the heuristic (confirmed: it was scoring ~0.09 average, essentially
random). This version uses a normal cold-start LR and no KL anchor.

Run this before train.py. train.py will automatically pick up the result
(snake_pretrained.zip) as its starting point if it exists.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from snake_env import SnakeEnv
from heuristic import heuristic_action
from policy_config import policy_kwargs

# Match train.py — see the note there on why GPU wins for this net size.
# CUDA also has no float64 restriction, unlike MPS on the Mac.
DEVICE = os.environ.get("TRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# Cap by STEPS, not episodes. Every observation is buffered in memory and v2
# observations are 5.2 KB each (1294 floats), so an episode-count cap is really
# an unbounded memory cap: the better the teacher plays, the longer its episodes
# and the more RAM it eats. 4000 episodes measured out at ~4M steps / ~20 GB,
# which swaps long before it finishes. 600k transitions is ~3 GB and is already
# far more than enough to clone a 3-action policy.
MAX_STEPS = 600_000
N_EPISODES = 4000          # upper bound; MAX_STEPS is what actually stops it

# The teacher plays ~79% STRAIGHT, ~9% LEFT, ~12% RIGHT. Unweighted, the v2 CNN
# collapsed straight onto the majority class: it predicted STRAIGHT for 100% of
# inputs, scored a "train_acc" of 0.787 that was exactly the class prior, and
# evaluated at 0.00 because a snake that never turns walks into a wall. Class
# weights make the two turn classes worth proportionally more, so ignoring them
# is no longer the cheapest way to reduce loss.
USE_CLASS_WEIGHTS = True

# 1e-3 was fine for the v1 MLP but is aggressive for the v2 conv net, and a
# too-high LR is the other half of how a policy collapses onto a constant.
LR = 3e-4
EPOCHS = 15
BATCH_SIZE = 512

OUT_PATH = "snake_pretrained.zip"


def collect_expert_data(n_episodes: int = N_EPISODES, max_steps: int = MAX_STEPS):
    env = SnakeEnv()
    obs_buf, act_buf = [], []
    episodes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = heuristic_action(env.body, env.dir_idx, env.food, env.tile_count)
            obs_buf.append(obs)
            act_buf.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        episodes += 1
        if episodes % 25 == 0:
            print(
                f"  {episodes} episodes, {len(obs_buf):,}/{max_steps:,} steps",
                flush=True,   # stdout is a pipe here; without this there is no
                              # visible progress until the process exits
            )
        if len(obs_buf) >= max_steps:
            print(f"  reached the {max_steps:,}-step cap after {episodes} episodes", flush=True)
            break
    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.int64)


def main():
    print("Collecting heuristic demonstrations...")
    obs_data, act_data = collect_expert_data()
    print(f"Collected {len(obs_data)} (obs, action) pairs")

    vec_env = make_vec_env(SnakeEnv, n_envs=1)
    model = PPO(
        "MlpPolicy",   # overridden by features_extractor_class below
        vec_env,
        policy_kwargs=policy_kwargs(),
        device=DEVICE,
        verbose=0,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    n_actions = int(model.action_space.n)
    counts = np.bincount(act_data, minlength=n_actions).astype(np.float64)
    print("teacher action mix: " + ", ".join(
        f"{i}:{counts[i] / counts.sum():.3f}" for i in range(n_actions)
    ), flush=True)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        # Inverse frequency, normalised to mean 1 so the loss scale — and
        # therefore the effective learning rate — stays comparable.
        inv = counts.sum() / np.maximum(counts, 1.0)
        inv *= n_actions / inv.sum()
        class_weight = torch.as_tensor(inv, dtype=torch.float32, device=DEVICE)
        print("class weights     : " + ", ".join(f"{w:.2f}" for w in inv), flush=True)

    n = len(obs_data)
    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        total_loss = 0.0
        correct = 0
        n_batches = 0
        per_class_correct = np.zeros(n_actions, dtype=np.int64)
        per_class_total = np.zeros(n_actions, dtype=np.int64)

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            # Must land on the same device as the policy — these used to be
            # implicitly CPU, which only worked because the model was too.
            obs_b = torch.as_tensor(obs_data[batch_idx], device=DEVICE)
            act_b = torch.as_tensor(act_data[batch_idx], device=DEVICE)

            dist = policy.get_distribution(obs_b)
            logits = dist.distribution.logits
            loss = F.cross_entropy(logits, act_b, weight=class_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            for a in range(n_actions):
                m = act_b == a
                per_class_total[a] += int(m.sum().item())
                per_class_correct[a] += int((pred[m] == a).sum().item())

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == act_b).sum().item()
            n_batches += 1

        acc = correct / n
        # Per-class recall, not just the aggregate. Aggregate accuracy hides a
        # collapse onto the majority class — 0.787 looked healthy while the
        # policy was in fact predicting STRAIGHT for every single input. If any
        # class here sits near 0.00, the run is broken no matter what the
        # headline number says.
        per_class = " ".join(
            f"a{a}={per_class_correct[a] / max(1, per_class_total[a]):.2f}"
            for a in range(n_actions)
        )
        print(
            f"epoch {epoch + 1}/{EPOCHS}  "
            f"loss={total_loss / max(1, n_batches):.4f}  "
            f"train_acc={acc:.3f}  recall[{per_class}]",
            flush=True,
        )

    model.save(OUT_PATH)
    print(f"Saved pretrained policy to {OUT_PATH}")


if __name__ == "__main__":
    main()
