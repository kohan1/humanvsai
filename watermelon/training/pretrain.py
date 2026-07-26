"""
Behavioural-cloning warm start for the Watermelon AI.

Clones heuristic.py into the policy network, so PPO starts from something that
already plays rather than from noise.

Two lessons baked in from the Snake build:

  1. Collection is capped by STEPS, not episodes. Every observation is buffered
     in memory (1332 floats = 5.3 KB each), so an episode cap is really an
     unbounded memory cap — the better the teacher plays, the longer its
     episodes and the more RAM it eats.
  2. NO KL anchor and a normal supervised learning rate. "Low LR + KL penalty"
     protects an already-competent policy; applied to a cold start it fights
     the entire point of pretraining and the network barely learns.

Run:
    python pretrain.py        # -> watermelon_pretrained.zip
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from watermelon_env import WatermelonEnv
from heuristic import heuristic_action_from_env
from policy_config import policy_kwargs

DEVICE = os.environ.get("TRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 300_000     # what actually stops collection
N_EPISODES = 6000       # upper bound

# Class weighting, learned the hard way on Snake. Its teacher played 79%
# STRAIGHT; unweighted, BC collapsed onto that majority class, reported a
# "train_acc" of 0.787 that was exactly the class prior, and evaluated at 0.00.
# A heuristic that favours particular drop columns has the same failure mode.
# Weights are clipped because with 24 actions a column the teacher almost never
# picks would otherwise get an enormous weight and dominate the gradient.
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_CLIP = 5.0

LR = 3e-4               # conservative for a conv net; 1e-3 helped Snake collapse
EPOCHS = 12

# Value-function warm-up. BC trains only the policy; without this the critic
# reaches PPO at random init and its noisy advantages destroy the clone before
# it learns anything useful. See fit_value_function() for the measurements.
# Measured: 40 episodes / 6 epochs was far too little — only ~40 gradient
# steps, and the critic just learned the mean (explained_variance 0.00). At
# 200 episodes / 30 epochs it reaches ~0.82, which is what PPO needs.
VF_EPISODES = 200
VF_EPOCHS = 30
VF_BATCH_SIZE = 256
BATCH_SIZE = 512

OUT_PATH = "watermelon_pretrained.zip"


def collect_expert_data(n_episodes=None, max_steps=None):
    # Resolved here rather than as default arguments, which bind at def time
    # and so cannot be overridden by reassigning the module constants.
    n_episodes = N_EPISODES if n_episodes is None else n_episodes
    max_steps = MAX_STEPS if max_steps is None else max_steps

    env = WatermelonEnv()
    obs_buf, act_buf = [], []
    episodes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = heuristic_action_from_env(env)
            obs_buf.append(obs)
            act_buf.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        episodes += 1

        if episodes % 100 == 0:
            print(f"  {episodes} episodes, {len(obs_buf):,}/{max_steps:,} steps", flush=True)
        if len(obs_buf) >= max_steps:
            print(f"  reached the {max_steps:,}-step cap after {episodes} episodes", flush=True)
            break

    return np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.int64)


def fit_value_function(model, n_episodes=VF_EPISODES, epochs=VF_EPOCHS, gamma=0.99):
    """
    Fit the critic to the cloned policy's own returns.

    WHY THIS MATTERS. Behavioural cloning trains only the POLICY network — it
    is a supervised classifier over actions and never touches the value head.
    So a BC checkpoint pairs a good policy with a randomly-initialised critic,
    and PPO's first updates compute advantages from noise. Measured here: PPO
    started from a BC policy scoring ~750, explained_variance began NEGATIVE
    (-0.016), and rollout score fell 721 -> 520 while the critic slowly learned
    (ending at 0.78). By then the policy had already been wrecked.

    Snake got away with it: 3 actions plus target_kl clamping leaves little
    room for a softmax to scatter. Watermelon has 24 actions and fell apart.

    Fitting the critic here means PPO starts with advantages that mean
    something, so its first updates build on the clone instead of destroying
    it.
    """
    import torch as _torch

    print(f"Fitting value function on {n_episodes} rollouts of the cloned policy...",
          flush=True)

    env = WatermelonEnv()
    obs_buf, ret_buf = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        ep_obs, ep_rew = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            ep_obs.append(obs)
            obs, reward, term, trunc, _ = env.step(int(action))
            ep_rew.append(reward)
            done = term or trunc

        # Discounted return-to-go, computed backwards.
        running = 0.0
        returns = [0.0] * len(ep_rew)
        for i in range(len(ep_rew) - 1, -1, -1):
            running = ep_rew[i] + gamma * running
            returns[i] = running

        obs_buf.extend(ep_obs)
        ret_buf.extend(returns)

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    ret_arr = np.asarray(ret_buf, dtype=np.float32)
    print(f"  {len(obs_arr):,} states, returns mean {ret_arr.mean():.2f} "
          f"std {ret_arr.std():.2f}", flush=True)

    # Only the value pathway is optimised; the cloned policy must not move.
    value_params = list(model.policy.mlp_extractor.value_net.parameters()) + \
        list(model.policy.value_net.parameters())
    optimizer = _torch.optim.Adam(value_params, lr=1e-3)

    n = len(obs_arr)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        batches = 0
        for start in range(0, n, VF_BATCH_SIZE):
            b = idx[start:start + VF_BATCH_SIZE]
            obs_b = _torch.as_tensor(obs_arr[b], device=DEVICE)
            ret_b = _torch.as_tensor(ret_arr[b], device=DEVICE)

            values = model.policy.predict_values(obs_b).squeeze(-1)
            loss = _torch.nn.functional.mse_loss(values, ret_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            batches += 1

        # Explained variance is the number PPO reports; matching it here makes
        # the two directly comparable. Near 0 means the critic is useless.
        with _torch.no_grad():
            preds = model.policy.predict_values(
                _torch.as_tensor(obs_arr, device=DEVICE)
            ).squeeze(-1).cpu().numpy()
        ev = 1.0 - np.var(ret_arr - preds) / max(1e-8, np.var(ret_arr))
        print(f"  vf epoch {epoch + 1}/{epochs}  mse={total / max(1, batches):.3f}  "
              f"explained_variance={ev:.3f}", flush=True)

    if ev < 0.3:
        print("  WARNING: critic still explains little of the return variance. "
              "PPO may degrade the policy before it catches up.", flush=True)


def main():
    print(f"Collecting heuristic demonstrations on {DEVICE}...", flush=True)
    obs_data, act_data = collect_expert_data()
    print(f"Collected {len(obs_data):,} (obs, action) pairs", flush=True)

    vec_env = make_vec_env(WatermelonEnv, n_envs=1)
    model = PPO(
        "MlpPolicy",   # overridden by features_extractor_class
        vec_env,
        policy_kwargs=policy_kwargs(),
        device=DEVICE,
        verbose=0,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    n_actions = int(model.action_space.n)
    counts = np.bincount(act_data, minlength=n_actions).astype(np.float64)
    used = int((counts > 0).sum())
    print(f"teacher uses {used}/{n_actions} columns; "
          f"most common takes {counts.max() / counts.sum():.3f} of actions", flush=True)

    class_weight = None
    if USE_CLASS_WEIGHTS:
        inv = counts.sum() / np.maximum(counts, 1.0)
        inv = np.minimum(inv, inv.min() * CLASS_WEIGHT_CLIP)
        inv *= n_actions / inv.sum()   # mean 1, so the LR stays comparable
        class_weight = torch.as_tensor(inv, dtype=torch.float32, device=DEVICE)

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
            # Same device as the policy — these being implicitly CPU is a
            # silent break the moment the model moves to CUDA.
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
            correct += (pred == act_b).sum().item()
            n_batches += 1

        # Worst recall across the classes the teacher actually uses. Aggregate
        # accuracy hides a collapse onto one action; this does not.
        active = [a for a in range(n_actions) if per_class_total[a] > 0]
        worst = min(per_class_correct[a] / per_class_total[a] for a in active)
        print(
            f"epoch {epoch + 1}/{EPOCHS}  "
            f"loss={total_loss / max(1, n_batches):.4f}  "
            f"train_acc={correct / n:.3f}  worst_class_recall={worst:.2f}",
            flush=True,
        )

    fit_value_function(model)

    model.save(OUT_PATH)
    print(f"Saved pretrained policy to {OUT_PATH}")


if __name__ == "__main__":
    main()
