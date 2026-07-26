"""
pretrain.py
-----------
Behavioural cloning on heuristic expert data.
Trains the neural network to imitate the Dellacherie heuristic
before RL fine-tuning takes over.

Usage:
    python3 pretrain.py                              # uses heuristic_data.json
    python3 pretrain.py --data heuristic_data.json  # explicit file

Run AFTER heuristic.py:
    python3 heuristic.py --games 5000
    python3 pretrain.py
    python3 train.py
"""

import os
import sys
import json
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

OBS_SIZE    = 238
MAX_ACTIONS = 40
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 1e-4   # safe learning rate
KL_WEIGHT   = 0.3    # how much to anchor to original weights

# ─── Dataset ──────────────────────────────────────────────────────────────────

class HeuristicDataset(Dataset):
    def __init__(self, records):
        self.obs     = torch.tensor(
            [r["obs"] for r in records], dtype=torch.float32
        )
        self.actions = torch.tensor(
            [min(r["action"], MAX_ACTIONS-1) for r in records], dtype=torch.long
        )

    def __len__(self): return len(self.obs)
    def __getitem__(self, idx): return self.obs[idx], self.actions[idx]

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Find data file
    data_file = "heuristic_data.json"
    if "--data" in sys.argv:
        data_file = sys.argv[sys.argv.index("--data")+1]

    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found.")
        print("Run heuristic.py first to generate training data.")
        sys.exit(1)

    print("=" * 55)
    print("  Behavioural Cloning — Heuristic Expert Data")
    print("=" * 55)

    with open(data_file) as f:
        records = json.load(f)
    print(f"\nLoaded {len(records):,} expert records from {data_file}")

    # Find model
    model_path = None
    for candidate in ["models/tetris_ppo_final.zip", "models/best_model.zip"]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if not model_path:
        print("\nNo existing model found — will pretrain from scratch.")
        from_scratch = True
    else:
        print(f"Loading model: {model_path}")
        from_scratch = False

    # Load model
    if not from_scratch:
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, device="cpu")
            print(f"Loaded with MaskablePPO — {model.num_timesteps:,} steps trained")
        except Exception as e1:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(model_path, device="cpu")
                print(f"Loaded with PPO — {model.num_timesteps:,} steps trained")
            except Exception as e2:
                print(f"ERROR loading model: {e1} / {e2}")
                sys.exit(1)

        policy     = model.policy
        orig_policy = copy.deepcopy(policy)
        orig_policy.eval()
    else:
        # Build a fresh policy from scratch
        from stable_baselines3 import PPO
        from tetris_env import TetrisEnv
        from stable_baselines3.common.monitor import Monitor
        env   = Monitor(TetrisEnv())
        model = PPO("MlpPolicy", env, device="cpu",
                    policy_kwargs=dict(net_arch=[512,512,256]))
        policy      = model.policy
        orig_policy = copy.deepcopy(policy)
        orig_policy.eval()

    policy.train()

    # Dataset and loader
    dataset    = HeuristicDataset(records)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {EPOCHS} epochs on {len(records):,} samples...")
    print(f"Batch size: {BATCH_SIZE}  |  LR: {LR}  |  KL weight: {KL_WEIGHT}")
    print("─" * 55)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total   = 0

        for obs_batch, action_batch in dataloader:
            optimizer.zero_grad()

            # Forward pass
            latent_pi, _ = policy.mlp_extractor(obs_batch)
            logits        = policy.action_net(latent_pi)

            # Cross entropy — imitate expert actions
            ce_loss = criterion(logits, action_batch)

            # KL penalty — stay close to original weights
            with torch.no_grad():
                orig_latent, _ = orig_policy.mlp_extractor(obs_batch)
                orig_logits    = orig_policy.action_net(orig_latent)

            kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(logits, dim=-1),
                nn.functional.softmax(orig_logits, dim=-1),
                reduction="batchmean"
            )

            loss = ce_loss + KL_WEIGHT * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += ce_loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == action_batch).sum().item()
            total      += len(action_batch)

        avg_loss = total_loss / len(dataloader)
        acc      = correct / total * 100
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Accuracy: {acc:.1f}%")

    print("─" * 55)
    print(f"\nBest accuracy: {best_acc:.1f}%")

    # Save
    os.makedirs("models", exist_ok=True)
    final = "models/tetris_ppo_final"

    if os.path.exists(final+".zip"):
        shutil.copy(final+".zip", final+"_pre_heuristic.zip")
        print(f"Backup saved: {final}_pre_heuristic.zip")

    # Copy trained weights back into model and save
    model2 = copy.deepcopy(model)
    model2.policy.load_state_dict(policy.state_dict())
    model2.save(final)
    print(f"Saved: {final}.zip")
    print("\nNow run: python3 train.py")

if __name__ == "__main__":
    main()
