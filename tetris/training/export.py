"""
export.py
---------
Converts the trained PPO model to ONNX format for use in the browser
via onnxruntime-web.

Usage:
    python3 export.py                        # exports models/tetris_ppo_final.zip
    python3 export.py models/best_model.zip  # exports a specific checkpoint

Output:
    tetris_ai.onnx  — drop this into your website folder

Requirements:
    pip3 install stable-baselines3 torch gymnasium numpy onnx

What this does:
    1. Loads the trained SB3 PPO model
    2. Extracts just the policy network (actor) — the part that picks moves
    3. Traces it with a dummy observation to get the computation graph
    4. Exports to ONNX format, exposing BOTH the final action logits AND
       the post-activation values of every hidden layer (for the live
       "neural network visualization" on the website)
    5. Verifies the export is valid

The exported model takes a 238-float observation vector and outputs:
    - action_logits   [batch, 40]   — one score per placement slot
    - layer1_act       [batch, 512]  — hidden layer 1 activations (post-ReLU)
    - layer2_act       [batch, 512]  — hidden layer 2 activations (post-ReLU)
    - layer3_act       [batch, 256]  — hidden layer 3 activations (post-ReLU)

The website picks the highest-scoring valid action from action_logits,
and can optionally use the layerN_act outputs to animate a live view of
neurons firing as the AI thinks.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

# ─── Load model ───────────────────────────────────────────────────────────────

def find_model_path():
    """Find the best available model to export."""
    candidates = [
        "models/tetris_ppo_final.zip",
        "models/best_model.zip",
    ]

    # Also check for command line argument
    if len(sys.argv) > 1:
        candidates.insert(0, sys.argv[1])

    for path in candidates:
        if os.path.exists(path):
            return path

    # Last resort: find any checkpoint
    if os.path.exists("models"):
        checkpoints = sorted([
            f for f in os.listdir("models")
            if f.endswith(".zip")
        ])
        if checkpoints:
            return os.path.join("models", checkpoints[-1])

    return None


def main():
    print("=" * 55)
    print("  Tetris AI — ONNX Export (with activations)")
    print("=" * 55)

    # ── Find model ────────────────────────────────────────────────
    model_path = find_model_path()
    if not model_path:
        print("\nERROR: No trained model found.")
        print("Make sure you have run train.py first and the")
        print("models/ folder contains tetris_ppo_final.zip")
        sys.exit(1)

    print(f"\nLoading model from: {model_path}")

    # ── Load SB3 model — try PPO first, then MaskablePPO ─────────
    model = None
    errors = []

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        print("Loaded with PPO")
    except Exception as e:
        errors.append(f"PPO: {e}")

    if model is None:
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, device="cpu")
            print("Loaded with MaskablePPO")
        except Exception as e:
            errors.append(f"MaskablePPO: {e}")

    if model is None:
        print("\nERROR: Could not load model.")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)

    print(f"Model loaded successfully.")
    print(f"Observation size: {model.observation_space.shape}")
    print(f"Action size: {model.action_space.n}")

    obs_size    = model.observation_space.shape[0]  # 238
    action_size = model.action_space.n               # 40

    # ── Extract the policy network ────────────────────────────────
    # SB3's PPO has an actor-critic network. We only need the actor
    # (the part that outputs action probabilities), but we tap into
    # its hidden layers along the way so the browser can visualize them.
    policy = model.policy
    policy.eval()

    # ── Build a clean export wrapper ──────────────────────────────
    # We wrap the policy to output the action logits PLUS the
    # post-activation output of every hidden layer in policy_net.

    class ActorWithActivations(nn.Module):
        """
        Wraps the SB3 policy to export action logits AND hidden layer
        activations, for live visualization in the browser.

        Input:  [batch, 238] float32 observation
        Output: (action_logits [batch, 40],
                 layer1_act [batch, 512],
                 layer2_act [batch, 512],
                 layer3_act [batch, 256])
        """
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.policy_net         = policy.mlp_extractor.policy_net
            self.action_net         = policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)

            x = features
            activations = []
            for layer in self.policy_net:
                x = layer(x)
                # Capture the tensor right after each activation function —
                # i.e. the post-ReLU values for each hidden layer. This is
                # what's meaningful to visualize (raw pre-activation values
                # are much less interpretable as "neuron firing").
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU,
                                       nn.LeakyReLU, nn.GELU, nn.SiLU)):
                    activations.append(x)

            logits = self.action_net(x)
            return (logits, *activations)

    actor = ActorWithActivations(policy)
    actor.eval()

    # ── Test with dummy input ─────────────────────────────────────
    print("\nTesting forward pass...")
    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)
    with torch.no_grad():
        outputs = actor(dummy_obs)

    logits = outputs[0]
    activations = outputs[1:]

    print(f"Output shape: {logits.shape}  (expected [1, {action_size}])")
    print(f"Output sample: {logits[0, :5].numpy()}...")
    print(f"\nFound {len(activations)} hidden layer(s) to expose:")
    for i, act in enumerate(activations, start=1):
        print(f"  layer{i}_act: shape {tuple(act.shape)}")

    if len(activations) == 0:
        print("\nWARNING: No activation layers detected in policy_net.")
        print("Check that policy.mlp_extractor.policy_net contains")
        print("standard activation modules (ReLU, Tanh, etc).")

    # ── Export to ONNX ────────────────────────────────────────────
    output_path = "tetris_ai.onnx"
    print(f"\nExporting to ONNX: {output_path}")

    output_names = ["action_logits"] + [f"layer{i+1}_act" for i in range(len(activations))]
    dynamic_axes = {name: {0: "batch_size"} for name in ["observation"] + output_names}

    torch.onnx.export(
        actor,
        dummy_obs,
        output_path,

        input_names=["observation"],
        output_names=output_names,

        dynamic_axes=dynamic_axes,

        opset_version=17,
        training=torch.onnx.TrainingMode.EVAL,
        # Force single-file export — prevents split .onnx + .data format
        # which browsers cannot load
        keep_initializers_as_inputs=False,
    )

    # If a .data file was created anyway, inline it into the .onnx file
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        print("Detected split export — inlining weights into single file...")
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
        model_proto = onnx.load(output_path)
        load_external_data_for_model(model_proto, os.path.dirname(os.path.abspath(output_path)))
        # Save as single file
        onnx.save(model_proto, output_path, save_as_external_data=False)
        os.remove(data_file)
        print("Weights inlined successfully.")

    print(f"Export complete.")

    # ── Verify the export ─────────────────────────────────────────
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX verification passed.")
    except ImportError:
        print("(onnx package not installed — skipping verification)")
        print("Install with: pip3 install onnx")
    except Exception as e:
        print(f"WARNING: ONNX verification failed: {e}")
        print("The file may still work in the browser.")

    # ── File size ─────────────────────────────────────────────────
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFile size: {size_mb:.1f} MB")

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Done! Next steps:")
    print("=" * 55)
    print(f"\n  1. Copy tetris_ai.onnx into your website folder")
    print(f"  2. Re-run embed_model.py to refresh model_data.js")
    print(f"  3. The website will load it automatically")
    print(f"\n  The AI picks moves by:")
    print(f"    - Building a {obs_size}-float observation of the board")
    print(f"    - Running it through this network")
    print(f"    - Getting {action_size} logits (one per placement slot)")
    print(f"    - Picking the highest-scoring valid placement")
    print(f"\n  It also now exposes {len(activations)} hidden layer(s)")
    print(f"  of activations, for the live neural network visualization.")
    print()


if __name__ == "__main__":
    main()
