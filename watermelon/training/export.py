"""
Export the trained Watermelon policy to a browser-ready ONNX file.

Wraps the policy's actual forward pass (features extractor -> mlp_extractor ->
action_net) in a plain nn.Module rather than calling the SB3 policy as a black
box, matching the Snake and Tetris exports. That also leaves room to expose
per-layer activations later for the neural-net visualisation.

Windows note: run with PYTHONIOENCODING=utf-8. torch.onnx prints status lines
containing non-cp1252 glyphs and the default console encoding crashes the
export part-way through — it looks like a code failure but is not.

    PYTHONIOENCODING=utf-8 python export.py
    cp watermelon_ai.onnx ../watermelon_ai.onnx
    cd .. && python embed_model.py
"""

import os

import onnx
import torch
import torch.nn as nn
from stable_baselines3 import PPO

MODEL_PATH = "watermelon_final.zip"
ONNX_PATH = "watermelon_ai.onnx"


class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    wrapper = PolicyWrapper(model.policy).eval()

    obs_size = model.observation_space.shape[0]
    dummy_input = torch.zeros(1, obs_size)

    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={"observation": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=17,
    )

    # Newer PyTorch can split weights into a sibling .onnx.data file; inline
    # them so the whole model embeds as one base64 blob.
    m = onnx.load(ONNX_PATH, load_external_data=True)
    onnx.save_model(m, ONNX_PATH, save_as_external_data=False)

    # Once inlined, the sidecar is dead weight — and leaving it behind is
    # actively misleading, since it looks like part of the model.
    sidecar = ONNX_PATH + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    n_actions = int(model.action_space.n)
    print(f"Exported to {ONNX_PATH} (obs {obs_size} -> {n_actions} actions)")


if __name__ == "__main__":
    main()
