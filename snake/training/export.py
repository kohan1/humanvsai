"""
Export the trained Snake PPO policy to a browser-ready ONNX file.

Wraps the policy's actual forward pass (feature extractor -> mlp_extractor
-> action_net) into a plain nn.Module, rather than calling the SB3 policy
as a black box — same approach as the Tetris export, and it means we can
easily add auxiliary per-layer outputs later if we build the same live
neural-net visualization for Snake.

Also inlines any split .onnx + .onnx.data files newer PyTorch versions can
produce, so the whole model embeds as a single base64 blob per the
embed_model.py pattern (see the snake build brief, section 5).
"""

import torch
import torch.nn as nn
import onnx
from stable_baselines3 import PPO

MODEL_PATH = "snake_final.zip"
ONNX_PATH = "snake_ai.onnx"


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

    # Inline any split external-data weights back into one file.
    m = onnx.load(ONNX_PATH, load_external_data=True)
    onnx.save_model(m, ONNX_PATH, save_as_external_data=False)

    print(f"Exported to {ONNX_PATH} (obs size: {obs_size})")


if __name__ == "__main__":
    main()
