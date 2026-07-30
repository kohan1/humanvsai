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

import os

import torch
import torch.nn as nn
import sys

import onnx
from stable_baselines3 import PPO

# Defaults to the current model, but accepts a path so a specific
# checkpoint can be exported:
#     python export.py archive/models/snake_final.129pt34.zip
# Used to add the value head to the weights already on the site without
# also swapping in a different (stronger) policy in the same change.
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "snake_final.zip"

# An explicit output path, so a checkpoint can be exported for the checkpoint
# switcher WITHOUT overwriting the model the site plays with:
#     python export.py archive/models/snake_final.70pt54.zip \
#                      ../checkpoints/s30m.onnx --no-critic
# Defaulting to the shipped name keeps the old one-argument behaviour, which
# install_model.sh relies on.
ONNX_PATH = sys.argv[2] if len(sys.argv) > 2 else "snake_ai.onnx"
CRITIC_PATH = ONNX_PATH.replace(".onnx", "_critic.onnx") \
    if len(sys.argv) > 2 else "snake_critic.onnx"

# The switcher's rungs do not ship a critic: it is another 34 MB per rung to
# power one readout, and the inspector already hides that readout when the
# fetch 404s.
WANT_CRITIC = "--no-critic" not in sys.argv


"""
TWO FILES, NOT ONE — see watermelon/training/export.py. Folding the critic in
as a second output of the playing model doubles the download, because the
value head carries its own copy of the conv stack. The inspector fetches the
critic separately, and only when someone opens it.
"""


class PolicyWrapper(nn.Module):
    """The playing model. This is what every visitor downloads."""

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = getattr(
            policy, "pi_features_extractor", policy.features_extractor)
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        return self.action_net(
            self.mlp_extractor.forward_actor(self.features_extractor(obs)))


class CriticWrapper(nn.Module):
    """Value head, using the VALUE extractor — feeding it the policy's
    features would produce numbers from a network that never saw them."""

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = getattr(
            policy, "vf_features_extractor", policy.features_extractor)
        self.mlp_extractor = policy.mlp_extractor
        self.value_net = policy.value_net

    def forward(self, obs):
        return self.value_net(
            self.mlp_extractor.forward_critic(self.features_extractor(obs)))


def write(wrapper, path, obs_size, output_names):
    torch.onnx.export(
        wrapper,
        torch.zeros(1, obs_size),
        path,
        input_names=["observation"],
        output_names=output_names,
        dynamic_axes={n: {0: "batch"} for n in ["observation"] + output_names},
        opset_version=17,
    )
    m = onnx.load(path, load_external_data=True)
    onnx.save_model(m, path, save_as_external_data=False)
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    return os.path.getsize(path) / 1048576


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    obs_size = model.observation_space.shape[0]

    mb = write(PolicyWrapper(model.policy).eval(), ONNX_PATH, obs_size, ["action_logits"])
    print(f"Exported to {ONNX_PATH} (obs {obs_size}, {mb:.1f} MB)")

    if not WANT_CRITIC:
        return
    mb = write(CriticWrapper(model.policy).eval(), CRITIC_PATH, obs_size, ["value"])
    print(f"Exported to {CRITIC_PATH} (obs {obs_size} -> value, {mb:.1f} MB) "
          f"— fetched only when the inspector is opened")


if __name__ == "__main__":
    main()
