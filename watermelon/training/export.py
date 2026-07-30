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
import sys

import onnx
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# Defaults to the current model, but accepts a path so a specific
# checkpoint can be exported:
#     python export.py archive/models/watermelon_final.902pt10.zip
# Used to add the value head to the weights already on the site without
# also swapping in a different (stronger) policy in the same change.
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "watermelon_final.zip"

# An explicit output path, so a checkpoint can be exported for the checkpoint
# switcher WITHOUT overwriting the model the site plays with:
#     python export.py archive/models/watermelon_ppo_614.zip \
#                      ../checkpoints/w5m.onnx --no-critic
# Defaulting to the shipped name keeps the old one-argument behaviour, which
# install_model.sh relies on.
ONNX_PATH = sys.argv[2] if len(sys.argv) > 2 else "watermelon_ai.onnx"
CRITIC_PATH = ONNX_PATH.replace(".onnx", "_critic.onnx") \
    if len(sys.argv) > 2 else "watermelon_critic.onnx"

# The switcher's rungs do not ship a critic — another 22.6 MB per rung for one
# readout the inspector already hides when the fetch 404s.
WANT_CRITIC = "--no-critic" not in sys.argv


"""
TWO FILES, NOT ONE.

The obvious thing is to add `value` as a second output of the policy model.
Measured: that takes the file from 22.6 MB to 45.3 MB, because
share_features_extractor=False (see policy_config.py) means the critic carries
its own complete copy of the conv stack.

Doubling what every visitor downloads, to power a readout inside a panel that
is closed by default, is a bad trade. So the critic is exported separately and
the page fetches it only when someone opens the inspector.
"""


class PolicyWrapper(nn.Module):
    """The playing model. Unchanged, and deliberately so — this is what every
    visitor downloads."""

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = getattr(
            policy, "pi_features_extractor", policy.features_extractor)
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        latent_pi = self.mlp_extractor.forward_actor(self.features_extractor(obs))
        return self.action_net(latent_pi)


class CriticWrapper(nn.Module):
    """
    The critic's estimate of the return expected from the current position —
    what the inspector shows as "expected score from here".

    Uses the VALUE extractor, not the policy's. Feeding the policy's features
    to the value head would produce plausible numbers from a network that never
    saw those inputs during training.
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = getattr(
            policy, "vf_features_extractor", policy.features_extractor)
        self.mlp_extractor = policy.mlp_extractor
        self.value_net = policy.value_net

    def forward(self, obs):
        return self.value_net(self.mlp_extractor.forward_critic(self.features_extractor(obs)))


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

    # Newer PyTorch can split weights into a sibling .onnx.data file; inline
    # them so the whole model embeds as one base64 blob.
    m = onnx.load(path, load_external_data=True)
    onnx.save_model(m, path, save_as_external_data=False)

    # Once inlined, the sidecar is dead weight — and leaving it behind is
    # actively misleading, since it looks like part of the model.
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    return os.path.getsize(path) / 1048576


def main():
    model = PPO.load(MODEL_PATH, device="cpu")
    obs_size = model.observation_space.shape[0]

    mb = write(PolicyWrapper(model.policy).eval(), ONNX_PATH, obs_size, ["action_logits"])
    n_actions = int(model.action_space.n)
    print(f"Exported to {ONNX_PATH} (obs {obs_size} -> {n_actions} actions, {mb:.1f} MB)")

    if not WANT_CRITIC:
        return
    mb = write(CriticWrapper(model.policy).eval(), CRITIC_PATH, obs_size, ["value"])
    print(f"Exported to {CRITIC_PATH} (obs {obs_size} -> value, {mb:.1f} MB) "
          f"— fetched only when the inspector is opened")


if __name__ == "__main__":
    main()
