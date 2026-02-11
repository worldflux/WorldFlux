#!/usr/bin/env python3
"""Smoke test for the minimal installable WorldFlux plugin."""

from __future__ import annotations

import torch

from worldflux import create_world_model
from worldflux.core import WorldModelRegistry


def main() -> int:
    WorldModelRegistry.load_entrypoint_plugins(force=True)

    alias_target = WorldModelRegistry.resolve_alias("minimalplugin-dreamer")
    if alias_target != "dreamerv3:ci":
        raise RuntimeError(f"Unexpected alias mapping: {alias_target}")

    model = create_world_model(
        "minimalplugin-dreamer",
        obs_shape=(8,),
        action_dim=2,
        encoder_type="mlp",
        decoder_type="mlp",
        component_overrides={"action_conditioner": "minimal_plugin.zero_action_conditioner"},
    )

    obs = torch.randn(1, 8)
    action = torch.randn(1, 2)
    state = model.encode(obs)
    _ = model.transition(state, action)
    print("minimal plugin smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
