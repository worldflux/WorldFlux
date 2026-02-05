#!/usr/bin/env python3
"""
Example: Planning with a world model using CEM.
"""

import argparse
import logging

import torch

from worldflux import create_world_model
from worldflux.core.payloads import normalize_planned_action
from worldflux.planners import CEMPlanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEM planning demo")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    model = create_world_model(
        "dreamerv3:ci",
        obs_shape=(4,),
        action_dim=args.action_dim,
        encoder_type="mlp",
        decoder_type="mlp",
    )
    obs = torch.randn(1, 4)
    state = model.encode(obs)

    planner = CEMPlanner(
        horizon=args.horizon, action_dim=args.action_dim, num_samples=64, num_elites=8
    )
    planned = planner.plan(model, state)
    action_seq = normalize_planned_action(planned, api_version="v0.2")
    if action_seq.tensor is None:
        raise RuntimeError("Planner returned payload without tensor sequence")
    logger.info("Planned action sequence shape: %s", tuple(action_seq.tensor.shape))


if __name__ == "__main__":
    main()
