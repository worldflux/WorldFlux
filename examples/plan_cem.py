#!/usr/bin/env python3
"""
Example: Planning with a world model using CEM.
"""

import argparse
import logging

import torch

from worldflux import create_world_model
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
    action_seq = planner.plan(model, state)
    logger.info("Planned action sequence shape: %s", action_seq.shape)


if __name__ == "__main__":
    main()
