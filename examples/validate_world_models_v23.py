#!/usr/bin/env python3
"""Validate WorldFlux v2.3 planner metadata behavior on key model families."""

from __future__ import annotations

import argparse
import warnings

import torch

from worldflux import create_world_model
from worldflux.core.payloads import (
    PLANNER_HORIZON_KEY,
    PLANNER_SEQUENCE_KEY,
    ActionPayload,
    first_action,
    normalize_planned_action,
)
from worldflux.planners import CEMPlanner


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_metadata_bridge_rules(action_dim: int) -> None:
    # v0.2: missing horizon -> infer + deprecation warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = ActionPayload(kind="continuous", tensor=torch.randn(3, action_dim))
        seq = normalize_planned_action(payload, api_version="v0.2")
    _assert(seq.tensor is not None, "Expected tensor sequence after v0.2 normalization")
    _assert(
        any(issubclass(w.category, DeprecationWarning) for w in caught),
        "Expected DeprecationWarning when horizon metadata is inferred in v0.2",
    )

    # v3: missing horizon -> hard error.
    try:
        normalize_planned_action(
            ActionPayload(kind="continuous", tensor=torch.randn(3, action_dim)),
            api_version="v3",
        )
    except ValueError as exc:
        _assert("Missing required planner metadata" in str(exc), str(exc))
    else:
        raise AssertionError("Expected ValueError in v3 when planner horizon is missing")

    # v3: legacy sequence key -> hard error.
    try:
        normalize_planned_action(
            ActionPayload(
                kind="continuous",
                tensor=torch.randn(3, action_dim),
                extras={PLANNER_SEQUENCE_KEY: True, PLANNER_HORIZON_KEY: 3},
            ),
            api_version="v3",
        )
    except ValueError as exc:
        _assert("removed in v0.3" in str(exc), str(exc))
    else:
        raise AssertionError("Expected ValueError in v3 when legacy sequence key is provided")


def _check_planner_model(
    *,
    model_id: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    api_version: str,
    device: str,
    horizon: int,
    extra_model_kwargs: dict[str, object] | None = None,
) -> None:
    kwargs = dict(extra_model_kwargs or {})
    model = create_world_model(
        model_id,
        obs_shape=obs_shape,
        action_dim=action_dim,
        api_version=api_version,
        device=device,
        **kwargs,
    )
    obs = torch.randn(1, *obs_shape, device=device)
    state = model.encode(obs)

    planner = CEMPlanner(horizon=horizon, action_dim=action_dim, num_samples=32, num_elites=8)
    planned = planner.plan(model, state)
    _assert(isinstance(planned, ActionPayload), f"{model_id}: planner output must be ActionPayload")
    _assert(
        int(planned.extras.get(PLANNER_HORIZON_KEY, -1)) == horizon,
        f"{model_id}: missing or invalid planner horizon metadata",
    )

    seq = normalize_planned_action(planned, api_version=api_version)
    _assert(seq.tensor is not None, f"{model_id}: expected tensor action sequence")
    _assert(
        int(seq.tensor.shape[0]) == horizon,
        f"{model_id}: expected horizon {horizon}, got {tuple(seq.tensor.shape)}",
    )

    first = first_action(planned, api_version=api_version)
    _assert(first.tensor is not None, f"{model_id}: first action tensor must exist")
    _assert(
        int(first.extras.get(PLANNER_HORIZON_KEY, -1)) == 1,
        f"{model_id}: first action horizon metadata must be 1",
    )

    print(f"[OK] {model_id}: planner metadata + normalization")


def _check_vjepa_rollout(*, action_dim: int, api_version: str, device: str, horizon: int) -> None:
    model = create_world_model(
        "vjepa2:ci",
        obs_shape=(4,),
        action_dim=action_dim,
        api_version=api_version,
        device=device,
    )
    state = model.encode(torch.randn(1, 4, device=device))

    planned = ActionPayload(
        kind="continuous",
        tensor=torch.randn(horizon, 1, action_dim, device=device),
        extras={PLANNER_HORIZON_KEY: horizon},
    )
    trajectory = model.rollout(state, planned)

    _assert(
        len(trajectory.states) == horizon + 1,
        f"V-JEPA2 rollout expected {horizon + 1} states, got {len(trajectory.states)}",
    )
    _assert(
        trajectory.actions.shape[0] == horizon,
        f"V-JEPA2 rollout expected action horizon {horizon}, got {tuple(trajectory.actions.shape)}",
    )
    print("[OK] vjepa2:ci: rollout accepts ActionPayload planner output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v2.3 world-model behavior")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--api-version", default="v0.2", choices=("v0.2", "v3"))
    args = parser.parse_args()

    _check_metadata_bridge_rules(action_dim=args.action_dim)
    print("[OK] metadata bridge rules: v0.2 warning / v3 hard errors")

    _check_planner_model(
        model_id="dreamerv3:ci",
        obs_shape=(4,),
        action_dim=args.action_dim,
        api_version=args.api_version,
        device=args.device,
        horizon=args.horizon,
        extra_model_kwargs={"encoder_type": "mlp", "decoder_type": "mlp"},
    )
    _check_planner_model(
        model_id="tdmpc2:ci",
        obs_shape=(4,),
        action_dim=args.action_dim,
        api_version=args.api_version,
        device=args.device,
        horizon=args.horizon,
    )
    _check_vjepa_rollout(
        action_dim=args.action_dim,
        api_version=args.api_version,
        device=args.device,
        horizon=args.horizon,
    )

    print("\nAll v2.3 checks passed.")


if __name__ == "__main__":
    main()
