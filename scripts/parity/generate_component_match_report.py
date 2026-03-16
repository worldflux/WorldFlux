#!/usr/bin/env python3
"""Generate a component match report from official checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldflux import create_world_model  # noqa: E402
from worldflux.parity.component_match import (  # noqa: E402
    ComponentMatchReport,
    run_dreamerv3_component_match,
    run_tdmpc2_component_match,
)
from worldflux.parity.weight_map import dreamerv3_weight_map, tdmpc2_weight_map  # noqa: E402


def _parse_obs_shape(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--obs-shape must contain at least one dimension")
    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"invalid --obs-shape: {raw!r}") from exc
    if any(dim <= 0 for dim in dims):
        raise SystemExit("--obs-shape dimensions must be positive")
    return dims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=str, choices=["dreamerv3", "tdmpc2"], required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-shape", type=str, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model-id",
        type=str,
        default="",
        help="Optional WorldFlux model id override. Defaults to the proof-canonical model for the family.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON file with create_world_model config overrides.",
    )
    return parser.parse_args()


def _to_tensor_map(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            out[str(key)] = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            out[str(key)] = torch.from_numpy(value)
    return out


def _load_official_state(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as loaded:
            return {str(key): torch.from_numpy(loaded[key]) for key in loaded.files}
    if suffix == ".pkl":
        with path.open("rb") as handle:
            # Trusted local checkpoint input produced by our own parity tooling.
            loaded = pickle.load(handle)  # nosec B301
        if isinstance(loaded, dict):
            if "params" in loaded and isinstance(loaded["params"], dict):
                loaded = loaded["params"]
            tensor_map = _to_tensor_map(loaded)
            if tensor_map:
                return tensor_map

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            loaded = loaded["model_state_dict"]
        elif "model" in loaded and isinstance(loaded["model"], dict):
            loaded = loaded["model"]
        tensor_map = _to_tensor_map(loaded)
        if tensor_map:
            return tensor_map
    raise SystemExit(f"unsupported checkpoint format: {path}")


def _build_worldflux_model(
    *,
    family: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    device: str,
    model_id: str = "",
    config_overrides: dict[str, Any] | None = None,
):
    resolved_model_id = model_id.strip()
    if not resolved_model_id:
        if family == "dreamerv3":
            resolved_model_id = "dreamerv3:official_xl"
        else:
            resolved_model_id = "tdmpc2:5m"
    return create_world_model(
        resolved_model_id,
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=device,
        **dict(config_overrides or {}),
    )


def _serialize_report(report: ComponentMatchReport) -> dict[str, Any]:
    return {
        "schema_version": "parity.component_match.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "family": report.family,
        "all_pass": report.all_pass,
        "results": [asdict(result) for result in report.results],
    }


def main() -> int:
    args = _parse_args()
    obs_shape = _parse_obs_shape(args.obs_shape)
    official_state = _load_official_state(args.official_checkpoint.resolve())
    config_overrides: dict[str, Any] = {}
    if args.config_json is not None:
        loaded = json.loads(args.config_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit("--config-json must contain a JSON object")
        config_overrides = dict(loaded)

    if args.family == "dreamerv3":
        mapped_state = dreamerv3_weight_map().official_to_worldflux(official_state)
    else:
        mapped_state = tdmpc2_weight_map().official_to_worldflux(official_state)

    model = _build_worldflux_model(
        family=args.family,
        obs_shape=obs_shape,
        action_dim=int(args.action_dim),
        device=args.device,
        model_id=args.model_id,
        config_overrides=config_overrides,
    )
    if args.family == "dreamerv3":
        report = run_dreamerv3_component_match(mapped_state, model)
    else:
        report = run_tdmpc2_component_match(mapped_state, model)

    payload = _serialize_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
