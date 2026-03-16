#!/usr/bin/env python3
"""Generate a TD-MPC2 structural alignment report from official/worldflux metadata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldflux.core.config import TDMPC2Config  # noqa: E402

_OFFICIAL_5M_ARCH = {
    "model_id": "tdmpc2:5m",
    "model_profile": "5m",
    "latent_dim": 256,
    "hidden_dim": 256,
}
_WORLD_FLUX_CANONICAL = "proof_5m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-input", type=Path, required=True)
    parser.add_argument("--worldflux-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return loaded


def _extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("metrics"), dict) and isinstance(
        payload["metrics"].get("metadata"), dict
    ):
        return dict(payload["metrics"]["metadata"])
    if isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def _compare_field(
    *,
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    checks.append(
        {
            "category": category,
            "field": name,
            "expected": expected,
            "actual": actual,
            "pass": expected == actual,
        }
    )


def _artifact_semantics(metadata: dict[str, Any]) -> dict[str, bool]:
    manifest = metadata.get("artifact_manifest")
    if not isinstance(manifest, dict):
        return {"artifact_manifest_present": False, "metrics_paths_present": False}
    metrics_paths = manifest.get("metrics_paths")
    return {
        "artifact_manifest_present": True,
        "metrics_paths_present": isinstance(metrics_paths, list) and bool(metrics_paths),
    }


def main() -> int:
    args = _parse_args()
    official_metadata = _extract_metadata(_load_payload(args.official_input.resolve()))
    worldflux_metadata = _extract_metadata(_load_payload(args.worldflux_input.resolve()))

    worldflux_profile = str(worldflux_metadata.get("model_profile", "")).strip().lower()
    worldflux_cfg = TDMPC2Config.from_size(worldflux_profile) if worldflux_profile else None

    checks: list[dict[str, Any]] = []
    _compare_field(
        checks=checks,
        category="architecture",
        name="official.model_id",
        expected=_OFFICIAL_5M_ARCH["model_id"],
        actual=str(official_metadata.get("model_id", "")),
    )
    _compare_field(
        checks=checks,
        category="architecture",
        name="official.model_profile",
        expected=_OFFICIAL_5M_ARCH["model_profile"],
        actual=str(official_metadata.get("model_profile", "")).strip().lower(),
    )
    _compare_field(
        checks=checks,
        category="architecture",
        name="worldflux.model_profile",
        expected=_WORLD_FLUX_CANONICAL,
        actual=worldflux_profile,
    )
    if worldflux_cfg is not None:
        _compare_field(
            checks=checks,
            category="architecture",
            name="worldflux.latent_dim",
            expected=_OFFICIAL_5M_ARCH["latent_dim"],
            actual=int(worldflux_cfg.latent_dim),
        )
        _compare_field(
            checks=checks,
            category="architecture",
            name="worldflux.hidden_dim",
            expected=_OFFICIAL_5M_ARCH["hidden_dim"],
            actual=int(worldflux_cfg.hidden_dim),
        )

    for field in ("steps", "eval_interval", "eval_episodes", "eval_window"):
        expected = None
        actual = None
        if field == "steps":
            expected = (
                (official_metadata.get("train_budget") or {}).get(field)
                if isinstance(official_metadata.get("train_budget"), dict)
                else None
            )
            actual = (
                (worldflux_metadata.get("train_budget") or {}).get(field)
                if isinstance(worldflux_metadata.get("train_budget"), dict)
                else None
            )
            category = "train_budget"
        else:
            expected = (
                (official_metadata.get("eval_protocol") or {}).get(field)
                if isinstance(official_metadata.get("eval_protocol"), dict)
                else None
            )
            actual = (
                (worldflux_metadata.get("eval_protocol") or {}).get(field)
                if isinstance(worldflux_metadata.get("eval_protocol"), dict)
                else None
            )
            category = "eval_protocol"
        _compare_field(
            checks=checks, category=category, name=field, expected=expected, actual=actual
        )

    official_artifacts = _artifact_semantics(official_metadata)
    worldflux_artifacts = _artifact_semantics(worldflux_metadata)
    for field, expected in official_artifacts.items():
        _compare_field(
            checks=checks,
            category="artifact_semantics",
            name=f"worldflux.{field}",
            expected=expected,
            actual=worldflux_artifacts.get(field),
        )

    status = "aligned" if all(bool(check["pass"]) for check in checks) else "mismatched"
    report = {
        "schema_version": "tdmpc2.alignment.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_input": str(args.official_input.resolve()),
        "worldflux_input": str(args.worldflux_input.resolve()),
        "official": {
            "model_id": official_metadata.get("model_id"),
            "model_profile": official_metadata.get("model_profile"),
        },
        "worldflux": {
            "model_id": worldflux_metadata.get("model_id"),
            "model_profile": worldflux_metadata.get("model_profile"),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
