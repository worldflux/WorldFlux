#!/usr/bin/env python3
"""Generate a machine-readable manifest for the public WorldFlux API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

SCHEMA_VERSION = "worldflux.public_api_manifest.v1"
DEFAULT_OUTPUT_PATH = Path("reports/public_api_manifest.json")


def _load_worldflux():
    import worldflux

    return worldflux


def _module_for_symbol(name: str) -> str:
    worldflux = _load_worldflux()
    if name in {"training", "execution"}:
        return f"worldflux.{name}"
    exports = getattr(worldflux, "_EXPORTS")
    return str(exports.get(name, "worldflux"))


def _stability_for_symbol(name: str) -> str:
    worldflux = _load_worldflux()
    stability_fn = getattr(worldflux, "_public_api_stability_for")
    return str(stability_fn(name))


def generate_manifest() -> dict[str, Any]:
    worldflux = _load_worldflux()
    symbols: dict[str, dict[str, str]] = {}
    for name in sorted(worldflux.__all__):
        symbols[f"worldflux.{name}"] = {
            "module": _module_for_symbol(name),
            "exported_from": "worldflux",
            "stability": _stability_for_symbol(name),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "package": "worldflux",
        "symbols": symbols,
    }


def write_manifest(output_path: Path) -> Path:
    manifest = generate_manifest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for manifest JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Generate the manifest and exit non-zero if the existing file differs.",
    )
    args = parser.parse_args()

    manifest = json.dumps(generate_manifest(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists():
            return 1
        existing = args.output.read_text(encoding="utf-8")
        return 0 if existing == manifest else 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
