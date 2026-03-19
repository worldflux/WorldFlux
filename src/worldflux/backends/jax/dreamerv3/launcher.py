# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Module entrypoint that bridges WorldFlux runtime to vendored DreamerV3 modules."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def _configure_vendor_paths(official_repo_root: Path) -> None:
    vendor_root = official_repo_root.resolve()
    path_entries = [str(vendor_root), str(vendor_root.parent)]
    for entry in reversed(path_entries):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _import_dreamer_main_module(official_repo_root: Path) -> ModuleType:
    _configure_vendor_paths(official_repo_root)
    return importlib.import_module("dreamerv3.main")


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo-root", type=Path, required=True)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, forwarded = _parse_args(argv)
    official_repo_root = args.official_repo_root.resolve()
    if not official_repo_root.exists():
        raise SystemExit(f"official Dreamer repo not found: {official_repo_root}")

    module = _import_dreamer_main_module(official_repo_root)

    previous_cwd = Path.cwd()
    try:
        os.chdir(official_repo_root)
        result = module.main(forwarded)
    finally:
        os.chdir(previous_cwd)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
