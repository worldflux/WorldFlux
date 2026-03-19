#!/usr/bin/env python3
"""Automated migration helper: v0.2 API to v3 API.

Performs safe textual replacements on Python source files. Complex
transformations (e.g. action_type -> action_spec dict) are flagged
for manual review.

Usage:
    python scripts/migrate_v02_to_v3.py [--dry-run] <file_or_dir> [...]

Exit codes:
    0 - all files migrated (or no changes needed)
    1 - files contain patterns that need manual review
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# Replacement rules: (pattern, replacement, description)
# -----------------------------------------------------------------------

_REPLACEMENTS: list[tuple[str, str, str]] = [
    # api_version="v0.2" removal
    (
        r',\s*api_version\s*=\s*["\']v0\.2["\']',
        "",
        "Remove api_version='v0.2' (v3 is now default)",
    ),
    (
        r'api_version\s*=\s*["\']v0\.2["\']',
        "",
        "Remove api_version='v0.2' (v3 is now default)",
    ),
    # RolloutEngine -> RolloutExecutor
    (
        r"from\s+worldflux\s+import\s+(.*\b)RolloutEngine(\b.*)",
        r"from worldflux import \1RolloutExecutor\2",
        "RolloutEngine renamed to RolloutExecutor",
    ),
    (
        r"\bRolloutEngine\b",
        "RolloutExecutor",
        "RolloutEngine -> RolloutExecutor",
    ),
    # PluginManifest moved
    (
        r"from\s+worldflux\s+import\s+PluginManifest",
        "from worldflux.core.registry import PluginManifest",
        "PluginManifest moved to worldflux.core.registry",
    ),
    # first_action moved
    (
        r"from\s+worldflux\s+import\s+first_action",
        "from worldflux.core.payloads import first_action",
        "first_action moved to worldflux.core.payloads",
    ),
    # is_namespaced_extra_key moved
    (
        r"from\s+worldflux\s+import\s+is_namespaced_extra_key",
        "from worldflux.core.payloads import is_namespaced_extra_key",
        "is_namespaced_extra_key moved to worldflux.core.payloads",
    ),
    # Skeleton configs - direct imports
    (
        r"from\s+worldflux\s+import\s+DiTSkeletonConfig",
        "from worldflux.core.config import DiTSkeletonConfig",
        "DiTSkeletonConfig moved to worldflux.core.config",
    ),
    (
        r"from\s+worldflux\s+import\s+SSMSkeletonConfig",
        "from worldflux.core.config import SSMSkeletonConfig",
        "SSMSkeletonConfig moved to worldflux.core.config",
    ),
    (
        r"from\s+worldflux\s+import\s+Renderer3DSkeletonConfig",
        "from worldflux.core.config import Renderer3DSkeletonConfig",
        "Renderer3DSkeletonConfig moved to worldflux.core.config",
    ),
    (
        r"from\s+worldflux\s+import\s+PhysicsSkeletonConfig",
        "from worldflux.core.config import PhysicsSkeletonConfig",
        "PhysicsSkeletonConfig moved to worldflux.core.config",
    ),
    (
        r"from\s+worldflux\s+import\s+GANSkeletonConfig",
        "from worldflux.core.config import GANSkeletonConfig",
        "GANSkeletonConfig moved to worldflux.core.config",
    ),
    # Skeleton world models
    (
        r"from\s+worldflux\s+import\s+DiTSkeletonWorldModel",
        "from worldflux.models.dit import DiTSkeletonWorldModel",
        "DiTSkeletonWorldModel moved to worldflux.models.dit",
    ),
    (
        r"from\s+worldflux\s+import\s+SSMSkeletonWorldModel",
        "from worldflux.models.ssm import SSMSkeletonWorldModel",
        "SSMSkeletonWorldModel moved to worldflux.models.ssm",
    ),
    (
        r"from\s+worldflux\s+import\s+Renderer3DSkeletonWorldModel",
        "from worldflux.models.renderer3d import Renderer3DSkeletonWorldModel",
        "Renderer3DSkeletonWorldModel moved to worldflux.models.renderer3d",
    ),
    (
        r"from\s+worldflux\s+import\s+PhysicsSkeletonWorldModel",
        "from worldflux.models.physics import PhysicsSkeletonWorldModel",
        "PhysicsSkeletonWorldModel moved to worldflux.models.physics",
    ),
    (
        r"from\s+worldflux\s+import\s+GANSkeletonWorldModel",
        "from worldflux.models.gan import GANSkeletonWorldModel",
        "GANSkeletonWorldModel moved to worldflux.models.gan",
    ),
]

# Patterns that require manual review (not auto-fixable).
_MANUAL_REVIEW: list[tuple[str, str]] = [
    (
        r'action_type\s*=\s*["\']hybrid["\']',
        "action_type='hybrid' is removed in v3. Refactor to use action_spec dict.",
    ),
    (
        r"obs_shape\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*[1-4]\s*\)",
        "Possible HWC obs_shape detected. v3 expects CHW (PyTorch convention).",
    ),
]


def migrate_file(path: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Migrate a single file. Returns (num_changes, num_warnings)."""
    text = path.read_text(encoding="utf-8")
    changes = 0
    warnings_count = 0

    for pattern, replacement, description in _REPLACEMENTS:
        new_text, n = re.subn(pattern, replacement, text)
        if n > 0:
            if dry_run:
                print(f"  [CHANGE] {path}: {description} ({n} occurrence(s))")
            changes += n
            text = new_text

    for pattern, message in _MANUAL_REVIEW:
        matches = list(re.finditer(pattern, text))
        for m in matches:
            lineno = text[: m.start()].count("\n") + 1
            print(f"  [MANUAL] {path}:{lineno}: {message}")
            warnings_count += 1

    if changes > 0 and not dry_run:
        path.write_text(text, encoding="utf-8")
        print(f"  Migrated {path} ({changes} change(s))")

    return changes, warnings_count


def migrate_path(target: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Migrate a file or all .py files in a directory."""
    total_changes = 0
    total_warnings = 0

    if target.is_file():
        c, w = migrate_file(target, dry_run=dry_run)
        total_changes += c
        total_warnings += w
    elif target.is_dir():
        for py_file in sorted(target.rglob("*.py")):
            c, w = migrate_file(py_file, dry_run=dry_run)
            total_changes += c
            total_warnings += w
    else:
        print(f"  [SKIP] {target}: not a file or directory")

    return total_changes, total_warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate WorldFlux code from v0.2 API to v3 API.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        help="Files or directories to migrate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files.",
    )
    args = parser.parse_args()

    total_changes = 0
    total_warnings = 0

    for target in args.targets:
        c, w = migrate_path(target, dry_run=args.dry_run)
        total_changes += c
        total_warnings += w

    print()
    if args.dry_run:
        print(f"Dry run complete: {total_changes} change(s) would be applied.")
    else:
        print(f"Migration complete: {total_changes} change(s) applied.")

    if total_warnings:
        print(f"{total_warnings} pattern(s) flagged for manual review.")
        sys.exit(1)


if __name__ == "__main__":
    main()
