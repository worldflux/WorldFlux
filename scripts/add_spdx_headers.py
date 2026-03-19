# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Add SPDX license headers to all Python files in the repository.

Usage:
    python scripts/add_spdx_headers.py [--check] [--verbose]

Options:
    --check     Check mode: report files missing headers without modifying them.
                Exits with code 1 if any files are missing headers.
    --verbose   Print each file as it is processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPDX_HEADER_LINES = [
    "# SPDX-License-Identifier: Apache-2.0\n",
    "# Copyright 2026 WorldFlux Contributors\n",
]

SPDX_MARKER = "SPDX-License-Identifier"

SCAN_DIRS = [
    "src/worldflux",
    "tests",
]


def _repo_root() -> Path:
    """Walk up from this script to find the repository root (contains pyproject.toml)."""
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    msg = "Cannot locate repository root from script location"
    raise RuntimeError(msg)


def _find_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        target = root / scan_dir
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


def _has_spdx_header(content: str) -> bool:
    for line in content.splitlines()[:5]:
        if SPDX_MARKER in line:
            return True
    return False


def _add_header(filepath: Path, *, verbose: bool = False) -> bool:
    """Add SPDX header to a file. Returns True if the file was modified."""
    content = filepath.read_text(encoding="utf-8")
    if _has_spdx_header(content):
        return False

    header = "".join(SPDX_HEADER_LINES)
    new_content = header + content
    filepath.write_text(new_content, encoding="utf-8")
    if verbose:
        print(f"  Added header: {filepath}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Add SPDX headers to Python files.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: report missing headers without modifying files.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each file processed.")
    args = parser.parse_args()

    root = _repo_root()
    py_files = _find_python_files(root)

    if args.check:
        missing: list[Path] = []
        for filepath in py_files:
            content = filepath.read_text(encoding="utf-8")
            if not _has_spdx_header(content):
                missing.append(filepath)
                if args.verbose:
                    print(f"  Missing header: {filepath}")
        if missing:
            print(f"SPDX header missing from {len(missing)} file(s):")
            for f in missing:
                print(f"  {f.relative_to(root)}")
            return 1
        print(f"All {len(py_files)} Python files have SPDX headers.")
        return 0

    modified = 0
    for filepath in py_files:
        if _add_header(filepath, verbose=args.verbose):
            modified += 1

    print(f"Processed {len(py_files)} files, added headers to {modified}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
