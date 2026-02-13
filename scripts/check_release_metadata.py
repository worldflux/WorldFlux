#!/usr/bin/env python3
"""Validate release tag/version/changelog consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def _normalize_tag(tag: str) -> str:
    normalized = tag.strip()
    if normalized.startswith("refs/tags/"):
        normalized = normalized[len("refs/tags/") :]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized


def _extract_project_version(pyproject: Path) -> str:
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml missing [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml missing project.version")
    return version.strip()


def _extract_changelog_versions(changelog: Path) -> set[str]:
    pattern = re.compile(r"^##\s+\[(?P<version>[^\]]+)\](?:\s+-\s+.*)?\s*$")
    found: set[str] = set()
    for line in changelog.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        found.add(match.group("version").strip())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag (e.g. v0.1.0)")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to CHANGELOG.md",
    )
    args = parser.parse_args()

    release_version = _normalize_tag(args.tag)
    if not release_version:
        print("[release-metadata] normalized release tag is empty")
        return 1

    try:
        project_version = _extract_project_version(args.pyproject)
    except Exception as exc:
        print(f"[release-metadata] failed to parse pyproject: {exc}")
        return 1

    changelog_versions = _extract_changelog_versions(args.changelog)

    failures: list[str] = []
    if release_version != project_version:
        failures.append(
            "release tag/version mismatch: "
            f"tag={release_version!r} project.version={project_version!r}"
        )

    if release_version not in changelog_versions:
        failures.append(f"CHANGELOG.md does not contain heading '## [{release_version}]'")

    if failures:
        print("[release-metadata] validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[release-metadata] validation passed")
    print(f"  - tag version: {release_version}")
    print(f"  - pyproject version: {project_version}")
    print(f"  - changelog section: [{release_version}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
