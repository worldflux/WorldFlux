"""Community metadata consistency checks."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_issue_templates_do_not_reference_worldloom() -> None:
    contents = [
        _read(".github/ISSUE_TEMPLATE/bug_report.md"),
        _read(".github/ISSUE_TEMPLATE/feature_request.md"),
        _read(".github/ISSUE_TEMPLATE/config.yml"),
    ]
    joined = "\n".join(contents)
    assert "WorldLoom" not in joined
    assert "worldloom" not in joined


def test_issue_template_contact_links_point_to_canonical_targets() -> None:
    config = _read(".github/ISSUE_TEMPLATE/config.yml")
    assert "https://github.com/worldflux/WorldFlux/discussions" in config
    assert "https://worldflux.ai/" in config
