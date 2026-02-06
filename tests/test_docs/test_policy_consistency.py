"""Consistency checks for maturity policy docs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readme_and_extensibility_define_same_maturity_families():
    readme = _read("README.md")
    ext = _read("docs/EXTENSIBILITY.md")
    assert "Reference" in readme
    assert "Experimental" in readme
    assert "DreamerV3, TD-MPC2" in readme
    assert "JEPA, V-JEPA2, Token, Diffusion" in readme
    assert "**reference**: DreamerV3, TD-MPC2" in ext
    assert "**experimental**: JEPA, V-JEPA2, Token, Diffusion" in ext


def test_roadmap_mentions_public_maturity_boundary():
    roadmap = _read("ROADMAP.md")
    assert "reference" in roadmap.lower()
    assert "experimental" in roadmap.lower()
    assert "DreamerV3" in roadmap
    assert "TD-MPC2" in roadmap
    assert "JEPA" in roadmap
    assert "V-JEPA2" in roadmap
    assert "Token" in roadmap
    assert "Diffusion" in roadmap


def test_extension_docs_emphasize_contract_first_policy():
    readme = _read("README.md")
    ext = _read("docs/EXTENSIBILITY.md")
    assert "contract-first" in readme.lower()
    assert "contract-first" in ext.lower()


def test_ci_includes_strict_docs_gate():
    ci = _read(".github/workflows/ci.yml")
    assert "docs:" in ci
    assert "uv sync --extra docs" in ci
    assert "uv run mkdocs build --strict" in ci


def test_quality_gates_doc_matches_uv_ci_commands():
    gates = _read("docs/reference/quality-gates.md")
    assert "uv run ruff check src/ tests/ examples/" in gates
    assert "uv run ruff format --check src/ tests/ examples/" in gates
    assert "uv run mypy src/worldflux/" in gates
    assert "uv run pytest tests/" in gates
    assert "uv run mkdocs build --strict" in gates
