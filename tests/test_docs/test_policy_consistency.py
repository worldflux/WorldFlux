"""Consistency checks for maturity policy docs."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

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
    assert "uvx ruff check src/ tests/ examples/" in gates
    assert "uvx ruff format --check src/ tests/ examples/" in gates
    assert "uv run mypy src/worldflux/" in gates
    assert "uv run pytest tests/" in gates
    assert "uv run mkdocs build --strict" in gates


def test_cli_installation_docs_prefer_direct_worldflux_command():
    readme = _read("README.md")
    installation = _read("docs/getting-started/installation.md")
    index = _read("docs/index.md")

    assert "uv tool install worldflux" in readme
    assert "worldflux init my-world-model" in readme

    assert "uv tool install worldflux" in installation
    assert "worldflux init my-world-model" in installation

    assert "uv tool install worldflux" in index
    assert "worldflux init my-world-model" in index

    assert "uv run worldflux init" not in readme
    assert "uv run worldflux init" not in installation


def test_pyproject_uses_default_cli_dependencies_and_optional_inquirer():
    pyproject = tomllib.loads(_read("pyproject.toml"))
    dependencies = pyproject["project"]["dependencies"]
    cli_extra = pyproject["project"]["optional-dependencies"]["cli"]

    assert any(dep.startswith("typer") for dep in dependencies)
    assert any(dep.startswith("rich") for dep in dependencies)
    assert cli_extra == ["inquirerpy>=0.3.4"]
