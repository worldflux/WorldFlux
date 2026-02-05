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
    assert "JEPA, Token, Diffusion" in readme
    assert "**reference**: DreamerV3, TD-MPC2" in ext
    assert "**experimental**: JEPA, Token, Diffusion" in ext


def test_roadmap_mentions_public_maturity_boundary():
    roadmap = _read("ROADMAP.md")
    assert "reference" in roadmap.lower()
    assert "experimental" in roadmap.lower()
    assert "DreamerV3" in roadmap
    assert "TD-MPC2" in roadmap
    assert "JEPA" in roadmap
    assert "Token" in roadmap
    assert "Diffusion" in roadmap


def test_extension_docs_emphasize_contract_first_policy():
    readme = _read("README.md")
    ext = _read("docs/EXTENSIBILITY.md")
    assert "contract-first" in readme.lower()
    assert "contract-first" in ext.lower()
