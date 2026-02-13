"""Tests for docs domain TLS health check script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_docs_domain_tls.py"
    spec = importlib.util.spec_from_file_location("check_docs_domain_tls", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_docs_domain_tls module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_dns_sans_filters_only_dns_entries() -> None:
    mod = _load_module()
    cert = {
        "subjectAltName": (
            ("DNS", "worldflux.ai"),
            ("DNS", "www.worldflux.ai"),
            ("IP Address", "185.199.108.153"),
        )
    }

    sans = mod._extract_dns_sans(cert)
    assert sans == {"worldflux.ai", "www.worldflux.ai"}


def test_validate_domain_tls_success_path(monkeypatch) -> None:
    mod = _load_module()

    monkeypatch.setattr(
        mod,
        "_fetch_certificate_sans",
        lambda host, port, timeout: {"worldflux.ai", "www.worldflux.ai"},
    )
    monkeypatch.setattr(mod, "_request_head_status", lambda url, timeout: 200)

    result = mod.validate_domain_tls(
        host="worldflux.ai",
        url="https://worldflux.ai/",
        expected_sans=("worldflux.ai",),
    )

    assert result.ok is True
    assert any("TLS SAN check passed" in message for message in result.messages)
    assert any("HTTPS HEAD check passed" in message for message in result.messages)


def test_validate_domain_tls_fails_when_expected_san_missing(monkeypatch) -> None:
    mod = _load_module()

    monkeypatch.setattr(
        mod,
        "_fetch_certificate_sans",
        lambda host, port, timeout: {"github.io"},
    )

    result = mod.validate_domain_tls(
        host="worldflux.ai",
        url="https://worldflux.ai/",
        expected_sans=("worldflux.ai",),
    )

    assert result.ok is False
    assert any("Missing expected SANs" in message for message in result.messages)
