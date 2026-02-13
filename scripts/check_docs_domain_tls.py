#!/usr/bin/env python3
"""Validate docs-domain TLS SAN entries and HTTPS reachability."""

from __future__ import annotations

import argparse
import socket
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainTlsCheckResult:
    ok: bool
    messages: tuple[str, ...]


def _extract_dns_sans(cert: dict[str, object]) -> set[str]:
    subject_alt_name = cert.get("subjectAltName", ())
    if not isinstance(subject_alt_name, tuple):
        return set()

    dns_names: set[str] = set()
    for entry in subject_alt_name:
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue
        key, value = entry
        if key == "DNS" and isinstance(value, str):
            dns_names.add(value.lower())
    return dns_names


def _fetch_certificate_sans(host: str, port: int, timeout: float) -> set[str]:
    context = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=timeout) as connection:
        with context.wrap_socket(connection, server_hostname=host) as tls_socket:
            cert = tls_socket.getpeercert()
    return _extract_dns_sans(cert)


def _request_head_status(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def validate_domain_tls(
    host: str,
    url: str,
    expected_sans: tuple[str, ...],
    port: int = 443,
    timeout: float = 10.0,
) -> DomainTlsCheckResult:
    messages: list[str] = []
    expected = tuple(name.lower() for name in expected_sans)

    try:
        actual_sans = _fetch_certificate_sans(host=host, port=port, timeout=timeout)
    except OSError as exc:
        return DomainTlsCheckResult(
            ok=False,
            messages=(f"Failed TLS handshake for {host}:{port}: {exc}",),
        )

    missing_sans = [name for name in expected if name not in actual_sans]
    if missing_sans:
        return DomainTlsCheckResult(
            ok=False,
            messages=(
                f"Certificate SAN mismatch for {host}:{port}.",
                f"Missing expected SANs: {', '.join(missing_sans)}",
                f"Actual SANs: {', '.join(sorted(actual_sans))}",
            ),
        )

    messages.append(f"TLS SAN check passed for {host}:{port}.")

    try:
        status_code = _request_head_status(url=url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        status_code = exc.code
    except urllib.error.URLError as exc:
        return DomainTlsCheckResult(
            ok=False,
            messages=(*messages, f"HTTPS HEAD request failed for {url}: {exc.reason}"),
        )

    if not 200 <= status_code < 400:
        return DomainTlsCheckResult(
            ok=False,
            messages=(*messages, f"HTTPS HEAD returned status {status_code} for {url}."),
        )

    messages.append(f"HTTPS HEAD check passed for {url} with status {status_code}.")
    return DomainTlsCheckResult(ok=True, messages=tuple(messages))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="worldflux.ai", help="Domain host to inspect.")
    parser.add_argument(
        "--url", default="https://worldflux.ai/", help="HTTPS URL to probe with HEAD."
    )
    parser.add_argument(
        "--expected-san",
        action="append",
        default=[],
        help="Expected DNS SAN entry (repeatable). Defaults to worldflux.ai if omitted.",
    )
    parser.add_argument("--port", type=int, default=443, help="TLS port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout in seconds.")
    args = parser.parse_args()

    expected_sans = tuple(args.expected_san) or ("worldflux.ai",)
    result = validate_domain_tls(
        host=args.host,
        url=args.url,
        expected_sans=expected_sans,
        port=args.port,
        timeout=args.timeout,
    )
    for message in result.messages:
        print(message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
