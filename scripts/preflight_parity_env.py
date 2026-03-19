#!/usr/bin/env python3
"""Pre-flight environment check for parity proof execution.

Validates that JAX, PyTorch, CUDA, and GPU resources are available and
compatible before launching a potentially expensive parity campaign.

Exit codes:
    0 - All checks passed
    1 - Critical check failed (parity run would fail)
    2 - Warning only (parity run may work but is not ideal)

Usage:
    python scripts/preflight_parity_env.py
    python scripts/preflight_parity_env.py --backend jax
    python scripts/preflight_parity_env.py --backend native_torch_reference
    python scripts/preflight_parity_env.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

CHECKS: list[dict[str, Any]] = []


def _record(name: str, status: str, detail: str) -> None:
    CHECKS.append({"name": name, "status": status, "detail": detail})
    icon = {"pass": "[OK]", "fail": "[FAIL]", "warn": "[WARN]", "skip": "[SKIP]"}
    print(f"  {icon.get(status, '[??]')} {name}: {detail}")


def check_python() -> None:
    ver = platform.python_version()
    impl = platform.python_implementation()
    major, minor = sys.version_info[:2]
    if major == 3 and 10 <= minor <= 12:
        _record("python", "pass", f"{impl} {ver}")
    else:
        _record("python", "warn", f"{impl} {ver} (expected 3.10-3.12)")


def check_pytorch() -> bool:
    try:
        import torch  # type: ignore[import-not-found]

        ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            _record("pytorch", "pass", f"{ver}, CUDA {cuda_version}, GPU: {device_name}")
        else:
            _record("pytorch", "warn", f"{ver}, CUDA not available (CPU only)")
        return cuda_available
    except ImportError:
        _record("pytorch", "fail", "Not installed")
        return False


def check_jax(require_gpu: bool = True) -> bool:
    try:
        import jax  # type: ignore[import-not-found]

        ver = jax.__version__
        try:
            devices = jax.devices("gpu")
            if devices:
                _record("jax_gpu", "pass", f"JAX {ver}, {len(devices)} GPU(s)")
                return True
            else:
                if require_gpu:
                    _record("jax_gpu", "fail", f"JAX {ver}, no GPU devices found")
                else:
                    _record("jax_gpu", "warn", f"JAX {ver}, no GPU (CPU fallback)")
                return False
        except RuntimeError:
            if require_gpu:
                _record("jax_gpu", "fail", f"JAX {ver}, GPU backend init failed")
            else:
                _record("jax_gpu", "warn", f"JAX {ver}, GPU unavailable (CPU fallback)")
            return False
    except ImportError:
        if require_gpu:
            _record("jax_gpu", "fail", "JAX not installed")
        else:
            _record("jax_gpu", "skip", "JAX not installed (not required for torch reference)")
        return False


def check_jax_cpu_fallback() -> bool:
    try:
        import os

        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import jax  # type: ignore[import-not-found]

        devices = jax.devices("cpu")
        if devices:
            _record("jax_cpu_fallback", "pass", f"JAX CPU with {len(devices)} device(s)")
            return True
        _record("jax_cpu_fallback", "warn", "JAX CPU fallback has no devices")
        return False
    except ImportError:
        _record("jax_cpu_fallback", "skip", "JAX not installed")
        return False
    except Exception as exc:
        _record("jax_cpu_fallback", "fail", f"JAX CPU fallback failed: {exc}")
        return False


def check_worldflux() -> bool:
    try:
        import importlib.util

        spec = importlib.util.find_spec("worldflux")
        if spec is None:
            _record("worldflux", "fail", "Package not found")
            return False
        _record("worldflux", "pass", "Importable")
        return True
    except ImportError as exc:
        _record("worldflux", "fail", f"Import failed: {exc}")
        return False


def check_gpu_memory() -> None:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            _record("gpu_memory", "pass" if total >= 16 else "warn", f"{total:.1f} GB")
        else:
            _record("gpu_memory", "skip", "No CUDA GPU")
    except ImportError:
        _record("gpu_memory", "skip", "PyTorch not installed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("jax", "native_torch_reference", "auto"),
        default="auto",
        help="Which backend to validate. Default: auto.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    print("WorldFlux Parity Environment Pre-flight Check")
    print("=" * 50)

    check_python()
    check_pytorch()
    check_worldflux()
    check_gpu_memory()

    jax_required = args.backend in ("jax", "auto")
    if args.backend == "native_torch_reference":
        _record("backend_mode", "pass", "native_torch_reference (self-parity)")
        check_jax(require_gpu=False)
    else:
        jax_gpu = check_jax(require_gpu=jax_required)
        if not jax_gpu and args.backend == "auto":
            check_jax_cpu_fallback()

    print("=" * 50)

    failures = [c for c in CHECKS if c["status"] == "fail"]
    warnings = [c for c in CHECKS if c["status"] == "warn"]

    if args.json:
        output = {
            "checks": CHECKS,
            "failures": len(failures),
            "warnings": len(warnings),
            "verdict": "fail" if failures else ("warn" if warnings else "pass"),
        }
        print(json.dumps(output, indent=2))

    if failures:
        print(f"\nFAILED: {len(failures)} critical check(s) failed.")
        for f in failures:
            print(f"  - {f['name']}: {f['detail']}")
        return 1

    if warnings:
        print(f"\nPASSED with {len(warnings)} warning(s).")
        return 2

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
