# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Execution contracts and executor implementations for backend-native flows."""

from .backend_policy import (
    DREAMER_MIN_LOCKED_SEEDS,
    DREAMER_MIN_PROOF_SEEDS,
    TDMPC2_COMPARE_ENABLED,
)
from .contracts import (
    BackendExecutionRequest,
    BackendExecutionResult,
    ExecutionMode,
    ExecutionStatus,
    ReasonCode,
)
from .executor import BackendExecutor, ParityBackedExecutor
from .manifest_routing import ManifestResolution, resolve_execution_manifest
from .proof_defaults import resolve_proof_backend_defaults
from .summary_normalization import (
    normalize_distributed_proof_summary,
    normalize_dreamer_batch_status,
    normalize_dreamer_official_batch_summary,
    normalize_parity_run_row,
)

__all__ = [
    "BackendExecutionRequest",
    "BackendExecutionResult",
    "BackendExecutor",
    "DREAMER_MIN_LOCKED_SEEDS",
    "DREAMER_MIN_PROOF_SEEDS",
    "ExecutionMode",
    "ExecutionStatus",
    "ManifestResolution",
    "normalize_distributed_proof_summary",
    "normalize_dreamer_batch_status",
    "normalize_dreamer_official_batch_summary",
    "normalize_parity_run_row",
    "ParityBackedExecutor",
    "ReasonCode",
    "TDMPC2_COMPARE_ENABLED",
    "resolve_proof_backend_defaults",
    "resolve_execution_manifest",
]
