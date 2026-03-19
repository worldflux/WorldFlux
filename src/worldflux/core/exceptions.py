"""Custom exceptions for WorldFlux.

Each exception class carries an ``error_code`` class attribute following the
scheme ``WF-<category><number>`` so that users and tooling can reference
specific failure modes unambiguously.

Category prefixes:
    C - Configuration
    S - Shape / tensor
    V - Validation / contract
    A - Capability
    M - Model registry
    K - Checkpoint
    T - Training
    B - Buffer
"""

from __future__ import annotations


class WorldFluxError(Exception):
    """Base exception for all WorldFlux errors.

    Attributes:
        error_code: Machine-readable error identifier (e.g. ``WF-C001``).
    """

    error_code: str = "WF-E000"

    pass


class ConfigurationError(WorldFluxError):
    """Raised when model configuration is invalid.

    Common causes:
        - Unknown or misspelled config parameter name
        - Value out of valid range (e.g. negative learning_rate)
        - Incompatible parameter combination

    Error codes:
        WF-C001 - model_type not found
        WF-C002 - invalid obs_shape
        WF-C003 - unknown config parameter (kwargs validation)
    """

    error_code: str = "WF-C001"

    def __init__(self, message: str, config_name: str | None = None):
        self.config_name = config_name
        if config_name:
            message = f"[{config_name}] {message}"
        super().__init__(message)


class ShapeMismatchError(WorldFluxError):
    """Raised when tensor shapes don't match expected dimensions.

    Common causes:
        - Batch dimension mismatch between observations and actions
        - Wrong channel ordering (HWC vs CHW)
        - Encoder/decoder dimension incompatibility

    Error codes:
        WF-S001 - batch dimension mismatch
        WF-S002 - spatial dimension mismatch
        WF-S003 - channel dimension mismatch
    """

    error_code: str = "WF-S001"

    def __init__(
        self,
        message: str,
        expected: tuple[int, ...] | None = None,
        got: tuple[int, ...] | None = None,
    ):
        self.expected = expected
        self.got = got
        if expected is not None and got is not None:
            message = f"{message} (expected {expected}, got {got})"
        super().__init__(message)


class StateError(WorldFluxError):
    """Raised when State is in an invalid state.

    Common causes:
        - Accessing state before initialization
        - State shape mismatch after reset
        - Attempting to modify frozen state

    Error codes:
        WF-S101 - invalid state access
    """

    error_code: str = "WF-S101"

    pass


class ValidationError(WorldFluxError):
    """Raised when runtime validation fails.

    Common causes:
        - Tensor value out of expected range
        - NaN or Inf detected in computation
        - Assertion failure in model forward pass

    Error codes:
        WF-V001 - generic validation failure
    """

    error_code: str = "WF-V001"

    pass


class ContractValidationError(ValidationError):
    """Raised when model I/O contract validation fails.

    Common causes:
        - Model output missing required fields
        - Output shape does not match contract spec
        - Incompatible modality types

    Error codes:
        WF-V002 - I/O contract violation
    """

    error_code: str = "WF-V002"

    pass


class CapabilityError(WorldFluxError):
    """Raised when a required model capability is missing.

    Common causes:
        - Calling predict() on a model without prediction capability
        - Requesting a component slot the model does not support
        - Using a feature gated behind a config flag

    Error codes:
        WF-A001 - missing capability
    """

    error_code: str = "WF-A001"

    pass


class ModelNotFoundError(WorldFluxError):
    """Raised when a requested model is not found in the registry.

    Common causes:
        - Misspelled model identifier
        - Model plugin not installed
        - Using an alias that was removed

    Error codes:
        WF-M001 - model not found
    """

    error_code: str = "WF-M001"

    def __init__(self, model_name: str, available: list[str] | None = None):
        self.model_name = model_name
        self.available = available
        message = f"Model '{model_name}' not found"
        if available:
            message += f". Available models: {available}"
        super().__init__(message)


class CheckpointError(WorldFluxError):
    """Raised when checkpoint loading/saving fails.

    Common causes:
        - Checkpoint file not found or corrupted
        - Architecture mismatch between saved and current model
        - Missing keys in state dict

    Error codes:
        WF-K001 - checkpoint load/save failure
    """

    error_code: str = "WF-K001"

    pass


class TrainingError(WorldFluxError):
    """Raised when training encounters an error.

    Common causes:
        - NaN detected in loss computation
        - Gradient explosion despite clipping
        - Out-of-memory during forward/backward pass

    Error codes:
        WF-T001 - NaN in loss
        WF-T002 - gradient anomaly
    """

    error_code: str = "WF-T001"

    pass


class BufferError(WorldFluxError):
    """Raised when replay buffer operations fail.

    Common causes:
        - Sampling from empty buffer
        - Shape mismatch when inserting transitions
        - Concurrent write access (ReplayBuffer is NOT thread-safe)

    Error codes:
        WF-B001 - buffer operation failure
    """

    error_code: str = "WF-B001"

    pass
