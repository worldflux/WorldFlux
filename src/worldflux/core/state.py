"""Unified latent state representation."""

from __future__ import annotations

import json
import math
import struct
import uuid
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any

import torch
from torch import Tensor

from .exceptions import ShapeMismatchError, StateError

_STATE_MAGIC = b"WFST"
_STATE_VERSION_TO_INT = {"v1": 1}
_STATE_INT_TO_VERSION = {1: "v1"}
_STATE_HEADER_STRUCT = struct.Struct(">4sBI")
_STATE_SHARED_MAGIC = "WFST-SHM"

_DTYPE_TO_NAME = {
    torch.bool: "bool",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _version_name_to_int(version: str) -> int:
    if version not in _STATE_VERSION_TO_INT:
        known = ", ".join(sorted(_STATE_VERSION_TO_INT.keys()))
        raise StateError(f"Unsupported State serialization version {version!r}. Known: {known}")
    return _STATE_VERSION_TO_INT[version]


def _version_int_to_name(version: int) -> str:
    if version not in _STATE_INT_TO_VERSION:
        known = ", ".join(str(v) for v in sorted(_STATE_INT_TO_VERSION.keys()))
        raise StateError(
            f"Unsupported State serialization version id {version!r}. Known ids: {known}"
        )
    return _STATE_INT_TO_VERSION[version]


def _dtype_to_name(dtype: torch.dtype) -> str:
    name = _DTYPE_TO_NAME.get(dtype)
    if name is None:
        raise StateError(f"Unsupported tensor dtype for State serialization: {dtype}")
    return name


def _name_to_dtype(name: str) -> torch.dtype:
    dtype = _NAME_TO_DTYPE.get(name)
    if dtype is None:
        known = ", ".join(sorted(_NAME_TO_DTYPE.keys()))
        raise StateError(f"Unsupported tensor dtype name {name!r}. Known: {known}")
    return dtype


def _tensor_numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(math.prod(shape))


def _shared_memory_name(namespace: str) -> str:
    # macOS posix shared-memory names have a strict small limit; keep names compact.
    safe_ns = "".join(ch for ch in namespace if ch.isalnum()).lower()
    safe_ns = safe_ns[:8] if safe_ns else "wfstate"
    token = uuid.uuid4().hex[:12]
    return f"{safe_ns}-{token}"


def _json_friendly_meta(meta: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(meta)
    except TypeError as exc:
        raise StateError(
            "State.meta must be JSON-serializable for binary serialization. "
            "Use simple scalar/list/dict metadata values."
        ) from exc
    return dict(meta)


@dataclass
class State:
    """Generic state container (tensor dictionary + metadata)."""

    tensors: dict[str, Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Tensor | None = None) -> Tensor | None:
        return self.tensors.get(key, default)

    @property
    def batch_size(self) -> int:
        for tensor in self.tensors.values():
            return tensor.shape[0]
        raise ValueError("State has no tensors to infer batch size")

    @property
    def device(self) -> torch.device:
        for tensor in self.tensors.values():
            return tensor.device
        raise ValueError("State has no tensors to infer device")

    def to(self, device: torch.device | str) -> State:
        device_obj = torch.device(device) if isinstance(device, str) else device
        return State(
            tensors={k: v.to(device_obj) for k, v in self.tensors.items()},
            meta=self.meta,
        )

    def detach(self) -> State:
        return State(
            tensors={k: v.detach() for k, v in self.tensors.items()},
            meta=self.meta,
        )

    def clone(self) -> State:
        return State(
            tensors={k: v.clone() for k, v in self.tensors.items()},
            meta=dict(self.meta),
        )

    def validate(self) -> None:
        """Validate state tensor shapes and batch consistency."""
        if not self.tensors:
            raise StateError("State has no tensors")
        batch_size = None
        for name, tensor in self.tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[0]
                continue
            if tensor.shape[0] != batch_size:
                raise ShapeMismatchError(
                    f"State tensor '{name}' batch size mismatch",
                    expected=(batch_size,),
                    got=(tensor.shape[0],),
                )

    def serialize(self, version: str = "v1", format: str = "binary") -> bytes:
        """Serialize state with a versioned binary envelope.

        Binary envelope layout:
            magic (4 bytes), version id (1 byte), metadata length (4 bytes),
            metadata JSON bytes, then raw tensor bytes.
        """
        if format != "binary":
            raise StateError(f"Unsupported State serialization format {format!r}. Use 'binary'.")
        version_id = _version_name_to_int(version)

        tensor_records: list[dict[str, Any]] = []
        raw_chunks: list[bytes] = []
        for name, tensor in self.tensors.items():
            dtype_name = _dtype_to_name(tensor.dtype)
            cpu_tensor = tensor.detach()
            source_device = str(cpu_tensor.device)
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.to("cpu")
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()

            raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
            tensor_records.append(
                {
                    "name": str(name),
                    "dtype": dtype_name,
                    "shape": list(cpu_tensor.shape),
                    "nbytes": len(raw),
                    "requires_grad": bool(tensor.requires_grad),
                    "source_device": source_device,
                }
            )
            raw_chunks.append(raw)

        metadata = {
            "format": "worldflux.state",
            "version": version,
            "meta": _json_friendly_meta(self.meta),
            "tensors": tensor_records,
        }
        meta_json = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
        header = _STATE_HEADER_STRUCT.pack(_STATE_MAGIC, version_id, len(meta_json))
        return header + meta_json + b"".join(raw_chunks)

    @classmethod
    def deserialize(cls, payload: bytes) -> State:
        """Deserialize state from `State.serialize(...)` payload."""
        if len(payload) < _STATE_HEADER_STRUCT.size:
            raise StateError("State payload is too short for binary header")

        magic, version_id, meta_len = _STATE_HEADER_STRUCT.unpack_from(payload, 0)
        if magic != _STATE_MAGIC:
            raise StateError("Invalid State payload magic header")

        version_name = _version_int_to_name(int(version_id))
        meta_start = _STATE_HEADER_STRUCT.size
        meta_end = meta_start + int(meta_len)
        if meta_end > len(payload):
            raise StateError("State payload is truncated (metadata length out of range)")

        try:
            metadata = json.loads(payload[meta_start:meta_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateError(f"Failed to decode State metadata JSON: {exc}") from exc
        if not isinstance(metadata, dict):
            raise StateError("State metadata must decode into a JSON object")

        declared_version = str(metadata.get("version", ""))
        if declared_version and declared_version != version_name:
            raise StateError(
                "State payload version mismatch between header and metadata: "
                f"{version_name!r} vs {declared_version!r}"
            )

        tensor_specs = metadata.get("tensors")
        if not isinstance(tensor_specs, list):
            raise StateError("State payload metadata is missing tensor specs list")

        view = memoryview(payload)
        offset = meta_end
        tensors: dict[str, Tensor] = {}

        for raw_spec in tensor_specs:
            if not isinstance(raw_spec, dict):
                raise StateError("Invalid tensor spec in State metadata")
            name = str(raw_spec.get("name", ""))
            if not name:
                raise StateError("Tensor spec missing non-empty 'name'")

            dtype_name = str(raw_spec.get("dtype", ""))
            dtype = _name_to_dtype(dtype_name)

            shape_raw = raw_spec.get("shape")
            if not isinstance(shape_raw, list):
                raise StateError(f"Tensor spec for {name!r} must include list 'shape'")
            shape = tuple(int(dim) for dim in shape_raw)
            if any(dim < 0 for dim in shape):
                raise StateError(f"Tensor spec for {name!r} has negative shape dimensions")

            nbytes = int(raw_spec.get("nbytes", -1))
            if nbytes < 0:
                raise StateError(f"Tensor spec for {name!r} has invalid 'nbytes'={nbytes}")

            numel = _tensor_numel(shape)
            expected_nbytes = numel * torch.empty((), dtype=dtype).element_size()
            if nbytes != expected_nbytes:
                raise StateError(
                    f"Tensor spec for {name!r} has inconsistent byte length: "
                    f"expected {expected_nbytes}, got {nbytes}"
                )
            if offset + nbytes > len(payload):
                raise StateError(f"State payload is truncated while reading tensor {name!r}")

            if nbytes == 0:
                tensor = torch.empty(shape, dtype=dtype)
            else:
                chunk = view[offset : offset + nbytes]
                bytes_view = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
                tensor = bytes_view.view(dtype).reshape(shape).clone()

            if bool(raw_spec.get("requires_grad", False)):
                tensor = tensor.requires_grad_(True)
            tensors[name] = tensor
            offset += nbytes

        if offset != len(payload):
            raise StateError(
                "State payload has trailing bytes after tensor decode. "
                "This usually indicates corruption or a version mismatch."
            )

        meta = metadata.get("meta", {})
        if not isinstance(meta, dict):
            raise StateError("State metadata field 'meta' must be an object")
        return cls(tensors=tensors, meta=dict(meta))

    def to_shared_memory(
        self,
        *,
        namespace: str = "worldflux-state",
        allow_copy_from_cuda: bool = False,
    ) -> dict[str, Any]:
        """Create shared-memory descriptor for zero-copy CPU state exchange.

        Notes:
            - CPU contiguous tensors retain zero-copy semantics when re-attached.
            - CUDA tensors require `allow_copy_from_cuda=True` and are copied to CPU.
        """
        descriptor_tensors: list[dict[str, Any]] = []

        for name, tensor in self.tensors.items():
            cpu_tensor = tensor.detach()
            source_device = str(cpu_tensor.device)
            if cpu_tensor.device.type != "cpu":
                if not allow_copy_from_cuda:
                    raise StateError(
                        "State.to_shared_memory received non-CPU tensor "
                        f"{name!r} on {source_device}. "
                        "Pass allow_copy_from_cuda=True to copy explicitly."
                    )
                cpu_tensor = cpu_tensor.to("cpu")
            if not cpu_tensor.is_contiguous():
                cpu_tensor = cpu_tensor.contiguous()

            numel = int(cpu_tensor.numel())
            nbytes = numel * cpu_tensor.element_size()
            segment_size = max(1, nbytes)
            shm_name = _shared_memory_name(namespace)

            shm = shared_memory.SharedMemory(create=True, size=segment_size, name=shm_name)
            try:
                if numel > 0:
                    target = torch.frombuffer(shm.buf, dtype=cpu_tensor.dtype, count=numel)
                    target.view(cpu_tensor.shape).copy_(cpu_tensor)
            finally:
                shm.close()

            descriptor_tensors.append(
                {
                    "name": str(name),
                    "dtype": _dtype_to_name(cpu_tensor.dtype),
                    "shape": list(cpu_tensor.shape),
                    "numel": numel,
                    "nbytes": nbytes,
                    "requires_grad": bool(tensor.requires_grad),
                    "source_device": source_device,
                    "shm_name": shm_name,
                }
            )

        return {
            "magic": _STATE_SHARED_MAGIC,
            "version": "v1",
            "meta": _json_friendly_meta(self.meta),
            "tensors": descriptor_tensors,
        }

    @classmethod
    def from_shared_memory(cls, descriptor: dict[str, Any], *, copy: bool = False) -> State:
        """Attach a state from shared-memory descriptor created by `to_shared_memory`."""
        if not isinstance(descriptor, dict):
            raise StateError("State shared-memory descriptor must be a dict")
        if descriptor.get("magic") != _STATE_SHARED_MAGIC:
            raise StateError("Invalid State shared-memory descriptor magic")
        _version_name_to_int(str(descriptor.get("version", "v1")))

        tensor_specs = descriptor.get("tensors")
        if not isinstance(tensor_specs, list):
            raise StateError("State shared-memory descriptor missing tensor list")

        tensors: dict[str, Tensor] = {}
        handles: list[shared_memory.SharedMemory] = []

        try:
            for raw_spec in tensor_specs:
                if not isinstance(raw_spec, dict):
                    raise StateError("Invalid tensor spec in shared-memory descriptor")
                name = str(raw_spec.get("name", ""))
                if not name:
                    raise StateError("Shared-memory tensor spec missing non-empty 'name'")

                dtype = _name_to_dtype(str(raw_spec.get("dtype", "")))
                shape_raw = raw_spec.get("shape")
                if not isinstance(shape_raw, list):
                    raise StateError(f"Shared-memory tensor {name!r} has invalid shape metadata")
                shape = tuple(int(dim) for dim in shape_raw)
                numel = int(raw_spec.get("numel", _tensor_numel(shape)))
                if numel < 0:
                    raise StateError(f"Shared-memory tensor {name!r} has invalid numel={numel}")

                shm_name = str(raw_spec.get("shm_name", ""))
                if not shm_name:
                    raise StateError(f"Shared-memory tensor {name!r} is missing shm_name")

                shm = shared_memory.SharedMemory(name=shm_name, create=False)
                handles.append(shm)

                if numel == 0:
                    tensor = torch.empty(shape, dtype=dtype)
                else:
                    tensor = torch.frombuffer(shm.buf, dtype=dtype, count=numel).reshape(shape)
                if copy:
                    tensor = tensor.clone()
                if bool(raw_spec.get("requires_grad", False)):
                    tensor = tensor.requires_grad_(True)
                tensors[name] = tensor
        except Exception:
            for shm in handles:
                shm.close()
            raise

        meta = descriptor.get("meta", {})
        if not isinstance(meta, dict):
            raise StateError("Shared-memory descriptor field 'meta' must be an object")
        state = cls(tensors=tensors, meta=dict(meta))

        if copy:
            for shm in handles:
                shm.close()
            return state

        state.meta["_wf_shared_memory_handles"] = handles
        state.meta["_wf_shared_memory_descriptor"] = descriptor
        return state

    def close_shared_memory(self, *, unlink: bool = False) -> None:
        """Close attached shared-memory handles, optionally unlinking segments."""
        handles_raw = self.meta.pop("_wf_shared_memory_handles", None)
        handles: list[shared_memory.SharedMemory] = (
            handles_raw if isinstance(handles_raw, list) else []
        )
        descriptor = self.meta.get("_wf_shared_memory_descriptor")

        for shm in handles:
            try:
                shm.close()
            except FileNotFoundError:
                continue

        if unlink and isinstance(descriptor, dict):
            self.unlink_shared_memory(descriptor)

    @staticmethod
    def unlink_shared_memory(descriptor: dict[str, Any]) -> None:
        """Unlink shared-memory segments created by `to_shared_memory`."""
        specs = descriptor.get("tensors")
        if not isinstance(specs, list):
            return
        for raw_spec in specs:
            if not isinstance(raw_spec, dict):
                continue
            shm_name = raw_spec.get("shm_name")
            if not isinstance(shm_name, str) or not shm_name:
                continue
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
            except FileNotFoundError:
                continue
            try:
                shm.unlink()
            finally:
                shm.close()
