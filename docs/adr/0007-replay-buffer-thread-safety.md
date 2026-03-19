# ADR 0007: ReplayBuffer Thread Safety

## Status

Accepted

## Context

The `ReplayBuffer` stores experience data for off-policy training.
Multi-threaded access patterns include:

- **Writer thread**: Environment step loop adding transitions.
- **Reader thread(s)**: Training loop sampling batches.

Making the buffer fully thread-safe (e.g. with locks around every
read/write) would:

- Add lock contention overhead on every `add()` and `sample()` call.
- Complicate the implementation with lock ordering, deadlock prevention,
  and conditional variables.
- Provide limited benefit since the primary use case is single-process
  training with sequential access.

## Decision

The `ReplayBuffer` is explicitly NOT thread-safe. The design enforces
single-writer semantics:

- Only one thread may call `add()` / `add_trajectory()`.
- `sample()` may be called from any thread, but concurrent `sample()`
  and `add()` are not guaranteed to be consistent.
- Users requiring multi-threaded access must provide external
  synchronization (e.g. `threading.Lock` wrapper).

This is documented in `CLAUDE.md` ("ReplayBuffer is NOT thread-safe.
Single writer thread only.") and enforced by convention rather than
runtime checks.

## Consequences

- `add()` and `sample()` have zero synchronization overhead.
- The buffer implementation remains simple: NumPy array writes with
  a circular index.
- Distributed training must use separate buffer instances per process
  (standard practice for DDP/FSDP workflows).
- The `BatchLoadingRetry` (ARCH-06) operates at the Trainer level,
  outside the buffer, so retry logic does not introduce thread safety
  concerns.
- Future multi-threaded buffer implementations can be provided as
  alternative `BatchProvider` implementations without changing the
  core buffer contract.
