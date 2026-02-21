"""CI smoke test harness for parity model training validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.batch import Batch
from ..core.model import WorldModel


@dataclass(frozen=True)
class SmokeCheckpoint:
    """Expected training behavior checkpoint for smoke tests."""

    family: str
    step_count: int
    loss_range: tuple[float, float]
    component_loss_ranges: dict[str, tuple[float, float]]
    gradient_norm_range: tuple[float, float]
    param_delta_norm_range: tuple[float, float]


@dataclass(frozen=True)
class SmokeResult:
    """Result of a smoke test run."""

    family: str
    passed: bool
    steps_run: int
    final_loss: float
    component_losses: dict[str, float]
    gradient_norms: list[float]
    param_delta_norm: float
    violations: list[str]


DREAMERV3_SMOKE_CHECKPOINT = SmokeCheckpoint(
    family="dreamerv3",
    step_count=100,
    loss_range=(0.01, 200.0),
    component_loss_ranges={
        "kl": (0.0, 100.0),
        "reconstruction": (0.0, 100.0),
        "reward": (0.0, 50.0),
        "continue": (0.0, 10.0),
    },
    gradient_norm_range=(1e-8, 1e6),
    param_delta_norm_range=(1e-10, 1e4),
)


TDMPC2_SMOKE_CHECKPOINT = SmokeCheckpoint(
    family="tdmpc2",
    step_count=100,
    loss_range=(0.01, 200.0),
    component_loss_ranges={
        "consistency": (0.0, 100.0),
        "reward": (0.0, 50.0),
        "td": (0.0, 100.0),
    },
    gradient_norm_range=(1e-8, 1e6),
    param_delta_norm_range=(1e-10, 1e4),
)


def _compute_gradient_norm(model: WorldModel) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def _compute_param_snapshot(model: WorldModel) -> dict[str, Tensor]:
    """Snapshot all parameters as detached clones."""
    return {name: p.data.clone() for name, p in model.named_parameters()}


def _compute_param_delta_norm(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    """Compute L2 norm of the parameter delta."""
    total = 0.0
    for name in before:
        delta = after[name] - before[name]
        total += delta.norm(2).item() ** 2
    return math.sqrt(total)


def _make_dreamerv3_batch(
    config: object, batch_size: int, seq_len: int, device: torch.device
) -> Batch:
    """Create a random batch for DreamerV3."""
    obs_shape = getattr(config, "obs_shape", (3, 64, 64))
    action_dim = getattr(config, "action_dim", 6)
    return Batch(
        obs=torch.randn(batch_size, seq_len, *obs_shape, device=device),
        actions=torch.randn(batch_size, seq_len, action_dim, device=device),
        rewards=torch.randn(batch_size, seq_len, device=device),
        terminations=torch.zeros(batch_size, seq_len, device=device),
    )


def _make_tdmpc2_batch(
    config: object, batch_size: int, seq_len: int, device: torch.device
) -> Batch:
    """Create a random batch for TD-MPC2."""
    obs_shape = getattr(config, "obs_shape", (39,))
    action_dim = getattr(config, "action_dim", 6)
    return Batch(
        obs=torch.randn(batch_size, seq_len, *obs_shape, device=device),
        actions=torch.randn(batch_size, seq_len, action_dim, device=device),
        rewards=torch.randn(batch_size, seq_len, device=device),
        terminations=torch.zeros(batch_size, seq_len, device=device),
    )


_BATCH_BUILDERS: dict[str, object] = {
    "dreamerv3": _make_dreamerv3_batch,
    "tdmpc2": _make_tdmpc2_batch,
}


def run_smoke_test(
    model: WorldModel,
    *,
    family: str,
    steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 8,
    seed: int = 42,
) -> SmokeResult:
    """Run a smoke test: forward + loss + backward for N steps.

    Parameters
    ----------
    model:
        The world model to test.
    family:
        Model family identifier (``"dreamerv3"`` or ``"tdmpc2"``).
    steps:
        Number of training steps.
    batch_size:
        Batch size for random data.
    seq_len:
        Sequence length for random data.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    SmokeResult
        Aggregated result including loss, gradients, and violations.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    build_batch = _BATCH_BUILDERS.get(family)
    if build_batch is None:
        raise ValueError(f"Unknown family: {family!r}. Expected one of {sorted(_BATCH_BUILDERS)}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    param_before = _compute_param_snapshot(model)
    gradient_norms: list[float] = []
    final_loss = float("nan")
    component_losses: dict[str, float] = {}

    for _step in range(steps):
        batch = build_batch(model.config, batch_size, seq_len, device)  # type: ignore[operator]
        optimizer.zero_grad()
        loss_out = model.loss(batch)
        loss_out.loss.backward()
        optimizer.step()

        grad_norm = _compute_gradient_norm(model)
        gradient_norms.append(grad_norm)
        final_loss = loss_out.loss.item()
        component_losses = {k: v.item() for k, v in loss_out.components.items()}

    param_after = _compute_param_snapshot(model)
    param_delta = _compute_param_delta_norm(param_before, param_after)

    # Validate against checkpoint
    if family == "dreamerv3":
        checkpoint = DREAMERV3_SMOKE_CHECKPOINT
    elif family == "tdmpc2":
        checkpoint = TDMPC2_SMOKE_CHECKPOINT
    else:
        raise ValueError(f"No checkpoint for family: {family!r}")

    violations: list[str] = []

    # Check NaN
    if math.isnan(final_loss):
        violations.append("final_loss is NaN")
    if math.isinf(final_loss):
        violations.append("final_loss is Inf")

    # Check loss range
    lo, hi = checkpoint.loss_range
    if not math.isnan(final_loss) and not (lo <= final_loss <= hi):
        violations.append(f"final_loss={final_loss:.4f} outside [{lo}, {hi}]")

    # Check component losses
    for comp_name, (c_lo, c_hi) in checkpoint.component_loss_ranges.items():
        comp_val = component_losses.get(comp_name)
        if comp_val is None:
            violations.append(f"missing component loss: {comp_name}")
        elif math.isnan(comp_val):
            violations.append(f"component {comp_name} is NaN")
        elif not (c_lo <= comp_val <= c_hi):
            violations.append(f"component {comp_name}={comp_val:.4f} outside [{c_lo}, {c_hi}]")

    # Check gradient norms (at least one non-zero)
    if all(g == 0.0 for g in gradient_norms):
        violations.append("all gradient norms are zero (no gradient flow)")
    else:
        g_lo, g_hi = checkpoint.gradient_norm_range
        for i, g in enumerate(gradient_norms):
            if math.isnan(g):
                violations.append(f"gradient norm at step {i} is NaN")
                break
            if not (g_lo <= g <= g_hi):
                violations.append(
                    f"gradient norm={g:.4e} at step {i} outside [{g_lo:.1e}, {g_hi:.1e}]"
                )
                break

    # Check parameter update
    p_lo, p_hi = checkpoint.param_delta_norm_range
    if not (p_lo <= param_delta <= p_hi):
        violations.append(f"param_delta_norm={param_delta:.4e} outside [{p_lo:.1e}, {p_hi:.1e}]")

    return SmokeResult(
        family=family,
        passed=len(violations) == 0,
        steps_run=steps,
        final_loss=final_loss,
        component_losses=component_losses,
        gradient_norms=gradient_norms,
        param_delta_norm=param_delta,
        violations=violations,
    )
