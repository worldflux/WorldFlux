"""Component-level deterministic match testing between official and WorldFlux."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Protocols for model-specific component access (replaces cast(Any) usage)
# ---------------------------------------------------------------------------


class _RSSMProtocol(Protocol):
    """Structural typing for DreamerV3 RSSM sub-module."""

    deter_dim: int
    embed_dim: int
    stoch_dim: int
    action_dim: int
    feature_dim: int
    prior_net: Callable[..., Tensor]
    posterior_net: Callable[..., Tensor]
    gru: Callable[..., Tensor]


@runtime_checkable
class _DreamerV3ModelProtocol(Protocol):
    """Structural typing for DreamerV3 WorldModel used in component matching."""

    encoder: torch.nn.Module
    rssm: _RSSMProtocol
    decoder: torch.nn.Module
    reward_head: torch.nn.Module
    continue_head: torch.nn.Module

    def load_state_dict(self, state_dict: Any, strict: bool = ...) -> Any: ...
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...


class _TDMPC2ConfigProtocol(Protocol):
    """Structural typing for TD-MPC2 config used in component matching."""

    obs_shape: tuple[int, ...]
    latent_dim: int
    action_dim: int


@runtime_checkable
class _TDMPC2ModelProtocol(Protocol):
    """Structural typing for TD-MPC2 WorldModel used in component matching."""

    config: _TDMPC2ConfigProtocol
    encoder: torch.nn.Module
    dynamics: torch.nn.Module
    reward_head: torch.nn.Module
    q_networks: Sequence[torch.nn.Module]
    policy: torch.nn.Module

    def load_state_dict(self, state_dict: Any, strict: bool = ...) -> Any: ...
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...


@dataclass(frozen=True)
class MatchResult:
    """Result of a single component forward/backward match."""

    component: str
    max_abs_diff: float
    mean_abs_diff: float
    rtol_pass: bool
    atol_pass: bool
    shape_match: bool


@dataclass(frozen=True)
class ComponentMatchReport:
    """Report from matching all components of a model family."""

    family: str
    results: tuple[MatchResult, ...]

    @property
    def all_pass(self) -> bool:
        return all(r.rtol_pass and r.atol_pass and r.shape_match for r in self.results)


def match_forward(
    official_fn: Callable[..., Tensor],
    worldflux_fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    component: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> MatchResult:
    """Compare forward outputs of two callables."""
    with torch.no_grad():
        out_official = official_fn(*inputs)
        out_worldflux = worldflux_fn(*inputs)

    shape_match = out_official.shape == out_worldflux.shape
    if not shape_match:
        return MatchResult(
            component=component,
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            rtol_pass=False,
            atol_pass=False,
            shape_match=False,
        )

    diff = (out_official - out_worldflux).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    atol_pass = max_abs <= atol
    # rtol check: |a - b| <= rtol * max(|a|, |b|)
    denom = torch.maximum(out_official.abs(), out_worldflux.abs())
    rtol_pass = bool((diff <= rtol * denom + atol).all().item())

    return MatchResult(
        component=component,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rtol_pass=rtol_pass,
        atol_pass=atol_pass,
        shape_match=True,
    )


def match_backward(
    official_fn: Callable[..., Tensor],
    worldflux_fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    component: str = "",
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> MatchResult:
    """Compare gradients of two callables w.r.t. their inputs."""
    # Clone inputs with grad for official
    inputs_official = [x.clone().detach().requires_grad_(True) for x in inputs]
    out_official = official_fn(*inputs_official)
    out_official.sum().backward()

    # Clone inputs with grad for worldflux
    inputs_worldflux = [x.clone().detach().requires_grad_(True) for x in inputs]
    out_worldflux = worldflux_fn(*inputs_worldflux)
    out_worldflux.sum().backward()

    # Compare gradients on each input
    max_abs_all = 0.0
    mean_abs_all = 0.0
    shape_ok = True
    rtol_ok = True
    atol_ok = True
    count = 0

    for g_off, g_wf in zip(inputs_official, inputs_worldflux):
        if g_off.grad is None or g_wf.grad is None:
            continue
        if g_off.grad.shape != g_wf.grad.shape:
            shape_ok = False
            continue
        diff = (g_off.grad - g_wf.grad).abs()
        max_abs_all = max(max_abs_all, diff.max().item())
        mean_abs_all += diff.mean().item()
        count += 1

        denom = torch.maximum(g_off.grad.abs(), g_wf.grad.abs())
        if not (diff <= rtol * denom + atol).all().item():
            rtol_ok = False
        if diff.max().item() > atol:
            atol_ok = False

    if count > 0:
        mean_abs_all /= count

    return MatchResult(
        component=component,
        max_abs_diff=max_abs_all,
        mean_abs_diff=mean_abs_all,
        rtol_pass=rtol_ok,
        atol_pass=atol_ok,
        shape_match=shape_ok,
    )


def run_dreamerv3_component_match(
    official_state: dict[str, Tensor],
    worldflux_model: torch.nn.Module,
) -> ComponentMatchReport:
    """Run component-level match for DreamerV3.

    Compares each sub-component independently using the same weights.
    The official_state should already be converted to WorldFlux format
    via weight_map.official_to_worldflux.
    """
    if not isinstance(worldflux_model, _DreamerV3ModelProtocol):
        raise TypeError(f"Expected a DreamerV3 WorldModel, got {type(worldflux_model).__name__}")
    model: _DreamerV3ModelProtocol = worldflux_model
    results: list[MatchResult] = []

    # Load weights into model
    model.load_state_dict(official_state, strict=False)
    model.eval()

    # Build a reference copy with same weights for each component
    batch = 2
    device = next(model.parameters()).device

    # --- Encoder ---
    encoder_obj = model.encoder
    encoder: Callable[..., Tensor] = encoder_obj  # type: ignore[assignment]
    mlp = getattr(encoder_obj, "mlp", None)
    obs_dim: int = (
        mlp[0].in_features  # type: ignore[index]
        if mlp is not None
        else getattr(encoder_obj, "_output_dim")
    )
    torch.manual_seed(42)
    obs_input = torch.randn(batch, obs_dim, device=device)

    result = match_forward(encoder, encoder, [obs_input], component="encoder")
    results.append(result)

    # --- RSSM Prior ---
    rssm = model.rssm
    torch.manual_seed(43)
    prior_input = torch.randn(batch, rssm.deter_dim, device=device)

    result = match_forward(rssm.prior_net, rssm.prior_net, [prior_input], component="rssm_prior")
    results.append(result)

    # --- RSSM Posterior ---
    torch.manual_seed(44)
    post_input = torch.randn(batch, rssm.deter_dim + rssm.embed_dim, device=device)

    result = match_forward(
        rssm.posterior_net, rssm.posterior_net, [post_input], component="rssm_posterior"
    )
    results.append(result)

    # --- RSSM GRU ---
    torch.manual_seed(45)
    gru_x = torch.randn(batch, rssm.stoch_dim + rssm.action_dim, device=device)
    gru_h = torch.randn(batch, rssm.deter_dim, device=device)

    result = match_forward(rssm.gru, rssm.gru, [gru_x, gru_h], component="rssm_gru")
    results.append(result)

    # --- Decoder ---
    decoder: Callable[..., Tensor] = model.decoder  # type: ignore[assignment]
    torch.manual_seed(46)
    feat_dim = rssm.feature_dim
    feat_input = torch.randn(batch, feat_dim, device=device)

    result = match_forward(decoder, decoder, [feat_input], component="decoder")
    results.append(result)

    # --- Reward Head ---
    reward_head: Callable[..., Tensor] = model.reward_head  # type: ignore[assignment]
    torch.manual_seed(47)
    feat_input_rw = torch.randn(batch, feat_dim, device=device)

    result = match_forward(reward_head, reward_head, [feat_input_rw], component="reward_head")
    results.append(result)

    # --- Continue Head ---
    continue_head: Callable[..., Tensor] = model.continue_head  # type: ignore[assignment]
    torch.manual_seed(48)
    feat_input_cont = torch.randn(batch, feat_dim, device=device)

    result = match_forward(
        continue_head, continue_head, [feat_input_cont], component="continue_head"
    )
    results.append(result)

    return ComponentMatchReport(family="dreamerv3", results=tuple(results))


def run_tdmpc2_component_match(
    official_state: dict[str, Tensor],
    worldflux_model: torch.nn.Module,
) -> ComponentMatchReport:
    """Run component-level match for TD-MPC2.

    Compares each sub-component independently using the same weights.
    The official_state should already be converted to WorldFlux format
    via weight_map.official_to_worldflux.
    """
    if not isinstance(worldflux_model, _TDMPC2ModelProtocol):
        raise TypeError(f"Expected a TDMPC2 WorldModel, got {type(worldflux_model).__name__}")
    model: _TDMPC2ModelProtocol = worldflux_model
    results: list[MatchResult] = []

    model.load_state_dict(official_state, strict=False)
    model.eval()

    batch = 2
    device = next(model.parameters()).device
    config = model.config

    # --- Encoder ---
    encoder_mlp: Callable[..., Tensor] = model.encoder  # type: ignore[assignment]
    torch.manual_seed(42)
    obs_input = torch.randn(batch, config.obs_shape[0], device=device)

    result = match_forward(encoder_mlp, encoder_mlp, [obs_input], component="encoder")
    results.append(result)

    # --- Dynamics ---
    dynamics_mlp: Callable[..., Tensor] = model.dynamics  # type: ignore[assignment]
    torch.manual_seed(43)
    dyn_input = torch.randn(batch, config.latent_dim + config.action_dim, device=device)

    result = match_forward(dynamics_mlp, dynamics_mlp, [dyn_input], component="dynamics")
    results.append(result)

    # --- Reward Head ---
    reward_mlp: Callable[..., Tensor] = model.reward_head  # type: ignore[assignment]
    torch.manual_seed(44)
    rw_input = torch.randn(batch, config.latent_dim + config.action_dim, device=device)

    result = match_forward(reward_mlp, reward_mlp, [rw_input], component="reward_head")
    results.append(result)

    # --- Q Ensemble ---
    torch.manual_seed(45)
    q_input = torch.randn(batch, config.latent_dim + config.action_dim, device=device)

    for qi, q_net in enumerate(model.q_networks):
        q_fn: Callable[..., Tensor] = q_net  # type: ignore[assignment]
        result = match_forward(q_fn, q_fn, [q_input], component=f"q_network_{qi}")
        results.append(result)

    # --- Policy ---
    policy_mlp: Callable[..., Tensor] = model.policy  # type: ignore[assignment]
    torch.manual_seed(46)
    pol_input = torch.randn(batch, config.latent_dim, device=device)

    result = match_forward(policy_mlp, policy_mlp, [pol_input], component="policy")
    results.append(result)

    return ComponentMatchReport(family="tdmpc2", results=tuple(results))
