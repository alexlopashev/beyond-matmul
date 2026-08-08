"""Eager-compatible route planning for the pinned Transformers OLMoE experts.

This is deliberately a narrow inference backend.  It replaces eager's dense
one-hot route mask and repeated per-expert ``where`` calls with one inspectable
plan while retaining eager's expert, router-slot, token, and accumulation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


BACKEND_NAME = "beyond_matmul_stable_route"
ROUTE_METADATA_ATTRIBUTE = "_beyond_matmul_last_route_plan"
ROUTE_ORDER = "expert_then_top_k_slot_then_token"
_EXECUTION_AUDIT: dict[str, Any] | None = None


class UnsupportedRoutePlan(ValueError):
    """Raised when a route cannot be represented by the narrow OLMoE plan."""


@dataclass(frozen=True)
class StableRoutePlan:
    """One expert-grouped view of an OLMoE router result."""

    expert_indices: torch.Tensor
    token_indices: torch.Tensor
    top_k_positions: torch.Tensor
    routing_weights: torch.Tensor
    expert_offsets: tuple[int, ...]
    active_experts: tuple[int, ...]
    num_tokens: int
    num_top_k: int
    valid_route_count: int
    sentinel_route_count: int

    def metadata(self) -> dict[str, Any]:
        return {
            "order": ROUTE_ORDER,
            "num_tokens": self.num_tokens,
            "num_top_k": self.num_top_k,
            "valid_route_count": self.valid_route_count,
            "sentinel_route_count": self.sentinel_route_count,
            "active_experts": list(self.active_experts),
            "expert_offsets": list(self.expert_offsets),
            "empty_expert_count": len(self.expert_offsets) - 1 - len(self.active_experts),
        }


def build_stable_route_plan(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    num_experts: int,
) -> StableRoutePlan:
    """Build one stable expert grouping in eager's slot-major traversal order.

    ``top_k_index`` is token-major, but eager transposes its one-hot mask to
    ``[expert, slot, token]`` before calling ``where``.  Flattening the router
    result through ``transpose(0, 1)`` before a stable expert sort is therefore
    essential: an ordinary token-major stable sort has different BF16
    accumulation semantics when a token reaches more than one expert.

    A route equal to ``num_experts`` is the Transformers expert-parallel
    sentinel.  It is accepted only with zero routing weight and omitted from
    real expert work.  Other out-of-range routes are rejected.
    """

    if num_experts <= 0:
        raise UnsupportedRoutePlan("num_experts_must_be_positive")
    if top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        raise UnsupportedRoutePlan("router_tensors_must_be_rank_two")
    if top_k_index.shape != top_k_weights.shape:
        raise UnsupportedRoutePlan("router_tensor_shape_mismatch")
    if top_k_index.device != top_k_weights.device:
        raise UnsupportedRoutePlan("router_tensor_device_mismatch")
    if top_k_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise UnsupportedRoutePlan("router_indices_must_be_integer")

    num_tokens, num_top_k = top_k_index.shape
    flat_experts = top_k_index.transpose(0, 1).reshape(-1)
    flat_weights = top_k_weights.transpose(0, 1).reshape(-1)
    flat_tokens = torch.arange(num_tokens, device=top_k_index.device).repeat(num_top_k)
    flat_slots = torch.arange(num_top_k, device=top_k_index.device).repeat_interleave(
        num_tokens
    )

    sentinel_mask = flat_experts == num_experts
    invalid_mask = (flat_experts < 0) | (flat_experts > num_experts)
    valid_mask = ~(sentinel_mask | invalid_mask)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_experts = flat_experts[valid_positions]
    stable_permutation = torch.argsort(valid_experts, stable=True)
    ordered_positions = valid_positions[stable_permutation]

    expert_indices = flat_experts[ordered_positions]
    token_indices = flat_tokens[ordered_positions]
    top_k_positions = flat_slots[ordered_positions]
    routing_weights = flat_weights[ordered_positions]
    counts = torch.bincount(expert_indices.to(torch.int64), minlength=num_experts)
    validation_and_counts = torch.cat(
        (
            counts,
            invalid_mask.sum().reshape(1),
            sentinel_mask.sum().reshape(1),
            ((flat_weights != 0) & sentinel_mask).sum().reshape(1),
        )
    ).to(device="cpu", dtype=torch.int64).tolist()
    count_values = validation_and_counts[:num_experts]
    invalid_route_count, sentinel_route_count, nonzero_sentinel_count = (
        validation_and_counts[num_experts:]
    )
    if invalid_route_count:
        raise UnsupportedRoutePlan("router_index_out_of_range")
    if nonzero_sentinel_count:
        raise UnsupportedRoutePlan("nonzero_sentinel_weight")

    cumulative = []
    running_count = 0
    for count in count_values:
        running_count += int(count)
        cumulative.append(running_count)
    expert_offsets = tuple([0, *(int(value) for value in cumulative)])
    active_experts = tuple(
        expert_idx
        for expert_idx in range(num_experts)
        if expert_offsets[expert_idx] != expert_offsets[expert_idx + 1]
    )

    return StableRoutePlan(
        expert_indices=expert_indices,
        token_indices=token_indices,
        top_k_positions=top_k_positions,
        routing_weights=routing_weights,
        expert_offsets=expert_offsets,
        active_experts=active_experts,
        num_tokens=num_tokens,
        num_top_k=num_top_k,
        valid_route_count=int(expert_indices.numel()),
        sentinel_route_count=int(sentinel_route_count),
    )


def eager_olmoe_experts_forward(
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Pinned OLMoE eager semantics, retained as an explicit fallback."""

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=experts.num_experts
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(
            current_state, experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, experts.down_proj[expert_idx]
        )
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    return final_hidden_states


def stable_route_experts_forward(
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Execute the pinned OLMoE expert program from one stable route plan."""

    fallback_reason = _fallback_reason(
        experts, hidden_states, top_k_index, top_k_weights
    )
    if fallback_reason is not None:
        _record_fallback(experts, fallback_reason)
        return eager_olmoe_experts_forward(
            experts, hidden_states, top_k_index, top_k_weights
        )

    try:
        plan = build_stable_route_plan(
            top_k_index,
            top_k_weights,
            num_experts=experts.num_experts,
        )
    except UnsupportedRoutePlan as exc:
        _record_fallback(experts, str(exc))
        return eager_olmoe_experts_forward(
            experts, hidden_states, top_k_index, top_k_weights
        )

    final_hidden_states = torch.zeros_like(hidden_states)
    for expert_idx in plan.active_experts:
        start = plan.expert_offsets[expert_idx]
        end = plan.expert_offsets[expert_idx + 1]
        token_idx = plan.token_indices[start:end]
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(
            current_state, experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, experts.down_proj[expert_idx]
        )
        current_hidden_states = (
            current_hidden_states * plan.routing_weights[start:end, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    _record_execution(
        experts,
        {
            "backend": BACKEND_NAME,
            "execution_path": "stable_route_plan",
            "fallback_used": False,
            "fallback_reason": None,
            "route_plan_build_count": 1,
            **plan.metadata(),
        },
    )
    return final_hidden_states


def register_transformers_backend(experts_interface: Any | None = None) -> str:
    """Register the backend through Transformers' public experts interface."""

    if experts_interface is None:
        try:
            from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
        except ImportError as exc:  # pragma: no cover - exercised by real preflight.
            raise RuntimeError("the pinned Transformers checkout is required") from exc
        experts_interface = ALL_EXPERTS_FUNCTIONS
    experts_interface.register(BACKEND_NAME, stable_route_experts_forward)
    return BACKEND_NAME


def begin_execution_audit() -> None:
    """Begin an aggregate backend-path audit for one candidate measurement."""

    global _EXECUTION_AUDIT
    if _EXECUTION_AUDIT is not None:
        raise RuntimeError("stable-route execution audit is already active")
    _EXECUTION_AUDIT = {
        "status": "observed",
        "backend": BACKEND_NAME,
        "calls": 0,
        "stable_route_plan_calls": 0,
        "eager_fallback_calls": 0,
        "route_plan_build_count": 0,
        "fallback_reasons": {},
    }


def end_execution_audit() -> dict[str, Any]:
    """Finish and return the current aggregate execution-path audit."""

    global _EXECUTION_AUDIT
    if _EXECUTION_AUDIT is None:
        raise RuntimeError("stable-route execution audit is not active")
    result = {
        **_EXECUTION_AUDIT,
        "fallback_reasons": dict(_EXECUTION_AUDIT["fallback_reasons"]),
    }
    _EXECUTION_AUDIT = None
    return result


def _fallback_reason(
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> str | None:
    if getattr(experts, "training", False):
        return "training_mode_not_supported"
    if not getattr(experts, "has_gate", True):
        return "ungated_experts_not_supported"
    if getattr(experts, "has_bias", False):
        return "expert_bias_not_supported"
    if getattr(experts, "is_transposed", False):
        return "transposed_expert_weights_not_supported"
    if not getattr(experts, "is_concatenated", True):
        return "interleaved_expert_weights_not_supported"
    if not all(
        hasattr(experts, attribute)
        for attribute in ("num_experts", "gate_up_proj", "down_proj", "act_fn")
    ):
        return "olmoe_expert_attributes_missing"
    if hidden_states.ndim != 2:
        return "hidden_states_must_be_rank_two"
    if top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        return "router_tensors_must_be_rank_two"
    if top_k_index.shape != top_k_weights.shape:
        return "router_tensor_shape_mismatch"
    if hidden_states.shape[0] != top_k_index.shape[0]:
        return "router_token_count_mismatch"
    if not (
        hidden_states.device == top_k_index.device == top_k_weights.device
    ):
        return "input_device_mismatch"
    return None


def _record_fallback(experts: torch.nn.Module, reason: str) -> None:
    _record_execution(
        experts,
        {
            "backend": BACKEND_NAME,
            "execution_path": "eager_fallback",
            "fallback_used": True,
            "fallback_reason": reason,
            "route_plan_build_count": 0,
        },
    )


def _record_execution(experts: torch.nn.Module, metadata: dict[str, Any]) -> None:
    setattr(experts, ROUTE_METADATA_ATTRIBUTE, metadata)
    if _EXECUTION_AUDIT is None:
        return
    _EXECUTION_AUDIT["calls"] += 1
    _EXECUTION_AUDIT["route_plan_build_count"] += int(
        metadata["route_plan_build_count"]
    )
    if metadata["execution_path"] == "stable_route_plan":
        _EXECUTION_AUDIT["stable_route_plan_calls"] += 1
        return
    _EXECUTION_AUDIT["eager_fallback_calls"] += 1
    reason = str(metadata["fallback_reason"])
    reasons = _EXECUTION_AUDIT["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
