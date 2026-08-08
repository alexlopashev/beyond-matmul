#!/usr/bin/env python3
"""Tiny deterministic OLMoE stable-route correctness and fallback demo."""

from __future__ import annotations

import json
from typing import Any

import torch

from beyond_matmul.olmoe_route_plan import (
    ROUTE_METADATA_ATTRIBUTE,
    eager_olmoe_experts_forward,
    stable_route_experts_forward,
)


class DemoExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.hidden_dim = 3
        self.intermediate_dim = 2
        self.gate_up_proj = torch.nn.Parameter(
            torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3) / 37
        )
        self.down_proj = torch.nn.Parameter(
            torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2) / 23
        )
        self.act_fn = torch.nn.functional.silu
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.eval()


def run_demo() -> dict[str, Any]:
    experts = DemoExperts()
    hidden_states = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [0.1, 0.5, -0.3],
            [-0.6, 0.8, 0.2],
            [0.9, -0.2, 0.4],
        ]
    )
    top_k_index = torch.tensor(
        [
            [2, 1, 2],
            [0, 2, 1],
            [2, 0, 2],
            [1, 2, 0],
        ],
        dtype=torch.long,
    )
    top_k_weights = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.4, 0.4, 0.2],
            [0.7, 0.2, 0.1],
        ]
    )

    reference = eager_olmoe_experts_forward(
        experts, hidden_states, top_k_index, top_k_weights
    )
    candidate = stable_route_experts_forward(
        experts, hidden_states, top_k_index, top_k_weights
    )
    stable_metadata = dict(getattr(experts, ROUTE_METADATA_ATTRIBUTE))
    difference = (candidate - reference).to(torch.float32)

    experts.train()
    fallback = stable_route_experts_forward(
        experts, hidden_states, top_k_index, top_k_weights
    )
    fallback_metadata = dict(getattr(experts, ROUTE_METADATA_ATTRIBUTE))

    return {
        "demo": "olmoe_stable_route",
        "correctness": {
            "exact_match": bool(torch.equal(candidate, reference)),
            "max_abs_error": float(torch.max(torch.abs(difference)).item()),
            "fallback_exact_match": bool(torch.equal(fallback, reference)),
        },
        "stable_execution": stable_metadata,
        "fallback_execution": fallback_metadata,
        "performance_claim": "none",
        "note": "This deterministic CPU demo validates semantics, not speed.",
    }


def main() -> int:
    result = run_demo()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["correctness"]["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
