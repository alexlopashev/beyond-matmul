#!/usr/bin/env python3
"""Tiny PyTorch adapter workload case study."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.frontend import capture_torch_fx_linear_operators
from beyond_matmul.ir import AffineOperator, DenseOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        print("PyTorch is not installed in this environment, so the adapter workload demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync")
        print("  mise exec -- uv run python examples/adapter_workload_demo.py")
        return 0

    class TinyMergedAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.Linear(6, 2, bias=False)
            self.lora_B = nn.Linear(2, 4, bias=True)
            with torch.no_grad():
                self.lora_A.weight.copy_(
                    torch.tensor([
                        [1.0, 0.0, -0.5, 0.25, 0.75, -1.0],
                        [0.2, -0.8, 0.4, 1.0, -0.3, 0.6],
                    ])
                )
                self.lora_B.weight.copy_(
                    torch.tensor([
                        [0.9, -0.2],
                        [0.4, 0.7],
                        [-0.6, 0.3],
                        [0.1, 0.8],
                    ])
                )
                self.lora_B.bias.copy_(torch.tensor([0.05, -0.1, 0.2, 0.0]))
            self.register_buffer("merged_weight", (self.lora_B.weight @ self.lora_A.weight).detach().clone())

        def forward(self, x):
            return F.linear(x, self.merged_weight, self.lora_B.bias)

    module = TinyMergedAdapter().eval()
    captured = capture_torch_fx_linear_operators(module)
    if "lora_B" not in captured:
        raise RuntimeError("no named adapter factors were captured")

    captured_operator = captured["lora_B"]
    operator = captured_operator.operator
    dense_operator = (
        AffineOperator(DenseOperator(operator.to_dense()), operator.bias)
        if isinstance(operator, AffineOperator)
        else DenseOperator(operator.to_dense())
    )

    inputs = torch.randn(8, operator.in_features, generator=torch.Generator().manual_seed(13))
    input_rows = inputs.tolist()
    request = PlanningRequest(
        batch_size=len(input_rows),
        calls=128,
        allow_approximate=False,
        sample_inputs=input_rows,
    )
    structured_plan = plan_fixed_weight(operator, request)
    dense_plan = plan_fixed_weight(dense_operator, request)

    torch_outputs = module(inputs).detach().tolist()
    operator_outputs = operator.apply(input_rows)
    rel_error = la.rms_relative_error(torch_outputs, operator_outputs)

    print("Tiny adapter workload demo")
    print()
    print(f"captured: {captured_operator.name}")
    print(f"operator: kind={operator.metadata.kind}, shape={operator.shape}")
    print(f"capture notes: {captured_operator.event.notes}")
    print()
    print(f"planner with adapter factors: {structured_plan.summary()}")
    print(f"planner after dense merge: {dense_plan.summary()}")
    print(f"operator output error vs torch module: {rel_error:.3g}")
    print()
    print("Takeaway: nearby adapter provenance recovers low-rank structure even when forward uses a merged dense weight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
