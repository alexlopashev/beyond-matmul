#!/usr/bin/env python3
"""Tiny PyTorch Conv1d workload case study."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.frontend import capture_torch_fx_operators
from beyond_matmul.ir import AffineOperator, DenseOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


def main() -> int:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print("PyTorch is not installed in this environment, so the Conv1d workload demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync")
        print("  mise exec -- uv run python examples/conv1d_workload_demo.py")
        return 0

    class TinyConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size=11, bias=True)
            with torch.no_grad():
                self.conv.weight.copy_(
                    torch.tensor([[
                        [0.75, -0.5, 1.25, 0.1, -1.0, 0.6, 1.5, -0.2, 0.35, -0.8, 0.95],
                    ]])
                )
                self.conv.bias.copy_(torch.tensor([0.125]))

        def forward(self, x):
            return self.conv(x)

    module = TinyConv1d().eval()
    inputs = torch.randn(8, 1, 12, generator=torch.Generator().manual_seed(19))
    captured = capture_torch_fx_operators(module, sample_inputs=inputs)
    if "conv" not in captured:
        raise RuntimeError("no Conv1d operator was captured from the FX graph")

    captured_operator = captured["conv"]
    operator = captured_operator.operator
    dense_operator = (
        AffineOperator(DenseOperator(operator.to_dense()), operator.bias)
        if isinstance(operator, AffineOperator)
        else DenseOperator(operator.to_dense())
    )

    input_rows = inputs.squeeze(1).tolist()
    request = PlanningRequest(
        batch_size=len(input_rows),
        calls=128,
        allow_approximate=False,
        sample_inputs=input_rows,
    )
    structured_plan = plan_fixed_weight(operator, request)
    dense_plan = plan_fixed_weight(dense_operator, request)

    torch_outputs = module(inputs).detach().squeeze(1).tolist()
    operator_outputs = operator.apply(input_rows)
    rel_error = la.rms_relative_error(torch_outputs, operator_outputs)

    linear_kind = operator.linear.metadata.kind if isinstance(operator, AffineOperator) else operator.metadata.kind
    print("Tiny Conv1d workload demo")
    print()
    print(f"captured: {captured_operator.name}")
    print(f"operator: kind={operator.metadata.kind}, linear_kind={linear_kind}, shape={operator.shape}")
    print(f"capture notes: {captured_operator.event.notes}")
    print()
    print(f"planner with convolution provenance: {structured_plan.summary()}")
    print(f"planner after dense materialization: {dense_plan.summary()}")
    print(f"operator output error vs torch module: {rel_error:.3g}")
    print()
    print("Takeaway: Conv1d provenance preserves the direct kernel before valid convolution is flattened to a dense Toeplitz matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
