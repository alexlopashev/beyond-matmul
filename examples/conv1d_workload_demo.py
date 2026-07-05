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


def _dense_operator_for(operator):
    dense = DenseOperator(operator.to_dense())
    if isinstance(operator, AffineOperator):
        return AffineOperator(dense, operator.bias)
    return dense


def _linear_kind(operator) -> str:
    if isinstance(operator, AffineOperator):
        return operator.linear.metadata.kind
    return operator.metadata.kind


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        print("PyTorch is not installed in this environment, so the Conv1d workload demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync --locked")
        print("  mise exec -- uv run python examples/conv1d_workload_demo.py")
        return 0

    weight = torch.tensor([
        [[0.75, -0.5, 1.25], [0.1, -1.0, 0.6]],
        [[1.5, -0.2, 0.35], [-0.8, 0.95, -0.4]],
    ])

    class ModuleConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, kernel_size=3, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(weight)

        def forward(self, x):
            return self.conv(x)

    class FunctionalConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("bias", torch.tensor([0.125, -0.25]))

        def forward(self, x):
            return F.conv1d(x, self.weight, self.bias)

    def run_case(title: str, module: nn.Module, inputs, capture_name: str | None = None) -> None:
        captured = capture_torch_fx_operators(module, sample_inputs=inputs)
        if capture_name is None:
            captured_operator = next((item for item in captured.values() if item.event.notes.get("capture") == "conv1d_function"), None)
        else:
            captured_operator = captured.get(capture_name)
        if captured_operator is None:
            raise RuntimeError(f"no Conv1d operator was captured for {title}")

        operator = captured_operator.operator
        dense_operator = _dense_operator_for(operator)
        input_rows = inputs.flatten(1).tolist()
        request = PlanningRequest(
            batch_size=len(input_rows),
            calls=128,
            allow_approximate=False,
            sample_inputs=input_rows,
        )
        structured_plan = plan_fixed_weight(operator, request)
        dense_plan = plan_fixed_weight(dense_operator, request)

        torch_outputs = module(inputs).detach().flatten(1).tolist()
        operator_outputs = operator.apply(input_rows)
        rel_error = la.rms_relative_error(torch_outputs, operator_outputs)

        print(title)
        print(f"  captured: {captured_operator.name}")
        print(f"  operator: kind={operator.metadata.kind}, linear_kind={_linear_kind(operator)}, shape={operator.shape}")
        print(f"  capture notes: {captured_operator.event.notes}")
        print(f"  planner with convolution provenance: {structured_plan.summary()}")
        print(f"  planner after dense materialization: {dense_plan.summary()}")
        print(f"  operator output error vs torch: {rel_error:.3g}")

    inputs = torch.randn(8, 2, 12, generator=torch.Generator().manual_seed(19))

    print("Conv1d workload coverage demo")
    print()
    run_case("Multi-channel nn.Conv1d module", ModuleConv1d().eval(), inputs, capture_name="conv")
    print()
    run_case("Functional F.conv1d with fixed bias", FunctionalConv1d().eval(), inputs)
    print()
    print("Takeaway: Conv1d provenance preserves direct channel-aware kernels before valid convolution is flattened to a dense block-Toeplitz matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
