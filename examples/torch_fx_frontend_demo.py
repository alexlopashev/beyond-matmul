#!/usr/bin/env python3
"""Torch FX frontend demo for provenance-aware low-rank linear capture."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.frontend import capture_torch_fx_linear_operators
from beyond_matmul.ir import DenseOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        print("PyTorch is not installed in this environment, so the Torch FX demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  uv sync")
        print("  uv run python examples/torch_fx_frontend_demo.py")
        return 0

    class LowRankProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.right = nn.Parameter(torch.tensor([
                [1.0, 0.0, -0.5, 0.25, 0.75, -1.0],
                [0.2, -0.8, 0.4, 1.0, -0.3, 0.6],
            ]))
            self.left = nn.Parameter(torch.tensor([
                [0.9, -0.2],
                [0.4, 0.7],
                [-0.6, 0.3],
                [0.1, 0.8],
            ]))

        def forward(self, x):
            hidden = F.linear(x, self.right)
            return F.linear(hidden, self.left)

    module = LowRankProjection().eval()
    captured = capture_torch_fx_linear_operators(module)
    if not captured:
        raise RuntimeError("no low-rank linear operator was captured from the FX graph")

    name, captured_operator = next(iter(captured.items()))
    operator = captured_operator.operator
    dense = DenseOperator(operator.to_dense())
    inputs = torch.randn(8, operator.in_features, generator=torch.Generator().manual_seed(7))
    input_rows = inputs.tolist()

    request = PlanningRequest(
        batch_size=len(input_rows),
        calls=32,
        allow_approximate=False,
        sample_inputs=input_rows,
    )
    plan = plan_fixed_weight(operator, request)
    dense_plan = plan_fixed_weight(dense, request)

    torch_outputs = module(inputs).detach().tolist()
    operator_outputs = operator.apply(input_rows)
    rel_error = la.rms_relative_error(torch_outputs, operator_outputs)

    print("Torch FX frontend demo")
    print()
    print(f"captured: {name}")
    print(f"operator: kind={operator.metadata.kind}, shape={operator.shape}, rank={operator.rank}")
    print(f"provenance: {operator.metadata.provenance.expression}")
    print()
    print(f"planner with provenance: {plan.summary()}")
    print(f"planner after dense materialization: {dense_plan.summary()}")
    print(f"operator output error vs torch module: {rel_error:.3g}")
    print()
    print("Takeaway: FX capture preserves the low-rank factors before they collapse into dense GEMM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
