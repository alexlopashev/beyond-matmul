#!/usr/bin/env python3
"""Torch FX coverage overview for fixed-weight frontend capture."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.frontend import capture_torch_fx_operators
from beyond_matmul.ir import AffineOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


def _operator_kind(operator) -> str:
    if isinstance(operator, AffineOperator):
        return f"affine({operator.linear.metadata.kind})"
    return operator.metadata.kind


def _report_case(label: str, captured_name: str, operator, input_rows, torch_outputs) -> None:
    request = PlanningRequest(
        batch_size=len(input_rows),
        calls=64,
        allow_approximate=False,
        sample_inputs=input_rows,
        codebook_sizes=(2,),
    )
    plan = plan_fixed_weight(operator, request)
    operator_outputs = operator.apply(input_rows)
    error = la.rms_relative_error(torch_outputs, operator_outputs)
    print(f"{label:<24} {captured_name:<12} {_operator_kind(operator):<18} {error:>10.3g} {plan.selected.name:<18}")


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        print("PyTorch is not installed in this environment, so the Torch coverage demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync")
        print("  mise exec -- uv run python examples/torch_coverage_demo.py")
        return 0

    class MatmulProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]))

        def forward(self, x):
            return x @ self.weight.T

    class AddmmProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [0.75, 1.25, -1.5]]))
            self.bias = nn.Parameter(torch.tensor([0.25, -0.5]))

        def forward(self, x):
            return torch.addmm(self.bias, x, self.weight.T)

    class LowRankProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Linear(6, 2, bias=False)
            self.up = nn.Linear(2, 4, bias=True)
            with torch.no_grad():
                self.down.weight.copy_(
                    torch.tensor([
                        [1.0, 0.0, -0.5, 0.25, 0.75, -1.0],
                        [0.2, -0.8, 0.4, 1.0, -0.3, 0.6],
                    ])
                )
                self.up.weight.copy_(
                    torch.tensor([
                        [0.9, -0.2],
                        [0.4, 0.7],
                        [-0.6, 0.3],
                        [0.1, 0.8],
                    ])
                )
                self.up.bias.copy_(torch.tensor([0.05, -0.1, 0.2, 0.0]))

        def forward(self, x):
            return self.up(self.down(x))

    class TinyConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel_size=3, bias=True)
            with torch.no_grad():
                self.conv.weight.copy_(torch.tensor([[[0.75, -0.5, 1.25]]]))
                self.conv.bias.copy_(torch.tensor([0.125]))

        def forward(self, x):
            return self.conv(x)

    print("Torch frontend coverage demo")
    print()
    print(f"{'pattern':<24} {'captured':<12} {'ir kind':<18} {'error':>10} {'lowering':<18}")
    print("-" * 86)

    dense_input = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
    matmul = MatmulProjection().eval()
    matmul_capture = capture_torch_fx_operators(matmul, sample_inputs=dense_input)["matmul"]
    _report_case("x @ weight.T", matmul_capture.name, matmul_capture.operator, dense_input.tolist(), matmul(dense_input).detach().tolist())

    addmm = AddmmProjection().eval()
    addmm_capture = capture_torch_fx_operators(addmm, sample_inputs=dense_input)["addmm"]
    _report_case("torch.addmm", addmm_capture.name, addmm_capture.operator, dense_input.tolist(), addmm(dense_input).detach().tolist())

    low_rank_input = torch.randn(4, 6, generator=torch.Generator().manual_seed(23))
    low_rank = LowRankProjection().eval()
    low_rank_capture = capture_torch_fx_operators(low_rank, sample_inputs=low_rank_input)["up"]
    _report_case("nested linear", low_rank_capture.name, low_rank_capture.operator, low_rank_input.tolist(), low_rank(low_rank_input).detach().tolist())

    conv_input = torch.randn(4, 1, 5, generator=torch.Generator().manual_seed(29))
    conv = TinyConv1d().eval()
    conv_capture = capture_torch_fx_operators(conv, sample_inputs=conv_input)["conv"]
    conv_rows = conv_input.squeeze(1).tolist()
    conv_outputs = conv(conv_input).detach().squeeze(1).tolist()
    _report_case("narrow Conv1d", conv_capture.name, conv_capture.operator, conv_rows, conv_outputs)

    print()
    print("Takeaway: coverage is now explicit and the dense matmul/addmm rows have executable capture checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
