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
    print(f"{label:<26} {captured_name:<12} {_operator_kind(operator):<24} {error:>10.3g} {plan.selected.name:<24}")


def main() -> int:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        print("PyTorch is not installed in this environment, so the Torch coverage demo was skipped.")
        print("Install project dependencies and rerun:")
        print("  mise exec -- uv sync --locked")
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

    channel_weight = torch.tensor([
        [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
        [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
    ])
    grouped_weight = torch.tensor([
        [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
        [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
        [[0.5, -0.5, 1.5], [1.0, 0.0, -1.0]],
        [[-0.25, 0.5, 0.75], [2.0, -1.0, 0.25]],
    ])
    depthwise_weight = torch.tensor([
        [[1.0, 0.0, -1.0]],
        [[0.5, -0.5, 1.5]],
        [[2.0, 1.0, 0.25]],
    ])

    class MultiChannelConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(2, 2, kernel_size=3, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(channel_weight)

        def forward(self, x):
            return self.conv(x)

    class FunctionalConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", channel_weight)
            self.register_buffer("bias", torch.tensor([0.1, -0.2]))

        def forward(self, x):
            return F.conv1d(x, self.weight, self.bias)

    class GroupedConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(4, 4, kernel_size=3, groups=2, bias=True)
            with torch.no_grad():
                self.conv.weight.copy_(grouped_weight)
                self.conv.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))

        def forward(self, x):
            return self.conv(x)

    class DepthwiseFunctionalConv1d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", depthwise_weight)

        def forward(self, x):
            return F.conv1d(x, self.weight, groups=3)

    print("Torch frontend coverage demo")
    print()
    print(f"{'pattern':<26} {'captured':<12} {'ir kind':<24} {'error':>10} {'lowering':<24}")
    print("-" * 102)

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

    channel_input = torch.randn(4, 2, 5, generator=torch.Generator().manual_seed(31))
    channel_conv = MultiChannelConv1d().eval()
    channel_capture = capture_torch_fx_operators(channel_conv, sample_inputs=channel_input)["conv"]
    channel_rows = channel_input.flatten(1).tolist()
    channel_outputs = channel_conv(channel_input).detach().flatten(1).tolist()
    _report_case("multi-channel Conv1d", channel_capture.name, channel_capture.operator, channel_rows, channel_outputs)

    functional_conv = FunctionalConv1d().eval()
    functional_captures = capture_torch_fx_operators(functional_conv, sample_inputs=channel_input)
    functional_capture = next(item for item in functional_captures.values() if item.event.notes.get("capture") == "conv1d_function")
    functional_outputs = functional_conv(channel_input).detach().flatten(1).tolist()
    _report_case("functional Conv1d", functional_capture.name, functional_capture.operator, channel_rows, functional_outputs)

    grouped_input = torch.randn(4, 4, 5, generator=torch.Generator().manual_seed(37))
    grouped_conv = GroupedConv1d().eval()
    grouped_capture = capture_torch_fx_operators(grouped_conv, sample_inputs=grouped_input)["conv"]
    grouped_rows = grouped_input.flatten(1).tolist()
    grouped_outputs = grouped_conv(grouped_input).detach().flatten(1).tolist()
    _report_case("grouped Conv1d", grouped_capture.name, grouped_capture.operator, grouped_rows, grouped_outputs)

    depthwise_input = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(41))
    depthwise_conv = DepthwiseFunctionalConv1d().eval()
    depthwise_captures = capture_torch_fx_operators(depthwise_conv, sample_inputs=depthwise_input)
    depthwise_capture = next(item for item in depthwise_captures.values() if item.event.notes.get("capture") == "conv1d_function")
    depthwise_rows = depthwise_input.flatten(1).tolist()
    depthwise_outputs = depthwise_conv(depthwise_input).detach().flatten(1).tolist()
    _report_case("depthwise Conv1d", depthwise_capture.name, depthwise_capture.operator, depthwise_rows, depthwise_outputs)

    print()
    print("Takeaway: coverage is explicit across dense matmul/addmm, low-rank, and tested Conv1d module/function rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
