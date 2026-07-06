#!/usr/bin/env python3
"""Machine-readable workload case-study artifacts for adapter, Conv1d, and fixed-mask demos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.frontend import capture_torch_fx_linear_operators, capture_torch_fx_operators
from beyond_matmul.ir import AffineOperator, DenseOperator, FixedMaskOperator, Provenance
from beyond_matmul.planner import PlanOption, PlanningRequest, plan_fixed_weight


TIMING_PROXY_BOUNDARY = (
    "Case-study artifacts record planner cost and memory proxies, not benchmark timings; "
    "use benchmark artifacts for measured Python timing proxies."
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - exercised only without project deps
        raise RuntimeError("PyTorch is required for workload case-study artifacts") from exc
    return torch, nn, F


def _dense_operator_for(operator):
    dense = DenseOperator(operator.to_dense())
    if isinstance(operator, AffineOperator):
        return AffineOperator(dense, operator.bias)
    return dense


def _linear_kind(operator) -> str:
    if isinstance(operator, AffineOperator):
        return operator.linear.metadata.kind
    return operator.metadata.kind


def _plan_fields(option: PlanOption) -> Dict[str, Any]:
    return {
        "selected_lowering": option.name,
        "valid": option.valid,
        "exact": option.exact,
        "output_relative_error": option.relative_error,
        "amortized_cost": option.amortized_cost,
        "estimated_apply_cost": option.estimated_apply_cost,
        "estimated_preprocessing_cost": option.estimated_preprocessing_cost,
        "estimated_memory_bytes": option.estimated_memory_bytes,
        "memory_bytes_moved": option.cost.memory_bytes_moved,
        "preprocessing_ops": option.cost.preprocessing_ops,
        "amortized_preprocessing_ops": option.cost.amortized_preprocessing_ops,
        "requested_calls": option.requested_calls,
    }


def _dense_fallback_option(options: Sequence[PlanOption]) -> PlanOption:
    for option in options:
        if option.name in {"dense_gemm", "dense_gemm_bias"}:
            return option
    raise RuntimeError("planner did not produce a dense GEMM fallback option")


def _case_record(
    *,
    case: str,
    workload: str,
    title: str,
    captured_operator,
    input_rows: Sequence[Sequence[float]],
    torch_outputs: Sequence[Sequence[float]],
    provenance_label: str,
) -> Dict[str, Any]:
    operator = captured_operator.operator
    dense_operator = _dense_operator_for(operator)
    request = PlanningRequest(
        batch_size=len(input_rows),
        calls=128,
        allow_approximate=False,
        sample_inputs=input_rows,
    )
    structured_plan = plan_fixed_weight(operator, request)
    dense_plan = plan_fixed_weight(dense_operator, request)
    operator_outputs = operator.apply(input_rows)
    output_relative_error = la.rms_relative_error(torch_outputs, operator_outputs)
    selected = structured_plan.selected
    dense_fallback = _dense_fallback_option(dense_plan.options)

    return {
        "case": case,
        "workload": workload,
        "title": title,
        "captured_operator": {
            "name": captured_operator.name,
            "kind": operator.metadata.kind,
            "linear_kind": _linear_kind(operator),
            "shape": list(operator.shape),
            "provenance": operator.metadata.provenance.to_dict()
            if hasattr(operator.metadata.provenance, "to_dict")
            else {
                "source": operator.metadata.provenance.source,
                "framework": operator.metadata.provenance.framework,
                "expression": operator.metadata.provenance.expression,
                "inputs": list(operator.metadata.provenance.inputs),
                "transform_history": list(operator.metadata.provenance.transform_history),
                "confidence": operator.metadata.provenance.confidence,
            },
            "lowerings": list(operator.metadata.lowerings),
        },
        "provenance_notes": dict(captured_operator.event.notes),
        "provenance_label": provenance_label,
        "dense_fallback": _plan_fields(dense_fallback),
        "selected_lowering": selected.name,
        "output_relative_error": output_relative_error,
        "cost_proxy": {
            "amortized_cost": selected.amortized_cost,
            "estimated_apply_cost": selected.estimated_apply_cost,
            "estimated_preprocessing_cost": selected.estimated_preprocessing_cost,
            "requested_calls": selected.requested_calls,
        },
        "memory_proxy": {
            "estimated_memory_bytes": selected.estimated_memory_bytes,
            "memory_bytes_moved": selected.cost.memory_bytes_moved,
            "cache_bytes": selected.cost.cache_bytes,
        },
        "timing_proxy_boundary": {
            "measured_timing": False,
            "unit": "not_measured",
            "note": TIMING_PROXY_BOUNDARY,
        },
        "_structured_plan_summary": structured_plan.summary(),
        "_dense_plan_summary": dense_plan.summary(),
    }


def collect_adapter_case() -> Dict[str, Any]:
    torch, nn, F = _require_torch()

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

    operator = captured["lora_B"].operator
    inputs = torch.randn(8, operator.in_features, generator=torch.Generator().manual_seed(13))
    return _case_record(
        case="adapter_merged_lora",
        workload="adapter",
        title="Tiny adapter workload demo",
        captured_operator=captured["lora_B"],
        input_rows=inputs.tolist(),
        torch_outputs=module(inputs).detach().tolist(),
        provenance_label="adapter factors captured from named modules even though forward uses merged dense weight",
    )


def _conv1d_cases() -> Iterable[Dict[str, Any]]:
    torch, nn, F = _require_torch()

    weight = torch.tensor([
        [[0.75, -0.5, 1.25], [0.1, -1.0, 0.6]],
        [[1.5, -0.2, 0.35], [-0.8, 0.95, -0.4]],
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

    class GroupedModuleConv1d(nn.Module):
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

    def run_case(case: str, title: str, module: nn.Module, inputs, capture_name: str | None = None) -> Dict[str, Any]:
        captured = capture_torch_fx_operators(module, sample_inputs=inputs)
        if capture_name is None:
            captured_operator = next(
                (item for item in captured.values() if item.event.notes.get("capture") == "conv1d_function"),
                None,
            )
        else:
            captured_operator = captured.get(capture_name)
        if captured_operator is None:
            raise RuntimeError(f"no Conv1d operator was captured for {title}")

        return _case_record(
            case=case,
            workload="conv1d",
            title=title,
            captured_operator=captured_operator,
            input_rows=inputs.flatten(1).tolist(),
            torch_outputs=module(inputs).detach().flatten(1).tolist(),
            provenance_label="Conv1d provenance preserved before valid convolution is flattened to a dense matrix",
        )

    inputs = torch.randn(8, 2, 12, generator=torch.Generator().manual_seed(19))
    grouped_inputs = torch.randn(8, 4, 12, generator=torch.Generator().manual_seed(21))
    depthwise_inputs = torch.randn(8, 3, 12, generator=torch.Generator().manual_seed(23))
    yield run_case("conv1d_module", "Multi-channel nn.Conv1d module", ModuleConv1d().eval(), inputs, capture_name="conv")
    yield run_case("conv1d_functional_bias", "Functional F.conv1d with fixed bias", FunctionalConv1d().eval(), inputs)
    yield run_case(
        "conv1d_grouped_module",
        "Grouped nn.Conv1d module with fixed bias",
        GroupedModuleConv1d().eval(),
        grouped_inputs,
        capture_name="conv",
    )
    yield run_case(
        "conv1d_depthwise_functional",
        "Depthwise functional F.conv1d",
        DepthwiseFunctionalConv1d().eval(),
        depthwise_inputs,
    )


def collect_conv1d_cases() -> List[Dict[str, Any]]:
    return list(_conv1d_cases())


def collect_fixed_mask_case() -> Dict[str, Any]:
    operator = FixedMaskOperator(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        pattern="causal_band",
        provenance=Provenance(
            source="fixed_mask_case_study",
            expression="constant band mask applied as a sparse linear map over values/features",
            inputs=("features",),
            transform_history=("mask_preserved_as_linear_operator",),
        ),
    )
    inputs = la.random_batch(8, operator.in_features, seed=31)
    captured_operator = SimpleNamespace(
        name="fixed_band_mask",
        operator=operator,
        event=SimpleNamespace(
            notes={
                "capture": "fixed_mask_literal",
                "mask_pattern": "causal_band",
                "scope": "fixed mask applied independent of attention scores",
            }
        ),
    )
    return _case_record(
        case="fixed_band_mask",
        workload="fixed_mask",
        title="Fixed band mask linear demo",
        captured_operator=captured_operator,
        input_rows=inputs,
        torch_outputs=operator.apply(inputs),
        provenance_label="fixed mask provenance preserves sparse linear structure before dense fallback materialization",
    )


def collect_results() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "workload_case_studies",
        "metadata": {
            "timing_unit": "not_measured",
            "timing_proxy_boundary": TIMING_PROXY_BOUNDARY,
        },
        "cases": [collect_adapter_case(), *collect_conv1d_cases(), collect_fixed_mask_case()],
    }


def write_json_artifact(output_path: str | os.PathLike[str]) -> Dict[str, Any]:
    artifact = _public_artifact(collect_results())
    _write_json(output_path, artifact)
    return artifact


def _write_json(output_path: str | os.PathLike[str], artifact: Dict[str, Any]) -> None:
    path = os.fspath(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(_public_artifact(artifact), output, indent=2, sort_keys=True)
        output.write("\n")


def _public_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(artifact)
    cleaned["cases"] = [{key: value for key, value in case.items() if not key.startswith("_")} for case in artifact["cases"]]
    return cleaned


def _print_adapter_case(case: Dict[str, Any]) -> None:
    print("Tiny adapter workload demo")
    print()
    print(f"captured: {case['captured_operator']['name']}")
    print(
        f"operator: kind={case['captured_operator']['kind']}, "
        f"shape={tuple(case['captured_operator']['shape'])}"
    )
    print(f"capture notes: {case['provenance_notes']}")
    print()
    print(f"planner with adapter factors: {case['_structured_plan_summary']}")
    print(f"planner after dense merge: {case['_dense_plan_summary']}")
    print(f"operator output error vs torch module: {case['output_relative_error']:.3g}")
    print()
    print("Takeaway: nearby adapter provenance recovers low-rank structure even when forward uses a merged dense weight.")


def _print_conv1d_case(case: Dict[str, Any]) -> None:
    print(case["title"])
    print(f"  captured: {case['captured_operator']['name']}")
    print(
        f"  operator: kind={case['captured_operator']['kind']}, "
        f"linear_kind={case['captured_operator']['linear_kind']}, "
        f"shape={tuple(case['captured_operator']['shape'])}"
    )
    print(f"  capture notes: {case['provenance_notes']}")
    print(f"  planner with convolution provenance: {case['_structured_plan_summary']}")
    print(f"  planner after dense materialization: {case['_dense_plan_summary']}")
    print(f"  operator output error vs torch: {case['output_relative_error']:.3g}")


def print_adapter_demo(case: Dict[str, Any] | None = None) -> None:
    _print_adapter_case(case or collect_adapter_case())


def print_conv1d_demo(cases: Sequence[Dict[str, Any]] | None = None) -> None:
    rows = list(cases or collect_conv1d_cases())
    print("Conv1d workload coverage demo")
    print()
    for index, case in enumerate(rows):
        if index:
            print()
        _print_conv1d_case(case)
    print()
    print("Takeaway: Conv1d provenance preserves direct channel-aware kernels before valid convolution is flattened to a dense block-Toeplitz matrix.")


def _print_table(artifact: Dict[str, Any]) -> None:
    cases = artifact["cases"]
    case_width = max([len("case"), *(len(case["case"]) for case in cases)])
    lowering_width = max([len("selected"), *(len(case["selected_lowering"]) for case in cases)])
    print(f"{'case':<{case_width}}  {'selected':<{lowering_width}}  rel_err  cost_proxy  memory_bytes")
    print(f"{'-' * case_width}  {'-' * lowering_width}  -------  ----------  ------------")
    for case in cases:
        print(
            f"{case['case']:<{case_width}}  "
            f"{case['selected_lowering']:<{lowering_width}}  "
            f"{case['output_relative_error']:7.3g}  "
            f"{case['cost_proxy']['amortized_cost']:10.1f}  "
            f"{case['memory_proxy']['estimated_memory_bytes']:12d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable case-study results to this JSON path")
    args = parser.parse_args()

    artifact = collect_results()
    _print_table(artifact)
    if args.json_output:
        _write_json(args.json_output, artifact)
        print(f"\nwrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
