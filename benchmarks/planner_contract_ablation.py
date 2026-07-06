#!/usr/bin/env python3
"""Planner contract ablation for exactness, reuse, backend, and dense fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.planner import LoweringPlan, PlanOption, PlanningRequest, plan_fixed_weight


def reference_matrix() -> la.Matrix:
    """Nearly rank-one dense matrix with small residual terms."""

    return [
        [2.0, 4.0, 6.0, 8.0],
        [1.02, 2.01, 2.99, 4.01],
        [-1.0, -2.0, -3.0, -4.0],
        [0.49, 1.0, 1.51, 2.02],
    ]


def sample_inputs() -> la.Matrix:
    return la.random_batch(8, 4, seed=3)


def codebook_matrix() -> la.Matrix:
    return [
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ]


def _dense_fallback(plan: LoweringPlan) -> PlanOption:
    for option in plan.options:
        if option.name in {"dense_gemm", "dense_gemm_bias"}:
            return option
    raise RuntimeError("planner did not produce a dense GEMM fallback option")


def _option_by_lowering(plan: LoweringPlan, lowering: str) -> PlanOption:
    for option in plan.options:
        if option.name == lowering:
            return option
    raise RuntimeError(f"planner did not produce lowering: {lowering}")


def _plan_fields(plan: LoweringPlan) -> Dict[str, Any]:
    dense = _dense_fallback(plan)
    return {
        "selected_lowering": plan.selected.name,
        "selected_valid": plan.selected.valid,
        "selected_relative_error": plan.selected.relative_error,
        "selected_cost": plan.selected.amortized_cost,
        "dense_fallback_valid": dense.valid,
        "dense_fallback_cost": dense.amortized_cost,
    }


def _request_fields(request: PlanningRequest) -> Dict[str, Any]:
    return {
        "batch_size": request.batch_size,
        "calls": request.calls,
        "max_relative_error": request.max_relative_error,
        "allow_approximate": request.allow_approximate,
        "backend": request.backend,
    }


def _target_fields(option: PlanOption) -> Dict[str, Any]:
    return {
        "target_lowering_valid": option.valid,
        "target_relative_error": option.relative_error,
        "target_cost": option.amortized_cost,
        "target_reasons": list(option.reasons),
    }


def exact_vs_bounded_error_scenario() -> Dict[str, Any]:
    matrix = reference_matrix()
    inputs = sample_inputs()
    exact_request = PlanningRequest(
        batch_size=len(inputs),
        calls=64,
        allow_approximate=False,
        sample_inputs=inputs,
        low_rank_ranks=(1,),
        sparse_densities=(0.25,),
        codebook_sizes=(2,),
    )
    bounded_request = PlanningRequest(
        batch_size=len(inputs),
        calls=64,
        max_relative_error=0.1,
        allow_approximate=True,
        sample_inputs=inputs,
        low_rank_ranks=(1,),
        sparse_densities=(0.25,),
        codebook_sizes=(2,),
    )
    exact_plan = plan_fixed_weight(matrix, exact_request)
    bounded_plan = plan_fixed_weight(matrix, bounded_request)
    bounded_fields = _plan_fields(bounded_plan)
    return {
        "scenario": "exact_vs_bounded_error",
        "case": "rank_one_plus_small_noise",
        "selected_lowering": bounded_fields["selected_lowering"],
        "selected_relative_error": bounded_fields["selected_relative_error"],
        "dense_fallback_valid": bounded_fields["dense_fallback_valid"],
        "dense_fallback_cost": bounded_fields["dense_fallback_cost"],
        "exact_only": {**_request_fields(exact_request), **_plan_fields(exact_plan)},
        "bounded_error": {**_request_fields(bounded_request), **bounded_fields},
    }


def reuse_sensitivity_scenario() -> Dict[str, Any]:
    matrix = reference_matrix()
    inputs = sample_inputs()
    before_request = PlanningRequest(
        batch_size=len(inputs),
        calls=7,
        max_relative_error=0.1,
        allow_approximate=True,
        sample_inputs=inputs,
        low_rank_ranks=(1,),
        sparse_densities=(0.25,),
        codebook_sizes=(2,),
    )
    after_request = PlanningRequest(
        batch_size=len(inputs),
        calls=8,
        max_relative_error=0.1,
        allow_approximate=True,
        sample_inputs=inputs,
        low_rank_ranks=(1,),
        sparse_densities=(0.25,),
        codebook_sizes=(2,),
    )
    before_plan = plan_fixed_weight(matrix, before_request)
    after_plan = plan_fixed_weight(matrix, after_request)
    before_target = _option_by_lowering(before_plan, "low_rank_product")
    after_target = _option_by_lowering(after_plan, "low_rank_product")
    after_fields = _plan_fields(after_plan)
    return {
        "scenario": "reuse_sensitivity",
        "case": "rank_one_plus_small_noise",
        "target_lowering": "low_rank_product",
        "selected_lowering": after_fields["selected_lowering"],
        "selected_relative_error": after_fields["selected_relative_error"],
        "dense_fallback_valid": after_fields["dense_fallback_valid"],
        "dense_fallback_cost": after_fields["dense_fallback_cost"],
        "before_amortization": {**_request_fields(before_request), **_plan_fields(before_plan), **_target_fields(before_target)},
        "after_amortization": {**_request_fields(after_request), **after_fields, **_target_fields(after_target)},
    }


def backend_support_scenario() -> Dict[str, Any]:
    request = PlanningRequest(batch_size=8, calls=16, backend="gpu", codebook_sizes=(2,))
    plan = plan_fixed_weight(codebook_matrix(), request)
    target = _option_by_lowering(plan, "codebook_kernel")
    fields = _plan_fields(plan)
    return {
        "scenario": "backend_support_sensitivity",
        "case": "exact_two-value_dense_matrix",
        "backend": request.backend,
        "target_lowering": "codebook_kernel",
        "selected_lowering": fields["selected_lowering"],
        "selected_relative_error": fields["selected_relative_error"],
        "dense_fallback_valid": fields["dense_fallback_valid"],
        "dense_fallback_cost": fields["dense_fallback_cost"],
        **_target_fields(target),
    }


def collect_results() -> Dict[str, Any]:
    scenarios = [
        exact_vs_bounded_error_scenario(),
        reuse_sensitivity_scenario(),
        backend_support_scenario(),
    ]
    return {
        "schema_version": 1,
        "benchmark": "planner_contract_ablation",
        "description": (
            "Small deterministic planner checks compare exact-only and bounded-error requests, "
            "reuse amortization thresholds, backend support, and dense GEMM fallback validity."
        ),
        "scenarios": scenarios,
    }


def write_json_artifact(output_path: str | os.PathLike[str]) -> Dict[str, Any]:
    artifact = collect_results()
    path = os.fspath(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(artifact, output, indent=2, sort_keys=True)
        output.write("\n")
    return artifact


def _print_table(artifact: Dict[str, Any]) -> None:
    print("scenario                     selected          rel_err   dense_valid  note")
    print("---------------------------  ----------------  --------  -----------  ----")
    for row in artifact["scenarios"]:
        note = row.get("target_lowering", row["case"])
        print(
            f"{row['scenario']:<27}  "
            f"{row['selected_lowering']:<16}  "
            f"{row['selected_relative_error']:8.3g}  "
            f"{str(row['dense_fallback_valid']):<11}  "
            f"{note}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable ablation results to this JSON path")
    args = parser.parse_args()

    artifact = collect_results()
    _print_table(artifact)
    if args.json_output:
        write_json_artifact(args.json_output)
        print(f"\nwrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
