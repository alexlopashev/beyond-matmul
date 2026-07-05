#!/usr/bin/env python3
"""Synthetic fixed-weight benchmark for planner sanity checks."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.ir import DiagonalOperator, LowRankOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


TimeApply = Callable[[object, Sequence[Sequence[float]], int], float]


def _time_apply(operator, inputs, repeats: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        operator.apply(inputs)
    end = time.perf_counter()
    return (end - start) / repeats


def _sparse_matrix(rows: int, cols: int) -> la.Matrix:
    matrix = la.zeros(rows, cols)
    for row in range(rows):
        matrix[row][(row * 7) % cols] = 1.0 + (row % 5) * 0.1
    return matrix


def _codebook_matrix(rows: int, cols: int) -> la.Matrix:
    values = [-1.0, -0.25, 0.0, 0.5]
    return [[values[(row * 3 + col * 5) % len(values)] for col in range(cols)] for row in range(rows)]


def cases() -> List[Tuple[str, object]]:
    left = la.random_matrix(64, 4, seed=3)
    right = la.random_matrix(4, 64, seed=4)
    return [
        ("diagonal", DiagonalOperator([1.0 + index * 0.01 for index in range(64)])),
        ("sparse", _sparse_matrix(64, 64)),
        ("low_rank", LowRankOperator(left, right)),
        ("codebook", _codebook_matrix(64, 64)),
        ("dense_random", la.random_matrix(64, 64, seed=5)),
    ]


def _benchmark_request(inputs: Sequence[Sequence[float]]) -> PlanningRequest:
    return PlanningRequest(
        batch_size=32,
        calls=32,
        max_relative_error=0.05,
        allow_approximate=True,
        sample_inputs=inputs,
    )


def collect_results(repeats: int = 20, time_apply: TimeApply = _time_apply) -> Dict[str, Any]:
    inputs = la.random_batch(32, 64, seed=12)
    request = _benchmark_request(inputs)
    case_results = []
    for name, weight in cases():
        plan = plan_fixed_weight(weight, request)
        dense_op = next(option.operator for option in plan.options if option.name == "dense_gemm")
        dense_time = time_apply(dense_op, inputs, repeats)
        chosen_time = time_apply(plan.selected.operator, inputs, repeats)
        selected = plan.selected
        case_results.append(
            {
                "case": name,
                "selected_lowering": selected.name,
                "valid": selected.valid,
                "exact": selected.exact,
                "estimated_cost": selected.amortized_cost,
                "relative_error": selected.relative_error,
                "dense_seconds_per_apply": dense_time,
                "chosen_seconds_per_apply": chosen_time,
                "estimated_apply_cost": selected.estimated_apply_cost,
                "estimated_preprocessing_cost": selected.estimated_preprocessing_cost,
                "estimated_memory_bytes": selected.estimated_memory_bytes,
                "memory_bytes_moved": selected.cost.memory_bytes_moved,
                "preprocessing_ops": selected.cost.preprocessing_ops,
                "amortized_preprocessing_ops": selected.cost.amortized_preprocessing_ops,
                "requested_calls": selected.requested_calls,
            }
        )

    return {
        "schema_version": 1,
        "benchmark": "fixed_weight",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "repeats": repeats,
            "timing_unit": "seconds_per_apply",
        },
        "request": {
            "batch_size": request.batch_size,
            "calls": request.calls,
            "max_relative_error": request.max_relative_error,
            "allow_approximate": request.allow_approximate,
            "backend": request.backend,
        },
        "cases": case_results,
    }


def write_json_artifact(
    output_path: str | os.PathLike[str],
    repeats: int = 20,
    time_apply: TimeApply = _time_apply,
) -> Dict[str, Any]:
    artifact = collect_results(repeats=repeats, time_apply=time_apply)
    _write_json(output_path, artifact)
    return artifact


def _write_json(output_path: str | os.PathLike[str], artifact: Dict[str, Any]) -> None:
    path = os.fspath(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(artifact, output, indent=2, sort_keys=True)
        output.write("\n")


def _print_table(artifact: Dict[str, Any]) -> None:
    print("case          selected           valid  est_cost    rel_err    dense_s    chosen_s")
    print("------------  -----------------  -----  ---------  --------  --------  --------")
    for result in artifact["cases"]:
        print(
            f"{result['case']:<12}  {result['selected_lowering']:<17}  "
            f"{str(result['valid']):<5}  "
            f"{result['estimated_cost']:9.1f}  "
            f"{result['relative_error']:8.3g}  "
            f"{result['dense_seconds_per_apply']:8.5f}  "
            f"{result['chosen_seconds_per_apply']:8.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable benchmark results to this JSON path")
    parser.add_argument("--repeats", type=int, default=20, help="timing repeats per case, default: 20")
    args = parser.parse_args()

    artifact = collect_results(repeats=args.repeats)
    _print_table(artifact)
    if args.json_output:
        _write_json(args.json_output, artifact)
        print(f"\nwrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
