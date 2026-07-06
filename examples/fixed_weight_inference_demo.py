#!/usr/bin/env python3
"""End-to-end demo for provenance-aware fixed-weight inference planning."""

from __future__ import annotations

import os
import sys
import time
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.analyzer import analyze_dense
from beyond_matmul.frontend import ProvenanceTracer
from beyond_matmul.ir import DenseOperator
from beyond_matmul.planner import LoweringPlan, PlanningRequest, plan_fixed_weight


def _time_apply(operator, inputs, repeats: int = 50) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        operator.apply(inputs)
    return (time.perf_counter() - start) / repeats


def _dense_option(plan: LoweringPlan):
    return next(option for option in plan.options if option.name == "dense_gemm")


def _print_options(plan: LoweringPlan, limit: int = 8) -> None:
    print("lowering             valid  exact  rel_error   cost      memory  notes")
    print("-------------------  -----  -----  ---------  --------  ------  ------------------------------")
    for option in sorted(plan.options, key=lambda item: (not item.valid, item.amortized_cost, item.relative_error))[:limit]:
        notes = ", ".join(option.reasons) if option.reasons else "ok"
        print(
            f"{option.name:<19}  "
            f"{str(option.valid):<5}  "
            f"{str(option.exact):<5}  "
            f"{option.relative_error:9.3g}  "
            f"{option.amortized_cost:8.1f}  "
            f"{option.estimated_memory_bytes:6d}  "
            f"{notes}"
        )


def _print_candidates(candidates: Iterable) -> None:
    print("candidate                         confidence  exact  validation                  evidence")
    print("--------------------------------  ----------  -----  --------------------------  ----------------------------------------")
    for candidate in list(candidates)[:6]:
        evidence_items = dict(candidate.evidence)
        validation = evidence_items.pop("validation", None)
        if validation is None:
            validation_text = "not_sample_validated"
        else:
            validation_text = (
                f"{validation['metric']}={validation['output_relative_error']:.3g}, "
                f"n={validation['sample_count']}, exact={validation['exact_on_samples']}, "
                f"bound={validation['confidence_bound']:.3g}"
            )
        evidence = ", ".join(f"{key}={value}" for key, value in evidence_items.items())
        print(
            f"{candidate.kind:<32}  "
            f"{candidate.confidence:10.3f}  "
            f"{str(candidate.exact):<5}  "
            f"{validation_text:<26}  "
            f"{evidence}"
        )


def main() -> None:
    print("Beyond Matmul: fixed-weight inference demo")
    print()

    tracer = ProvenanceTracer(framework="demo")

    left = la.random_matrix(64, 4, seed=3)
    right = la.random_matrix(4, 64, seed=4)
    structured = tracer.low_rank("adapter.proj", left, right, expression="LoRA-style U @ V fixed projection")
    dense_weight = structured.to_dense()
    inputs = la.random_batch(batch_size=16, width=structured.in_features, seed=11)

    request = PlanningRequest(
        batch_size=len(inputs),
        calls=64,
        allow_approximate=True,
        max_relative_error=0.02,
        sample_inputs=inputs,
        low_rank_ranks=(1, 2, 4, 8),
        sparse_densities=(0.2, 0.4),
        codebook_sizes=(2, 4, 8),
    )

    print("1. Capture provenance before densification")
    event = tracer.events[0]
    print(f"captured: name={event.name}, op_type={event.op_type}, expression={event.provenance.expression}")
    print(f"operator: kind={structured.metadata.kind}, shape={structured.shape}, rank={structured.rank}")
    print()

    print("2. Plan with provenance preserved")
    preserved_plan = plan_fixed_weight(structured, request)
    print(f"selected: {preserved_plan.summary()}")
    _print_options(preserved_plan)
    print()

    print("3. Lose provenance by materializing a dense matrix")
    anonymous_dense = DenseOperator(dense_weight)
    dense_plan = plan_fixed_weight(anonymous_dense, request)
    print(f"dense fallback is available, but recovery candidates are now needed")
    print(f"selected after recovery: {dense_plan.summary()}")
    print()

    print("4. Recover cheap structure from dense weights")
    _print_candidates(analyze_dense(dense_weight, ranks=(1, 2, 4, 8), sample_inputs=inputs))
    print()

    print("5. Check output behavior and proxy timings")
    dense_option = _dense_option(preserved_plan)
    dense_outputs = dense_option.operator.apply(inputs)
    selected_outputs = preserved_plan.selected.operator.apply(inputs)
    rel_error = la.rms_relative_error(dense_outputs, selected_outputs)
    dense_seconds = _time_apply(dense_option.operator, inputs)
    selected_seconds = _time_apply(preserved_plan.selected.operator, inputs)
    recovered_seconds = _time_apply(dense_plan.selected.operator, inputs)
    print(f"preserved selected output error vs dense: {rel_error:.3g}")
    print()
    print("path                  selected           rel_error   seconds/apply")
    print("--------------------  -----------------  ---------  -------------")
    print(f"dense fallback         {dense_option.name:<17}  {dense_option.relative_error:9.3g}  {dense_seconds:13.6f}")
    print(
        f"provenance preserved   {preserved_plan.selected.name:<17}  "
        f"{preserved_plan.selected.relative_error:9.3g}  {selected_seconds:13.6f}"
    )
    print(
        f"provenance recovered   {dense_plan.selected.name:<17}  "
        f"{dense_plan.selected.relative_error:9.3g}  {recovered_seconds:13.6f}"
    )
    print()
    print("Takeaway: dense GEMM stays as a valid fallback, but it is no longer the only representation.")


if __name__ == "__main__":
    main()
