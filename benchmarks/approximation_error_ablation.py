#!/usr/bin/env python3
"""Approximation error ablation for matrix-vs-output scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from beyond_matmul import _linalg as la
from beyond_matmul.planner import PlanOption, PlanningRequest, plan_fixed_weight


def reference_matrix() -> la.Matrix:
    """Small dense matrix with a dominant feature omitted by the sample inputs."""

    return [
        [20.0, 1.0, 0.2, -0.1],
        [-18.0, 0.3, 1.0, 0.1],
        [16.0, -0.2, 0.4, 1.0],
        [-14.0, 1.0, -1.0, 0.5],
    ]


def sample_inputs() -> la.Matrix:
    return [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, -1.0, 0.5],
    ]


def _request(inputs: Sequence[Sequence[float]]) -> PlanningRequest:
    return PlanningRequest(
        batch_size=len(inputs),
        calls=32,
        max_relative_error=0.25,
        allow_approximate=True,
        sample_inputs=inputs,
        low_rank_ranks=(1,),
        sparse_densities=(0.25,),
        codebook_sizes=(2,),
    )


def _candidate_kind(option: PlanOption) -> str | None:
    kind = option.operator.metadata.kind
    source = option.operator.metadata.provenance.source
    if kind == "low_rank" and source == "planner_low_rank":
        return "low_rank"
    if kind == "sparse_coo" and source == "planner_sparse_topk":
        return "sparse_topk"
    if kind == "codebook" and source == "planner_codebook":
        return "codebook"
    if kind == "bitpacked_binary" and source == "planner_bitpacked_binary":
        return "bitpacked"
    return None


def _parameters(option: PlanOption, candidate_kind: str) -> Dict[str, Any]:
    structure = option.operator.metadata.structure
    if candidate_kind == "low_rank":
        return {"rank": int(structure["rank"])}
    if candidate_kind == "sparse_topk":
        return {"density": float(structure["density"]), "nnz": int(structure["nnz"])}
    if candidate_kind == "codebook":
        return {"codebook_size": int(structure["codebook_size"])}
    if candidate_kind == "bitpacked":
        return {"values": list(structure["values"]), "scale": float(structure["scale"])}
    raise ValueError(f"unsupported candidate kind: {candidate_kind}")


def _decision(error: float, epsilon: float) -> str:
    return "accept" if error <= epsilon else "reject"


def _reason(option: PlanOption, output_decision: str, selected: bool) -> str:
    if output_decision == "reject":
        return "; ".join(option.reasons) if option.reasons else "output error exceeds request"
    if selected:
        return "accepted and selected by output-aware planner"
    return "accepted by output error but not lowest-cost valid option"


def _candidate_rows(reference: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    inputs = sample_inputs()
    request = _request(inputs)
    plan = plan_fixed_weight(reference, request)
    rows: List[Dict[str, Any]] = []
    for option in plan.options:
        candidate_kind = _candidate_kind(option)
        if candidate_kind is None:
            continue
        reconstruction_error = la.relative_frobenius_error(reference, option.operator.to_dense())
        output_error = option.relative_error
        matrix_decision = _decision(reconstruction_error, request.max_relative_error)
        output_decision = _decision(output_error, request.max_relative_error)
        selected = option is plan.selected
        rows.append(
            {
                "candidate_kind": candidate_kind,
                "parameters": _parameters(option, candidate_kind),
                "reconstruction_error": reconstruction_error,
                "output_error": output_error,
                "matrix_error_decision": matrix_decision,
                "output_error_decision": output_decision,
                "selected_lowering": option.name,
                "selected_by_output_aware_planner": selected,
                "reason": _reason(option, output_decision, selected),
            }
        )
    return rows


def _qualitative_difference(rows: Sequence[Dict[str, Any]]) -> str:
    divergent = [
        row["candidate_kind"]
        for row in rows
        if row["matrix_error_decision"] != row["output_error_decision"]
    ]
    if divergent:
        names = ", ".join(divergent)
        return f"matrix error would accept {names}, but output error rejects them on the sampled inputs"
    return "no qualitative matrix-vs-output decision difference in this small deterministic artifact"


def collect_results() -> Dict[str, Any]:
    matrix = reference_matrix()
    inputs = sample_inputs()
    request = _request(inputs)
    plan = plan_fixed_weight(matrix, request)
    rows = _candidate_rows(matrix)
    return {
        "schema_version": 1,
        "benchmark": "approximation_error_ablation",
        "case": "dominant_unused_feature",
        "description": (
            "A dense matrix has a dominant feature that representative sample inputs do not exercise; "
            "matrix-relative error can therefore accept candidates that output-relative error rejects."
        ),
        "request": {
            "batch_size": request.batch_size,
            "calls": request.calls,
            "max_relative_error": request.max_relative_error,
            "allow_approximate": request.allow_approximate,
            "backend": request.backend,
        },
        "sample_inputs": inputs,
        "output_aware_selected_lowering": plan.selected.name,
        "qualitative_difference": _qualitative_difference(rows),
        "candidates": rows,
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
    rows = artifact["candidates"]
    kind_width = max([len("candidate"), *(len(row["candidate_kind"]) for row in rows)])
    lowering_width = max([len("lowering"), *(len(row["selected_lowering"]) for row in rows)])
    print(
        f"{'candidate':<{kind_width}}  {'lowering':<{lowering_width}}  "
        "matrix_err  output_err  matrix  output  reason"
    )
    print(f"{'-' * kind_width}  {'-' * lowering_width}  ----------  ----------  ------  ------  ------")
    for row in rows:
        print(
            f"{row['candidate_kind']:<{kind_width}}  "
            f"{row['selected_lowering']:<{lowering_width}}  "
            f"{row['reconstruction_error']:10.3g}  "
            f"{row['output_error']:10.3g}  "
            f"{row['matrix_error_decision']:<6}  "
            f"{row['output_error_decision']:<6}  "
            f"{row['reason']}"
        )
    print(f"\nselected by output-aware planner: {artifact['output_aware_selected_lowering']}")
    print(f"qualitative difference: {artifact['qualitative_difference']}")


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
