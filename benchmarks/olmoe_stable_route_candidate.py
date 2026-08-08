#!/usr/bin/env python3
"""Measure the eager-compatible stable-route OLMoE candidate."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from beyond_matmul.olmoe_route_plan import (
    BACKEND_NAME,
    begin_execution_audit,
    end_execution_audit,
    register_transformers_backend,
)


def _load_sibling_module(filename: str, module_name: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = _load_sibling_module(
    "olmoe_stock_baseline.py", "_beyond_matmul_candidate_stock_baseline"
)
profile = _load_sibling_module(
    "olmoe_stock_profile.py", "_beyond_matmul_candidate_stock_profile"
)

BENCHMARK = "olmoe_stable_route_candidate"
CONTRACT_PATH = baseline.CONTRACT_PATH
MODEL = baseline.MODEL
MODEL_REVISION = baseline.MODEL_REVISION
TRANSFORMERS_REVISION = baseline.TRANSFORMERS_REVISION
DTYPE = baseline.DTYPE
DEFAULT_WARMUP_REPETITIONS = 5
DEFAULT_MEASURED_REPETITIONS = 20
MINIMUM_WIN_FRACTION = 0.10
MAXIMUM_REGRESSION_FRACTION = 0.05
ENVIRONMENT_BINDING_KEYS = profile.ENVIRONMENT_BINDING_KEYS

CandidateExecutor = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], int, int],
    Mapping[str, Any],
]


def candidate_configuration() -> Dict[str, Any]:
    return {
        "configuration_id": f"{BACKEND_NAME}__uncompiled",
        "experts_backend": BACKEND_NAME,
        "compiled": False,
        "compile_mode": None,
        "fullgraph": None,
        "eligibility": "required",
        "exclusion_reason": None,
    }


def collect_results(
    *,
    mode: str,
    baseline_artifact: Mapping[str, Any] | None = None,
    baseline_artifact_path: str | None = None,
    environment: Mapping[str, Any] | None = None,
    run_candidate: CandidateExecutor | None = None,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    if mode not in {"contract-smoke", "real"}:
        raise ValueError(f"unsupported mode: {mode}")
    if warmup_repetitions < 0 or measured_repetitions <= 0:
        raise ValueError("repetition counts must be non-negative warmups and positive measurements")

    regimes = baseline.required_regimes()
    configuration = candidate_configuration()
    generated_at = generated_at_utc or _utc_now()
    measurement_contract = _measurement_contract(
        warmup_repetitions, measured_repetitions
    )
    if mode == "contract-smoke":
        return {
            "schema_version": 1,
            "benchmark": BENCHMARK,
            "contract": CONTRACT_PATH,
            "mode": mode,
            "generated_at_utc": generated_at,
            "command": list(command or sys.argv),
            "pins": _pins(),
            "baseline_binding": {
                "status": "not_provided",
                "path": None,
                "sha256": None,
                "cohort_complete": False,
            },
            "measurement_contract": measurement_contract,
            "environment": _smoke_environment(),
            "candidate_configuration": configuration,
            "execution_audit": _empty_execution_audit(),
            "results": [_empty_result(regime) for regime in regimes],
            "summary": {
                "row_inventory_complete": True,
                "benchmark_complete": False,
                "candidate_measurements_present": False,
                "correctness_all_passed": False,
                "success_threshold_met": False,
                "regression_guard_met": False,
                "fallback_free": False,
                "decision": "pending",
                "performance_claim": "none",
                "readiness_blockers": ["contract_smoke_not_performance_evidence"],
            },
        }

    if (
        warmup_repetitions != DEFAULT_WARMUP_REPETITIONS
        or measured_repetitions != DEFAULT_MEASURED_REPETITIONS
    ):
        raise ValueError("real mode requires five warmups and 20 measured repetitions")
    if baseline_artifact is None:
        raise ValueError("real mode requires the complete stock baseline artifact")
    best_stock_rows = profile.validate_baseline_artifact(baseline_artifact)
    resolved_environment = dict(environment or baseline.probe_environment())
    environment_blockers = candidate_readiness_blockers(
        resolved_environment, baseline_artifact.get("environment", {})
    )
    if environment_blockers:
        raise RuntimeError(
            "candidate preflight blocked: " + ", ".join(environment_blockers)
        )

    executor = run_candidate or RealCandidateRunner()
    payload = dict(
        executor(
            configuration,
            regimes,
            warmup_repetitions,
            measured_repetitions,
        )
    )
    measured_rows = [dict(row) for row in payload.get("results", [])]
    execution_audit = dict(payload.get("execution_audit", _empty_execution_audit()))
    results = _bind_and_normalize_results(
        regimes,
        best_stock_rows,
        measured_rows,
        warmup_repetitions=warmup_repetitions,
        measured_repetitions=measured_repetitions,
    )

    expected_regimes = {regime["regime_id"] for regime in regimes}
    observed_regimes = {row["regime_id"] for row in results}
    row_inventory_complete = (
        len(results) == len(expected_regimes) and observed_regimes == expected_regimes
    )
    successful_rows = [row for row in results if row["status"] == "ok"]
    candidate_measurements_present = bool(successful_rows)
    measurements_complete = (
        row_inventory_complete
        and len(successful_rows) == len(expected_regimes)
        and all(row["measurement_contract_satisfied"] for row in results)
    )
    correctness_all_passed = measurements_complete and all(
        row["correctness"].get("status") == "passed" for row in results
    )
    success_threshold_met = correctness_all_passed and any(
        row["comparison"]["eligible"]
        and row["comparison"]["latency_improvement_fraction"]
        >= MINIMUM_WIN_FRACTION
        for row in results
    )
    regression_guard_met = correctness_all_passed and all(
        row["comparison"]["eligible"]
        and row["comparison"]["latency_regression_fraction"]
        <= MAXIMUM_REGRESSION_FRACTION
        for row in results
    )
    fallback_free = _fallback_free(execution_audit)
    benchmark_complete = measurements_complete and execution_audit.get("status") == "observed"
    accepted = (
        benchmark_complete
        and correctness_all_passed
        and success_threshold_met
        and regression_guard_met
        and fallback_free
    )
    if accepted:
        decision = "accept"
    elif benchmark_complete:
        decision = "reject"
    else:
        decision = "incomplete"

    readiness_blockers: List[str] = []
    if not measurements_complete:
        readiness_blockers.append("candidate_measurements_incomplete")
    if execution_audit.get("status") != "observed":
        readiness_blockers.append("candidate_execution_path_unobserved")
    if benchmark_complete and not correctness_all_passed:
        readiness_blockers.append("candidate_correctness_failed")
    if benchmark_complete and not fallback_free:
        readiness_blockers.append("candidate_fallback_observed")
    if benchmark_complete and correctness_all_passed and not success_threshold_met:
        readiness_blockers.append("candidate_success_threshold_not_met")
    if benchmark_complete and correctness_all_passed and not regression_guard_met:
        readiness_blockers.append("candidate_regression_guard_failed")

    return {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "contract": CONTRACT_PATH,
        "mode": mode,
        "generated_at_utc": generated_at,
        "command": list(command or sys.argv),
        "pins": _pins(),
        "baseline_binding": {
            "status": "validated_complete_stock_cohort",
            "path": baseline_artifact_path,
            "sha256": artifact_sha256(baseline_artifact),
            "cohort_complete": True,
        },
        "measurement_contract": measurement_contract,
        "environment": resolved_environment,
        "candidate_configuration": configuration,
        "execution_audit": execution_audit,
        "results": results,
        "summary": {
            "row_inventory_complete": row_inventory_complete,
            "benchmark_complete": benchmark_complete,
            "candidate_measurements_present": candidate_measurements_present,
            "correctness_all_passed": correctness_all_passed,
            "success_threshold_met": success_threshold_met,
            "regression_guard_met": regression_guard_met,
            "fallback_free": fallback_free,
            "decision": decision,
            "performance_claim": (
                "qualified_candidate_speedup" if accepted else "none"
            ),
            "readiness_blockers": readiness_blockers,
        },
    }


class RealCandidateRunner:
    """Reuse the stock runner's frozen inputs, timing, and eager reference."""

    def __call__(
        self,
        configuration: Mapping[str, Any],
        regimes: Sequence[Mapping[str, Any]],
        warmup_repetitions: int,
        measured_repetitions: int,
    ) -> Mapping[str, Any]:
        register_transformers_backend()
        begin_execution_audit()
        try:
            runner = baseline.RealConfigurationRunner(
                warmup_repetitions=warmup_repetitions,
                measured_repetitions=measured_repetitions,
            )
            rows = runner(configuration, regimes)
        finally:
            execution_audit = end_execution_audit()
        return {"results": list(rows), "execution_audit": execution_audit}


def candidate_readiness_blockers(
    environment: Mapping[str, Any],
    baseline_environment: Mapping[str, Any],
) -> List[str]:
    blockers = list(environment.get("readiness_blockers", []))
    if environment.get("preflight_status") != "ready":
        blockers.append("stock_preflight_not_ready")
    if environment.get("cuda_available") is not True:
        blockers.append("cuda_unavailable")
    for key in ENVIRONMENT_BINDING_KEYS:
        expected = baseline_environment.get(key)
        observed = environment.get(key)
        if expected is not None and observed != expected:
            blockers.append(f"environment_mismatch:{key}")
    return _deduplicate(blockers)


def artifact_sha256(artifact: Mapping[str, Any]) -> str:
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _bind_and_normalize_results(
    regimes: Sequence[Mapping[str, Any]],
    best_stock_rows: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    *,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> List[Dict[str, Any]]:
    by_regime: Dict[str, Mapping[str, Any]] = {}
    for row in measured_rows:
        regime_id = str(row.get("regime_id"))
        if regime_id in by_regime:
            raise ValueError(f"duplicate candidate regime: {regime_id}")
        by_regime[regime_id] = row
    best_by_regime = {str(row["regime_id"]): row for row in best_stock_rows}

    results: List[Dict[str, Any]] = []
    for regime in regimes:
        regime_id = str(regime["regime_id"])
        measured = by_regime.get(regime_id)
        if measured is None:
            results.append(_empty_result(regime, reason="candidate_missing_required_regime"))
            continue
        best = best_by_regime[regime_id]
        row = dict(measured)
        timing = dict(row.get("timing", {}))
        samples = timing.get("cuda_event_seconds")
        contract_satisfied = (
            row.get("status") == "ok"
            and timing.get("warmup_repetitions") == warmup_repetitions
            and timing.get("measured_repetitions") == measured_repetitions
            and isinstance(samples, list)
            and len(samples) == measured_repetitions
            and all(_finite_positive(value) for value in samples)
        )
        candidate_median = None
        if contract_satisfied:
            candidate_median = float(statistics.median(float(value) for value in samples))
            recorded_median = timing.get("cuda_event_median_seconds")
            if not _close_number(recorded_median, candidate_median):
                raise ValueError(
                    f"candidate median does not match samples for {regime_id}"
                )
            timing["cuda_event_median_seconds"] = candidate_median
            row["timing"] = timing

        baseline_median = float(best["cuda_event_median_seconds"])
        correctness_passed = row.get("correctness", {}).get("status") == "passed"
        eligible = bool(
            contract_satisfied
            and correctness_passed
            and candidate_median is not None
            and _finite_positive(baseline_median)
        )
        if eligible:
            latency_improvement = (baseline_median - candidate_median) / baseline_median
            latency_regression = (candidate_median - baseline_median) / baseline_median
            throughput_improvement = baseline_median / candidate_median - 1.0
        else:
            latency_improvement = None
            latency_regression = None
            throughput_improvement = None

        row.update(
            {
                "phase": regime["phase"],
                "batch_size": regime["batch_size"],
                "sequence_length": regime["sequence_length"],
                "candidate_configuration_id": f"{BACKEND_NAME}__uncompiled",
                "measurement_contract_satisfied": contract_satisfied,
                "baseline": {
                    "configuration_id": best["configuration_id"],
                    "experts_backend": best["experts_backend"],
                    "compiled": best.get("compiled"),
                    "compile_mode": best.get("compile_mode"),
                    "cuda_event_median_seconds": baseline_median,
                    "throughput_tokens_per_second": best.get(
                        "throughput_tokens_per_second"
                    ),
                },
                "comparison": {
                    "eligible": eligible,
                    "latency_improvement_fraction": latency_improvement,
                    "throughput_improvement_fraction": throughput_improvement,
                    "latency_regression_fraction": latency_regression,
                },
            }
        )
        results.append(row)

    unexpected = set(by_regime) - {str(regime["regime_id"]) for regime in regimes}
    if unexpected:
        raise ValueError(
            "unexpected candidate regimes: " + ", ".join(sorted(unexpected))
        )
    return results


def _empty_result(
    regime: Mapping[str, Any], *, reason: str = "contract_smoke_no_execution"
) -> Dict[str, Any]:
    return {
        "regime_id": regime["regime_id"],
        "phase": regime["phase"],
        "batch_size": regime["batch_size"],
        "sequence_length": regime["sequence_length"],
        "candidate_configuration_id": f"{BACKEND_NAME}__uncompiled",
        "status": "not_measured",
        "reason": reason,
        "measurement_contract_satisfied": False,
        "correctness": {
            "status": "not_measured",
            "reference": "eager__uncompiled",
            "max_abs_error": None,
            "relative_l2_error": None,
            "max_abs_tolerance": baseline.CORRECTNESS_MAX_ABS_TOLERANCE,
            "relative_l2_tolerance": baseline.CORRECTNESS_RELATIVE_L2_TOLERANCE,
        },
        "timing": {
            "cuda_event_median_seconds": None,
            "cuda_event_seconds": [],
            "warmup_repetitions": None,
            "measured_repetitions": None,
        },
        "baseline": None,
        "comparison": {
            "eligible": False,
            "latency_improvement_fraction": None,
            "throughput_improvement_fraction": None,
            "latency_regression_fraction": None,
        },
    }


def _measurement_contract(
    warmup_repetitions: int, measured_repetitions: int
) -> Dict[str, Any]:
    return {
        "warmup_repetitions": warmup_repetitions,
        "measured_repetitions": measured_repetitions,
        "primary_timing": "cuda_event_median_seconds",
        "reference": "eager__uncompiled_outputs_and_frozen_best_stock_timings",
        "correctness_first": True,
        "success_threshold": (
            "at_least_10_percent_latency_or_throughput_improvement_in_one_regime"
        ),
        "minimum_win_fraction": MINIMUM_WIN_FRACTION,
        "regression_guard": (
            "no_more_than_5_percent_median_latency_regression_in_every_other_regime"
        ),
        "maximum_regression_fraction": MAXIMUM_REGRESSION_FRACTION,
    }


def _pins() -> Dict[str, Any]:
    return {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "dtype": DTYPE,
        "input_seed": baseline.INPUT_SEED,
        "candidate_backend": BACKEND_NAME,
    }


def _smoke_environment() -> Dict[str, Any]:
    return {
        "preflight_status": "not_run",
        "cuda_available": None,
    }


def _empty_execution_audit() -> Dict[str, Any]:
    return {
        "status": "not_observed",
        "backend": BACKEND_NAME,
        "calls": 0,
        "stable_route_plan_calls": 0,
        "eager_fallback_calls": 0,
        "route_plan_build_count": 0,
        "fallback_reasons": {},
    }


def _fallback_free(audit: Mapping[str, Any]) -> bool:
    calls = audit.get("calls")
    return bool(
        audit.get("status") == "observed"
        and isinstance(calls, int)
        and calls > 0
        and audit.get("stable_route_plan_calls") == calls
        and audit.get("eager_fallback_calls") == 0
        and audit.get("route_plan_build_count") == calls
        and not audit.get("fallback_reasons")
    )


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _close_number(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and math.isclose(
        float(left), right, rel_tol=1e-12, abs_tol=1e-15
    )


def _deduplicate(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json_artifact(
    path: str | os.PathLike[str], **collect_kwargs: Any
) -> Dict[str, Any]:
    artifact = collect_results(**collect_kwargs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--real", action="store_true")
    parser.add_argument(
        "--baseline-artifact", default="docs/results/olmoe_stock_baseline.json"
    )
    parser.add_argument(
        "--warmup-repetitions", type=int, default=DEFAULT_WARMUP_REPETITIONS
    )
    parser.add_argument(
        "--measured-repetitions", type=int, default=DEFAULT_MEASURED_REPETITIONS
    )
    parser.add_argument("--json-output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "contract-smoke" if args.smoke else "real"
    baseline_artifact = None
    baseline_path = None
    if mode == "real":
        baseline_path = args.baseline_artifact
        baseline_artifact = json.loads(
            Path(baseline_path).read_text(encoding="utf-8")
        )
    artifact = write_json_artifact(
        args.json_output,
        mode=mode,
        baseline_artifact=baseline_artifact,
        baseline_artifact_path=baseline_path,
        warmup_repetitions=args.warmup_repetitions,
        measured_repetitions=args.measured_repetitions,
        command=sys.argv if argv is None else [sys.argv[0], *argv],
    )
    print(
        f"row_inventory_complete={str(artifact['summary']['row_inventory_complete']).lower()} "
        f"benchmark_complete={str(artifact['summary']['benchmark_complete']).lower()} "
        f"decision={artifact['summary']['decision']} "
        f"claim={artifact['summary']['performance_claim']}"
    )
    if mode == "contract-smoke":
        return 0
    return 0 if artifact["summary"]["decision"] == "accept" else 2


if __name__ == "__main__":
    raise SystemExit(main())
