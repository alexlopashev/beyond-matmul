#!/usr/bin/env python3
"""Offline, integrity-checked presentation of the measured OLMoE result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ARTIFACT = REPO_ROOT / "docs/results/olmoe_stock_baseline.json"
DEFAULT_CANDIDATE_ARTIFACT = (
    REPO_ROOT / "docs/results/olmoe_stable_route_candidate.json"
)
CANDIDATE_ARTIFACT_SHA256 = (
    "ac462efc4127b5274379aa21c450137234eab049b0cd189503069d1e7d73299a"
)
BASELINE_CANONICAL_SHA256 = (
    "6b630ce7a174e0b29e21a3df2ab1358cf3b6c14dcf3d548c171eff228ba8436e"
)
CANDIDATE_SOURCE_REVISION = "34b6f14967cc5dc80f3d436e75d59c7bfae278f9"
CANDIDATE_MODULE_SHA256 = (
    "359921a81a213eba35e3ab49661836b1ee94580cbdbb91639b14ae0dace95a91"
)
REQUIRED_REGIME_ORDER = (
    "prefill_b1_s128",
    "prefill_b1_s512",
    "prefill_b4_s128",
    "prefill_b4_s512",
    "decode_b1_p128",
    "decode_b1_p512",
    "decode_b8_p128",
    "decode_b8_p512",
)
REQUIRED_REGIME_IDS = set(REQUIRED_REGIME_ORDER)
ENVIRONMENT_BINDING_KEYS = (
    "gpu_uuid",
    "nvidia_driver_version",
    "cuda_runtime",
    "torch_version",
    "transformers_revision",
    "model_revision",
    "dtype",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}: {error}") from error
    _require(isinstance(value, dict), f"{name} must contain one JSON object")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_bindings(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_path: Path,
) -> None:
    actual_candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    _require(
        actual_candidate_sha == CANDIDATE_ARTIFACT_SHA256,
        "candidate artifact SHA-256 does not match the reviewed evidence",
    )
    actual_baseline_sha = _canonical_sha256(baseline)
    _require(
        actual_baseline_sha == BASELINE_CANONICAL_SHA256,
        "baseline canonical SHA-256 does not match the reviewed cohort",
    )
    baseline_binding = candidate.get("baseline_binding", {})
    _require(
        baseline_binding.get("sha256") == actual_baseline_sha,
        "candidate baseline binding does not match the loaded stock artifact",
    )
    implementation = candidate.get("implementation_binding", {})
    _require(
        implementation.get("status") == "bound",
        "candidate source revision is not bound",
    )
    _require(
        implementation.get("revision") == CANDIDATE_SOURCE_REVISION,
        "candidate source revision does not match the reviewed implementation",
    )
    _require(
        implementation.get("dirty") is False,
        "candidate source tree was dirty during measurement",
    )
    _require(
        implementation.get("candidate_module_sha256")
        == CANDIDATE_MODULE_SHA256,
        "candidate module checksum does not match the reviewed implementation",
    )
    candidate_module = REPO_ROOT / "benchmarks/olmoe_stable_route_candidate.py"
    _require(
        hashlib.sha256(candidate_module.read_bytes()).hexdigest()
        == CANDIDATE_MODULE_SHA256,
        "current candidate harness differs from the measured module",
    )
    for key in ENVIRONMENT_BINDING_KEYS:
        _require(
            candidate.get("environment", {}).get(key)
            == baseline.get("environment", {}).get(key),
            f"candidate environment differs from stock for {key}",
        )


def _best_stock_rows(baseline: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    _require(
        baseline.get("benchmark") == "olmoe_stock_baseline"
        and baseline.get("mode") == "real",
        "stock artifact is not the real OLMoE baseline",
    )
    summary = baseline.get("summary", {})
    _require(
        summary.get("row_inventory_complete") is True
        and summary.get("cohort_complete") is True,
        "stock cohort is incomplete",
    )
    rows = baseline.get("best_stock_by_regime", [])
    _require(isinstance(rows, list), "stock best-row inventory is missing")
    by_regime = {str(row.get("regime_id")): row for row in rows}
    _require(
        len(rows) == len(by_regime) == 8
        and set(by_regime) == REQUIRED_REGIME_IDS,
        "stock best-row inventory does not contain the eight required regimes",
    )
    _require(
        all(row.get("configuration_id") == "eager__uncompiled" for row in rows),
        "stock best-row inventory is not the reviewed eager cohort",
    )
    return by_regime


def _recompute_results(
    candidate: Mapping[str, Any],
    best_stock: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(
        candidate.get("benchmark") == "olmoe_stable_route_candidate"
        and candidate.get("mode") == "real",
        "candidate artifact is not the real stable-route result",
    )
    contract = candidate.get("measurement_contract", {})
    _require(
        contract.get("warmup_repetitions") == 5
        and contract.get("measured_repetitions") == 20,
        "candidate repetition contract changed",
    )
    _require(
        contract.get("minimum_win_fraction") == 0.10
        and contract.get("maximum_regression_fraction") == 0.05,
        "candidate decision thresholds changed",
    )
    rows = candidate.get("results", [])
    _require(isinstance(rows, list), "candidate result inventory is missing")
    by_regime = {str(row.get("regime_id")): row for row in rows}
    _require(
        len(rows) == len(by_regime) == 8
        and set(by_regime) == REQUIRED_REGIME_IDS,
        "candidate result inventory does not contain the eight required regimes",
    )

    presented: list[dict[str, Any]] = []
    for regime_id in REQUIRED_REGIME_ORDER:
        row = by_regime[regime_id]
        samples = row.get("timing", {}).get("cuda_event_seconds", [])
        _require(
            isinstance(samples, list)
            and len(samples) == 20
            and all(isinstance(sample, (int, float)) and sample > 0 for sample in samples),
            f"{regime_id}: expected 20 positive CUDA-event samples",
        )
        candidate_seconds = statistics.median(samples)
        recorded_seconds = row.get("timing", {}).get("cuda_event_median_seconds")
        _require(
            isinstance(recorded_seconds, (int, float))
            and math.isclose(
                candidate_seconds,
                recorded_seconds,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{regime_id}: recorded candidate median does not match raw samples",
        )
        phase = row.get("phase")
        timed_tokens = int(row.get("batch_size", 0)) * (
            int(row.get("sequence_length", 0)) if phase == "prefill" else 1
        )
        throughput = timed_tokens / candidate_seconds
        _require(
            math.isclose(
                throughput,
                float(row.get("throughput_tokens_per_second", 0.0)),
                rel_tol=1e-12,
            ),
            f"{regime_id}: candidate throughput does not match raw timing",
        )

        stock = best_stock[regime_id]
        embedded_stock = row.get("baseline", {})
        stock_seconds = float(stock.get("cuda_event_median_seconds", 0.0))
        _require(stock_seconds > 0, f"{regime_id}: stock median is invalid")
        _require(
            embedded_stock.get("configuration_id") == stock.get("configuration_id")
            and math.isclose(
                float(embedded_stock.get("cuda_event_median_seconds", 0.0)),
                stock_seconds,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{regime_id}: embedded stock comparison is not the reviewed best row",
        )
        improvement = (stock_seconds - candidate_seconds) / stock_seconds
        comparison = row.get("comparison", {})
        _require(
            comparison.get("eligible") is True
            and math.isclose(
                float(comparison.get("latency_improvement_fraction", 0.0)),
                improvement,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(comparison.get("latency_regression_fraction", 0.0)),
                -improvement,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{regime_id}: comparison summary does not match raw medians",
        )
        correctness = row.get("correctness", {})
        _require(
            row.get("status") == "ok"
            and row.get("measurement_contract_satisfied") is True
            and correctness.get("status") == "passed"
            and float(correctness.get("max_abs_error", math.inf))
            <= float(correctness.get("max_abs_tolerance", -math.inf))
            and float(correctness.get("relative_l2_error", math.inf))
            <= float(correctness.get("relative_l2_tolerance", -math.inf)),
            f"{regime_id}: correctness or measurement contract failed",
        )
        presented.append(
            {
                "regime_id": regime_id,
                "stock_median_ms": stock_seconds * 1000.0,
                "candidate_median_ms": candidate_seconds * 1000.0,
                "latency_improvement_percent": improvement * 100.0,
                "candidate_tokens_per_second": throughput,
                "correctness": "passed",
            }
        )
    return presented


def _validate_audit(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = candidate.get("execution_audit", {})
    _require(audit.get("status") == "observed", "execution audit is not observed")
    calls = audit.get("calls")
    _require(
        calls == 4800
        and audit.get("stable_route_plan_calls") == calls
        and audit.get("route_plan_build_count") == calls
        and audit.get("eager_fallback_calls") == 0
        and audit.get("fallback_reasons") == {},
        "execution audit does not prove one stable plan per call and zero fallback",
    )
    return audit


def run_demo(
    *,
    baseline_path: Path = DEFAULT_BASELINE_ARTIFACT,
    candidate_path: Path = DEFAULT_CANDIDATE_ARTIFACT,
) -> dict[str, Any]:
    """Validate and summarize committed GPU evidence without executing a model."""

    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    baseline = _load_json(baseline_path, "stock artifact")
    candidate = _load_json(candidate_path, "candidate artifact")
    _validate_bindings(baseline, candidate, candidate_path)
    results = _recompute_results(candidate, _best_stock_rows(baseline))
    audit = _validate_audit(candidate)

    improvements = [row["latency_improvement_percent"] for row in results]
    correctness_all_passed = all(row["correctness"] == "passed" for row in results)
    success_threshold_met = max(improvements) >= 10.0
    regression_guard_met = min(improvements) >= -5.0
    fallback_free = audit["eager_fallback_calls"] == 0
    accepted = (
        correctness_all_passed
        and success_threshold_met
        and regression_guard_met
        and fallback_free
    )
    summary = candidate.get("summary", {})
    _require(
        accepted
        and summary.get("decision") == "accept"
        and summary.get("performance_claim") == "qualified_candidate_speedup"
        and summary.get("correctness_all_passed") is correctness_all_passed
        and summary.get("success_threshold_met") is success_threshold_met
        and summary.get("regression_guard_met") is regression_guard_met
        and summary.get("fallback_free") is fallback_free,
        "candidate conclusion does not match recomputed gates",
    )

    return {
        "demo": "olmoe_capstone",
        "mode": "offline_committed_evidence",
        "story": {
            "problem": (
                "Stock eager repeatedly discovers routed token membership around "
                "the expert contractions."
            ),
            "change": (
                "Preserve expert, selected-slot, token, and routing-weight provenance; "
                "build one stable route plan and reuse it through projection, gating, "
                "weighting, and aggregation."
            ),
            "fallback": (
                "Unsupported execution remains explicit and correct through stock eager."
            ),
        },
        "evidence_binding": {
            "candidate_artifact_sha256": CANDIDATE_ARTIFACT_SHA256,
            "baseline_canonical_sha256": BASELINE_CANONICAL_SHA256,
            "candidate_source_revision": CANDIDATE_SOURCE_REVISION,
            "candidate_module_sha256": CANDIDATE_MODULE_SHA256,
        },
        "results": results,
        "execution_audit": dict(audit),
        "conclusion": {
            "decision": "accept",
            "performance_claim": "qualified_candidate_speedup",
            "correctness_all_passed": correctness_all_passed,
            "fallback_free": fallback_free,
            "minimum_latency_improvement_percent": min(improvements),
            "maximum_latency_improvement_percent": max(improvements),
        },
        "boundary": (
            "one H100 PCIe GPU, one OLMoE revision, and one pinned software cohort"
        ),
        "not_proven": [
            "other GPUs, models, sequence regimes, or future library revisions",
            "compiled-candidate gains or memory savings",
            "production readiness or upstream acceptance",
        ],
    }


def render_demo(result: Mapping[str, Any]) -> str:
    lines = [
        "Beyond Matmul: measured OLMoE capstone",
        "",
        "1. The problem",
        str(result["story"]["problem"]),
        "",
        "2. The changed execution",
        str(result["story"]["change"]),
        f"Fallback: {result['story']['fallback']}",
        "",
        "3. The eight H100 races (recomputed from committed raw samples)",
        "regime              stock ms  route ms  faster  correctness",
        "------------------  --------  --------  ------  -----------",
    ]
    for row in result["results"]:
        lines.append(
            f"{row['regime_id']:<18}  "
            f"{row['stock_median_ms']:>8.3f}  "
            f"{row['candidate_median_ms']:>8.3f}  "
            f"{row['latency_improvement_percent']:>5.2f}%  "
            f"{row['correctness']}"
        )
    audit = result["execution_audit"]
    conclusion = result["conclusion"]
    lines.extend(
        [
            "",
            "4. Safety checks",
            (
                f"{audit['stable_route_plan_calls']:,} stable calls, "
                f"{audit['eager_fallback_calls']:,} eager fallbacks; "
                "all eight correctness gates passed."
            ),
            "Artifact, baseline, source revision, module, medians, throughput, and "
            "decision math were checked before this table was shown.",
            "",
            "5. Qualified conclusion",
            (
                "Qualified conclusion: median latency was "
                f"{conclusion['minimum_latency_improvement_percent']:.2f}% to "
                f"{conclusion['maximum_latency_improvement_percent']:.2f}% lower "
                "than the frozen best-correct-stock rows."
            ),
            f"Boundary: {result['boundary']}.",
            "",
            "6. What this does not prove",
        ]
    )
    lines.extend(f"- {item}" for item in result["not_proven"])
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT,
    )
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        default=DEFAULT_CANDIDATE_ARTIFACT,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the validated summary as JSON instead of the human demo",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_demo(
            baseline_path=args.baseline_artifact,
            candidate_path=args.candidate_artifact,
        )
    except ValueError as error:
        print(f"evidence validation failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_demo(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
