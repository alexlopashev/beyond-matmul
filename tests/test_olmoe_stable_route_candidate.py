import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_candidate_module():
    module_name = "olmoe_stable_route_candidate"
    module_path = REPO_ROOT / "benchmarks" / "olmoe_stable_route_candidate.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _real_baseline_artifact():
    return json.loads(
        (REPO_ROOT / "docs" / "results" / "olmoe_stock_baseline.json").read_text(
            encoding="utf-8"
        )
    )


def _candidate_payload(candidate, baseline, *, ratio=0.88, correctness="passed"):
    rows = []
    for selected in baseline["best_stock_by_regime"]:
        baseline_seconds = selected["cuda_event_median_seconds"]
        candidate_seconds = baseline_seconds * ratio
        tokens = next(
            regime["tokens_per_timed_forward"]
            for regime in candidate.baseline.required_regimes()
            if regime["regime_id"] == selected["regime_id"]
        )
        rows.append(
            {
                "regime_id": selected["regime_id"],
                "status": "ok",
                "reason": None,
                "resolved_experts_backend": candidate.BACKEND_NAME,
                "stage_experts_backends": {
                    "setup": candidate.BACKEND_NAME,
                    "timed": candidate.BACKEND_NAME,
                },
                "correctness": {
                    "status": correctness,
                    "reference": "eager__uncompiled",
                    "max_abs_error": 0.0 if correctness == "passed" else 1.0,
                    "relative_l2_error": 0.0 if correctness == "passed" else 1.0,
                    "max_abs_tolerance": 0.125,
                    "relative_l2_tolerance": 0.01,
                    "reason": None if correctness == "passed" else "correctness_tolerance_exceeded",
                },
                "timing": {
                    "cuda_event_median_seconds": candidate_seconds,
                    "cuda_event_seconds": [candidate_seconds] * 20,
                    "wall_median_seconds": candidate_seconds,
                    "wall_seconds": [candidate_seconds] * 20,
                    "warmup_repetitions": 5,
                    "measured_repetitions": 20,
                },
                "throughput_tokens_per_second": tokens / candidate_seconds,
                "allocator": {"status": "measured_cuda_allocator"},
            }
        )
    return {
        "results": rows,
        "execution_audit": {
            "status": "observed",
            "backend": candidate.BACKEND_NAME,
            "calls": 400,
            "stable_route_plan_calls": 400,
            "eager_fallback_calls": 0,
            "route_plan_build_count": 400,
            "fallback_reasons": {},
        },
    }


class OlmoeStableRouteCandidateTests(unittest.TestCase):
    def test_smoke_has_all_eight_rows_and_cannot_make_a_performance_claim(self):
        candidate = _load_candidate_module()

        artifact = candidate.collect_results(
            mode="contract-smoke",
            generated_at_utc="2026-08-08T00:00:00Z",
            command=["python", "benchmarks/olmoe_stable_route_candidate.py", "--smoke"],
        )

        self.assertEqual(artifact["benchmark"], "olmoe_stable_route_candidate")
        self.assertEqual(len(artifact["results"]), 8)
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertFalse(artifact["summary"]["candidate_measurements_present"])
        self.assertEqual(artifact["summary"]["decision"], "pending")
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertEqual(
            artifact["measurement_contract"]["success_threshold"],
            "at_least_10_percent_latency_or_throughput_improvement_in_one_regime",
        )
        self.assertEqual(
            artifact["measurement_contract"]["regression_guard"],
            "no_more_than_5_percent_median_latency_regression_in_every_other_regime",
        )

    def test_real_candidate_binds_to_baseline_and_accepts_only_a_complete_qualified_win(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()

        artifact = candidate.collect_results(
            mode="real",
            baseline_artifact=baseline,
            baseline_artifact_path="docs/results/olmoe_stock_baseline.json",
            environment=baseline["environment"],
            run_candidate=lambda *_args, **_kwargs: _candidate_payload(candidate, baseline),
            warmup_repetitions=5,
            measured_repetitions=20,
            generated_at_utc="2026-08-08T00:00:00Z",
        )

        self.assertEqual(
            artifact["baseline_binding"]["sha256"],
            candidate.artifact_sha256(baseline),
        )
        self.assertTrue(artifact["summary"]["candidate_measurements_present"])
        self.assertTrue(artifact["summary"]["correctness_all_passed"])
        self.assertTrue(artifact["summary"]["success_threshold_met"])
        self.assertTrue(artifact["summary"]["regression_guard_met"])
        self.assertTrue(artifact["summary"]["fallback_free"])
        self.assertEqual(artifact["summary"]["decision"], "accept")
        self.assertEqual(
            artifact["summary"]["performance_claim"],
            "qualified_candidate_speedup",
        )
        self.assertTrue(all(row["comparison"]["eligible"] for row in artifact["results"]))

    def test_correctness_failure_is_ineligible_and_never_claims_speedup(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()

        artifact = candidate.collect_results(
            mode="real",
            baseline_artifact=baseline,
            environment=baseline["environment"],
            run_candidate=lambda *_args, **_kwargs: _candidate_payload(
                candidate, baseline, ratio=0.5, correctness="failed"
            ),
            warmup_repetitions=5,
            measured_repetitions=20,
        )

        self.assertFalse(artifact["summary"]["correctness_all_passed"])
        self.assertEqual(artifact["summary"]["decision"], "reject")
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertTrue(all(not row["comparison"]["eligible"] for row in artifact["results"]))

    def test_regression_guard_rejects_a_candidate_with_one_large_regression(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()
        payload = _candidate_payload(candidate, baseline, ratio=0.88)
        selected = baseline["best_stock_by_regime"][0]
        regressed_seconds = selected["cuda_event_median_seconds"] * 1.06
        payload["results"][0]["timing"]["cuda_event_median_seconds"] = regressed_seconds
        payload["results"][0]["timing"]["cuda_event_seconds"] = [
            regressed_seconds
        ] * 20

        artifact = candidate.collect_results(
            mode="real",
            baseline_artifact=baseline,
            environment=baseline["environment"],
            run_candidate=lambda *_args, **_kwargs: payload,
            warmup_repetitions=5,
            measured_repetitions=20,
        )

        self.assertFalse(artifact["summary"]["regression_guard_met"])
        self.assertEqual(artifact["summary"]["decision"], "reject")
        self.assertEqual(artifact["summary"]["performance_claim"], "none")

    def test_real_candidate_rejects_environment_drift_from_frozen_baseline(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()
        environment = copy.deepcopy(baseline["environment"])
        environment["gpu_uuid"] = "GPU-different"

        with self.assertRaisesRegex(RuntimeError, "environment_mismatch:gpu_uuid"):
            candidate.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=environment,
                run_candidate=lambda *_args, **_kwargs: {},
                warmup_repetitions=5,
                measured_repetitions=20,
            )

    def test_observed_eager_fallback_rejects_an_otherwise_fast_candidate(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()
        payload = _candidate_payload(candidate, baseline, ratio=0.5)
        payload["execution_audit"].update(
            {
                "stable_route_plan_calls": 399,
                "eager_fallback_calls": 1,
                "route_plan_build_count": 399,
                "fallback_reasons": {"training_mode_not_supported": 1},
            }
        )

        artifact = candidate.collect_results(
            mode="real",
            baseline_artifact=baseline,
            environment=baseline["environment"],
            run_candidate=lambda *_args, **_kwargs: payload,
            warmup_repetitions=5,
            measured_repetitions=20,
        )

        self.assertFalse(artifact["summary"]["fallback_free"])
        self.assertEqual(artifact["summary"]["decision"], "reject")
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertIn(
            "candidate_fallback_observed",
            artifact["summary"]["readiness_blockers"],
        )

    def test_real_mode_requires_the_frozen_five_warmups_and_twenty_samples(self):
        candidate = _load_candidate_module()
        baseline = _real_baseline_artifact()

        with self.assertRaisesRegex(ValueError, "five warmups and 20 measured repetitions"):
            candidate.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=baseline["environment"],
                run_candidate=lambda *_args, **_kwargs: {},
                warmup_repetitions=1,
                measured_repetitions=2,
            )


if __name__ == "__main__":
    unittest.main()
