import collections
import hashlib
import json
import math
import statistics
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class OlmoeMeasuredArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(
            (REPO_ROOT / "docs/results/olmoe_stock_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        cls.profile = json.loads(
            (REPO_ROOT / "docs/results/olmoe_stock_profile.json").read_text(
                encoding="utf-8"
            )
        )
        cls.candidate_path = (
            REPO_ROOT / "docs/results/olmoe_stable_route_candidate.json"
        )
        cls.candidate = json.loads(cls.candidate_path.read_text(encoding="utf-8"))

    def test_real_baseline_is_complete_and_selects_only_correct_rows(self):
        artifact = self.baseline

        self.assertEqual(artifact["benchmark"], "olmoe_stock_baseline")
        self.assertEqual(artifact["mode"], "real")
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertTrue(artifact["summary"]["cohort_complete"])
        self.assertEqual(artifact["summary"]["readiness_blockers"], [])
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertFalse(artifact["summary"]["candidate_measurements_present"])
        self.assertEqual(artifact["measurement_contract"]["warmup_repetitions"], 5)
        self.assertEqual(artifact["measurement_contract"]["measured_repetitions"], 20)
        self.assertEqual(
            artifact["pins"]["dependency_versions"]["apache-tvm-ffi"],
            "0.1.13.post2",
        )

        inventory = artifact["configuration_inventory"]
        self.assertEqual(len(inventory), 36)
        self.assertEqual(
            collections.Counter(row["eligibility"] for row in inventory),
            {"required": 20, "excluded": 16},
        )

        results = artifact["results"]
        self.assertEqual(len(results), 288)
        self.assertEqual(
            collections.Counter(row["status"] for row in results),
            {"ok": 96, "failed": 64, "not_applicable": 128},
        )
        successful = [row for row in results if row["status"] == "ok"]
        self.assertEqual(
            collections.Counter(row["correctness"]["status"] for row in successful),
            {"passed": 8, "failed": 88},
        )

        best_rows = artifact["best_stock_by_regime"]
        self.assertEqual(len(best_rows), 8)
        self.assertEqual(
            {row["configuration_id"] for row in best_rows},
            {"eager__uncompiled"},
        )
        for best in best_rows:
            matching = [
                row
                for row in results
                if row["regime_id"] == best["regime_id"]
                and row["configuration_id"] == best["configuration_id"]
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["correctness"]["status"], "passed")

    def test_real_profile_is_complete_bound_and_conservative(self):
        artifact = self.profile

        self.assertEqual(artifact["benchmark"], "olmoe_stock_profile")
        self.assertEqual(artifact["mode"], "real")
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertTrue(artifact["summary"]["profile_complete"])
        self.assertEqual(artifact["summary"]["readiness_blockers"], [])
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertFalse(artifact["summary"]["candidate_measurements_present"])
        self.assertTrue(artifact["baseline_binding"]["cohort_complete"])

        profiles = artifact["full_model_profiles"]
        self.assertEqual(len(profiles), 8)
        self.assertEqual(
            {(row["regime_id"], row["configuration_id"]) for row in profiles},
            {
                (row["regime_id"], row["configuration_id"])
                for row in self.baseline["best_stock_by_regime"]
            },
        )
        for row in profiles:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(
                row["attribution"]["cupti_trace_status"],
                "cuda_kernel_events_present",
            )
            self.assertGreater(
                row["attribution"]["totals"]["self_device_time_us"], 0.0
            )

        diagnostic = artifact["expert_layer_diagnostic"]
        self.assertEqual(diagnostic["status"], "ok")
        self.assertEqual(diagnostic["layer_index"], 8)
        self.assertEqual(diagnostic["regime_id"], "prefill_b1_s512")
        self.assertEqual(diagnostic["correctness"]["status"], "passed")
        self.assertEqual(diagnostic["correctness"]["max_abs_error"], 0.0)
        self.assertEqual(diagnostic["correctness"]["relative_l2_error"], 0.0)

        attribution = diagnostic["attribution"]
        categories = {row["category"]: row for row in attribution["categories"]}
        self.assertAlmostEqual(
            categories["sorting_permutation"]["device_time_proportion"],
            0.294045,
            places=5,
        )
        self.assertAlmostEqual(
            categories["expert_contractions"]["device_time_proportion"],
            0.286341,
            places=5,
        )
        for field in ("self_cpu_time_us", "self_device_time_us"):
            self.assertAlmostEqual(
                sum(row[field] for row in attribution["categories"]),
                attribution["totals"][field],
                places=6,
            )

    def test_real_candidate_is_source_bound_complete_correct_and_accepted(self):
        artifact = self.candidate

        self.assertEqual(artifact["benchmark"], "olmoe_stable_route_candidate")
        self.assertEqual(artifact["mode"], "real")
        self.assertEqual(
            hashlib.sha256(self.candidate_path.read_bytes()).hexdigest(),
            "ac462efc4127b5274379aa21c450137234eab049b0cd189503069d1e7d73299a",
        )
        self.assertEqual(
            artifact["baseline_binding"]["sha256"],
            "6b630ce7a174e0b29e21a3df2ab1358cf3b6c14dcf3d548c171eff228ba8436e",
        )
        implementation = artifact["implementation_binding"]
        self.assertEqual(implementation["status"], "bound")
        self.assertEqual(
            implementation["revision"],
            "34b6f14967cc5dc80f3d436e75d59c7bfae278f9",
        )
        self.assertFalse(implementation["dirty"])
        candidate_module = REPO_ROOT / "benchmarks/olmoe_stable_route_candidate.py"
        self.assertEqual(
            implementation["candidate_module_sha256"],
            hashlib.sha256(candidate_module.read_bytes()).hexdigest(),
        )

        results = artifact["results"]
        self.assertEqual(len(results), 8)
        self.assertEqual(
            {row["regime_id"] for row in results},
            {
                "prefill_b1_s128",
                "prefill_b1_s512",
                "prefill_b4_s128",
                "prefill_b4_s512",
                "decode_b1_p128",
                "decode_b1_p512",
                "decode_b8_p128",
                "decode_b8_p512",
            },
        )
        for row in results:
            samples = row["timing"]["cuda_event_seconds"]
            median = statistics.median(samples)
            timed_tokens = row["batch_size"] * (
                row["sequence_length"] if row["phase"] == "prefill" else 1
            )
            self.assertEqual(len(samples), 20)
            self.assertTrue(all(sample > 0.0 for sample in samples))
            self.assertTrue(
                math.isclose(
                    median,
                    row["timing"]["cuda_event_median_seconds"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            self.assertTrue(
                math.isclose(
                    timed_tokens / median,
                    row["throughput_tokens_per_second"],
                    rel_tol=1e-12,
                )
            )
            baseline_seconds = row["baseline"]["cuda_event_median_seconds"]
            improvement = (baseline_seconds - median) / baseline_seconds
            self.assertTrue(
                math.isclose(
                    improvement,
                    row["comparison"]["latency_improvement_fraction"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
            self.assertEqual(row["status"], "ok")
            self.assertTrue(row["measurement_contract_satisfied"])
            self.assertEqual(row["correctness"]["status"], "passed")
            self.assertGreaterEqual(improvement, 0.10)

        for key in (
            "gpu_uuid",
            "nvidia_driver_version",
            "cuda_runtime",
            "torch_version",
            "transformers_revision",
            "model_revision",
            "dtype",
        ):
            self.assertEqual(
                artifact["environment"][key], self.baseline["environment"][key]
            )

        audit = artifact["execution_audit"]
        self.assertEqual(audit["status"], "observed")
        self.assertEqual(audit["calls"], 4800)
        self.assertEqual(audit["stable_route_plan_calls"], 4800)
        self.assertEqual(audit["route_plan_build_count"], 4800)
        self.assertEqual(audit["eager_fallback_calls"], 0)
        self.assertEqual(audit["fallback_reasons"], {})

        summary = artifact["summary"]
        self.assertTrue(summary["row_inventory_complete"])
        self.assertTrue(summary["benchmark_complete"])
        self.assertTrue(summary["correctness_all_passed"])
        self.assertTrue(summary["success_threshold_met"])
        self.assertTrue(summary["regression_guard_met"])
        self.assertTrue(summary["fallback_free"])
        self.assertEqual(summary["decision"], "accept")
        self.assertEqual(
            summary["performance_claim"], "qualified_candidate_speedup"
        )
        self.assertEqual(summary["readiness_blockers"], [])


if __name__ == "__main__":
    unittest.main()
