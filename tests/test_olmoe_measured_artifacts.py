import collections
import json
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


if __name__ == "__main__":
    unittest.main()
