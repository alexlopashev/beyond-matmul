import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_ablation_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "planner_contract_ablation.py"
    spec = importlib.util.spec_from_file_location("planner_contract_ablation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PlannerContractAblationTests(unittest.TestCase):
    def test_collects_contract_scenarios_with_dense_fallbacks(self):
        ablation = _load_ablation_module()

        first = ablation.collect_results()
        second = ablation.collect_results()

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["benchmark"], "planner_contract_ablation")
        self.assertEqual(
            {scenario["scenario"] for scenario in first["scenarios"]},
            {"exact_vs_bounded_error", "reuse_sensitivity", "backend_support_sensitivity"},
        )

        for scenario in first["scenarios"]:
            self.assertTrue(scenario["dense_fallback_valid"], scenario["scenario"])
            self.assertGreaterEqual(scenario["dense_fallback_cost"], 0.0)
            self.assertIsInstance(scenario["selected_lowering"], str)
            self.assertIsInstance(scenario["selected_relative_error"], numbers.Real)
            self.assertGreaterEqual(scenario["selected_relative_error"], 0.0)

    def test_exact_only_and_bounded_error_share_fixed_weight_case(self):
        ablation = _load_ablation_module()

        artifact = ablation.collect_results()
        scenario = next(row for row in artifact["scenarios"] if row["scenario"] == "exact_vs_bounded_error")

        self.assertEqual(scenario["case"], "rank_one_plus_small_noise")
        self.assertEqual(scenario["exact_only"]["selected_lowering"], "dense_gemm")
        self.assertEqual(scenario["bounded_error"]["selected_lowering"], "low_rank_product")
        self.assertEqual(scenario["bounded_error"]["max_relative_error"], 0.1)
        self.assertLessEqual(
            scenario["bounded_error"]["selected_relative_error"],
            scenario["bounded_error"]["max_relative_error"],
        )
        self.assertTrue(scenario["exact_only"]["dense_fallback_valid"])
        self.assertTrue(scenario["bounded_error"]["dense_fallback_valid"])

    def test_reuse_and_backend_sensitivity_record_rejections(self):
        ablation = _load_ablation_module()

        artifact = ablation.collect_results()
        reuse = next(row for row in artifact["scenarios"] if row["scenario"] == "reuse_sensitivity")
        backend = next(row for row in artifact["scenarios"] if row["scenario"] == "backend_support_sensitivity")

        self.assertEqual(reuse["before_amortization"]["calls"], 7)
        self.assertFalse(reuse["before_amortization"]["target_lowering_valid"])
        self.assertTrue(
            any("preprocessing does not amortize" in reason for reason in reuse["before_amortization"]["target_reasons"])
        )
        self.assertEqual(reuse["after_amortization"]["calls"], 8)
        self.assertTrue(reuse["after_amortization"]["target_lowering_valid"])
        self.assertTrue(reuse["before_amortization"]["dense_fallback_valid"])
        self.assertTrue(reuse["after_amortization"]["dense_fallback_valid"])

        self.assertEqual(backend["backend"], "gpu")
        self.assertEqual(backend["target_lowering"], "codebook_kernel")
        self.assertFalse(backend["target_lowering_valid"])
        self.assertIn("backend does not support lowering", backend["target_reasons"])
        self.assertTrue(backend["dense_fallback_valid"])

    def test_writes_json_artifact(self):
        ablation = _load_ablation_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "planner_contract_ablation.json"
            artifact = ablation.write_json_artifact(output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)


if __name__ == "__main__":
    unittest.main()
