import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_case_study_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "case_study_artifacts.py"
    spec = importlib.util.spec_from_file_location("case_study_artifacts", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CaseStudyArtifactTests(unittest.TestCase):
    def test_collects_adapter_and_conv1d_case_study_schema(self):
        artifacts = _load_case_study_module()

        artifact = artifacts.collect_results()

        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["artifact"], "workload_case_studies")
        self.assertEqual(
            {case["case"] for case in artifact["cases"]},
            {
                "adapter_merged_lora",
                "conv1d_module",
                "conv1d_functional_bias",
                "conv1d_grouped_module",
                "conv1d_depthwise_functional",
                "fixed_band_mask",
            },
        )
        self.assertEqual(artifact["metadata"]["timing_unit"], "not_measured")
        self.assertIn("not benchmark timings", artifact["metadata"]["timing_proxy_boundary"])

        required_fields = {
            "case",
            "captured_operator",
            "provenance_notes",
            "dense_fallback",
            "selected_lowering",
            "output_relative_error",
            "cost_proxy",
            "memory_proxy",
            "timing_proxy_boundary",
        }
        for case in artifact["cases"]:
            self.assertTrue(required_fields.issubset(case))
            self.assertIsInstance(case["provenance_notes"], dict)
            self.assertIsInstance(case["captured_operator"]["shape"], list)
            self.assertIsInstance(case["captured_operator"]["kind"], str)
            self.assertIsInstance(case["dense_fallback"]["selected_lowering"], str)
            self.assertIsInstance(case["output_relative_error"], numbers.Real)
            self.assertLess(case["output_relative_error"], 1e-6)
            self.assertIsInstance(case["cost_proxy"]["amortized_cost"], numbers.Real)
            self.assertGreaterEqual(case["cost_proxy"]["amortized_cost"], 0.0)
            self.assertIsInstance(case["memory_proxy"]["estimated_memory_bytes"], numbers.Integral)
            self.assertGreaterEqual(case["memory_proxy"]["estimated_memory_bytes"], 0)
            self.assertEqual(case["timing_proxy_boundary"]["measured_timing"], False)

        rows = {case["case"]: case for case in artifact["cases"]}
        self.assertEqual(rows["adapter_merged_lora"]["selected_lowering"], "low_rank_product_bias")
        self.assertEqual(rows["adapter_merged_lora"]["captured_operator"]["kind"], "affine")
        self.assertEqual(rows["adapter_merged_lora"]["captured_operator"]["linear_kind"], "low_rank")
        self.assertEqual(rows["adapter_merged_lora"]["provenance_notes"]["capture"], "named_adapter_pair")
        self.assertEqual(rows["adapter_merged_lora"]["dense_fallback"]["selected_lowering"], "dense_gemm_bias")
        self.assertEqual(rows["conv1d_module"]["selected_lowering"], "conv1d_channel_direct")
        self.assertEqual(rows["conv1d_module"]["captured_operator"]["linear_kind"], "conv1d_channel")
        self.assertEqual(rows["conv1d_module"]["dense_fallback"]["selected_lowering"], "dense_gemm")
        self.assertEqual(rows["conv1d_functional_bias"]["selected_lowering"], "conv1d_channel_direct_bias")
        self.assertEqual(rows["conv1d_functional_bias"]["captured_operator"]["kind"], "affine")
        self.assertEqual(rows["conv1d_functional_bias"]["dense_fallback"]["selected_lowering"], "dense_gemm_bias")
        self.assertEqual(rows["conv1d_grouped_module"]["selected_lowering"], "conv1d_grouped_direct_bias")
        self.assertEqual(rows["conv1d_grouped_module"]["captured_operator"]["linear_kind"], "conv1d_channel")
        self.assertEqual(rows["conv1d_grouped_module"]["provenance_notes"]["group_type"], "grouped")
        self.assertEqual(rows["conv1d_grouped_module"]["dense_fallback"]["selected_lowering"], "dense_gemm_bias")
        self.assertEqual(rows["conv1d_depthwise_functional"]["selected_lowering"], "conv1d_depthwise_direct")
        self.assertEqual(rows["conv1d_depthwise_functional"]["captured_operator"]["linear_kind"], "conv1d_channel")
        self.assertEqual(rows["conv1d_depthwise_functional"]["provenance_notes"]["group_type"], "depthwise")
        self.assertEqual(rows["conv1d_depthwise_functional"]["dense_fallback"]["selected_lowering"], "dense_gemm")
        self.assertEqual(rows["fixed_band_mask"]["selected_lowering"], "fixed_mask_sparse")
        self.assertEqual(rows["fixed_band_mask"]["captured_operator"]["linear_kind"], "fixed_mask")
        self.assertEqual(rows["fixed_band_mask"]["provenance_notes"]["mask_pattern"], "causal_band")
        self.assertEqual(rows["fixed_band_mask"]["dense_fallback"]["selected_lowering"], "dense_gemm")

    def test_writes_json_artifact(self):
        artifacts = _load_case_study_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "workload_case_studies.json"
            artifact = artifacts.write_json_artifact(output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)


if __name__ == "__main__":
    unittest.main()
