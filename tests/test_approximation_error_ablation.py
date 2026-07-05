import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_ablation_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "approximation_error_ablation.py"
    spec = importlib.util.spec_from_file_location("approximation_error_ablation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ApproximationErrorAblationTests(unittest.TestCase):
    def test_collects_stable_schema_for_approximation_candidates(self):
        ablation = _load_ablation_module()

        first = ablation.collect_results()
        second = ablation.collect_results()

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["benchmark"], "approximation_error_ablation")
        self.assertEqual(first["case"], "dominant_unused_feature")
        self.assertEqual(first["request"]["max_relative_error"], 0.25)
        self.assertEqual(
            first["sample_inputs"],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, -1.0, 0.5],
            ],
        )

        required_fields = {
            "candidate_kind",
            "parameters",
            "reconstruction_error",
            "output_error",
            "matrix_error_decision",
            "output_error_decision",
            "selected_lowering",
            "selected_by_output_aware_planner",
            "reason",
        }
        rows = first["candidates"]
        self.assertEqual(
            {row["candidate_kind"] for row in rows},
            {"low_rank", "sparse_topk", "codebook", "bitpacked"},
        )
        for row in rows:
            self.assertTrue(required_fields.issubset(row))
            self.assertIsInstance(row["parameters"], dict)
            self.assertIsInstance(row["reconstruction_error"], numbers.Real)
            self.assertIsInstance(row["output_error"], numbers.Real)
            self.assertGreaterEqual(row["reconstruction_error"], 0.0)
            self.assertGreaterEqual(row["output_error"], 0.0)
            self.assertIn(row["matrix_error_decision"], {"accept", "reject"})
            self.assertIn(row["output_error_decision"], {"accept", "reject"})
            self.assertIsInstance(row["selected_by_output_aware_planner"], bool)

        low_rank = next(row for row in rows if row["candidate_kind"] == "low_rank")
        self.assertEqual(low_rank["parameters"], {"rank": 1})
        self.assertEqual(low_rank["matrix_error_decision"], "accept")
        self.assertEqual(low_rank["output_error_decision"], "reject")
        self.assertIn("matrix error would accept", first["qualitative_difference"])

    def test_writes_json_artifact(self):
        ablation = _load_ablation_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "approximation_error_ablation.json"
            artifact = ablation.write_json_artifact(output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)


if __name__ == "__main__":
    unittest.main()
