import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "fixed_weight.py"
    spec = importlib.util.spec_from_file_location("fixed_weight_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedWeightBenchmarkArtifactTests(unittest.TestCase):
    def test_writes_stable_json_artifact_schema(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "fixed_weight.json"
            artifact = benchmark.write_json_artifact(
                output_path,
                repeats=1,
                time_apply=lambda _operator, _inputs, _repeats: 0.25,
            )

            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)
        self.assertEqual(loaded["schema_version"], 1)
        self.assertEqual(loaded["benchmark"], "fixed_weight")
        self.assertEqual(loaded["metadata"]["timing_unit"], "seconds_per_apply")
        self.assertEqual(loaded["metadata"]["repeats"], 1)
        self.assertEqual(loaded["request"]["batch_size"], 32)
        self.assertEqual(loaded["request"]["calls"], 32)

        required_case_fields = {
            "case",
            "selected_lowering",
            "valid",
            "exact",
            "estimated_cost",
            "relative_error",
            "dense_seconds_per_apply",
            "chosen_seconds_per_apply",
            "estimated_apply_cost",
            "estimated_preprocessing_cost",
            "estimated_memory_bytes",
            "memory_bytes_moved",
            "preprocessing_ops",
            "requested_calls",
        }
        cases = loaded["cases"]
        self.assertGreaterEqual(len(cases), 5)
        self.assertTrue(required_case_fields.issubset(cases[0]))
        for case in cases:
            self.assertIsInstance(case["dense_seconds_per_apply"], numbers.Real)
            self.assertIsInstance(case["chosen_seconds_per_apply"], numbers.Real)
            self.assertGreaterEqual(case["dense_seconds_per_apply"], 0.0)
            self.assertGreaterEqual(case["chosen_seconds_per_apply"], 0.0)


if __name__ == "__main__":
    unittest.main()
