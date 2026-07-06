import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from beyond_matmul.analyzer import analyze_dense


def _load_demo_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "fixed_weight_inference_demo.py"
    spec = importlib.util.spec_from_file_location("fixed_weight_inference_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedWeightDemoTests(unittest.TestCase):
    def test_recovery_candidate_table_exposes_validation_fields(self):
        demo = _load_demo_module()
        matrix = [[2.0, 4.0], [1.0, 2.0]]
        candidates = analyze_dense(matrix, ranks=(1,), sample_inputs=[[1.0, 0.0], [0.0, 1.0]])

        output = io.StringIO()
        with redirect_stdout(output):
            demo._print_candidates(candidates)

        text = output.getvalue()
        self.assertIn("candidate", text)
        self.assertIn("confidence", text)
        self.assertIn("exact", text)
        self.assertIn("validation", text)
        self.assertIn("output_relative_l2", text)


if __name__ == "__main__":
    unittest.main()
