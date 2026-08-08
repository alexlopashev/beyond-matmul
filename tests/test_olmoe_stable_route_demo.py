import importlib.util
import unittest
from pathlib import Path


def _load_demo_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "olmoe_stable_route_demo.py"
    spec = importlib.util.spec_from_file_location("olmoe_stable_route_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OlmoeStableRouteDemoTests(unittest.TestCase):
    def test_demo_exercises_stable_plan_and_explicit_eager_fallback(self):
        demo = _load_demo_module()

        result = demo.run_demo()

        self.assertEqual(result["demo"], "olmoe_stable_route")
        self.assertTrue(result["correctness"]["exact_match"])
        self.assertEqual(result["correctness"]["max_abs_error"], 0.0)
        self.assertEqual(
            result["stable_execution"]["execution_path"], "stable_route_plan"
        )
        self.assertEqual(result["stable_execution"]["route_plan_build_count"], 1)
        self.assertEqual(
            result["stable_execution"]["order"],
            "expert_then_top_k_slot_then_token",
        )
        self.assertEqual(
            result["fallback_execution"]["execution_path"], "eager_fallback"
        )
        self.assertEqual(
            result["fallback_execution"]["fallback_reason"],
            "training_mode_not_supported",
        )
        self.assertEqual(result["performance_claim"], "none")


if __name__ == "__main__":
    unittest.main()
