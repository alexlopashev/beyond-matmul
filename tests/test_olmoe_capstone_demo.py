import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_demo_module():
    path = REPO_ROOT / "examples" / "olmoe_capstone_demo.py"
    spec = importlib.util.spec_from_file_location("olmoe_capstone_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OlmoeCapstoneDemoTests(unittest.TestCase):
    def test_demo_recomputes_and_presents_the_committed_h100_result(self):
        demo = _load_demo_module()

        result = demo.run_demo()

        self.assertEqual(result["demo"], "olmoe_capstone")
        self.assertEqual(result["mode"], "offline_committed_evidence")
        self.assertEqual(len(result["results"]), 8)
        self.assertEqual(
            {row["regime_id"] for row in result["results"]},
            demo.REQUIRED_REGIME_IDS,
        )
        self.assertAlmostEqual(
            result["conclusion"]["minimum_latency_improvement_percent"],
            19.255236004116036,
        )
        self.assertAlmostEqual(
            result["conclusion"]["maximum_latency_improvement_percent"],
            63.302320025346326,
        )
        self.assertTrue(result["conclusion"]["correctness_all_passed"])
        self.assertTrue(result["conclusion"]["fallback_free"])
        self.assertEqual(result["conclusion"]["decision"], "accept")
        self.assertEqual(
            result["conclusion"]["performance_claim"],
            "qualified_candidate_speedup",
        )
        self.assertEqual(result["execution_audit"]["stable_route_plan_calls"], 4800)
        self.assertEqual(result["execution_audit"]["eager_fallback_calls"], 0)
        self.assertIn("one H100 PCIe", result["boundary"])

        rendered = demo.render_demo(result)
        self.assertIn("1. The problem", rendered)
        self.assertIn("2. The changed execution", rendered)
        self.assertIn("prefill_b1_s128", rendered)
        self.assertIn("decode_b8_p512", rendered)
        self.assertIn("19.26%", rendered)
        self.assertIn("63.30%", rendered)
        self.assertIn("4,800 stable calls, 0 eager fallbacks", rendered)
        self.assertIn("Qualified conclusion", rendered)
        self.assertIn("What this does not prove", rendered)

    def test_demo_rejects_a_tampered_candidate_artifact(self):
        demo = _load_demo_module()
        source = REPO_ROOT / "docs/results/olmoe_stable_route_candidate.json"
        artifact = json.loads(source.read_text(encoding="utf-8"))
        artifact["results"][0]["timing"]["cuda_event_seconds"][0] *= 2

        with tempfile.TemporaryDirectory() as temp_dir:
            candidate_path = Path(temp_dir) / "candidate.json"
            candidate_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate artifact SHA-256"):
                demo.run_demo(candidate_path=candidate_path)


if __name__ == "__main__":
    unittest.main()
