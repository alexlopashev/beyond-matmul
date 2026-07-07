import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "peft_transformers_lora_inference.py"
    spec = importlib.util.spec_from_file_location("peft_transformers_lora_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PeftTransformersLoraInferenceTests(unittest.TestCase):
    def test_smoke_artifact_matches_contract_shape(self):
        benchmark = _load_benchmark_module()

        first = benchmark.collect_results(
            sequence_lengths=[4],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=2,
            mode="synthetic-smoke",
            time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01, 0.02],
        )
        second = benchmark.collect_results(
            sequence_lengths=[4],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=2,
            mode="synthetic-smoke",
            time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01, 0.02],
        )

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["benchmark"], "peft_transformers_lora_inference")
        self.assertEqual(first["contract"], "docs/peft_capstone_benchmark_contract.md")
        self.assertEqual(first["workload"]["base_model"], "hf-internal-testing/tiny-random-OPTForCausalLM")
        self.assertEqual(first["workload"]["adapter"], "peft-internal-testing/tiny-OPTForCausalLM-lora")
        self.assertEqual(first["workload"]["sequence_lengths"], [4])
        self.assertEqual(first["workload"]["batch_sizes"], [1])
        self.assertEqual(first["workload"]["warmup_repetitions"], 1)
        self.assertEqual(first["workload"]["measured_repetitions"], 2)
        self.assertEqual(first["workload"]["input_seed"], 20260707)
        self.assertEqual(first["dependencies"]["peft_upstream"]["repository"], "huggingface/peft")
        self.assertEqual(first["dependencies"]["peft_fork"]["repository"], "alexlopashev/peft")
        self.assertIn("python", first["dependencies"])
        self.assertIn("torch", first["dependencies"])
        self.assertIn("platform", first["environment"])

        results = first["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual(
            {row["baseline"] for row in results},
            {
                "upstream_peft_unmerged",
                "upstream_peft_merged_dense",
                "beyond_matmul_peft_fork",
            },
        )

        required_result_fields = {
            "case",
            "baseline",
            "status",
            "sequence_length",
            "batch_size",
            "latency_seconds",
            "peak_memory_bytes",
            "peak_memory_status",
            "adapter_switch_seconds",
            "adapter_switch_status",
            "correctness",
            "lowering",
        }
        for row in results:
            self.assertTrue(required_result_fields.issubset(row))
            self.assertEqual(row["case"], "seq4_batch1")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["peak_memory_bytes"], None)
            self.assertEqual(row["peak_memory_status"], "not_measurable_on_cpu")
            self.assertEqual(row["adapter_switch_seconds"], None)
            self.assertEqual(row["adapter_switch_status"], "not_measured_single_adapter")
            self.assertEqual(row["latency_seconds"]["median"], 0.015)
            self.assertEqual(row["latency_seconds"]["p50"], 0.015)
            self.assertEqual(row["latency_seconds"]["p90"], 0.019)
            self.assertEqual(row["latency_seconds"]["p95"], 0.0195)
            self.assertEqual(row["latency_seconds"]["p99"], 0.0199)
            for value in row["latency_seconds"].values():
                self.assertIsInstance(value, numbers.Real)
                self.assertGreaterEqual(value, 0.0)
            self.assertEqual(row["correctness"]["reference_baseline"], "upstream_peft_unmerged")
            self.assertEqual(row["correctness"]["tolerance_profile"], "cpu_fp32")
            self.assertTrue(row["correctness"]["passed"])
            self.assertLessEqual(row["correctness"]["max_abs_error"], 1e-4)
            self.assertLessEqual(row["correctness"]["relative_l2_error"], 1e-5)

        fork = next(row for row in results if row["baseline"] == "beyond_matmul_peft_fork")
        self.assertEqual(fork["lowering"]["kind"], "provenance_lora_fork")
        self.assertTrue(fork["lowering"]["dense_fallback_available"])
        self.assertFalse(fork["lowering"]["dense_fallback_used"])
        self.assertTrue(first["summary"]["all_required_cases_present"])
        self.assertTrue(first["summary"]["all_correctness_checks_passed"])
        self.assertEqual(first["summary"]["performance_claim"], "none")

    def test_writes_json_artifact(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "peft_transformers_lora_inference.json"
            artifact = benchmark.write_json_artifact(
                output_path,
                sequence_lengths=[4],
                batch_sizes=[1],
                warmup_repetitions=1,
                measured_repetitions=1,
                mode="synthetic-smoke",
                time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01],
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)

    def test_target_specs_accept_paths_and_git_refs(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            fork = Path(temp_dir) / "fork"
            upstream.mkdir()
            fork.mkdir()

            artifact = benchmark.collect_results(
                sequence_lengths=[4],
                batch_sizes=[1],
                warmup_repetitions=1,
                measured_repetitions=1,
                mode="synthetic-smoke",
                upstream_peft_path=str(upstream),
                fork_peft_path=str(fork),
                upstream_peft_ref="main",
                fork_peft_ref="beyond-matmul/provenance-lora-inference",
                time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01],
            )

        self.assertEqual(artifact["dependencies"]["peft_upstream"]["path"], str(upstream))
        self.assertEqual(artifact["dependencies"]["peft_fork"]["path"], str(fork))
        self.assertEqual(artifact["dependencies"]["peft_upstream"]["requested_ref"], "main")
        self.assertEqual(
            artifact["dependencies"]["peft_fork"]["requested_ref"],
            "beyond-matmul/provenance-lora-inference",
        )


if __name__ == "__main__":
    unittest.main()
