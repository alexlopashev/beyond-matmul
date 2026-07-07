import importlib.util
import json
import numbers
import os
import tempfile
import unittest
from unittest import mock
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
        self.assertTrue(first["summary"]["all_fork_fallback_cases_explicit"])
        self.assertFalse(first["summary"]["benchmark_ready"])
        self.assertEqual(first["summary"]["readiness_blockers"], ["synthetic_smoke_not_benchmark_evidence"])
        self.assertLessEqual(first["summary"]["max_abs_error"], 1e-4)
        self.assertLessEqual(first["summary"]["max_relative_l2_error"], 1e-5)
        self.assertEqual(first["summary"]["fallback_cases"], [])
        self.assertEqual(first["summary"]["negative_cases"], [])
        self.assertEqual(first["summary"]["performance_claim"], "none")

    def test_summary_blocks_readiness_when_fork_changes_outputs(self):
        benchmark = _load_benchmark_module()

        class DivergentSyntheticLoraModel:
            def __init__(self, baseline, vocab_size=16):
                self.baseline = baseline
                self.vocab_size = vocab_size

            def __call__(self, input_ids, attention_mask):
                torch = benchmark._torch()
                logits = torch.ones((*input_ids.shape, self.vocab_size), dtype=torch.float32)
                if self.baseline == "beyond_matmul_peft_fork":
                    logits = logits + 0.01
                return logits

        with mock.patch.object(benchmark, "_SyntheticLoraModel", DivergentSyntheticLoraModel):
            artifact = benchmark.collect_results(
                sequence_lengths=[4],
                batch_sizes=[1],
                warmup_repetitions=1,
                measured_repetitions=1,
                mode="synthetic-smoke",
                time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01],
            )

        fork = next(row for row in artifact["results"] if row["baseline"] == "beyond_matmul_peft_fork")
        self.assertEqual(fork["status"], "failed_correctness")
        self.assertFalse(fork["correctness"]["passed"])
        self.assertFalse(artifact["summary"]["all_correctness_checks_passed"])
        self.assertFalse(artifact["summary"]["benchmark_ready"])
        self.assertIn("correctness_checks_failed", artifact["summary"]["readiness_blockers"])
        self.assertGreater(artifact["summary"]["max_abs_error"], 1e-4)
        self.assertGreater(artifact["summary"]["max_relative_l2_error"], 1e-5)
        self.assertEqual(
            artifact["summary"]["negative_cases"],
            [
                {
                    "case": "seq4_batch1",
                    "baseline": "beyond_matmul_peft_fork",
                    "status": "failed_correctness",
                    "reason": "correctness tolerance failed",
                    "correctness_passed": False,
                }
            ],
        )

    def test_real_summary_records_explicit_not_applicable_and_fallback_cases(self):
        benchmark = _load_benchmark_module()

        fallback_event = {
            "schema_version": 1,
            "kind": "beyond_matmul_lora_provenance",
            "path": "dense_fallback",
            "module_path": "model.layers.0.self_attn.q_proj",
            "adapter": "default",
            "dense_fallback_available": True,
            "dense_fallback_used": True,
            "fallback_reason": "unsupported_adapter_composition",
        }
        reference_payload = {
            "baseline": "upstream_peft_unmerged",
            "sequence_length": 4,
            "batch_size": 1,
            "status": "ok",
            "reason": None,
            "latencies": [0.01],
            "logits": [[[0.0, 1.0]]],
            "peft_provenance_events": [],
        }
        merged_payload = {
            "baseline": "upstream_peft_merged_dense",
            "sequence_length": 4,
            "batch_size": 1,
            "status": "not_applicable",
            "reason": "merge_and_unload failed: unsupported adapter",
            "latencies": None,
            "logits": None,
        }
        fork_payload = {
            "baseline": "beyond_matmul_peft_fork",
            "sequence_length": 4,
            "batch_size": 1,
            "status": "ok",
            "reason": None,
            "latencies": [0.02],
            "logits": [[[0.0, 1.0]]],
            "peft_provenance_events": [fallback_event],
        }

        with mock.patch.object(benchmark, "_resolve_checkout", side_effect=["/tmp/upstream-peft", "/tmp/fork-peft"]):
            with mock.patch.object(
                benchmark,
                "_run_real_worker",
                side_effect=[reference_payload, merged_payload, fork_payload],
            ):
                artifact = benchmark.collect_results(
                    sequence_lengths=[4],
                    batch_sizes=[1],
                    warmup_repetitions=0,
                    measured_repetitions=1,
                    mode="real",
                )

        self.assertTrue(artifact["summary"]["all_correctness_checks_passed"])
        self.assertTrue(artifact["summary"]["all_fork_fallback_cases_explicit"])
        self.assertTrue(artifact["summary"]["benchmark_ready"])
        self.assertEqual(artifact["summary"]["readiness_blockers"], [])
        self.assertEqual(
            artifact["summary"]["fallback_cases"],
            [
                {
                    "case": "seq4_batch1",
                    "baseline": "beyond_matmul_peft_fork",
                    "status": "ok",
                    "kind": "peft_dense_fallback",
                    "fallback_reasons": ["unsupported_adapter_composition"],
                    "correctness_passed": True,
                }
            ],
        )
        self.assertEqual(
            artifact["summary"]["negative_cases"],
            [
                {
                    "case": "seq4_batch1",
                    "baseline": "upstream_peft_merged_dense",
                    "status": "not_applicable",
                    "reason": "merge_and_unload failed: unsupported adapter",
                    "correctness_passed": True,
                }
            ],
        )

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

    def test_artifact_can_record_regeneration_command(self):
        benchmark = _load_benchmark_module()

        artifact = benchmark.collect_results(
            sequence_lengths=[4],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=1,
            mode="synthetic-smoke",
            command=[
                "python",
                "benchmarks/peft_transformers_lora_inference.py",
                "--json-output",
                "docs/results/peft_transformers_lora_inference.json",
            ],
            generated_at_utc="2026-07-07T18:30:00Z",
            time_forward=lambda _baseline, _inputs, _warmup, _repetitions: [0.01],
        )

        self.assertEqual(
            artifact["run"]["command"],
            [
                "python",
                "benchmarks/peft_transformers_lora_inference.py",
                "--json-output",
                "docs/results/peft_transformers_lora_inference.json",
            ],
        )
        self.assertEqual(
            artifact["run"]["command_text"],
            "python benchmarks/peft_transformers_lora_inference.py --json-output "
            "docs/results/peft_transformers_lora_inference.json",
        )
        self.assertEqual(artifact["run"]["generated_at_utc"], "2026-07-07T18:30:00Z")
        self.assertEqual(artifact["run"]["mode"], "synthetic-smoke")

    def test_real_revision_label_resolves_huggingface_sha(self):
        benchmark = _load_benchmark_module()

        with mock.patch.object(benchmark, "_huggingface_revision", return_value="abc123"):
            self.assertEqual(benchmark._revision_label("model-or-adapter", "real"), "abc123")

    def test_real_worker_prefers_src_layout_checkout_over_installed_peft(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkout = temp_path / "peft-checkout"
            checkout_src = checkout / "src" / "peft"
            installed_peft = temp_path / "installed" / "peft"
            transformers = temp_path / "installed" / "transformers"
            checkout_src.mkdir(parents=True)
            installed_peft.mkdir(parents=True)
            transformers.mkdir(parents=True)
            (checkout_src / "__init__.py").write_text(
                "\n".join(
                    [
                        "class PeftModel:",
                        "    @staticmethod",
                        "    def from_pretrained(model, adapter):",
                        "        model.peft_source = 'checkout-src'",
                        "        return model",
                    ]
                ),
                encoding="utf-8",
            )
            (installed_peft / "__init__.py").write_text(
                "\n".join(
                    [
                        "class PeftModel:",
                        "    @staticmethod",
                        "    def from_pretrained(model, adapter):",
                        "        raise RuntimeError('wrong peft imported')",
                    ]
                ),
                encoding="utf-8",
            )
            (transformers / "__init__.py").write_text(
                "\n".join(
                    [
                        "class _Config:",
                        "    vocab_size = 8",
                        "",
                        "class _Output:",
                        "    def __init__(self, logits):",
                        "        self.logits = logits",
                        "",
                        "class _Model:",
                        "    config = _Config()",
                        "",
                        "    def eval(self):",
                        "        return self",
                        "",
                        "    def merge_and_unload(self):",
                        "        return self",
                        "",
                        "    def __call__(self, input_ids, attention_mask):",
                        "        import torch",
                        "        shape = (*input_ids.shape, self.config.vocab_size)",
                        "        return _Output(torch.zeros(shape, dtype=torch.float32))",
                        "",
                        "class AutoModelForCausalLM:",
                        "    @staticmethod",
                        "    def from_pretrained(name):",
                        "        return _Model()",
                    ]
                ),
                encoding="utf-8",
            )

            old_pythonpath = os.environ.get("PYTHONPATH")
            pythonpath = str(temp_path / "installed")
            if old_pythonpath:
                pythonpath = pythonpath + os.pathsep + old_pythonpath
            with mock.patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                payload = benchmark._run_real_worker(
                    "upstream_peft_unmerged",
                    sequence_length=2,
                    batch_size=1,
                    warmup_repetitions=0,
                    measured_repetitions=1,
                    peft_path=str(checkout),
                    merge_dense=False,
                )

        self.assertEqual(payload["status"], "ok")
        self.assertIsNone(payload["reason"])
        self.assertIsNotNone(payload["logits"])

    def test_worker_payload_copies_fork_provenance_events_into_row(self):
        benchmark = _load_benchmark_module()

        event = {
            "schema_version": 1,
            "kind": "beyond_matmul_lora_provenance",
            "path": "dense_fallback",
            "module_path": "model.layers.0.self_attn.q_proj",
            "adapter": "default",
            "base_module": "Linear",
            "rank": 8,
            "in_features": 32,
            "out_features": 32,
            "input_shape": [1, 16, 32],
            "a_shape": [8, 32],
            "b_shape": [32, 8],
            "scaling": 1.0,
            "dtype": "torch.float32",
            "device": "cpu",
            "dense_fallback_available": True,
            "dense_fallback_used": True,
            "fallback_reason": "lora_bias",
        }
        payload = {
            "baseline": "beyond_matmul_peft_fork",
            "sequence_length": 16,
            "batch_size": 1,
            "status": "ok",
            "reason": None,
            "latencies": [0.01],
            "logits": [[[0.0, 1.0]]],
            "peft_provenance_events": [event],
        }

        row = benchmark._worker_payload_to_row(payload, reference_logits=payload["logits"])

        self.assertEqual(row["peft_provenance_events"], [event])
        self.assertEqual(row["lowering"]["kind"], "peft_dense_fallback")
        self.assertTrue(row["lowering"]["dense_fallback_available"])
        self.assertTrue(row["lowering"]["dense_fallback_used"])
        self.assertEqual(row["lowering"]["fallback_reasons"], ["lora_bias"])


if __name__ == "__main__":
    unittest.main()
