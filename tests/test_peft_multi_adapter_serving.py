import importlib.util
import json
import numbers
import tempfile
import unittest
from unittest import mock
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "peft_multi_adapter_serving.py"
    spec = importlib.util.spec_from_file_location("peft_multi_adapter_serving", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PeftMultiAdapterServingTests(unittest.TestCase):
    def test_worker_model_preloads_both_serving_adapters(self):
        benchmark = _load_benchmark_module()

        class FakePeftModel:
            def __init__(self):
                self.loaded_adapters = []

            @classmethod
            def from_pretrained(cls, _base_model, repository, *, adapter_name, revision):
                model = cls()
                model.loaded_adapters.append((adapter_name, repository, revision))
                return model

            def load_adapter(self, repository, *, adapter_name, revision):
                self.loaded_adapters.append((adapter_name, repository, revision))

            def set_adapter(self, _adapter_name):
                pass

        model = benchmark._load_worker_model(
            FakePeftModel,
            object(),
            benchmark.ADAPTERS[0],
            "upstream_peft_unmerged",
        )

        self.assertEqual(
            model.loaded_adapters,
            [
                (adapter.name, adapter.repository, adapter.revision)
                for adapter in benchmark.ADAPTERS
            ],
        )

    def test_worker_switch_measurement_transitions_from_other_adapter(self):
        benchmark = _load_benchmark_module()

        class FakeModel:
            def __init__(self):
                self.set_adapter_calls = []

            def set_adapter(self, adapter_name):
                self.set_adapter_calls.append(adapter_name)

        args = mock.Mock(_worker_warmup=1, _worker_repetitions=2)
        model = FakeModel()
        latencies = benchmark._measure_worker_switch(
            args,
            model,
            benchmark.ADAPTERS[0],
            "upstream_peft_unmerged",
        )

        self.assertEqual(
            model.set_adapter_calls,
            ["gaisb", "merchant", "gaisb", "merchant", "gaisb", "merchant"],
        )
        self.assertEqual(len(latencies), 2)
        self.assertTrue(all(latency >= 0.0 for latency in latencies))

    def test_dense_cache_switch_measurement_uses_both_cached_adapters(self):
        benchmark = _load_benchmark_module()

        class RecordingCache(dict):
            def __init__(self):
                super().__init__({"merchant": object(), "gaisb": object()})
                self.accesses = []

            def __getitem__(self, key):
                self.accesses.append(key)
                return super().__getitem__(key)

        args = mock.Mock(_worker_warmup=1, _worker_repetitions=2)
        cache = RecordingCache()
        latencies = benchmark._measure_worker_switch(
            args,
            cache,
            benchmark.ADAPTERS[0],
            "upstream_peft_merged_dense_cache",
        )

        self.assertEqual(
            cache.accesses,
            ["gaisb", "merchant", "gaisb", "merchant", "gaisb", "merchant"],
        )
        self.assertEqual(len(latencies), 2)
        self.assertTrue(all(latency >= 0.0 for latency in latencies))

    def test_process_peak_memory_sampler_reports_measured_or_unavailable(self):
        benchmark = _load_benchmark_module()

        class FakeUsage:
            ru_maxrss = 123

        class FakeResource:
            RUSAGE_SELF = object()

            @staticmethod
            def getrusage(_target):
                return FakeUsage()

        class MissingResource:
            pass

        linux_sample = benchmark._process_peak_memory_sample(FakeResource, platform_name="linux")
        mac_sample = benchmark._process_peak_memory_sample(FakeResource, platform_name="darwin")
        unavailable = benchmark._process_peak_memory_sample(MissingResource, platform_name="linux")

        self.assertEqual(linux_sample["peak_memory_bytes"], 123 * 1024)
        self.assertEqual(linux_sample["peak_memory_status"], "measured_process_maxrss")
        self.assertEqual(mac_sample["peak_memory_bytes"], 123)
        self.assertEqual(mac_sample["peak_memory_status"], "measured_process_maxrss")
        self.assertEqual(unavailable["peak_memory_bytes"], None)
        self.assertEqual(unavailable["peak_memory_status"], "unavailable_platform_api")

    def test_dense_cache_merges_the_named_adapter(self):
        benchmark = _load_benchmark_module()

        class FakeConfig:
            vocab_size = 16

        class FakeLogits:
            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[[0.0]]]

        class FakeOutput:
            logits = FakeLogits()

        class FakeDenseModel:
            config = FakeConfig()

            def __call__(self, *, input_ids, attention_mask):
                del input_ids, attention_mask
                return FakeOutput()

            def eval(self):
                pass

        class FakePeftModel:
            merged_adapters = []

            def __init__(self, adapter_name):
                self.adapter_name = adapter_name
                self.active_adapter = None

            @classmethod
            def from_pretrained(cls, _base_model, _repository, *, adapter_name, revision):
                del revision
                return cls(adapter_name)

            def set_adapter(self, adapter_name):
                self.active_adapter = adapter_name

            def merge_and_unload(self):
                self.merged_adapters.append((self.adapter_name, self.active_adapter))
                return FakeDenseModel()

        class FakeBaseModel:
            @classmethod
            def from_pretrained(cls, _base_model, *, revision, dtype):
                del revision, dtype
                return cls()

        args = mock.Mock(
            _worker_baseline="upstream_peft_merged_dense_cache",
            _worker_sequence_length=4,
            _worker_batch_size=1,
            _worker_warmup=0,
            _worker_repetitions=1,
        )

        with (
            mock.patch.object(benchmark, "_worker_inputs", return_value={"input_ids": object(), "attention_mask": object()}),
            mock.patch.object(benchmark, "_measure_worker_switch", return_value=[0.0]),
            mock.patch.object(benchmark, "_time_forward", return_value=[0.01]),
            mock.patch.object(benchmark, "_worker_storage", return_value={}),
            mock.patch.object(benchmark, "_torch"),
        ):
            result = benchmark._run_dense_cache_worker(
                args,
                FakeBaseModel,
                FakePeftModel,
                benchmark.ADAPTERS[0],
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            FakePeftModel.merged_adapters,
            [(adapter.name, adapter.name) for adapter in benchmark.ADAPTERS],
        )

    def test_base_worker_model_load_forces_contract_fp32_dtype(self):
        benchmark = _load_benchmark_module()

        class FakeTorch:
            float32 = object()

        class FakeBaseModel:
            load_kwargs = None

            @classmethod
            def from_pretrained(cls, _base_model, **kwargs):
                cls.load_kwargs = kwargs
                return cls()

        with mock.patch.object(benchmark, "_torch", return_value=FakeTorch):
            model = benchmark._load_base_worker_model(FakeBaseModel)

        self.assertIsInstance(model, FakeBaseModel)
        self.assertEqual(
            FakeBaseModel.load_kwargs,
            {
                "revision": benchmark.BASE_MODEL_REVISION,
                "dtype": FakeTorch.float32,
            },
        )

    def test_smoke_artifact_matches_contract_shape(self):
        benchmark = _load_benchmark_module()

        first = benchmark.collect_results(
            sequence_lengths=[4],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=2,
            mode="synthetic-smoke",
            time_forward=lambda _baseline, _adapter, _inputs, _warmup, _repetitions: [0.01, 0.02],
            time_switch=lambda _baseline, _adapter, _warmup, _repetitions: [0.001, 0.002],
        )
        second = benchmark.collect_results(
            sequence_lengths=[4],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=2,
            mode="synthetic-smoke",
            time_forward=lambda _baseline, _adapter, _inputs, _warmup, _repetitions: [0.01, 0.02],
            time_switch=lambda _baseline, _adapter, _warmup, _repetitions: [0.001, 0.002],
        )

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["benchmark"], "peft_multi_adapter_serving")
        self.assertEqual(first["contract"], "docs/peft_multi_adapter_serving_benchmark_contract.md")
        self.assertEqual(first["workload"]["base_model"], "facebook/opt-125m")
        self.assertEqual(
            first["workload"]["base_model_revision"],
            "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        )
        self.assertEqual(first["workload"]["model_context_limit"], 2048)
        self.assertEqual([adapter["name"] for adapter in first["workload"]["adapters"]], ["merchant", "gaisb"])
        self.assertEqual(first["workload"]["sequence_lengths"], [4])
        self.assertEqual(first["workload"]["batch_sizes"], [1])
        self.assertEqual(first["workload"]["input_seed"], 20260708)
        self.assertEqual(first["workload"]["warmup_repetitions"], 1)
        self.assertEqual(first["workload"]["measured_repetitions"], 2)
        self.assertEqual(first["dependencies"]["peft_upstream"]["repository"], "huggingface/peft")
        self.assertEqual(first["dependencies"]["peft_fork"]["repository"], "alexlopashev/peft")
        self.assertIn("python", first["dependencies"])
        self.assertIn("torch", first["dependencies"])
        self.assertIn("platform", first["environment"])

        results = first["results"]
        self.assertEqual(len(results), 8)
        self.assertEqual(
            {row["baseline"] for row in results},
            {
                "upstream_peft_unmerged",
                "upstream_peft_merged_dense_cache",
                "upstream_peft_repeated_merge_unmerge",
                "beyond_matmul_factor_provenance",
            },
        )
        self.assertEqual({row["adapter"] for row in results}, {"merchant", "gaisb"})

        required_result_fields = {
            "case",
            "adapter",
            "baseline",
            "status",
            "sequence_length",
            "batch_size",
            "latency_seconds",
            "adapter_switch_seconds",
            "adapter_switch_status",
            "peak_memory_bytes",
            "peak_memory_status",
            "storage",
            "correctness",
            "lowering",
        }
        for row in results:
            self.assertTrue(required_result_fields.issubset(row))
            self.assertEqual(row["case"], f"{row['adapter']}_seq4_batch1")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["peak_memory_bytes"], None)
            self.assertEqual(row["peak_memory_status"], "not_measured_synthetic_smoke")
            self.assertEqual(row["latency_seconds"]["median"], 0.015)
            self.assertEqual(row["adapter_switch_seconds"]["median"], 0.0015)
            self.assertIn(row["adapter_switch_status"], {"measured_loaded_adapters", "measured_dense_cache_pointer_swap"})
            for stats in (row["latency_seconds"], row["adapter_switch_seconds"]):
                for value in stats.values():
                    self.assertIsInstance(value, numbers.Real)
                    self.assertGreaterEqual(value, 0.0)
            self.assertEqual(row["correctness"]["reference_baseline"], "upstream_peft_unmerged")
            self.assertEqual(row["correctness"]["tolerance_profile"], "cpu_fp32")
            self.assertTrue(row["correctness"]["passed"])
            self.assertLessEqual(row["correctness"]["max_abs_error"], 1e-4)
            self.assertLessEqual(row["correctness"]["relative_l2_error"], 1e-5)
            self.assertIsInstance(row["storage"]["adapter_payload_bytes"], int)
            self.assertIn("resident_adapter_bytes", row["storage"])

        provenance = [
            row
            for row in results
            if row["baseline"] == "beyond_matmul_factor_provenance"
        ]
        self.assertEqual(len(provenance), 2)
        for row in provenance:
            self.assertEqual(row["lowering"]["kind"], "provenance_lora_factors")
            self.assertTrue(row["lowering"]["dense_fallback_available"])
            self.assertFalse(row["lowering"]["dense_fallback_used"])
            self.assertEqual(row["lowering"]["active_adapter"], row["adapter"])

        self.assertTrue(first["summary"]["all_required_cases_present"])
        self.assertTrue(first["summary"]["all_correctness_checks_passed"])
        self.assertTrue(first["summary"]["all_switching_cases_present"])
        self.assertTrue(first["summary"]["all_dense_fallback_cases_explicit"])
        self.assertFalse(first["summary"]["all_peak_memory_cases_measured"])
        self.assertTrue(first["summary"]["all_adapter_switch_cases_measured"])
        self.assertFalse(first["summary"]["memory_control_claim_ready"])
        self.assertEqual(
            first["summary"]["memory_control_readiness_blockers"],
            ["synthetic_smoke_not_benchmark_evidence", "peak_memory_cases_unavailable"],
        )
        self.assertFalse(first["summary"]["benchmark_ready"])
        self.assertEqual(first["summary"]["readiness_blockers"], ["synthetic_smoke_not_benchmark_evidence"])
        self.assertEqual(first["summary"]["fallback_cases"], [])
        self.assertEqual(first["summary"]["negative_cases"], [])
        self.assertEqual(first["summary"]["memory_or_control_claim"], "none")
        self.assertEqual(first["summary"]["performance_claim"], "none")

    def test_context_limit_preflight_records_blocker_rows(self):
        benchmark = _load_benchmark_module()

        artifact = benchmark.collect_results(
            sequence_lengths=[4, 8],
            batch_sizes=[1],
            warmup_repetitions=1,
            measured_repetitions=1,
            mode="synthetic-smoke",
            model_context_limit=6,
            time_forward=lambda _baseline, _adapter, _inputs, _warmup, _repetitions: [0.01],
            time_switch=lambda _baseline, _adapter, _warmup, _repetitions: [0.001],
        )

        blocked = [row for row in artifact["results"] if row["sequence_length"] == 8]
        self.assertEqual(len(blocked), 8)
        for row in blocked:
            self.assertEqual(row["status"], "blocked")
            self.assertEqual(row["latency_seconds"], None)
            self.assertEqual(row["adapter_switch_seconds"], None)
            self.assertEqual(row["peak_memory_bytes"], None)
            self.assertEqual(row["peak_memory_status"], "not_measured_blocked")
            self.assertIn("context limit", row["reason"])
            self.assertFalse(row["correctness"]["passed"])
        self.assertIn("context_limit_exceeded", artifact["summary"]["readiness_blockers"])
        self.assertIn("context_limit_exceeded", artifact["summary"]["memory_control_readiness_blockers"])
        self.assertFalse(artifact["summary"]["benchmark_ready"])
        self.assertFalse(artifact["summary"]["memory_control_claim_ready"])
        self.assertTrue(artifact["summary"]["all_required_cases_present"])

    def test_worker_payload_records_measured_peak_memory_readiness(self):
        benchmark = _load_benchmark_module()

        payloads = [
            {
                "baseline": "upstream_peft_unmerged",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.01],
                "switch_latencies": [0.001],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 2365968},
                "peak_memory_bytes": 10_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "upstream_peft_merged_dense_cache",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 500000000},
                "peak_memory_bytes": 20_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "upstream_peft_repeated_merge_unmerge",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.03],
                "switch_latencies": [0.002],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": None},
                "peak_memory_bytes": 30_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "beyond_matmul_factor_provenance",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0005],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 2365968},
                "peft_provenance_events": [
                    {
                        "kind": "beyond_matmul_lora_provenance",
                        "path": "structured_low_rank",
                        "adapter": "merchant",
                    }
                ],
                "peak_memory_bytes": 40_000,
                "peak_memory_status": "measured_process_maxrss",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            fork = Path(temp_dir) / "fork"
            upstream.mkdir()
            fork.mkdir()
            with mock.patch.object(benchmark, "_run_real_worker", side_effect=payloads):
                artifact = benchmark.collect_results(
                    adapters=[benchmark.ADAPTERS[0]],
                    sequence_lengths=[4],
                    batch_sizes=[1],
                    warmup_repetitions=0,
                    measured_repetitions=1,
                    mode="real",
                    upstream_peft_path=str(upstream),
                    fork_peft_path=str(fork),
                    model_context_limit=2048,
                )

        self.assertEqual(
            {row["peak_memory_bytes"] for row in artifact["results"]},
            {10_000, 20_000, 30_000, 40_000},
        )
        self.assertEqual(
            {row["peak_memory_status"] for row in artifact["results"]},
            {"measured_process_maxrss"},
        )
        self.assertTrue(artifact["summary"]["all_correctness_checks_passed"])
        self.assertTrue(artifact["summary"]["all_peak_memory_cases_measured"])
        self.assertTrue(artifact["summary"]["all_adapter_switch_cases_measured"])
        self.assertTrue(artifact["summary"]["memory_control_claim_ready"])
        self.assertEqual(artifact["summary"]["memory_control_readiness_blockers"], [])

    def test_memory_control_claim_readiness_requires_correctness(self):
        benchmark = _load_benchmark_module()

        payloads = [
            {
                "baseline": "upstream_peft_unmerged",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.01],
                "switch_latencies": [0.001],
                "logits": [[[0.0, 1.0]]],
                "peak_memory_bytes": 10_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "upstream_peft_merged_dense_cache",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0],
                "logits": [[[0.0, 1.0]]],
                "peak_memory_bytes": 20_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "upstream_peft_repeated_merge_unmerge",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.03],
                "switch_latencies": [0.002],
                "logits": [[[0.0, 1.0]]],
                "peak_memory_bytes": 30_000,
                "peak_memory_status": "measured_process_maxrss",
            },
            {
                "baseline": "beyond_matmul_factor_provenance",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0005],
                "logits": [[[0.5, 1.5]]],
                "peak_memory_bytes": 40_000,
                "peak_memory_status": "measured_process_maxrss",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            fork = Path(temp_dir) / "fork"
            upstream.mkdir()
            fork.mkdir()
            with mock.patch.object(benchmark, "_run_real_worker", side_effect=payloads):
                artifact = benchmark.collect_results(
                    adapters=[benchmark.ADAPTERS[0]],
                    sequence_lengths=[4],
                    batch_sizes=[1],
                    warmup_repetitions=0,
                    measured_repetitions=1,
                    mode="real",
                    upstream_peft_path=str(upstream),
                    fork_peft_path=str(fork),
                    model_context_limit=2048,
                )

        self.assertFalse(artifact["summary"]["all_correctness_checks_passed"])
        self.assertTrue(artifact["summary"]["all_peak_memory_cases_measured"])
        self.assertTrue(artifact["summary"]["all_adapter_switch_cases_measured"])
        self.assertFalse(artifact["summary"]["memory_control_claim_ready"])
        self.assertEqual(artifact["summary"]["memory_control_readiness_blockers"], ["correctness_checks_failed"])

    def test_summary_records_not_applicable_and_fallback_cases(self):
        benchmark = _load_benchmark_module()

        payloads = [
            {
                "baseline": "upstream_peft_unmerged",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.01],
                "switch_latencies": [0.001],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 2365968},
            },
            {
                "baseline": "upstream_peft_merged_dense_cache",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0],
                "logits": [[[0.0, 1.0]]],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 500000000},
            },
            {
                "baseline": "upstream_peft_repeated_merge_unmerge",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "not_applicable",
                "reason": "merge/unmerge transition is unsupported",
                "latencies": None,
                "switch_latencies": None,
                "logits": None,
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": None},
            },
            {
                "baseline": "beyond_matmul_factor_provenance",
                "adapter": "merchant",
                "sequence_length": 4,
                "batch_size": 1,
                "status": "ok",
                "reason": None,
                "latencies": [0.02],
                "switch_latencies": [0.0005],
                "logits": [[[0.0, 1.0]]],
                "peft_provenance_events": [
                    {
                        "kind": "beyond_matmul_lora_provenance",
                        "path": "dense_fallback",
                        "adapter": "merchant",
                        "dense_fallback_available": True,
                        "dense_fallback_used": True,
                        "fallback_reason": "unsupported_adapter_composition",
                    }
                ],
                "storage": {"adapter_config_bytes": 128, "resident_adapter_bytes": 2365968},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream = Path(temp_dir) / "upstream"
            fork = Path(temp_dir) / "fork"
            upstream.mkdir()
            fork.mkdir()
            with mock.patch.object(benchmark, "_run_real_worker", side_effect=payloads):
                artifact = benchmark.collect_results(
                    adapters=[benchmark.ADAPTERS[0]],
                    sequence_lengths=[4],
                    batch_sizes=[1],
                    warmup_repetitions=0,
                    measured_repetitions=1,
                    mode="real",
                    upstream_peft_path=str(upstream),
                    fork_peft_path=str(fork),
                    model_context_limit=2048,
                )

        self.assertTrue(artifact["summary"]["all_correctness_checks_passed"])
        self.assertTrue(artifact["summary"]["all_dense_fallback_cases_explicit"])
        self.assertTrue(artifact["summary"]["benchmark_ready"])
        self.assertEqual(artifact["summary"]["readiness_blockers"], [])
        self.assertEqual(
            artifact["summary"]["fallback_cases"],
            [
                {
                    "case": "merchant_seq4_batch1",
                    "adapter": "merchant",
                    "baseline": "beyond_matmul_factor_provenance",
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
                    "case": "merchant_seq4_batch1",
                    "adapter": "merchant",
                    "baseline": "upstream_peft_repeated_merge_unmerge",
                    "status": "not_applicable",
                    "reason": "merge/unmerge transition is unsupported",
                    "correctness_passed": True,
                }
            ],
        )

    def test_writes_json_artifact(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "peft_multi_adapter_serving.json"
            artifact = benchmark.write_json_artifact(
                output_path,
                sequence_lengths=[4],
                batch_sizes=[1],
                warmup_repetitions=1,
                measured_repetitions=1,
                mode="synthetic-smoke",
                time_forward=lambda _baseline, _adapter, _inputs, _warmup, _repetitions: [0.01],
                time_switch=lambda _baseline, _adapter, _warmup, _repetitions: [0.001],
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)


if __name__ == "__main__":
    unittest.main()
