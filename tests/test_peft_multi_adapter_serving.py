import importlib.util
import json
import numbers
import tempfile
import types
import unittest
import weakref
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
            self.assertEqual(row["lowering"]["execution_path"], "structured_low_rank")
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
        self.assertEqual(
            first["summary"]["structured_low_rank_cases"],
            [
                {
                    "case": "merchant_seq4_batch1",
                    "adapter": "merchant",
                    "baseline": "beyond_matmul_factor_provenance",
                    "status": "ok",
                    "kind": "provenance_lora_factors",
                    "execution_path": "structured_low_rank",
                    "correctness_passed": True,
                },
                {
                    "case": "gaisb_seq4_batch1",
                    "adapter": "gaisb",
                    "baseline": "beyond_matmul_factor_provenance",
                    "status": "ok",
                    "kind": "provenance_lora_factors",
                    "execution_path": "structured_low_rank",
                    "correctness_passed": True,
                },
            ],
        )
        self.assertEqual(first["summary"]["negative_cases"], [])
        self.assertEqual(first["summary"]["memory_or_control_claim"], "none")
        self.assertEqual(first["summary"]["performance_claim"], "none")

    def test_structured_low_rank_requires_fp32_cpu_contract(self):
        benchmark = _load_benchmark_module()

        row = benchmark._result_row(
            baseline="beyond_matmul_factor_provenance",
            adapter=benchmark.ADAPTERS[0],
            sequence_length=4,
            batch_size=1,
            latencies=[0.01],
            switch_latencies=[0.001],
            logits=[[[0.0, 1.0]]],
            reference_logits=[[[0.0, 1.0]]],
            peft_provenance_events=[
                {
                    "kind": "beyond_matmul_lora_provenance",
                    "path": "structured_low_rank",
                    "adapter": "merchant",
                    "device": "cpu",
                    "dtype": "torch.float16",
                    "a_device": "cpu",
                    "a_dtype": "torch.float16",
                    "b_device": "cpu",
                    "b_dtype": "torch.float16",
                    "base_module": "Linear",
                    "fan_in_fan_out": False,
                }
            ],
        )

        self.assertTrue(row["correctness"]["passed"])
        self.assertEqual(row["lowering"]["kind"], "peft_dense_fallback")
        self.assertEqual(row["lowering"]["execution_path"], "dense_fallback")
        self.assertTrue(row["lowering"]["dense_fallback_used"])
        self.assertEqual(row["lowering"]["fallback_reasons"], ["non_fp32_dtype"])

    def test_structured_low_rank_requires_correctness_parity(self):
        benchmark = _load_benchmark_module()

        row = benchmark._result_row(
            baseline="beyond_matmul_factor_provenance",
            adapter=benchmark.ADAPTERS[0],
            sequence_length=4,
            batch_size=1,
            latencies=[0.01],
            switch_latencies=[0.001],
            logits=[[[1.0, 2.0]]],
            reference_logits=[[[0.0, 1.0]]],
            peft_provenance_events=[
                {
                    "kind": "beyond_matmul_lora_provenance",
                    "path": "structured_low_rank",
                    "adapter": "merchant",
                    "device": "cpu",
                    "dtype": "torch.float32",
                    "a_device": "cpu",
                    "a_dtype": "torch.float32",
                    "b_device": "cpu",
                    "b_dtype": "torch.float32",
                    "base_module": "Linear",
                    "fan_in_fan_out": False,
                }
            ],
        )

        self.assertEqual(row["status"], "failed_correctness")
        self.assertEqual(row["lowering"]["kind"], "peft_dense_fallback")
        self.assertEqual(row["lowering"]["execution_path"], "dense_fallback")
        self.assertTrue(row["lowering"]["dense_fallback_used"])
        self.assertEqual(row["lowering"]["fallback_reasons"], ["correctness_failed"])

    def test_structured_low_rank_rejects_unsupported_device_and_layout(self):
        benchmark = _load_benchmark_module()

        row = benchmark._result_row(
            baseline="beyond_matmul_factor_provenance",
            adapter=benchmark.ADAPTERS[0],
            sequence_length=4,
            batch_size=1,
            latencies=[0.01],
            switch_latencies=[0.001],
            logits=[[[0.0, 1.0]]],
            reference_logits=[[[0.0, 1.0]]],
            peft_provenance_events=[
                {
                    "kind": "beyond_matmul_lora_provenance",
                    "path": "structured_low_rank",
                    "adapter": "merchant",
                    "device": "cuda:0",
                    "dtype": "torch.float32",
                    "a_device": "cuda:0",
                    "a_dtype": "torch.float32",
                    "b_device": "cuda:0",
                    "b_dtype": "torch.float32",
                    "base_module": "Linear",
                    "fan_in_fan_out": True,
                }
            ],
        )

        self.assertTrue(row["correctness"]["passed"])
        self.assertEqual(row["lowering"]["kind"], "peft_dense_fallback")
        self.assertEqual(row["lowering"]["execution_path"], "dense_fallback")
        self.assertTrue(row["lowering"]["dense_fallback_used"])
        self.assertEqual(row["lowering"]["fallback_reasons"], ["non_cpu_device", "unsupported_layout"])

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

    def test_hardware_contract_smoke_records_cuda_schema_and_required_grid(self):
        benchmark = _load_benchmark_module()

        artifact = benchmark.collect_hardware_contract_results(
            mode="synthetic-smoke",
            cuda_available=False,
        )

        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["benchmark"], "hardware_backed_peft_multi_adapter_serving")
        self.assertEqual(
            artifact["contract"],
            "docs/hardware_backed_production_benchmark_contract.md",
        )
        self.assertEqual(artifact["workload"]["device"], "cuda")
        self.assertEqual(artifact["workload"]["sequence_lengths"], [16, 64, 128])
        self.assertEqual(artifact["workload"]["batch_sizes"], [1, 2])
        self.assertIn("accelerate", artifact["dependencies"])
        self.assertIn("safetensors", artifact["dependencies"])
        self.assertIn("torch_backends", artifact["hardware"])
        self.assertEqual(artifact["hardware"]["gpu"], None)
        self.assertFalse(artifact["summary"]["production_contract_ready"])
        self.assertFalse(artifact["summary"]["performance_fields_interpretable"])
        self.assertFalse(artifact["summary"]["memory_fields_interpretable"])
        self.assertIn("missing_cuda", artifact["summary"]["readiness_blockers"])
        self.assertTrue(artifact["summary"]["all_required_cases_present"])

        cases = {
            (row["adapter"], row["baseline"], row["sequence_length"], row["batch_size"])
            for row in artifact["results"]
        }
        self.assertEqual(len(cases), 2 * 4 * 3 * 2)
        self.assertIn(("merchant", "beyond_matmul_structured_low_rank", 128, 2), cases)
        for row in artifact["results"]:
            self.assertEqual(row["status"], "blocked")
            self.assertIn("missing_cuda", row["readiness_blockers"])
            self.assertIn("forward_latency_seconds", row)
            self.assertIn("adapter_switch_seconds", row)
            self.assertIn("preprocessing_seconds", row)
            self.assertIn("cuda_memory", row)
            self.assertIn("storage", row)
            self.assertEqual(row["correctness"]["tolerance_profile"], "cuda_fp32")
            self.assertFalse(row["correctness"]["passed"])
            self.assertIn("dense_fallback_available", row["lowering"])

    def test_hardware_contract_preflight_blockers_are_stable(self):
        benchmark = _load_benchmark_module()

        bad_adapter = benchmark.AdapterSpec(
            name="merchant",
            repository="choyiny/opt-125m-lora-merchant-finetune",
            revision="not-the-contract-revision",
            payload_file="adapter_model.safetensors",
            payload_bytes=2_365_968,
        )
        artifact = benchmark.collect_hardware_contract_results(
            adapters=[bad_adapter],
            sequence_lengths=[128],
            batch_sizes=[1],
            mode="synthetic-smoke",
            cuda_available=True,
            model_context_limit=64,
            hardware_metadata={
                "gpu": "NVIDIA Test GPU",
                "cuda_device": 0,
                "compute_capability": None,
                "total_memory_bytes": None,
                "driver": None,
                "mig_partition": None,
            },
        )

        blockers = artifact["summary"]["readiness_blockers"]
        self.assertIn("contract_pin_mismatch", blockers)
        self.assertIn("context_limit_exceeded", blockers)
        self.assertIn("incomplete_hardware_metadata", blockers)
        self.assertFalse(artifact["summary"]["production_contract_ready"])
        self.assertFalse(artifact["summary"]["performance_fields_interpretable"])
        self.assertFalse(artifact["summary"]["memory_fields_interpretable"])
        self.assertTrue(artifact["summary"]["all_required_cases_present"])
        for row in artifact["results"]:
            self.assertEqual(row["status"], "blocked")
            self.assertIn("contract_pin_mismatch", row["readiness_blockers"])
            self.assertIn("context_limit_exceeded", row["readiness_blockers"])
            self.assertIn("incomplete_hardware_metadata", row["readiness_blockers"])

    def test_latency_stats_include_min_and_max(self):
        benchmark = _load_benchmark_module()

        stats = benchmark._latency_stats([0.003, 0.001, 0.002])

        self.assertEqual(stats["min"], 0.001)
        self.assertEqual(stats["max"], 0.003)

    def test_hardware_contract_fake_backend_measures_upstream_rows_in_isolated_processes(self):
        benchmark = _load_benchmark_module()
        calls = []

        def fake_runner(*, adapter, baseline, sequence_length, batch_size, forward_warmup_repetitions,
                        forward_measured_repetitions, switch_warmup_repetitions,
                        switch_measured_repetitions, **_kwargs):
            calls.append((adapter.name, baseline, sequence_length, batch_size))
            return {
                "status": "ok",
                "reason": None,
                "isolated_process": True,
                "process_id": 40_000 + len(calls),
                "forward_latencies_seconds": [0.010, 0.012, 0.014],
                "forward_wall_seconds": [0.011, 0.013, 0.015],
                "switch_latencies_seconds": [0.001, 0.002, 0.003],
                "switch_wall_seconds": [0.0015, 0.0025, 0.0035],
                "preprocessing_seconds": {
                    "model_load": 3.0,
                    "tokenization": 0.2,
                    "dense_cache_build": 0.4
                    if baseline == "upstream_peft_merged_dense_cache"
                    else None,
                    "structured_factor_pack": None,
                    "compilation_or_graph_capture": None,
                },
                "cuda_memory": {
                    "setup_peak_allocated_bytes": 100,
                    "setup_peak_reserved_bytes": 200,
                    "steady_peak_allocated_bytes": 120,
                    "steady_peak_reserved_bytes": 240,
                    "post_setup_allocated_bytes": 80,
                    "post_setup_reserved_bytes": 160,
                    "post_loop_allocated_bytes": 90,
                    "post_loop_reserved_bytes": 180,
                },
                "correctness": {
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                    "passed": True,
                },
                "storage": {"resident_adapter_bytes": adapter.payload_bytes},
                "timing_protocol": {
                    "cuda_events": True,
                    "synchronized_before_after": True,
                    "allocator_reset": True,
                    "setup_excluded_from_steady_state": True,
                },
            }

        artifact = benchmark.collect_hardware_contract_results(
            adapters=[benchmark.ADAPTERS[0]],
            sequence_lengths=[16],
            batch_sizes=[1],
            mode="fake-backend",
            cuda_available=True,
            hardware_metadata={
                "gpu": "Fake CUDA GPU",
                "cuda_device": 0,
                "compute_capability": "9.0",
                "total_memory_bytes": 80_000_000_000,
                "driver": "999.0",
            },
            hardware_case_runner=fake_runner,
        )

        self.assertEqual(
            calls,
            [
                ("merchant", "upstream_peft_unmerged", 16, 1),
                ("merchant", "upstream_peft_merged_dense_cache", 16, 1),
                ("merchant", "upstream_peft_repeated_merge_unmerge", 16, 1),
            ],
        )
        measured = [row for row in artifact["results"] if row["baseline"] != "beyond_matmul_structured_low_rank"]
        self.assertEqual(len(measured), 3)
        for row in measured:
            self.assertEqual(row["status"], "ok")
            self.assertTrue(row["measurement"]["isolated_process"])
            self.assertTrue(row["timing_protocol"]["cuda_events"])
            self.assertTrue(row["timing_protocol"]["synchronized_before_after"])
            self.assertTrue(row["timing_protocol"]["allocator_reset"])
            self.assertEqual(row["forward_repetitions"]["warmup"], 25)
            self.assertEqual(row["forward_repetitions"]["measured"], 100)
            self.assertEqual(row["adapter_switch_repetitions"]["warmup"], 25)
            self.assertEqual(row["adapter_switch_repetitions"]["measured"], 100)
            self.assertEqual(row["forward_latency_seconds"]["min"], 0.010)
            self.assertEqual(row["forward_latency_seconds"]["max"], 0.014)
            self.assertEqual(row["forward_latency_wall_seconds"]["median"], 0.013)
            self.assertEqual(row["cuda_memory"]["post_setup_reserved_bytes"], 160)
            self.assertEqual(row["cuda_memory"]["post_loop_reserved_bytes"], 180)

        structured = [
            row for row in artifact["results"]
            if row["baseline"] == "beyond_matmul_structured_low_rank"
        ]
        self.assertEqual(len(structured), 1)
        self.assertEqual(structured[0]["status"], "blocked")
        self.assertIn("structured_path_blocked_milestone_2", structured[0]["readiness_blockers"])
        self.assertFalse(artifact["summary"]["production_contract_ready"])
        self.assertIn("structured_path_blocked_milestone_2", artifact["summary"]["readiness_blockers"])

    def test_cuda_measurement_backend_uses_events_sync_and_allocator_resets(self):
        benchmark = _load_benchmark_module()

        class FakeEvent:
            def __init__(self, cuda):
                self.cuda = cuda

            def record(self):
                self.cuda.calls.append("record")

            def elapsed_time(self, _other):
                return 12.5

        class FakeCuda:
            def __init__(self):
                self.calls = []

            def empty_cache(self):
                self.calls.append("empty_cache")

            def synchronize(self):
                self.calls.append("synchronize")

            def reset_peak_memory_stats(self):
                self.calls.append("reset_peak_memory_stats")

            def memory_allocated(self):
                return 10

            def memory_reserved(self):
                return 20

            def max_memory_allocated(self):
                return 30

            def max_memory_reserved(self):
                return 40

            def Event(self, *, enable_timing):
                self.calls.append(("event", enable_timing))
                return FakeEvent(self)

        fake_cuda = FakeCuda()
        backend = benchmark._CudaMeasurementBackend(mock.Mock(cuda=fake_cuda))
        backend.empty_cache()
        backend.reset_peak_memory_stats()
        timing = backend.time_cuda_region(lambda: fake_cuda.calls.append("callback"))
        allocator = backend.allocator_values()

        self.assertEqual(timing["cuda_seconds"], 0.0125)
        self.assertGreaterEqual(timing["wall_seconds"], 0.0)
        self.assertEqual(
            fake_cuda.calls[:7],
            [
                "empty_cache",
                "reset_peak_memory_stats",
                ("event", True),
                ("event", True),
                "synchronize",
                "record",
                "callback",
            ],
        )
        self.assertEqual(fake_cuda.calls[7:9], ["record", "synchronize"])
        self.assertEqual(
            allocator,
            {
                "allocated_bytes": 10,
                "reserved_bytes": 20,
                "peak_allocated_bytes": 30,
                "peak_reserved_bytes": 40,
            },
        )

    def test_hardware_dense_cache_loader_retains_both_merged_models(self):
        benchmark = _load_benchmark_module()

        class FakeDenseModel:
            def __init__(self, adapter_name):
                self.adapter_name = adapter_name

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                return self

        class FakePeftModel:
            def __init__(self, adapter_name):
                self.adapter_name = adapter_name
                self.active_adapter = None

            def set_adapter(self, adapter_name):
                self.active_adapter = adapter_name

            def merge_and_unload(self):
                self.assert_active_adapter()
                return FakeDenseModel(self.adapter_name)

            def assert_active_adapter(self):
                if self.active_adapter != self.adapter_name:
                    raise AssertionError("the named adapter must be selected before merge")

        args = mock.Mock(_hardware_worker_baseline="upstream_peft_merged_dense_cache")
        with (
            mock.patch.object(benchmark, "_load_base_worker_model", side_effect=lambda _model_class: object()),
            mock.patch.object(
                benchmark,
                "_load_primary_worker_adapter",
                side_effect=lambda _peft_class, _base, cached_adapter, _baseline: FakePeftModel(
                    cached_adapter.name
                ),
            ),
        ):
            dense_cache, preprocessing = benchmark._load_hardware_worker_model(
                args,
                object(),
                object(),
                benchmark.ADAPTERS[0],
            )

        self.assertEqual(set(dense_cache), {adapter.name for adapter in benchmark.ADAPTERS})
        self.assertIsNot(dense_cache[benchmark.ADAPTERS[0].name], dense_cache[benchmark.ADAPTERS[1].name])
        self.assertEqual(
            [dense_cache[adapter.name].adapter_name for adapter in benchmark.ADAPTERS],
            [adapter.name for adapter in benchmark.ADAPTERS],
        )
        self.assertTrue(all(dense_cache[adapter.name].device == "cuda" for adapter in benchmark.ADAPTERS))
        self.assertIsNotNone(preprocessing["dense_cache_build"])

    def test_hardware_dense_cache_switch_selects_between_cached_models(self):
        benchmark = _load_benchmark_module()

        class RecordingCache(dict):
            def __init__(self):
                super().__init__({adapter.name: object() for adapter in benchmark.ADAPTERS})
                self.selections = []

            def __getitem__(self, key):
                selected = super().__getitem__(key)
                self.selections.append(selected)
                return selected

        class FakeBackend:
            @staticmethod
            def time_cuda_region(callback):
                callback()
                return {"cuda_seconds": 0.0, "wall_seconds": 0.0}

        args = mock.Mock(
            _hardware_worker_baseline="upstream_peft_merged_dense_cache",
            _hardware_worker_switch_warmup=1,
            _hardware_worker_switch_repetitions=2,
        )
        cache = RecordingCache()
        result = benchmark._measure_hardware_worker_switch(
            args,
            cache,
            benchmark.ADAPTERS[0],
            FakeBackend(),
        )
        other_model = dict.__getitem__(cache, benchmark.ADAPTERS[1].name)
        selected_model = dict.__getitem__(cache, benchmark.ADAPTERS[0].name)
        expected_selections = [
            other_model,
            selected_model,
        ] * 3

        self.assertEqual(cache.selections, expected_selections)
        self.assertIs(result["selected_model"], selected_model)

    def test_hardware_real_worker_clears_cache_and_retains_dense_models_through_sampling(self):
        benchmark = _load_benchmark_module()
        calls = []
        resident_models = []
        captured_payload = {}

        class FakeTensor:
            def to(self, _device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[[0.0]]]

        class FakeModel:
            def __init__(self, adapter_name):
                self.adapter_name = adapter_name
                self.config = mock.Mock(vocab_size=16)

            def __call__(self, *, input_ids, attention_mask):
                del input_ids, attention_mask
                return mock.Mock(logits=FakeTensor())

        class FakeInferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        class FakeBackend:
            def __init__(self, _torch_module):
                pass

            def empty_cache(self):
                calls.append("empty_cache")

            def synchronize(self):
                calls.append("synchronize")

            def reset_peak_memory_stats(self):
                calls.append("reset_peak_memory_stats")

            def allocator_values(self):
                calls.append("allocator_values")
                self.assert_cache_resident()
                return {
                    "allocated_bytes": 10,
                    "reserved_bytes": 20,
                    "peak_allocated_bytes": 30,
                    "peak_reserved_bytes": 40,
                }

            @staticmethod
            def assert_cache_resident():
                if not resident_models or any(model_ref() is None for model_ref in resident_models):
                    raise AssertionError("both dense models must remain resident")

        def load_models(*_args):
            calls.append("setup")
            dense_cache = {adapter.name: FakeModel(adapter.name) for adapter in benchmark.ADAPTERS}
            resident_models.extend(weakref.ref(model) for model in dense_cache.values())
            return dense_cache, {
                "model_load": 0.1,
                "hub_download": None,
                "tokenization": None,
                "dense_cache_build": 0.1,
                "structured_factor_pack": None,
                "compilation_or_graph_capture": None,
            }

        def measure_switch(_args, dense_cache, adapter, _backend):
            calls.append("switch")
            FakeBackend.assert_cache_resident()
            return {
                "cuda_seconds": [0.0],
                "wall_seconds": [0.0],
                "selected_model": dense_cache[adapter.name],
            }

        def measure_forward(_args, model, _inputs, _backend):
            calls.append(("forward", model.adapter_name))
            FakeBackend.assert_cache_resident()
            return {"cuda_seconds": [0.01], "wall_seconds": [0.02]}

        peft_module = types.ModuleType("peft")
        peft_module.PeftModel = object
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoModelForCausalLM = object
        args = mock.Mock(
            _hardware_worker_adapter_name=benchmark.ADAPTERS[0].name,
            _hardware_worker_baseline="upstream_peft_merged_dense_cache",
            _hardware_worker_peft_path="/tmp/peft",
            _hardware_worker_sequence_length=16,
            _hardware_worker_batch_size=1,
            _hardware_worker_json_output="/tmp/result.json",
        )
        with (
            mock.patch.dict("sys.modules", {"peft": peft_module, "transformers": transformers_module}),
            mock.patch.object(benchmark, "_prepend_peft_import_paths"),
            mock.patch.object(benchmark, "_torch", return_value=FakeTorch()),
            mock.patch.object(benchmark, "_CudaMeasurementBackend", FakeBackend),
            mock.patch.object(benchmark, "_load_hardware_worker_model", side_effect=load_models),
            mock.patch.object(
                benchmark,
                "_worker_inputs",
                return_value={"input_ids": FakeTensor(), "attention_mask": FakeTensor()},
            ),
            mock.patch.object(benchmark, "_measure_hardware_worker_switch", side_effect=measure_switch),
            mock.patch.object(benchmark, "_measure_hardware_worker_forward", side_effect=measure_forward),
            mock.patch.object(benchmark, "_worker_storage", return_value={}),
            mock.patch.object(
                benchmark,
                "_write_json",
                side_effect=lambda _path, payload: captured_payload.update(payload),
            ),
        ):
            benchmark._hardware_real_worker(args)

        self.assertEqual(captured_payload["status"], "ok")
        self.assertEqual(
            calls,
            [
                "empty_cache",
                "synchronize",
                "reset_peak_memory_stats",
                "setup",
                "synchronize",
                "allocator_values",
                "reset_peak_memory_stats",
                "switch",
                ("forward", benchmark.ADAPTERS[0].name),
                "synchronize",
                "allocator_values",
            ],
        )

    def test_hardware_contract_keeps_failed_correctness_rows_out_of_claim_summary(self):
        benchmark = _load_benchmark_module()

        def fake_runner(*, adapter, baseline, **_kwargs):
            passed = baseline != "upstream_peft_merged_dense_cache"
            return {
                "status": "ok",
                "reason": None,
                "isolated_process": True,
                "process_id": 50_000,
                "forward_latencies_seconds": [0.010],
                "forward_wall_seconds": [0.011],
                "switch_latencies_seconds": [0.001],
                "switch_wall_seconds": [0.0015],
                "preprocessing_seconds": {},
                "cuda_memory": {
                    "setup_peak_allocated_bytes": 100,
                    "setup_peak_reserved_bytes": 200,
                    "steady_peak_allocated_bytes": 120,
                    "steady_peak_reserved_bytes": 240,
                    "post_setup_allocated_bytes": 80,
                    "post_setup_reserved_bytes": 160,
                    "post_loop_allocated_bytes": 90,
                    "post_loop_reserved_bytes": 180,
                },
                "correctness": {
                    "max_abs_error": 0.2 if not passed else 0.0,
                    "relative_l2_error": 0.1 if not passed else 0.0,
                    "passed": passed,
                },
                "storage": {"resident_adapter_bytes": adapter.payload_bytes},
                "timing_protocol": {
                    "cuda_events": True,
                    "synchronized_before_after": True,
                    "allocator_reset": True,
                    "setup_excluded_from_steady_state": True,
                },
            }

        artifact = benchmark.collect_hardware_contract_results(
            adapters=[benchmark.ADAPTERS[0]],
            sequence_lengths=[16],
            batch_sizes=[1],
            mode="fake-backend",
            cuda_available=True,
            hardware_metadata={
                "gpu": "Fake CUDA GPU",
                "cuda_device": 0,
                "compute_capability": "9.0",
                "total_memory_bytes": 80_000_000_000,
                "driver": "999.0",
            },
            hardware_case_runner=fake_runner,
        )

        failed = [
            row for row in artifact["results"]
            if row["baseline"] == "upstream_peft_merged_dense_cache"
        ]
        self.assertEqual(failed[0]["status"], "failed_correctness")
        self.assertFalse(failed[0]["correctness"]["passed"])
        self.assertIn("correctness_checks_failed", artifact["summary"]["readiness_blockers"])
        self.assertNotIn(
            "upstream_peft_merged_dense_cache",
            {row["baseline"] for row in artifact["summary"]["claim_summary_rows"]},
        )

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
