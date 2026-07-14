import importlib.util
import json
import tempfile
import unittest
from contextlib import nullcontext
from unittest import mock
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "olmoe_stock_baseline.py"
    spec = importlib.util.spec_from_file_location("olmoe_stock_baseline", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OlmoeStockBaselineTests(unittest.TestCase):
    def test_pins_external_model_and_transformers_revision(self):
        benchmark = _load_benchmark_module()

        self.assertEqual(benchmark.MODEL, "allenai/OLMoE-1B-7B-0924")
        self.assertEqual(
            benchmark.MODEL_REVISION,
            "bd1c52f59153f724c1ad11ca1791edc77bab3806",
        )
        self.assertEqual(
            benchmark.TRANSFORMERS_REVISION,
            "a6895655b289cc3fdd29afec36904e0b8545ef92",
        )
        self.assertEqual(benchmark.PINNED_DEPENDENCY_VERSIONS["kernels"], "0.15.2")

    def test_required_regimes_cover_prefill_and_decode_grid(self):
        benchmark = _load_benchmark_module()

        regimes = benchmark.required_regimes()

        self.assertEqual(len(regimes), 8)
        self.assertEqual(
            {(row["phase"], row["batch_size"], row["sequence_length"]) for row in regimes},
            {
                ("prefill", 1, 128),
                ("prefill", 1, 512),
                ("prefill", 4, 128),
                ("prefill", 4, 512),
                ("decode", 1, 128),
                ("decode", 1, 512),
                ("decode", 8, 128),
                ("decode", 8, 512),
            },
        )
        self.assertEqual(len({row["regime_id"] for row in regimes}), 8)

    def test_configuration_inventory_makes_compile_exclusions_explicit(self):
        benchmark = _load_benchmark_module()
        compile_modes = [
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ]

        configurations = benchmark.configuration_inventory(compile_modes)
        rows = {row["configuration_id"]: row for row in configurations}

        for backend in benchmark.STOCK_BACKENDS:
            self.assertEqual(rows[f"{backend}__uncompiled"]["eligibility"], "required")

        for backend in ("eager", "batched_mm"):
            for mode in compile_modes:
                row = rows[f"{backend}__compiled__{mode}"]
                self.assertEqual(row["eligibility"], "required")
                self.assertIsNone(row["exclusion_reason"])

        for mode in compile_modes:
            grouped = rows[f"grouped_mm__compiled__{mode}"]
            default = rows[f"default__compiled__{mode}"]
            if mode in {"default", "max-autotune-no-cudagraphs"}:
                self.assertEqual(grouped["eligibility"], "required")
                self.assertEqual(default["eligibility"], "required")
            else:
                self.assertEqual(grouped["eligibility"], "excluded")
                self.assertIn("audited compile contract", grouped["exclusion_reason"])
                self.assertEqual(default["eligibility"], "excluded")
                self.assertIn("resolves to grouped_mm", default["exclusion_reason"])

            for backend in ("deepgemm", "sonicmoe"):
                optimized = rows[f"{backend}__compiled__{mode}"]
                self.assertEqual(optimized["eligibility"], "excluded")
                self.assertIn("external CUDA kernel", optimized["exclusion_reason"])

        expected_count = len(benchmark.STOCK_BACKENDS) * (1 + len(compile_modes))
        self.assertEqual(len(configurations), expected_count)

    def test_default_compile_inventory_matches_every_mode_in_pinned_torch(self):
        benchmark = _load_benchmark_module()
        torch = benchmark._torch()

        self.assertEqual(
            set(benchmark.DEFAULT_COMPILE_MODES),
            set(torch._inductor.list_mode_options()),
        )

    def test_contract_smoke_emits_every_row_without_claiming_evidence(self):
        benchmark = _load_benchmark_module()
        compile_modes = ["default", "max-autotune-no-cudagraphs"]

        artifact = benchmark.collect_results(
            mode="contract-smoke",
            compile_modes=compile_modes,
            command=["python", "benchmarks/olmoe_stock_baseline.py", "--smoke"],
            generated_at_utc="2026-07-14T00:00:00Z",
        )

        regimes = benchmark.required_regimes()
        configurations = benchmark.configuration_inventory(compile_modes)
        self.assertEqual(len(artifact["results"]), len(regimes) * len(configurations))
        self.assertEqual(artifact["mode"], "contract-smoke")
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertFalse(artifact["summary"]["cohort_complete"])
        self.assertFalse(artifact["summary"]["target_decision_ready"])
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertEqual(artifact["summary"]["candidate_measurements_present"], False)
        self.assertIn("contract_smoke_not_performance_evidence", artifact["summary"]["readiness_blockers"])
        self.assertEqual(artifact["best_stock_by_regime"], [])
        self.assertTrue(artifact["measurement_contract"]["prefill_builds_kv_cache"])
        self.assertEqual(
            artifact["pins"]["compile_modes"],
            list(benchmark.DEFAULT_COMPILE_MODES),
        )

        required_fields = {
            "regime_id",
            "phase",
            "batch_size",
            "sequence_length",
            "configuration_id",
            "experts_backend",
            "compiled",
            "compile_mode",
            "status",
            "reason",
            "correctness",
            "timing",
            "throughput_tokens_per_second",
            "preprocessing",
            "routing_overhead",
            "allocator",
        }
        for row in artifact["results"]:
            self.assertTrue(required_fields.issubset(row))
            if row["configuration_eligibility"] == "excluded":
                self.assertEqual(row["status"], "not_applicable")
                self.assertIsNotNone(row["reason"])
            else:
                self.assertEqual(row["status"], "blocked")
                self.assertEqual(row["reason"], "contract_smoke_not_performance_evidence")
            self.assertEqual(row["correctness"]["status"], "not_measured")

    def test_real_collection_selects_best_correct_stock_configuration_per_regime(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)

        def run_configuration(configuration, regimes):
            backend_cost = {
                "default": 0.8,
                "eager": 1.2,
                "batched_mm": 0.7,
                "grouped_mm": 0.6,
                "deepgemm": 0.5,
                "sonicmoe": 0.4,
            }[configuration["experts_backend"]]
            if configuration["compiled"]:
                backend_cost -= 0.1
            return [
                _successful_measurement(regime, backend_cost)
                for regime in regimes
            ]

        artifact = benchmark.collect_results(
            mode="real",
            compile_modes=benchmark.DEFAULT_COMPILE_MODES,
            environment=environment,
            run_configuration=run_configuration,
            command=["python", "benchmarks/olmoe_stock_baseline.py", "--real"],
            generated_at_utc="2026-07-14T00:00:00Z",
        )

        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertTrue(artifact["summary"]["cohort_complete"])
        self.assertFalse(artifact["summary"]["target_decision_ready"])
        self.assertEqual(len(artifact["best_stock_by_regime"]), 8)
        self.assertEqual(
            {row["configuration_id"] for row in artifact["best_stock_by_regime"]},
            {"sonicmoe__uncompiled"},
        )

    def test_real_collection_rejects_a_reduced_compile_inventory(self):
        benchmark = _load_benchmark_module()

        with self.assertRaisesRegex(ValueError, "full pinned compile mode inventory"):
            benchmark.collect_results(
                mode="real",
                compile_modes=["default"],
                environment=_available_environment(benchmark),
                run_configuration=lambda _configuration, regimes: [
                    _successful_measurement(regime, 1.0) for regime in regimes
                ],
            )

    def test_missing_executor_regime_becomes_an_explicit_failure(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)

        def omit_last_regime(_configuration, regimes):
            return [
                _successful_measurement(regime, 1.0)
                for regime in regimes[:-1]
            ]

        artifact = benchmark.collect_results(
            mode="real",
            compile_modes=benchmark.DEFAULT_COMPILE_MODES,
            environment=environment,
            run_configuration=omit_last_regime,
        )

        failures = [
            row
            for row in artifact["results"]
            if row["status"] == "failed" and row["reason"] == "executor_missing_required_regime"
        ]
        required_configuration_count = sum(
            row["eligibility"] == "required"
            for row in benchmark.configuration_inventory(benchmark.DEFAULT_COMPILE_MODES)
        )
        self.assertEqual(len(failures), required_configuration_count)
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertFalse(artifact["summary"]["cohort_complete"])
        self.assertIn("required_measurements_incomplete", artifact["summary"]["readiness_blockers"])
        self.assertEqual(artifact["best_stock_by_regime"], [])

    def test_interpretable_stock_failure_still_allows_best_successful_selection(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)

        def fail_deepgemm(configuration, regimes):
            if configuration["experts_backend"] == "deepgemm":
                raise RuntimeError("upstream kernel rejected this configuration")
            return [_successful_measurement(regime, 1.0) for regime in regimes]

        artifact = benchmark.collect_results(
            mode="real",
            compile_modes=benchmark.DEFAULT_COMPILE_MODES,
            environment=environment,
            run_configuration=fail_deepgemm,
        )

        self.assertTrue(artifact["summary"]["cohort_complete"])
        self.assertIn("stock_configuration_failures", artifact["summary"]["warnings"])
        self.assertNotIn("required_measurements_failed", artifact["summary"]["readiness_blockers"])
        self.assertEqual(len(artifact["best_stock_by_regime"]), 8)

    def test_dependency_failure_keeps_the_real_cohort_incomplete(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)

        def fail_deepgemm_dependency(configuration, regimes):
            if configuration["experts_backend"] == "deepgemm":
                raise ImportError("kernel package could not load")
            return [_successful_measurement(regime, 1.0) for regime in regimes]

        artifact = benchmark.collect_results(
            mode="real",
            compile_modes=benchmark.DEFAULT_COMPILE_MODES,
            environment=environment,
            run_configuration=fail_deepgemm_dependency,
        )

        self.assertFalse(artifact["summary"]["cohort_complete"])
        self.assertIn(
            "required_measurements_incomplete",
            artifact["summary"]["readiness_blockers"],
        )
        self.assertEqual(artifact["best_stock_by_regime"], [])

    def test_incorrect_faster_row_is_never_selected_as_best_stock(self):
        benchmark = _load_benchmark_module()
        regime = benchmark.required_regimes()[0]
        correct = _successful_measurement(regime, 1.0)
        correct.update(
            {
                "configuration_id": "eager__uncompiled",
                "experts_backend": "eager",
                "configuration_eligibility": "required",
            }
        )
        incorrect = _successful_measurement(regime, 0.1)
        incorrect.update(
            {
                "configuration_id": "batched_mm__uncompiled",
                "experts_backend": "batched_mm",
                "configuration_eligibility": "required",
            }
        )
        incorrect["correctness"]["status"] = "failed"

        selected = benchmark.select_best_stock_rows([incorrect, correct])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["configuration_id"], "eager__uncompiled")

    def test_write_json_artifact_round_trips_contract_smoke(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "olmoe_stock_baseline_smoke.json"
            artifact = benchmark.write_json_artifact(
                output_path,
                mode="contract-smoke",
                compile_modes=["default"],
                generated_at_utc="2026-07-14T00:00:00Z",
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)
        self.assertEqual(loaded["benchmark"], "olmoe_stock_baseline")
        self.assertEqual(loaded["schema_version"], 1)

    def test_runtime_backend_exclusion_is_explicit_and_does_not_invalidate_cohort(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)
        environment["backend_availability"] = {
            "deepgemm": {
                "status": "not_applicable",
                "reason": "requires_compute_capability_9_0",
            },
            "sonicmoe": {
                "status": "not_applicable",
                "reason": "requires_compute_capability_9_0",
            },
        }

        artifact = benchmark.collect_results(
            mode="real",
            compile_modes=benchmark.DEFAULT_COMPILE_MODES,
            environment=environment,
            run_configuration=lambda _configuration, regimes: [
                _successful_measurement(regime, 1.0) for regime in regimes
            ],
        )

        optimized_rows = [
            row
            for row in artifact["results"]
            if row["experts_backend"] in {"deepgemm", "sonicmoe"}
            and not row["compiled"]
        ]
        self.assertTrue(optimized_rows)
        self.assertTrue(all(row["status"] == "not_applicable" for row in optimized_rows))
        self.assertTrue(
            all(row["reason"] == "requires_compute_capability_9_0" for row in optimized_rows)
        )
        self.assertTrue(artifact["summary"]["cohort_complete"])

    def test_real_collection_builds_default_cuda_runner_after_preflight(self):
        benchmark = _load_benchmark_module()
        environment = _available_environment(benchmark)

        class FakeRunner:
            instances = []

            def __init__(self, **kwargs):
                self.instances.append(kwargs)

            def __call__(self, _configuration, regimes):
                return [_successful_measurement(regime, 1.0) for regime in regimes]

        with mock.patch.object(benchmark, "RealConfigurationRunner", FakeRunner):
            artifact = benchmark.collect_results(
                mode="real",
                compile_modes=benchmark.DEFAULT_COMPILE_MODES,
                environment=environment,
                warmup_repetitions=2,
                measured_repetitions=3,
            )

        self.assertEqual(
            FakeRunner.instances,
            [{"warmup_repetitions": 2, "measured_repetitions": 3}],
        )
        self.assertTrue(artifact["summary"]["cohort_complete"])

    def test_model_load_and_compile_kwargs_preserve_pins_and_backend_contract(self):
        benchmark = _load_benchmark_module()

        class FakeTorch:
            bfloat16 = object()

        default = benchmark.configuration_inventory(["default"])[0]
        eager = next(
            row
            for row in benchmark.configuration_inventory(["default"])
            if row["configuration_id"] == "eager__uncompiled"
        )
        grouped_compiled = next(
            row
            for row in benchmark.configuration_inventory(["default"])
            if row["configuration_id"] == "grouped_mm__compiled__default"
        )

        default_kwargs = benchmark.model_load_kwargs(default, FakeTorch)
        eager_kwargs = benchmark.model_load_kwargs(eager, FakeTorch)
        compile_kwargs = benchmark.compile_kwargs(grouped_compiled, "grouped_mm")

        self.assertEqual(default_kwargs["revision"], benchmark.MODEL_REVISION)
        self.assertIs(default_kwargs["dtype"], FakeTorch.bfloat16)
        self.assertEqual(default_kwargs["device_map"], {"": "cuda:0"})
        self.assertNotIn("experts_implementation", default_kwargs)
        self.assertEqual(eager_kwargs["experts_implementation"], "eager")
        self.assertEqual(compile_kwargs, {"mode": None, "fullgraph": True})

    def test_correctness_metrics_apply_both_predeclared_tolerances(self):
        benchmark = _load_benchmark_module()
        torch = benchmark._torch()
        reference = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        passed = benchmark.correctness_metrics(reference + 0.001, reference)
        failed = benchmark.correctness_metrics(reference + 1.0, reference)

        self.assertEqual(passed["status"], "passed")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(passed["max_abs_tolerance"], 0.125)
        self.assertEqual(passed["relative_l2_tolerance"], 0.01)

    def test_backend_preflight_distinguishes_hardware_exclusion_from_missing_dependency(self):
        benchmark = _load_benchmark_module()
        compatible_dependencies = {
            "kernels": benchmark.PINNED_DEPENDENCY_VERSIONS["kernels"],
            "nvidia-cutlass-dsl": benchmark.PINNED_DEPENDENCY_VERSIONS[
                "nvidia-cutlass-dsl"
            ],
        }

        ampere = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(8, 0),
            cuda_runtime="12.8",
            available_modules=set(),
            cuda_toolkit_version=None,
            nvcc_available=False,
            dependency_versions={},
        )
        hopper_without_kernels = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(9, 0),
            cuda_runtime="12.8",
            available_modules=set(),
            cuda_toolkit_version="12.8",
            nvcc_available=True,
            dependency_versions={},
        )
        hopper_ready = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(9, 0),
            cuda_runtime="12.8",
            available_modules={"kernels", "cutlass"},
            cuda_toolkit_version="12.8",
            nvcc_available=True,
            dependency_versions=compatible_dependencies,
        )
        blackwell_old_runtime = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(10, 0),
            cuda_runtime="12.8",
            available_modules={"kernels", "cutlass"},
            cuda_toolkit_version="12.8",
            nvcc_available=True,
            dependency_versions=compatible_dependencies,
        )
        blackwell_old_toolkit = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(10, 0),
            cuda_runtime="13.0",
            available_modules={"kernels", "cutlass"},
            cuda_toolkit_version="12.8",
            nvcc_available=True,
            dependency_versions=compatible_dependencies,
        )
        hopper_without_toolkit = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(9, 0),
            cuda_runtime="12.8",
            available_modules={"kernels", "cutlass"},
            cuda_toolkit_version=None,
            nvcc_available=False,
            dependency_versions=compatible_dependencies,
        )
        incompatible_kernels = benchmark.backend_availability(
            cuda_available=True,
            compute_capability=(9, 0),
            cuda_runtime="12.8",
            available_modules={"kernels", "cutlass"},
            cuda_toolkit_version="12.8",
            nvcc_available=True,
            dependency_versions={
                **compatible_dependencies,
                "kernels": "0.16.0",
            },
        )

        self.assertEqual(ampere["deepgemm"]["status"], "not_applicable")
        self.assertEqual(ampere["sonicmoe"]["status"], "not_applicable")
        self.assertEqual(hopper_without_kernels["deepgemm"]["status"], "blocked")
        self.assertEqual(hopper_without_kernels["sonicmoe"]["status"], "blocked")
        self.assertEqual(hopper_ready["deepgemm"]["status"], "available")
        self.assertEqual(hopper_ready["sonicmoe"]["status"], "available")
        self.assertEqual(hopper_without_toolkit["deepgemm"]["status"], "blocked")
        self.assertIn("nvcc", hopper_without_toolkit["deepgemm"]["reason"])
        self.assertEqual(incompatible_kernels["deepgemm"]["status"], "blocked")
        self.assertIn("incompatible_kernels_version", incompatible_kernels["deepgemm"]["reason"])
        self.assertEqual(blackwell_old_runtime["deepgemm"]["status"], "blocked")
        self.assertIn("12_9", blackwell_old_runtime["deepgemm"]["reason"])
        self.assertEqual(blackwell_old_toolkit["deepgemm"]["status"], "blocked")
        self.assertIn("nvcc_toolkit_12_9", blackwell_old_toolkit["deepgemm"]["reason"])

    def test_cuda_toolkit_probe_records_the_nvcc_version_used_by_deepgemm(self):
        benchmark = _load_benchmark_module()
        completed = mock.Mock(
            returncode=0,
            stdout="Cuda compilation tools, release 12.9, V12.9.41\n",
            stderr="",
        )

        with mock.patch.object(benchmark.os.path, "isfile", return_value=True):
            metadata = benchmark.probe_cuda_toolkit_metadata(
                environ={"CUDA_HOME": "/opt/cuda"},
                run_command=mock.Mock(return_value=completed),
            )

        self.assertEqual(metadata["status"], "measured")
        self.assertEqual(metadata["nvcc_path"], "/opt/cuda/bin/nvcc")
        self.assertEqual(metadata["cuda_toolkit_version"], "12.9")

    def test_core_dependency_preflight_requires_accelerate_and_exact_pins(self):
        benchmark = _load_benchmark_module()
        dependencies = dict(benchmark.PINNED_DEPENDENCY_VERSIONS)

        self.assertEqual(benchmark.dependency_readiness_blockers(dependencies), [])
        dependencies["accelerate"] = None
        self.assertIn(
            "accelerate_unavailable",
            benchmark.dependency_readiness_blockers(dependencies),
        )
        dependencies["accelerate"] = "0.1.0"
        self.assertIn(
            "accelerate_version_mismatch",
            benchmark.dependency_readiness_blockers(dependencies),
        )

    def test_prefill_builds_cache_and_decode_resets_allocator_after_prompt_setup(self):
        benchmark = _load_benchmark_module()

        prefill_log = []
        prefill_torch = _FakeTorch(prefill_log)
        prefill_model = _FakeModel(prefill_log)
        with mock.patch.object(
            benchmark,
            "_deterministic_tokens",
            return_value=_FakeTensor(),
        ):
            prefill_measurement, _ = benchmark._measure_regime(
                prefill_model,
                benchmark.required_regimes()[0],
                prefill_torch,
                resolved_backend="eager",
                warmup_repetitions=0,
                measured_repetitions=1,
            )

        prefill_calls = [entry for entry in prefill_log if entry[0] == "model"]
        self.assertEqual(len(prefill_calls), 1)
        self.assertTrue(prefill_calls[0][1]["use_cache"])
        self.assertTrue(prefill_measurement["allocator"]["cache_construction_in_timed_region"])

        decode_log = []
        decode_torch = _FakeTorch(decode_log)
        decode_model = _FakeModel(decode_log)
        decode_regime = next(
            regime for regime in benchmark.required_regimes() if regime["phase"] == "decode"
        )
        with mock.patch.object(
            benchmark,
            "_deterministic_tokens",
            return_value=_FakeTensor(),
        ):
            decode_measurement, _ = benchmark._measure_regime(
                decode_model,
                decode_regime,
                decode_torch,
                resolved_backend="grouped_mm",
                warmup_repetitions=0,
                measured_repetitions=2,
            )

        decode_calls = [entry for entry in decode_log if entry[0] == "model"]
        self.assertEqual(len(decode_calls), 4)
        for prompt, token in zip(decode_calls[::2], decode_calls[1::2]):
            self.assertTrue(prompt[1]["use_cache"])
            self.assertNotIn("past_key_values", prompt[1])
            self.assertTrue(token[1]["use_cache"])
            self.assertIn("past_key_values", token[1])
        reset_indices = [index for index, entry in enumerate(decode_log) if entry[0] == "reset_peak"]
        prompt_indices = [
            index
            for index, entry in enumerate(decode_log)
            if entry[0] == "model" and "past_key_values" not in entry[1]
        ]
        token_indices = [
            index
            for index, entry in enumerate(decode_log)
            if entry[0] == "model" and "past_key_values" in entry[1]
        ]
        self.assertEqual(len(reset_indices), 2)
        self.assertTrue(
            all(prompt < reset < token for prompt, reset, token in zip(prompt_indices, reset_indices, token_indices))
        )
        self.assertTrue(decode_measurement["allocator"]["decode_prompt_setup_excluded"])
        self.assertEqual(
            decode_measurement["allocator"]["cache_resident_allocated_bytes"],
            [100, 100],
        )

    def test_loaded_model_revision_must_match_the_pin(self):
        benchmark = _load_benchmark_module()

        good_model = mock.Mock(config=mock.Mock(_commit_hash=benchmark.MODEL_REVISION))
        bad_model = mock.Mock(config=mock.Mock(_commit_hash="wrong"))

        self.assertEqual(
            benchmark.validate_loaded_model_revision(good_model),
            benchmark.MODEL_REVISION,
        )
        with self.assertRaisesRegex(RuntimeError, "model revision mismatch"):
            benchmark.validate_loaded_model_revision(bad_model)

    def test_real_cli_freezes_warmup_and_measurement_counts(self):
        benchmark = _load_benchmark_module()

        args = benchmark._parse_args(
            [
                "--real",
                "--warmup-repetitions",
                "5",
                "--measured-repetitions",
                "20",
                "--json-output",
                "result.json",
            ]
        )

        self.assertEqual(args.warmup_repetitions, 5)
        self.assertEqual(args.measured_repetitions, 20)

    def test_decode_stage_mirrors_stock_grouped_to_batched_switch(self):
        benchmark = _load_benchmark_module()

        self.assertEqual(
            benchmark.stage_experts_backends("decode", "grouped_mm"),
            {"setup": "grouped_mm", "timed": "batched_mm"},
        )
        self.assertEqual(
            benchmark.stage_experts_backends("prefill", "grouped_mm"),
            {"setup": "grouped_mm", "timed": "grouped_mm"},
        )
        self.assertEqual(
            benchmark.stage_experts_backends("decode", "deepgemm"),
            {"setup": "deepgemm", "timed": "deepgemm"},
        )

    def test_nvidia_driver_probe_records_driver_and_gpu_uuid(self):
        benchmark = _load_benchmark_module()

        completed = mock.Mock(
            returncode=0,
            stdout="GPU-1234, 580.42.01\n",
            stderr="",
        )
        run_command = mock.Mock(return_value=completed)

        metadata = benchmark.probe_nvidia_smi_metadata(run_command=run_command)

        self.assertEqual(metadata["status"], "measured")
        self.assertEqual(metadata["gpu_uuid"], "GPU-1234")
        self.assertEqual(metadata["driver_version"], "580.42.01")
        command = run_command.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")
        self.assertIn("--id=0", command)


def _available_environment(benchmark):
    return {
        "preflight_status": "ready",
        "readiness_blockers": [],
        "cuda_available": True,
        "cuda_device_name": "synthetic-hopper",
        "cuda_compute_capability": [9, 0],
        "cuda_runtime": "12.4",
        "torch_version": "2.12.1",
        "transformers_version": "test",
        "transformers_revision": benchmark.TRANSFORMERS_REVISION,
        "model_revision": benchmark.MODEL_REVISION,
        "dtype": "bfloat16",
    }


def _successful_measurement(regime, median_cuda_seconds):
    return {
        "regime_id": regime["regime_id"],
        "status": "ok",
        "reason": None,
        "correctness": {
            "status": "passed",
            "reference": "eager__uncompiled",
            "max_abs_error": 0.0,
            "relative_l2_error": 0.0,
            "max_abs_tolerance": 0.125,
            "relative_l2_tolerance": 0.01,
        },
        "timing": {
            "cuda_event_median_seconds": median_cuda_seconds,
            "wall_median_seconds": median_cuda_seconds + 0.01,
            "warmup_repetitions": 2,
            "measured_repetitions": 3,
        },
        "throughput_tokens_per_second": 100.0 / median_cuda_seconds,
        "preprocessing": {"status": "measured", "median_seconds": 0.01},
        "routing_overhead": {"status": "not_measured", "median_seconds": None},
        "allocator": {"status": "measured", "peak_allocated_bytes": 1024},
    }


class _FakeTensor:
    def detach(self):
        return self

    def to(self, **_kwargs):
        return self


class _FakeOutput:
    def __init__(self):
        self.logits = _FakeTensor()
        self.past_key_values = object()


class _FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self):
        self.log.append(("event_record",))

    def elapsed_time(self, _other):
        return 1.0


class _FakeCuda:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append(("synchronize",))

    def reset_peak_memory_stats(self, _device):
        self.log.append(("reset_peak",))

    def memory_allocated(self, _device):
        self.log.append(("memory_allocated",))
        return 100

    def max_memory_allocated(self, _device):
        self.log.append(("max_memory_allocated",))
        return 150

    def Event(self, *, enable_timing):
        assert enable_timing
        return _FakeEvent(self.log)


class _FakeTorch:
    long = object()
    float32 = object()

    def __init__(self, log):
        self.cuda = _FakeCuda(log)

    def inference_mode(self):
        return nullcontext()

    def ones_like(self, *_args, **_kwargs):
        return _FakeTensor()

    def ones(self, *_args, **_kwargs):
        return _FakeTensor()


class _FakeModel:
    class config:
        vocab_size = 16

    def __init__(self, log):
        self.log = log

    def __call__(self, **kwargs):
        self.log.append(("model", kwargs))
        return _FakeOutput()

    def set_experts_implementation(self, backend):
        self.log.append(("set_backend", backend))


if __name__ == "__main__":
    unittest.main()
