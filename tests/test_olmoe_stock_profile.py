import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_module(filename, module_name):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_profile_module():
    return _load_module("olmoe_stock_profile.py", "olmoe_stock_profile")


class _LegacyCudaEvent:
    def __init__(self, name, cpu_time, cuda_time, *, count=1, scope="expert_layer"):
        self.key = name
        self.self_cpu_time_total = cpu_time
        self.self_cuda_time_total = cuda_time
        self.count = count
        self.scope = scope


class OlmoeStockProfileTests(unittest.TestCase):
    def test_diagnostic_contract_is_pinned_to_middle_layer_and_real_regime(self):
        profile = _load_profile_module()

        self.assertEqual(profile.DIAGNOSTIC_LAYER_INDEX, 8)
        self.assertEqual(profile.DIAGNOSTIC_LAYER_PATH, "model.layers.8.mlp")
        self.assertEqual(profile.DIAGNOSTIC_REGIME_ID, "prefill_b1_s512")
        self.assertEqual(profile.MODEL, "allenai/OLMoE-1B-7B-0924")
        self.assertEqual(len(profile.ATTRIBUTION_CATEGORIES), 10)
        self.assertEqual(profile.ATTRIBUTION_CATEGORIES[-1], "unclassified")

    def test_event_classification_has_stable_precedence_and_explicit_unknowns(self):
        profile = _load_profile_module()
        cases = {
            "aten::topk": "routing_top_k",
            "aten::sort_copy": "sorting_permutation",
            "aten::bincount": "offsets_histogram",
            "aten::grouped_mm": "expert_contractions",
            "aten::silu": "activation_gating",
            "aten::mul": "activation_gating",
            "aten::index": "sorting_permutation",
            "aten::index_add_empty": "aggregation_scatter",
            "aten::contiguous": "layout_copy_conversion",
            "aten::empty": "allocation",
            "TorchDynamo Cache Lookup": "compilation",
            "vendor::mystery_kernel": "unclassified",
            "Torch-Compiled Region": "unclassified",
        }

        observed = {
            name: profile.classify_event_name(name, scope="expert_layer")
            for name in cases
        }

        self.assertEqual(observed, cases)
        self.assertEqual(
            profile.classify_event_name("aten::linear", scope="full_model"),
            "unclassified",
        )
        self.assertEqual(
            profile.classify_event_name("aten::linear", scope="expert_layer"),
            "expert_contractions",
        )
        self.assertEqual(
            profile.classify_event_name("aten::linear", scope="routing_top_k"),
            "routing_top_k",
        )

    def test_attribution_classifies_every_event_once_and_conserves_totals(self):
        profile = _load_profile_module()
        events = [
            SimpleNamespace(
                key="aten::topk",
                self_cpu_time_total=3.0,
                self_device_time_total=5.0,
                count=2,
                scope="routing_top_k",
            ),
            _LegacyCudaEvent("aten::mm", 7.0, 11.0),
            _LegacyCudaEvent("unknown::kernel", 13.0, 17.0),
        ]

        attribution = profile.summarize_events(events, default_scope="expert_layer")

        self.assertEqual(attribution["event_group_count"], 3)
        self.assertEqual(sum(row["event_group_count"] for row in attribution["categories"]), 3)
        self.assertEqual(attribution["totals"]["self_cpu_time_us"], 23.0)
        self.assertEqual(attribution["totals"]["self_device_time_us"], 33.0)
        self.assertAlmostEqual(
            sum(row["device_time_proportion"] for row in attribution["categories"]),
            1.0,
        )
        self.assertEqual(attribution["unclassified_event_names"], ["unknown::kernel"])
        self.assertEqual(
            [row["category"] for row in attribution["events"]],
            ["routing_top_k", "expert_contractions", "unclassified"],
        )

    def test_native_profiler_scope_does_not_override_the_declared_semantic_scope(self):
        profile = _load_profile_module()
        event = SimpleNamespace(
            key="aten::linear",
            self_cpu_time_total=1.0,
            self_device_time_total=2.0,
            count=1,
            scope=0,
        )

        attribution = profile.summarize_events(
            [event],
            default_scope="expert_layer",
        )

        self.assertEqual(attribution["events"][0]["scope"], "expert_layer")
        self.assertEqual(attribution["events"][0]["category"], "expert_contractions")

    def test_merged_router_and_expert_attribution_preserves_self_times(self):
        profile = _load_profile_module()
        routing = profile.summarize_events(
            [
                SimpleNamespace(
                    key="aten::topk",
                    self_cpu_time_total=3.0,
                    self_device_time_total=5.0,
                )
            ],
            default_scope="routing_top_k",
        )
        expert = profile.summarize_events(
            [SimpleNamespace(key="aten::mm", self_cpu_time_total=7.0, self_device_time_total=11.0)],
            default_scope="expert_layer",
        )

        merged = profile.merge_attributions([routing, expert])

        self.assertEqual(merged["totals"]["self_cpu_time_us"], 10.0)
        self.assertEqual(merged["totals"]["self_device_time_us"], 16.0)
        self.assertEqual(
            [row["category"] for row in merged["events"]],
            ["routing_top_k", "expert_contractions"],
        )

    def test_device_timing_is_required_for_real_cupti_attribution(self):
        profile = _load_profile_module()
        attribution = profile.summarize_events(
            [SimpleNamespace(key="aten::mm", self_cpu_time_total=2.0, count=1)],
            default_scope="expert_layer",
        )

        with self.assertRaisesRegex(RuntimeError, "CUPTI.*device events"):
            profile.require_device_attribution(attribution)

    def test_cupti_trace_requires_an_actual_cuda_kernel_event(self):
        profile = _load_profile_module()

        class Event:
            def __init__(self, device_type):
                self._device_type = device_type

            def device_type(self):
                return self._device_type

        def profiler(events):
            results = SimpleNamespace(events=lambda: events)
            return SimpleNamespace(profiler=SimpleNamespace(kineto_results=results))

        with self.assertRaisesRegex(RuntimeError, "CUPTI.*CUDA kernel trace"):
            profile.require_cupti_trace(profiler([Event("DeviceType.CPU")]))

        profile.require_cupti_trace(profiler([Event("DeviceType.CUDA")]))

    def test_contract_smoke_is_row_complete_but_contains_no_measurements(self):
        profile = _load_profile_module()

        artifact = profile.collect_results(
            mode="contract-smoke",
            command=["python", "benchmarks/olmoe_stock_profile.py", "--smoke"],
            generated_at_utc="2026-07-14T00:00:00Z",
        )

        self.assertEqual(len(artifact["full_model_profiles"]), 8)
        self.assertTrue(artifact["summary"]["row_inventory_complete"])
        self.assertFalse(artifact["summary"]["profile_complete"])
        self.assertFalse(artifact["summary"]["target_decision_ready"])
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertEqual(artifact["summary"]["candidate_measurements_present"], False)
        self.assertEqual(
            artifact["summary"]["readiness_blockers"],
            ["contract_smoke_not_performance_evidence"],
        )
        self.assertTrue(
            all(row["status"] == "not_measured" for row in artifact["full_model_profiles"])
        )
        self.assertTrue(
            all(
                row["attribution"]["totals"]["self_device_time_us"] is None
                for row in artifact["full_model_profiles"]
            )
        )
        self.assertEqual(artifact["expert_layer_diagnostic"]["status"], "not_measured")
        self.assertEqual(
            artifact["expert_layer_diagnostic"]["evidence_boundary"],
            "diagnostic_only_not_end_to_end_evidence",
        )

    def test_real_collection_requires_a_complete_matching_stock_artifact(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)
        baseline["summary"]["cohort_complete"] = False

        with self.assertRaisesRegex(ValueError, "complete real stock baseline"):
            profile.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=_ready_environment(),
                run_profile=lambda _artifact: {},
            )

    def test_real_collection_rejects_a_different_hardware_environment(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)
        current = _ready_environment()
        current["gpu_uuid"] = "GPU-different"

        with self.assertRaisesRegex(RuntimeError, "environment_mismatch:gpu_uuid"):
            profile.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=current,
                run_profile=lambda _artifact: {},
            )

    def test_real_collection_fails_explicitly_without_cuda_profiler_activity(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)
        current = _ready_environment()
        current["profiler_cuda_activity_available"] = False

        with self.assertRaisesRegex(RuntimeError, "cuda_profiler_activity_unavailable"):
            profile.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=current,
                run_profile=lambda _artifact: {},
            )

    def test_real_collection_binds_every_profile_to_best_stock_and_stays_nonclaiming(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)

        def run_profile(_artifact):
            return {
                "full_model_profiles": [
                    _successful_profile_row(best_row)
                    for best_row in baseline["best_stock_by_regime"]
                ],
                "expert_layer_diagnostic": _successful_diagnostic(profile),
            }

        artifact = profile.collect_results(
            mode="real",
            baseline_artifact=baseline,
            environment=_ready_environment(),
            run_profile=run_profile,
            generated_at_utc="2026-07-14T00:00:00Z",
        )

        self.assertTrue(artifact["summary"]["profile_complete"])
        self.assertFalse(artifact["summary"]["target_decision_ready"])
        self.assertEqual(artifact["summary"]["performance_claim"], "none")
        self.assertEqual(
            {
                (row["regime_id"], row["configuration_id"])
                for row in artifact["full_model_profiles"]
            },
            {
                (row["regime_id"], row["configuration_id"])
                for row in baseline["best_stock_by_regime"]
            },
        )
        self.assertEqual(
            artifact["expert_layer_diagnostic"]["correctness"]["status"],
            "passed",
        )
        self.assertTrue(
            all(
                row["attribution"]["cupti_trace_status"]
                == "cuda_kernel_events_present"
                for row in artifact["full_model_profiles"]
            )
        )

    def test_real_collection_rejects_profile_metadata_that_diverges_from_best_stock(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)

        def run_profile(_artifact):
            rows = [
                _successful_profile_row(best_row)
                for best_row in baseline["best_stock_by_regime"]
            ]
            rows[0]["experts_backend"] = "grouped_mm"
            return {
                "full_model_profiles": rows,
                "expert_layer_diagnostic": _successful_diagnostic(profile),
            }

        with self.assertRaisesRegex(ValueError, "profile metadata diverges"):
            profile.collect_results(
                mode="real",
                baseline_artifact=baseline,
                environment=_ready_environment(),
                run_profile=run_profile,
            )

    def test_real_collection_is_incomplete_without_recorded_cupti_trace_status(self):
        profile = _load_profile_module()
        baseline = _complete_baseline_artifact(profile)

        def run_profile(_artifact):
            rows = [
                _successful_profile_row(best_row)
                for best_row in baseline["best_stock_by_regime"]
            ]
            rows[0]["attribution"].pop("cupti_trace_status")
            return {
                "full_model_profiles": rows,
                "expert_layer_diagnostic": _successful_diagnostic(profile),
            }

        artifact = profile.collect_results(
            mode="real",
            baseline_artifact=baseline,
            environment=_ready_environment(),
            run_profile=run_profile,
        )

        self.assertFalse(artifact["summary"]["profile_complete"])
        self.assertEqual(
            artifact["summary"]["readiness_blockers"],
            ["profile_measurements_incomplete"],
        )

    def test_selected_expert_layer_is_discovered_and_real_io_is_captured(self):
        profile = _load_profile_module()
        torch = profile._torch()

        class SparseLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Identity()
                self.experts = torch.nn.Identity()

            def forward(self, hidden_states):
                return hidden_states * 2

        selected = SparseLayer()
        layers = [SimpleNamespace(mlp=torch.nn.Identity()) for _ in range(16)]
        layers[profile.DIAGNOSTIC_LAYER_INDEX] = SimpleNamespace(mlp=selected)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=layers),
            config=SimpleNamespace(num_hidden_layers=16),
        )

        discovered = profile.find_expert_layer(model)
        captured = profile.capture_real_activation(
            discovered,
            lambda: discovered(torch.ones((1, 4, 3))),
        )

        self.assertIs(discovered, selected)
        self.assertEqual(tuple(captured["input"].shape), (1, 4, 3))
        self.assertEqual(tuple(captured["output"].shape), (1, 4, 3))
        self.assertEqual(captured["call_count"], 1)
        self.assertFalse(captured["input"].requires_grad)
        self.assertTrue(torch.equal(captured["output"], captured["input"] * 2))

    def test_write_json_artifact_round_trips_profile_smoke(self):
        profile = _load_profile_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "profile_smoke.json"
            artifact = profile.write_json_artifact(
                output_path,
                mode="contract-smoke",
                generated_at_utc="2026-07-14T00:00:00Z",
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)
        self.assertEqual(loaded["benchmark"], "olmoe_stock_profile")
        self.assertEqual(loaded["schema_version"], 1)


def _ready_environment():
    return {
        "preflight_status": "ready",
        "readiness_blockers": [],
        "cuda_available": True,
        "profiler_cuda_activity_available": True,
        "kineto_available": True,
        "gpu_uuid": "GPU-pinned",
        "nvidia_driver_version": "580.42.01",
        "cuda_runtime": "13.0",
        "torch_version": "2.12.1",
        "transformers_revision": "a6895655b289cc3fdd29afec36904e0b8545ef92",
        "model_revision": "bd1c52f59153f724c1ad11ca1791edc77bab3806",
        "dtype": "bfloat16",
        "cupti_validation": "required_device_events_at_runtime",
    }


def _complete_baseline_artifact(profile):
    configuration = {
        "configuration_id": "eager__uncompiled",
        "experts_backend": "eager",
        "compiled": False,
        "compile_mode": None,
        "fullgraph": False,
        "eligibility": "required",
        "exclusion_reason": None,
    }
    best_rows = [
        {
            "regime_id": regime["regime_id"],
            "configuration_id": configuration["configuration_id"],
            "experts_backend": "eager",
            "compiled": False,
            "compile_mode": None,
            "cuda_event_median_seconds": 0.01,
            "throughput_tokens_per_second": 100.0,
        }
        for regime in profile.baseline.required_regimes()
    ]
    results = [
        {
            **row,
            "status": "ok",
            "correctness": {"status": "passed"},
            "timing": {"cuda_event_median_seconds": 0.01},
        }
        for row in best_rows
    ]
    return {
        "schema_version": 1,
        "benchmark": profile.baseline.BENCHMARK,
        "mode": "real",
        "pins": {
            "model": profile.MODEL,
            "model_revision": profile.MODEL_REVISION,
            "transformers_revision": profile.TRANSFORMERS_REVISION,
            "dtype": profile.DTYPE,
        },
        "environment": _ready_environment(),
        "regimes": profile.baseline.required_regimes(),
        "configuration_inventory": [configuration],
        "results": results,
        "best_stock_by_regime": best_rows,
        "summary": {"cohort_complete": True, "performance_claim": "none"},
    }


def _attribution():
    return {
        "timing_status": "profiled_self_time",
        "cupti_trace_status": "cuda_kernel_events_present",
        "event_group_count": 1,
        "events": [
            {
                "name": "aten::mm",
                "scope": "expert_layer",
                "category": "expert_contractions",
                "count": 1,
                "self_cpu_time_us": 1.0,
                "self_device_time_us": 2.0,
            }
        ],
        "categories": [],
        "totals": {"self_cpu_time_us": 1.0, "self_device_time_us": 2.0},
        "unclassified_event_names": [],
    }


def _successful_profile_row(best_row):
    return {
        "regime_id": best_row["regime_id"],
        "configuration_id": best_row["configuration_id"],
        "experts_backend": best_row["experts_backend"],
        "compiled": best_row["compiled"],
        "compile_mode": best_row["compile_mode"],
        "status": "ok",
        "reason": None,
        "attribution": _attribution(),
    }


def _successful_diagnostic(profile):
    return {
        "regime_id": profile.DIAGNOSTIC_REGIME_ID,
        "layer_index": profile.DIAGNOSTIC_LAYER_INDEX,
        "layer_path": profile.DIAGNOSTIC_LAYER_PATH,
        "status": "ok",
        "reason": None,
        "configuration_id": "eager__uncompiled",
        "experts_backend": "eager",
        "selected_full_model_compiled": False,
        "replay_compiled": False,
        "input": {"shape": [1, 512, 2048], "dtype": "torch.bfloat16"},
        "output": {"shape": [1, 512, 2048], "dtype": "torch.bfloat16"},
        "correctness": {"status": "passed"},
        "attribution": _attribution(),
        "evidence_boundary": "diagnostic_only_not_end_to_end_evidence",
    }


if __name__ == "__main__":
    unittest.main()
