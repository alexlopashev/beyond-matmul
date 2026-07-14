#!/usr/bin/env python3
"""Profile the best pinned OLMoE stock paths and one real-activation expert layer."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


def _load_baseline_module():
    module_path = Path(__file__).with_name("olmoe_stock_baseline.py")
    spec = importlib.util.spec_from_file_location(
        "_beyond_matmul_olmoe_stock_baseline",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline_module()

BENCHMARK = "olmoe_stock_profile"
CONTRACT_PATH = baseline.CONTRACT_PATH
MODEL = baseline.MODEL
MODEL_REVISION = baseline.MODEL_REVISION
TRANSFORMERS_REVISION = baseline.TRANSFORMERS_REVISION
DTYPE = baseline.DTYPE
DIAGNOSTIC_REGIME_ID = "prefill_b1_s512"
DIAGNOSTIC_LAYER_INDEX = 8
DIAGNOSTIC_LAYER_PATH = f"model.layers.{DIAGNOSTIC_LAYER_INDEX}.mlp"
DEFAULT_PROFILE_WARMUPS = 1
ATTRIBUTION_CATEGORIES = (
    "routing_top_k",
    "sorting_permutation",
    "offsets_histogram",
    "expert_contractions",
    "activation_gating",
    "aggregation_scatter",
    "layout_copy_conversion",
    "allocation",
    "compilation",
    "unclassified",
)
ENVIRONMENT_BINDING_KEYS = (
    "gpu_uuid",
    "nvidia_driver_version",
    "cuda_runtime",
    "torch_version",
    "transformers_revision",
    "model_revision",
    "dtype",
)

ProfileExecutor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def classify_event_name(name: str, *, scope: str = "full_model") -> str:
    """Classify one profiler event with ordered, reviewable precedence."""

    normalized = str(name).lower()
    if scope == "routing_top_k":
        return "routing_top_k"

    if _contains_any(
        normalized,
        (
            "torchdynamo",
            "graph break",
            "compile_to_module",
            "compile_fx",
            "inductor compilation",
        ),
    ):
        return "compilation"
    if _contains_any(normalized, ("topk", "top_k", "router", "routing")):
        return "routing_top_k"
    if _contains_any(normalized, ("sort", "argsort", "permute")):
        return "sorting_permutation"
    if _contains_any(
        normalized,
        ("bincount", "histogram", "histc", "cumsum", "prefix_sum", "offset"),
    ):
        return "offsets_histogram"
    if _contains_any(
        normalized,
        ("index_add", "scatter", "segment_reduce", "index_put", "atomic_add"),
    ):
        return "aggregation_scatter"
    if scope == "expert_layer" and _contains_any(
        normalized,
        ("aten::index", "index_select", "aten::where", "gather", "one_hot", "nonzero"),
    ):
        return "sorting_permutation"
    if scope == "expert_layer" and _contains_any(normalized, ("aten::mul", "mul_")):
        return "activation_gating"
    if _contains_any(
        normalized,
        ("silu", "gelu", "relu", "sigmoid", "swiglu", "activation"),
    ):
        return "activation_gating"
    if _contains_any(
        normalized,
        (
            "contiguous",
            "_to_copy",
            "copy_",
            "transpose",
            "reshape",
            "view",
            "as_strided",
            "convert",
        ),
    ):
        return "layout_copy_conversion"
    if _contains_any(
        normalized,
        ("empty", "zeros", "ones", "allocate", "allocation", "malloc"),
    ):
        return "allocation"

    contraction_tokens = (
        "grouped_mm",
        "grouped_gemm",
        "batched_mm",
        "deepgemm",
        "sonicmoe",
        "cutlass",
        "gemm",
        "matmul",
        "linear",
        "aten::mm",
        "aten::bmm",
    )
    if scope == "expert_layer" and _contains_any(normalized, contraction_tokens):
        return "expert_contractions"
    if scope == "full_model" and _contains_any(
        normalized,
        ("grouped_mm", "grouped_gemm", "deepgemm", "sonicmoe"),
    ):
        return "expert_contractions"
    return "unclassified"


def summarize_events(
    events: Sequence[Any],
    *,
    default_scope: str = "full_model",
) -> Dict[str, Any]:
    """Assign each aggregated profiler event once and conserve self-time totals."""

    normalized_events: List[Dict[str, Any]] = []
    for event in events:
        name = str(_event_value(event, "key", _event_value(event, "name", "")))
        scope = str(
            event.get("scope", default_scope)
            if isinstance(event, Mapping)
            else default_scope
        )
        cpu_value = _event_value(event, "self_cpu_time_total", None)
        if cpu_value is None:
            cpu_value = _event_value(event, "self_cpu_time_us", 0.0)
        cpu_time = _nonnegative_float(cpu_value)
        device_value = _event_value(event, "self_device_time_total", None)
        if device_value is None:
            device_value = _event_value(event, "self_cuda_time_total", None)
        if device_value is None:
            device_value = _event_value(event, "self_device_time_us", 0.0)
        device_time = _nonnegative_float(device_value)
        count = int(_event_value(event, "count", 1) or 0)
        normalized_events.append(
            {
                "name": name,
                "scope": scope,
                "category": classify_event_name(name, scope=scope),
                "count": count,
                "self_cpu_time_us": cpu_time,
                "self_device_time_us": device_time,
            }
        )

    cpu_total = sum(row["self_cpu_time_us"] for row in normalized_events)
    device_total = sum(row["self_device_time_us"] for row in normalized_events)
    categories = []
    for category in ATTRIBUTION_CATEGORIES:
        selected = [row for row in normalized_events if row["category"] == category]
        category_cpu = sum(row["self_cpu_time_us"] for row in selected)
        category_device = sum(row["self_device_time_us"] for row in selected)
        categories.append(
            {
                "category": category,
                "event_group_count": len(selected),
                "call_count": sum(row["count"] for row in selected),
                "self_cpu_time_us": category_cpu,
                "self_device_time_us": category_device,
                "cpu_time_proportion": category_cpu / cpu_total if cpu_total > 0.0 else None,
                "device_time_proportion": (
                    category_device / device_total if device_total > 0.0 else None
                ),
            }
        )

    return {
        "timing_status": "profiled_self_time",
        "event_group_count": len(normalized_events),
        "events": normalized_events,
        "categories": categories,
        "totals": {
            "self_cpu_time_us": cpu_total,
            "self_device_time_us": device_total,
        },
        "unclassified_event_names": sorted(
            {row["name"] for row in normalized_events if row["category"] == "unclassified"}
        ),
    }


def merge_attributions(attributions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    events: List[Mapping[str, Any]] = []
    for attribution in attributions:
        events.extend(attribution.get("events", []))
    return summarize_events(events)


def summarize_profiler(
    profiler: Any,
    *,
    default_scope: str,
) -> Dict[str, Any]:
    """Attribute frontend CPU rows whose device time already owns linked kernels."""

    frontend_events = [
        event
        for event in profiler.key_averages()
        if _device_type_name(getattr(event, "device_type", None)) == "CPU"
    ]
    return summarize_events(frontend_events, default_scope=default_scope)


def require_device_attribution(attribution: Mapping[str, Any]) -> None:
    device_time = attribution.get("totals", {}).get("self_device_time_us")
    if not isinstance(device_time, (int, float)) or float(device_time) <= 0.0:
        raise RuntimeError(
            "CUPTI did not produce device events; CUDA profiler attribution is unavailable"
        )


def require_cupti_trace(profiler: Any) -> None:
    """Reject PyTorch's legacy CUDA timing fallback, which has no kernel trace."""

    wrapped = getattr(profiler, "profiler", None)
    results = getattr(wrapped, "kineto_results", None)
    events_method = getattr(results, "events", None)
    events = events_method() if callable(events_method) else []
    has_cuda_kernel = any(
        _device_type_name(event.device_type()) == "CUDA"
        and str(event.activity_type()).lower() == "kernel"
        for event in events
        if callable(getattr(event, "device_type", None))
        and callable(getattr(event, "activity_type", None))
    )
    if not has_cuda_kernel:
        raise RuntimeError(
            "CUPTI did not produce a CUDA kernel trace; legacy CUDA timing is insufficient"
        )


def collect_results(
    *,
    mode: str,
    baseline_artifact: Mapping[str, Any] | None = None,
    baseline_artifact_path: str | None = None,
    environment: Mapping[str, Any] | None = None,
    run_profile: ProfileExecutor | None = None,
    profile_warmups: int = DEFAULT_PROFILE_WARMUPS,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    if mode not in {"contract-smoke", "real"}:
        raise ValueError(f"unsupported mode: {mode}")
    if profile_warmups < 0:
        raise ValueError("profile warmups must be non-negative")

    regimes = baseline.required_regimes()
    generated_at = generated_at_utc or _utc_now()
    if mode == "contract-smoke":
        return {
            "schema_version": 1,
            "benchmark": BENCHMARK,
            "contract": CONTRACT_PATH,
            "mode": mode,
            "generated_at_utc": generated_at,
            "command": list(command or sys.argv),
            "pins": _pins(),
            "baseline_binding": {
                "status": "not_provided",
                "path": None,
                "sha256": None,
                "cohort_complete": False,
            },
            "profiler_contract": _profiler_contract(profile_warmups),
            "environment": _smoke_environment(),
            "full_model_profiles": [_empty_profile_row(regime) for regime in regimes],
            "expert_layer_diagnostic": _empty_diagnostic(),
            "summary": {
                "row_inventory_complete": True,
                "profile_complete": False,
                "target_decision_ready": False,
                "candidate_measurements_present": False,
                "performance_claim": "none",
                "readiness_blockers": ["contract_smoke_not_performance_evidence"],
            },
        }

    if baseline_artifact is None:
        raise ValueError("real mode requires a complete real stock baseline artifact")
    best_rows = validate_baseline_artifact(baseline_artifact)
    resolved_environment = dict(environment or probe_profiler_environment())
    blockers = profile_readiness_blockers(
        resolved_environment,
        baseline_artifact.get("environment", {}),
    )
    if blockers:
        raise RuntimeError("profiler preflight blocked: " + ", ".join(blockers))

    executor = run_profile or RealProfileRunner(profile_warmups=profile_warmups)
    payload = dict(executor(baseline_artifact))
    full_profiles = [dict(row) for row in payload.get("full_model_profiles", [])]
    diagnostic = dict(payload.get("expert_layer_diagnostic", {}))
    expected_bindings = {
        (str(row["regime_id"]), str(row["configuration_id"])) for row in best_rows
    }
    observed_bindings = {
        (str(row.get("regime_id")), str(row.get("configuration_id")))
        for row in full_profiles
    }
    row_inventory_complete = (
        len(full_profiles) == len(expected_bindings)
        and observed_bindings == expected_bindings
    )
    if not row_inventory_complete:
        raise ValueError("profile rows must bind exactly once to every best stock regime")
    best_by_binding = {
        (str(row["regime_id"]), str(row["configuration_id"])): row
        for row in best_rows
    }
    for row in full_profiles:
        binding = (str(row["regime_id"]), str(row["configuration_id"]))
        selected = best_by_binding[binding]
        if any(
            row.get(key) != selected.get(key)
            for key in ("experts_backend", "compiled", "compile_mode")
        ):
            raise ValueError("full-model profile metadata diverges from its best stock row")
    diagnostic_best = next(
        row for row in best_rows if row["regime_id"] == DIAGNOSTIC_REGIME_ID
    )
    if (
        diagnostic.get("regime_id") != DIAGNOSTIC_REGIME_ID
        or diagnostic.get("layer_index") != DIAGNOSTIC_LAYER_INDEX
        or diagnostic.get("layer_path") != DIAGNOSTIC_LAYER_PATH
        or diagnostic.get("configuration_id") != diagnostic_best["configuration_id"]
        or diagnostic.get("experts_backend") != diagnostic_best["experts_backend"]
        or diagnostic.get("selected_full_model_compiled")
        != bool(diagnostic_best.get("compiled"))
    ):
        raise ValueError("expert diagnostic is not bound to the pinned layer and best stock row")

    profile_complete = (
        all(_successful_profile(row) for row in full_profiles)
        and _successful_diagnostic(diagnostic)
    )
    readiness_blockers = [] if profile_complete else ["profile_measurements_incomplete"]
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "contract": CONTRACT_PATH,
        "mode": mode,
        "generated_at_utc": generated_at,
        "command": list(command or sys.argv),
        "pins": _pins(),
        "baseline_binding": {
            "status": "validated_complete_stock_cohort",
            "path": baseline_artifact_path,
            "sha256": _artifact_sha256(baseline_artifact),
            "cohort_complete": True,
        },
        "profiler_contract": _profiler_contract(profile_warmups),
        "environment": resolved_environment,
        "full_model_profiles": full_profiles,
        "expert_layer_diagnostic": diagnostic,
        "summary": {
            "row_inventory_complete": row_inventory_complete,
            "profile_complete": profile_complete,
            "target_decision_ready": False,
            "candidate_measurements_present": False,
            "performance_claim": "none",
            "readiness_blockers": readiness_blockers,
        },
    }


def validate_baseline_artifact(
    artifact: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    invalid = (
        artifact.get("benchmark") != baseline.BENCHMARK
        or artifact.get("mode") != "real"
        or artifact.get("summary", {}).get("cohort_complete") is not True
        or artifact.get("summary", {}).get("performance_claim") != "none"
    )
    pins = artifact.get("pins", {})
    invalid = invalid or any(
        pins.get(key) != value
        for key, value in {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "dtype": DTYPE,
        }.items()
    )
    expected_regimes = {row["regime_id"] for row in baseline.required_regimes()}
    best_rows = [dict(row) for row in artifact.get("best_stock_by_regime", [])]
    invalid = invalid or len(best_rows) != len(expected_regimes)
    invalid = invalid or {row.get("regime_id") for row in best_rows} != expected_regimes
    configurations = {
        row.get("configuration_id"): row
        for row in artifact.get("configuration_inventory", [])
    }
    results = artifact.get("results", [])
    for best in best_rows:
        binding = (best.get("regime_id"), best.get("configuration_id"))
        matching = [
            row
            for row in results
            if (row.get("regime_id"), row.get("configuration_id")) == binding
        ]
        if best.get("configuration_id") not in configurations:
            invalid = True
        else:
            configuration = configurations[best["configuration_id"]]
            if any(
                best.get(key) != configuration.get(key)
                for key in ("experts_backend", "compiled", "compile_mode")
            ):
                invalid = True
        if len(matching) != 1:
            invalid = True
        elif (
            matching[0].get("status") != "ok"
            or matching[0].get("correctness", {}).get("status") != "passed"
        ):
            invalid = True
    if invalid:
        raise ValueError("real profiling requires a complete real stock baseline artifact")
    return best_rows


def probe_profiler_environment() -> Dict[str, Any]:
    environment = dict(baseline.probe_environment())
    torch = _torch()
    supported = set(torch.profiler.supported_activities())
    environment.update(
        {
            "profiler_cuda_activity_available": (
                torch.profiler.ProfilerActivity.CUDA in supported
            ),
            "kineto_available": bool(torch.profiler.kineto_available()),
            "cupti_validation": "required_device_events_at_runtime",
        }
    )
    return environment


def profile_readiness_blockers(
    environment: Mapping[str, Any],
    baseline_environment: Mapping[str, Any],
) -> List[str]:
    blockers = list(environment.get("readiness_blockers", []))
    if environment.get("preflight_status") != "ready":
        blockers.append("stock_preflight_not_ready")
    if environment.get("cuda_available") is not True:
        blockers.append("cuda_unavailable")
    if environment.get("profiler_cuda_activity_available") is not True:
        blockers.append("cuda_profiler_activity_unavailable")
    if environment.get("kineto_available") is not True:
        blockers.append("kineto_unavailable")
    for key in ENVIRONMENT_BINDING_KEYS:
        expected = baseline_environment.get(key)
        observed = environment.get(key)
        if expected is not None and observed != expected:
            blockers.append(f"environment_mismatch:{key}")
    return _deduplicate(blockers)


def find_expert_layer(model: Any) -> Any:
    if getattr(getattr(model, "config", None), "num_hidden_layers", None) != 16:
        raise RuntimeError("pinned OLMoE diagnostic requires exactly 16 model layers")
    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None or len(layers) <= DIAGNOSTIC_LAYER_INDEX:
        raise RuntimeError(f"selected expert layer is unavailable at {DIAGNOSTIC_LAYER_PATH}")
    layer = getattr(layers[DIAGNOSTIC_LAYER_INDEX], "mlp", None)
    if layer is None or not callable(layer):
        raise RuntimeError(f"selected expert layer is unavailable at {DIAGNOSTIC_LAYER_PATH}")
    if not callable(getattr(layer, "gate", None)):
        raise RuntimeError("selected OLMoE expert layer does not expose its router")
    if not callable(getattr(layer, "experts", None)):
        raise RuntimeError("selected OLMoE expert layer does not expose its experts")
    return layer


def capture_real_activation(layer: Any, forward_call: Callable[[], Any]) -> Dict[str, Any]:
    captured: Dict[str, Any] = {"call_count": 0}

    def pre_hook(_module: Any, args: Sequence[Any]) -> None:
        captured["call_count"] += 1
        if len(args) != 1 or not hasattr(args[0], "detach"):
            raise RuntimeError("selected expert layer must receive one positional tensor input")
        if captured["call_count"] == 1:
            captured["input"] = args[0].detach().clone()

    def post_hook(_module: Any, _args: Sequence[Any], output: Any) -> None:
        if captured["call_count"] == 1:
            if not hasattr(output, "detach"):
                raise RuntimeError("selected expert layer must return one tensor output")
            captured["output"] = output.detach().clone()

    pre_handle = layer.register_forward_pre_hook(pre_hook)
    post_handle = layer.register_forward_hook(post_hook)
    try:
        forward_call()
    finally:
        pre_handle.remove()
        post_handle.remove()
    if captured.get("call_count") != 1 or "input" not in captured or "output" not in captured:
        raise RuntimeError("full-model forward did not call the selected expert layer exactly once")
    return captured


class RealProfileRunner:
    """Run full-model profiles for best rows and one isolated attribution diagnostic."""

    def __init__(self, *, profile_warmups: int) -> None:
        self.profile_warmups = profile_warmups

    def __call__(self, artifact: Mapping[str, Any]) -> Mapping[str, Any]:
        best_rows = [dict(row) for row in artifact["best_stock_by_regime"]]
        configurations = {
            row["configuration_id"]: dict(row)
            for row in artifact["configuration_inventory"]
        }
        regimes = {row["regime_id"]: row for row in baseline.required_regimes()}
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in best_rows:
            grouped.setdefault(str(row["configuration_id"]), []).append(row)

        full_profiles: List[Dict[str, Any]] = []
        for configuration_id, selected_rows in grouped.items():
            configuration = configurations[configuration_id]
            model = None
            try:
                model, resolved_backend = _load_model(configuration, compile_model=True)
                for selected in selected_rows:
                    try:
                        full_profiles.append(
                            _profile_full_model_regime(
                                model,
                                regimes[selected["regime_id"]],
                                selected,
                                resolved_backend=resolved_backend,
                                profile_warmups=self.profile_warmups,
                            )
                        )
                    except Exception as exc:  # pragma: no cover - CUDA integration path.
                        full_profiles.append(_failed_profile_row(selected, exc))
            except Exception as exc:  # pragma: no cover - CUDA integration path.
                full_profiles.extend(_failed_profile_row(row, exc) for row in selected_rows)
            finally:
                if model is not None:
                    del model
                gc.collect()
                baseline._reset_compiler_and_allocator(_torch())

        selected_diagnostic = next(
            row for row in best_rows if row["regime_id"] == DIAGNOSTIC_REGIME_ID
        )
        try:
            diagnostic = _profile_expert_diagnostic(
                configurations[selected_diagnostic["configuration_id"]],
                selected_diagnostic,
                regimes[DIAGNOSTIC_REGIME_ID],
                profile_warmups=self.profile_warmups,
            )
        except Exception as exc:  # pragma: no cover - CUDA integration path.
            diagnostic = _failed_diagnostic(selected_diagnostic, exc)
        return {
            "full_model_profiles": full_profiles,
            "expert_layer_diagnostic": diagnostic,
        }


def _load_model(
    configuration: Mapping[str, Any],
    *,
    compile_model: bool,
) -> tuple[Any, str]:
    torch = _torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - real preflight prevents this.
        raise RuntimeError("the pinned Transformers checkout is required") from exc

    baseline._reset_compiler_and_allocator(torch)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        **baseline.model_load_kwargs(configuration, torch),
    )
    model.eval()
    baseline.validate_loaded_model_revision(model)
    resolved_backend = baseline._resolved_experts_backend(model, configuration)
    if compile_model and configuration.get("compiled"):
        model.forward = torch.compile(
            model.forward,
            **baseline.compile_kwargs(configuration, resolved_backend),
        )
    torch.cuda.synchronize()
    return model, resolved_backend


def _profile_full_model_regime(
    model: Any,
    regime: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    resolved_backend: str,
    profile_warmups: int,
) -> Dict[str, Any]:
    torch = _torch()
    output = None
    with torch.inference_mode():
        for _ in range(profile_warmups):
            call = _prepare_regime_forward(model, regime, resolved_backend, torch)
            warmup_output = call()
            torch.cuda.synchronize()
            del warmup_output
        call = _prepare_regime_forward(model, regime, resolved_backend, torch)
        with _cuda_profiler(torch) as profiler:
            output = call()
            torch.cuda.synchronize()
    require_cupti_trace(profiler)
    attribution = summarize_profiler(profiler, default_scope="full_model")
    require_device_attribution(attribution)
    attribution["cupti_trace_status"] = "cuda_kernel_events_present"
    assert output is not None
    row = {
        **_selected_binding(selected),
        "status": "ok",
        "reason": None,
        "resolved_experts_backend": resolved_backend,
        "profiled_output_shape": list(output.logits.shape),
        "attribution": attribution,
    }
    del output
    return row


def _profile_expert_diagnostic(
    configuration: Mapping[str, Any],
    selected: Mapping[str, Any],
    regime: Mapping[str, Any],
    *,
    profile_warmups: int,
) -> Dict[str, Any]:
    torch = _torch()
    model = None
    try:
        model, resolved_backend = _load_model(configuration, compile_model=False)
        layer = find_expert_layer(model)
        with torch.inference_mode():
            call = _prepare_regime_forward(model, regime, resolved_backend, torch)
            captured = capture_real_activation(layer, call)
            flat_input = captured["input"].reshape(-1, captured["input"].shape[-1])
            route_output, routing = _profile_callable(
                torch,
                lambda: layer.gate(flat_input),
                scope="routing_top_k",
                profile_warmups=profile_warmups,
            )
            _router_logits, top_k_weights, top_k_index = route_output
            expert_output, expert = _profile_callable(
                torch,
                lambda: layer.experts(flat_input, top_k_index, top_k_weights),
                scope="expert_layer",
                profile_warmups=profile_warmups,
            )
            replay_output = expert_output.reshape(captured["output"].shape)
        attribution = merge_attributions([routing, expert])
        require_device_attribution(attribution)
        attribution["cupti_trace_status"] = "cuda_kernel_events_present"
        correctness = _diagnostic_correctness(replay_output, captured["output"], torch)
        return {
            "regime_id": DIAGNOSTIC_REGIME_ID,
            "layer_index": DIAGNOSTIC_LAYER_INDEX,
            "layer_path": DIAGNOSTIC_LAYER_PATH,
            "status": "ok" if correctness["status"] == "passed" else "failed",
            "reason": None if correctness["status"] == "passed" else correctness["reason"],
            "configuration_id": selected["configuration_id"],
            "experts_backend": selected["experts_backend"],
            "resolved_experts_backend": resolved_backend,
            "selected_full_model_compiled": bool(selected.get("compiled")),
            "selected_full_model_compile_mode": selected.get("compile_mode"),
            "replay_compiled": False,
            "compilation_boundary": (
                "layer replay preserves the selected experts backend but does not reproduce "
                "full-model torch.compile; compiled cost is attributed by the bound "
                "full-model profile"
            ),
            "input": _tensor_metadata(captured["input"]),
            "output": _tensor_metadata(captured["output"]),
            "routing": {
                "top_k_index_shape": list(top_k_index.shape),
                "top_k_weights_shape": list(top_k_weights.shape),
            },
            "correctness": correctness,
            "attribution": attribution,
            "evidence_boundary": "diagnostic_only_not_end_to_end_evidence",
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        baseline._reset_compiler_and_allocator(torch)


def _profile_callable(
    torch: Any,
    call: Callable[[], Any],
    *,
    scope: str,
    profile_warmups: int,
) -> tuple[Any, Dict[str, Any]]:
    for _ in range(profile_warmups):
        warmup_output = call()
        torch.cuda.synchronize()
        del warmup_output
    with _cuda_profiler(torch) as profiler:
        output = call()
        torch.cuda.synchronize()
    require_cupti_trace(profiler)
    attribution = summarize_profiler(profiler, default_scope=scope)
    require_device_attribution(attribution)
    attribution["cupti_trace_status"] = "cuda_kernel_events_present"
    return output, attribution


def _cuda_profiler(torch: Any):
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    )


def _prepare_regime_forward(
    model: Any,
    regime: Mapping[str, Any],
    resolved_backend: str,
    torch: Any,
) -> Callable[[], Any]:
    input_ids = baseline._deterministic_tokens(
        torch,
        batch_size=int(regime["batch_size"]),
        sequence_length=int(regime["sequence_length"]),
        vocab_size=int(model.config.vocab_size),
        salt=0,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda:0")
    if regime["phase"] == "prefill":
        return lambda: model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )

    stage_backends = baseline.stage_experts_backends("decode", resolved_backend)
    if stage_backends["setup"] != stage_backends["timed"]:
        baseline._set_model_experts_backend(model, stage_backends["setup"])
    prompt_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    past_key_values = prompt_output.past_key_values
    del prompt_output
    if stage_backends["setup"] != stage_backends["timed"]:
        baseline._set_model_experts_backend(model, stage_backends["timed"])
    decode_token = baseline._deterministic_tokens(
        torch,
        batch_size=int(regime["batch_size"]),
        sequence_length=1,
        vocab_size=int(model.config.vocab_size),
        salt=1,
    )
    decode_attention_mask = torch.ones(
        (int(regime["batch_size"]), int(regime["sequence_length"]) + 1),
        dtype=torch.long,
        device="cuda:0",
    )
    return lambda: model(
        input_ids=decode_token,
        attention_mask=decode_attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )


def _diagnostic_correctness(candidate: Any, reference: Any, torch: Any) -> Dict[str, Any]:
    candidate_float = candidate.detach().to(dtype=torch.float32, device="cpu")
    reference_float = reference.detach().to(dtype=torch.float32, device="cpu")
    if tuple(candidate_float.shape) != tuple(reference_float.shape):
        return {
            "status": "failed",
            "reason": "output_shape_mismatch",
            "max_abs_error": None,
            "relative_l2_error": None,
        }
    difference = candidate_float - reference_float
    max_abs_error = float(torch.max(torch.abs(difference)).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    reference_norm = float(torch.linalg.vector_norm(reference_float).item())
    relative_l2_error = difference_norm / max(reference_norm, sys.float_info.epsilon)
    passed = (
        math.isfinite(max_abs_error)
        and math.isfinite(relative_l2_error)
        and max_abs_error <= baseline.CORRECTNESS_MAX_ABS_TOLERANCE
        and relative_l2_error <= baseline.CORRECTNESS_RELATIVE_L2_TOLERANCE
    )
    return {
        "status": "passed" if passed else "failed",
        "reason": None if passed else "correctness_tolerance_exceeded",
        "reference": "captured_full_model_expert_layer_output",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "max_abs_tolerance": baseline.CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": baseline.CORRECTNESS_RELATIVE_L2_TOLERANCE,
    }


def _successful_profile(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    if row.get("attribution", {}).get("cupti_trace_status") != "cuda_kernel_events_present":
        return False
    try:
        require_device_attribution(row.get("attribution", {}))
    except RuntimeError:
        return False
    return True


def _successful_diagnostic(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "ok" or row.get("correctness", {}).get("status") != "passed":
        return False
    if row.get("attribution", {}).get("cupti_trace_status") != "cuda_kernel_events_present":
        return False
    try:
        require_device_attribution(row.get("attribution", {}))
    except RuntimeError:
        return False
    return True


def _empty_profile_row(regime: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "regime_id": regime["regime_id"],
        "phase": regime["phase"],
        "batch_size": regime["batch_size"],
        "sequence_length": regime["sequence_length"],
        "configuration_id": None,
        "experts_backend": None,
        "compiled": None,
        "compile_mode": None,
        "status": "not_measured",
        "reason": "contract_smoke_not_performance_evidence",
        "attribution": _empty_attribution(),
    }


def _empty_diagnostic() -> Dict[str, Any]:
    return {
        "regime_id": DIAGNOSTIC_REGIME_ID,
        "layer_index": DIAGNOSTIC_LAYER_INDEX,
        "layer_path": DIAGNOSTIC_LAYER_PATH,
        "status": "not_measured",
        "reason": "contract_smoke_not_performance_evidence",
        "configuration_id": None,
        "experts_backend": None,
        "selected_full_model_compiled": None,
        "replay_compiled": False,
        "input": {"shape": None, "dtype": None, "device": None},
        "output": {"shape": None, "dtype": None, "device": None},
        "correctness": {"status": "not_measured"},
        "attribution": _empty_attribution(),
        "evidence_boundary": "diagnostic_only_not_end_to_end_evidence",
    }


def _empty_attribution() -> Dict[str, Any]:
    return {
        "timing_status": "not_measured",
        "cupti_trace_status": "not_measured",
        "event_group_count": 0,
        "events": [],
        "categories": [
            {
                "category": category,
                "event_group_count": 0,
                "call_count": 0,
                "self_cpu_time_us": None,
                "self_device_time_us": None,
                "cpu_time_proportion": None,
                "device_time_proportion": None,
            }
            for category in ATTRIBUTION_CATEGORIES
        ],
        "totals": {"self_cpu_time_us": None, "self_device_time_us": None},
        "unclassified_event_names": [],
    }


def _failed_profile_row(selected: Mapping[str, Any], exc: Exception) -> Dict[str, Any]:
    return {
        **_selected_binding(selected),
        "status": "failed",
        "reason": f"{type(exc).__name__}:{exc}",
        "attribution": _empty_attribution(),
    }


def _failed_diagnostic(selected: Mapping[str, Any], exc: Exception) -> Dict[str, Any]:
    row = _empty_diagnostic()
    row.update(
        {
            "status": "failed",
            "reason": f"{type(exc).__name__}:{exc}",
            "configuration_id": selected["configuration_id"],
            "experts_backend": selected["experts_backend"],
            "selected_full_model_compiled": bool(selected.get("compiled")),
        }
    )
    return row


def _selected_binding(selected: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "regime_id": selected["regime_id"],
        "configuration_id": selected["configuration_id"],
        "experts_backend": selected["experts_backend"],
        "compiled": selected.get("compiled"),
        "compile_mode": selected.get("compile_mode"),
    }


def _pins() -> Dict[str, Any]:
    return {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "dtype": DTYPE,
        "diagnostic_regime_id": DIAGNOSTIC_REGIME_ID,
        "diagnostic_layer_index": DIAGNOSTIC_LAYER_INDEX,
        "diagnostic_layer_path": DIAGNOSTIC_LAYER_PATH,
    }


def _profiler_contract(profile_warmups: int) -> Dict[str, Any]:
    return {
        "profile_warmups": profile_warmups,
        "activities": ["CPU", "CUDA"],
        "time_basis": "key_averages_self_time",
        "device_time_required": True,
        "record_shapes": True,
        "categories": list(ATTRIBUTION_CATEGORIES),
        "unclassified_events_preserved": True,
        "full_model_profiles_bind_to": "best_stock_by_regime",
        "expert_replay_boundary": "diagnostic_only_not_end_to_end_evidence",
    }


def _smoke_environment() -> Dict[str, Any]:
    return {
        "preflight_status": "not_run",
        "cuda_available": None,
        "profiler_cuda_activity_available": None,
        "kineto_available": None,
        "cupti_validation": "not_run",
    }


def _tensor_metadata(tensor: Any) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _artifact_sha256(artifact: Mapping[str, Any]) -> str:
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_any(value: str, tokens: Sequence[str]) -> bool:
    return any(token in value for token in tokens)


def _event_value(event: Any, key: str, default: Any) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _device_type_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].upper()


def _nonnegative_float(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(resolved) or resolved < 0.0:
        return 0.0
    return resolved


def _deduplicate(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def _torch():
    return baseline._torch()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json_artifact(
    path: str | os.PathLike[str],
    **collect_kwargs: Any,
) -> Dict[str, Any]:
    artifact = collect_results(**collect_kwargs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="write a schema-only artifact")
    mode.add_argument("--real", action="store_true", help="profile a complete stock cohort")
    parser.add_argument(
        "--baseline-artifact",
        default="docs/results/olmoe_stock_baseline.json",
    )
    parser.add_argument("--profile-warmups", type=int, default=DEFAULT_PROFILE_WARMUPS)
    parser.add_argument("--json-output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "contract-smoke" if args.smoke else "real"
    baseline_artifact = None
    baseline_path = None
    if mode == "real":
        baseline_path = args.baseline_artifact
        baseline_artifact = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    artifact = write_json_artifact(
        args.json_output,
        mode=mode,
        baseline_artifact=baseline_artifact,
        baseline_artifact_path=baseline_path,
        profile_warmups=args.profile_warmups,
        command=sys.argv if argv is None else [sys.argv[0], *argv],
    )
    print(
        f"row_inventory_complete={str(artifact['summary']['row_inventory_complete']).lower()} "
        f"profile_complete={str(artifact['summary']['profile_complete']).lower()} "
        f"blockers={','.join(artifact['summary']['readiness_blockers']) or 'none'}"
    )
    return 0 if mode == "contract-smoke" or artifact["summary"]["profile_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
