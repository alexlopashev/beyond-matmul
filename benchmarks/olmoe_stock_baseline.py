#!/usr/bin/env python3
"""Pinned OLMoE stock-backend target-validation harness."""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


BENCHMARK = "olmoe_stock_baseline"
CONTRACT_PATH = "docs/olmoe_tensor_contraction_capstone.md"
MODEL = "allenai/OLMoE-1B-7B-0924"
MODEL_REVISION = "bd1c52f59153f724c1ad11ca1791edc77bab3806"
TRANSFORMERS_REVISION = "a6895655b289cc3fdd29afec36904e0b8545ef92"
DTYPE = "bfloat16"
INPUT_SEED = 20260714
CORRECTNESS_MAX_ABS_TOLERANCE = 0.125
CORRECTNESS_RELATIVE_L2_TOLERANCE = 0.01
DEFAULT_WARMUP_REPETITIONS = 3
DEFAULT_MEASURED_REPETITIONS = 10
STOCK_BACKENDS = (
    "default",
    "eager",
    "batched_mm",
    "grouped_mm",
    "deepgemm",
    "sonicmoe",
)
DEFAULT_COMPILE_MODES = (
    "default",
    "lite",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
GROUPED_MM_COMPILE_MODES = {"default", "max-autotune-no-cudagraphs"}
EXTERNAL_KERNEL_BACKENDS = {"deepgemm", "sonicmoe"}
PINNED_DEPENDENCY_VERSIONS = {
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "huggingface-hub": "1.23.0",
    "kernels": "0.15.2",
    "nvidia-cutlass-dsl": "4.6.0",
}
CORE_DEPENDENCIES = {"accelerate", "safetensors", "huggingface-hub"}
KERNELS_MIN_VERSION = "0.15.2"
KERNELS_MAX_VERSION = "0.16.0"

RunConfiguration = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]


def required_regimes() -> List[Dict[str, Any]]:
    regimes: List[Dict[str, Any]] = []
    for batch_size in (1, 4):
        for sequence_length in (128, 512):
            regimes.append(
                {
                    "regime_id": f"prefill_b{batch_size}_s{sequence_length}",
                    "phase": "prefill",
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "tokens_per_timed_forward": batch_size * sequence_length,
                }
            )
    for batch_size in (1, 8):
        for prompt_length in (128, 512):
            regimes.append(
                {
                    "regime_id": f"decode_b{batch_size}_p{prompt_length}",
                    "phase": "decode",
                    "batch_size": batch_size,
                    "sequence_length": prompt_length,
                    "prompt_length": prompt_length,
                    "decode_tokens_per_timed_forward": batch_size,
                    "tokens_per_timed_forward": batch_size,
                }
            )
    return regimes


def configuration_inventory(compile_modes: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    modes = _validate_compile_modes(compile_modes or DEFAULT_COMPILE_MODES)
    configurations: List[Dict[str, Any]] = []
    for backend in STOCK_BACKENDS:
        configurations.append(
            {
                "configuration_id": f"{backend}__uncompiled",
                "experts_backend": backend,
                "compiled": False,
                "compile_mode": None,
                "fullgraph": None,
                "eligibility": "required",
                "exclusion_reason": None,
            }
        )
        for mode in modes:
            eligibility = "required"
            exclusion_reason = None
            if backend in EXTERNAL_KERNEL_BACKENDS:
                eligibility = "excluded"
                exclusion_reason = (
                    "audited Transformers contract routes this backend through an external CUDA kernel "
                    "and does not define a torch.compile comparison"
                )
            elif backend == "grouped_mm" and mode not in GROUPED_MM_COMPILE_MODES:
                eligibility = "excluded"
                exclusion_reason = (
                    "mode is excluded by the audited compile contract for grouped_mm"
                )
            elif backend == "default" and mode not in GROUPED_MM_COMPILE_MODES:
                eligibility = "excluded"
                exclusion_reason = (
                    "audited CUDA default resolves to grouped_mm, whose compile contract excludes this mode"
                )

            if backend == "eager":
                fullgraph: bool | str = False
            elif backend in {"batched_mm", "grouped_mm"}:
                fullgraph = True
            elif backend == "default":
                fullgraph = "resolved_backend"
            else:
                fullgraph = False

            configurations.append(
                {
                    "configuration_id": f"{backend}__compiled__{mode}",
                    "experts_backend": backend,
                    "compiled": True,
                    "compile_mode": mode,
                    "fullgraph": fullgraph,
                    "eligibility": eligibility,
                    "exclusion_reason": exclusion_reason,
                }
            )
    return configurations


def collect_results(
    *,
    mode: str,
    compile_modes: Sequence[str] | None = None,
    environment: Mapping[str, Any] | None = None,
    run_configuration: RunConfiguration | None = None,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    if mode not in {"contract-smoke", "real"}:
        raise ValueError(f"unsupported mode: {mode}")

    resolved_compile_modes = _validate_compile_modes(
        compile_modes or DEFAULT_COMPILE_MODES
    )
    if mode == "real" and set(resolved_compile_modes) != set(DEFAULT_COMPILE_MODES):
        raise ValueError(
            "real mode requires the full pinned compile mode inventory: "
            + ", ".join(DEFAULT_COMPILE_MODES)
        )

    regimes = required_regimes()
    configurations = configuration_inventory(resolved_compile_modes)
    resolved_environment = dict(
        environment
        if environment is not None
        else (_smoke_environment() if mode == "contract-smoke" else probe_environment())
    )
    if warmup_repetitions < 0:
        raise ValueError("warmup repetitions must be non-negative")
    if measured_repetitions <= 0:
        raise ValueError("measured repetitions must be positive")
    if run_configuration is not None:
        executor = run_configuration
    elif mode == "real" and resolved_environment.get("preflight_status") == "ready":
        executor = RealConfigurationRunner(
            warmup_repetitions=warmup_repetitions,
            measured_repetitions=measured_repetitions,
        )
    else:
        executor = _unconfigured_real_executor
    results: List[Dict[str, Any]] = []

    for configuration in configurations:
        if configuration["eligibility"] == "excluded":
            results.extend(
                _empty_result_row(
                    regime,
                    configuration,
                    status="not_applicable",
                    reason=str(configuration["exclusion_reason"]),
                )
                for regime in regimes
            )
            continue

        runtime_status, runtime_reason = _runtime_configuration_status(
            configuration,
            resolved_environment,
        )
        if runtime_status in {"not_applicable", "blocked"}:
            results.extend(
                _empty_result_row(
                    regime,
                    configuration,
                    status=runtime_status,
                    reason=runtime_reason,
                    runtime_applicability=runtime_status,
                )
                for regime in regimes
            )
            continue

        if resolved_environment.get("preflight_status") != "ready":
            reason = _preflight_row_reason(mode, resolved_environment)
            results.extend(
                _empty_result_row(regime, configuration, status="blocked", reason=reason)
                for regime in regimes
            )
            continue

        measured = _execute_configuration(executor, configuration, regimes)
        by_regime = {
            str(row.get("regime_id")): dict(row)
            for row in measured
            if row.get("regime_id") is not None
        }
        for regime in regimes:
            measurement = by_regime.get(regime["regime_id"])
            if measurement is None:
                results.append(
                    _empty_result_row(
                        regime,
                        configuration,
                        status="failed",
                        reason="executor_missing_required_regime",
                    )
                )
            else:
                results.append(_normalize_measured_row(regime, configuration, measurement))

    row_inventory_complete = _row_inventory_complete(results, regimes, configurations)
    required_results = [
        row
        for row in results
        if row["configuration_eligibility"] == "required"
        and row["runtime_applicability"] != "not_applicable"
    ]
    incomplete_failures = [
        row
        for row in required_results
        if row["status"] == "failed" and _is_incomplete_failure_reason(row["reason"])
    ]
    blocked_results = [row for row in required_results if row["status"] == "blocked"]
    nonterminal_results = [
        row
        for row in required_results
        if row["status"] not in {"ok", "failed", "blocked"}
    ]
    regimes_with_correct_stock = {
        row["regime_id"]
        for row in required_results
        if row["status"] == "ok" and row["correctness"]["status"] == "passed"
    }
    all_regimes_have_correct_stock = regimes_with_correct_stock == {
        regime["regime_id"] for regime in regimes
    }
    cohort_complete = (
        mode == "real"
        and bool(required_results)
        and row_inventory_complete
        and not blocked_results
        and not nonterminal_results
        and not incomplete_failures
        and all_regimes_have_correct_stock
    )
    readiness_blockers = list(resolved_environment.get("readiness_blockers", []))
    warnings: List[str] = []
    if mode == "contract-smoke":
        readiness_blockers.append("contract_smoke_not_performance_evidence")
    if not row_inventory_complete:
        readiness_blockers.append("required_row_inventory_incomplete")
    if blocked_results and mode == "real":
        readiness_blockers.append("required_measurements_blocked")
    if (incomplete_failures or nonterminal_results) and mode == "real":
        readiness_blockers.append("required_measurements_incomplete")
    if not all_regimes_have_correct_stock and mode == "real":
        readiness_blockers.append("no_correct_stock_configuration_for_every_regime")
    if any(
        row["status"] == "failed" and row not in incomplete_failures
        for row in required_results
    ):
        warnings.append("stock_configuration_failures")
    if any(
        row["status"] == "ok" and row["correctness"]["status"] != "passed"
        for row in required_results
    ):
        warnings.append("stock_configuration_correctness_failures")
    if not cohort_complete and mode == "real" and not readiness_blockers:
        readiness_blockers.append("required_measurements_incomplete")
    readiness_blockers = _deduplicate(readiness_blockers)
    warnings = _deduplicate(warnings)

    best_stock = select_best_stock_rows(results) if mode == "real" and cohort_complete else []
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "contract": CONTRACT_PATH,
        "mode": mode,
        "generated_at_utc": generated_at_utc or _utc_now(),
        "command": list(command or sys.argv),
        "pins": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "transformers_revision": TRANSFORMERS_REVISION,
            "dtype": DTYPE,
            "input_seed": INPUT_SEED,
            "compile_modes": list(DEFAULT_COMPILE_MODES),
            "dependency_versions": dict(PINNED_DEPENDENCY_VERSIONS),
        },
        "correctness_contract": {
            "reference_configuration": "eager__uncompiled",
            "observed_output": "last_token_logits",
            "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
            "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        },
        "measurement_contract": {
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "input_preparation_in_timed_region": False,
            "prefill_builds_kv_cache": True,
            "prefill_cache_construction_in_timed_region": True,
            "decode_prompt_prefill_in_timed_region": False,
            "decode_allocator_peak_excludes_prompt_prefill": True,
            "primary_timing": "cuda_event_median_seconds",
        },
        "environment": resolved_environment,
        "regimes": regimes,
        "configuration_inventory": configurations,
        "results": results,
        "best_stock_by_regime": best_stock,
        "summary": {
            "row_inventory_complete": row_inventory_complete,
            "cohort_complete": cohort_complete,
            "target_decision_ready": False,
            "candidate_measurements_present": False,
            "performance_claim": "none",
            "readiness_blockers": readiness_blockers,
            "warnings": warnings,
        },
    }


def select_best_stock_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    regime_ids = [regime["regime_id"] for regime in required_regimes()]
    for regime_id in regime_ids:
        candidates = []
        for row in results:
            if row.get("regime_id") != regime_id or row.get("status") != "ok":
                continue
            if row.get("correctness", {}).get("status") != "passed":
                continue
            median = row.get("timing", {}).get("cuda_event_median_seconds")
            if not isinstance(median, (int, float)) or not math.isfinite(float(median)):
                continue
            candidates.append(row)
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                float(row["timing"]["cuda_event_median_seconds"]),
                str(row["configuration_id"]),
            ),
        )
        selected.append(
            {
                "regime_id": regime_id,
                "configuration_id": best["configuration_id"],
                "experts_backend": best["experts_backend"],
                "compiled": best.get("compiled"),
                "compile_mode": best.get("compile_mode"),
                "stage_experts_backends": best.get("stage_experts_backends"),
                "cuda_event_median_seconds": best["timing"]["cuda_event_median_seconds"],
                "throughput_tokens_per_second": best.get("throughput_tokens_per_second"),
            }
        )
    return selected


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the project pins torch.
        raise RuntimeError("PyTorch is required for the OLMoE stock benchmark") from exc
    return torch


def model_load_kwargs(
    configuration: Mapping[str, Any],
    torch_module: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "revision": MODEL_REVISION,
        "dtype": torch_module.bfloat16,
        "device_map": {"": "cuda:0"},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
    }
    backend = str(configuration["experts_backend"])
    if backend != "default":
        kwargs["experts_implementation"] = backend
    return kwargs


def compile_kwargs(
    configuration: Mapping[str, Any],
    resolved_backend: str,
) -> Dict[str, Any]:
    if not configuration["compiled"]:
        raise ValueError("compile kwargs requested for an uncompiled configuration")
    mode = configuration["compile_mode"]
    fullgraph = configuration["fullgraph"]
    if fullgraph == "resolved_backend":
        fullgraph = resolved_backend in {"grouped_mm", "batched_mm"}
    return {
        "mode": None if mode == "default" else mode,
        "fullgraph": bool(fullgraph),
    }


def backend_availability(
    *,
    cuda_available: bool,
    compute_capability: Sequence[int] | None,
    cuda_runtime: str | None,
    available_modules: set[str],
    cuda_toolkit_version: str | None,
    nvcc_available: bool,
    dependency_versions: Mapping[str, str | None],
) -> Dict[str, Dict[str, str | None]]:
    if not cuda_available:
        return {
            backend: {"status": "blocked", "reason": "cuda_unavailable"}
            for backend in STOCK_BACKENDS
        }

    availability = {
        backend: {"status": "available", "reason": None}
        for backend in STOCK_BACKENDS
    }
    capability = tuple(compute_capability or ())
    capability_major = capability[0] if capability else 0
    if capability < (9, 0):
        for backend in EXTERNAL_KERNEL_BACKENDS:
            availability[backend] = {
                "status": "not_applicable",
                "reason": "requires_compute_capability_9_0",
            }
        return availability

    if capability_major not in {9, 10}:
        availability["deepgemm"] = {
            "status": "not_applicable",
            "reason": "deepgemm_supports_only_compute_capability_9_or_10",
        }
    else:
        minimum_cuda = (12, 9) if capability_major == 10 else (12, 3)
        minimum_label = "12_9" if capability_major == 10 else "12_3"
        if _version_tuple(cuda_runtime) < minimum_cuda:
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": f"deepgemm_requires_cuda_runtime_{minimum_label}_or_newer",
            }
        elif not nvcc_available or cuda_toolkit_version is None:
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": "deepgemm_requires_nvcc_cuda_toolkit",
            }
        elif _version_tuple(cuda_toolkit_version) < minimum_cuda:
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": f"deepgemm_requires_nvcc_toolkit_{minimum_label}_or_newer",
            }
        elif "kernels" not in available_modules:
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": "deepgemm_requires_kernels_package",
            }
        elif not _version_in_half_open_range(
            dependency_versions.get("kernels"),
            KERNELS_MIN_VERSION,
            KERNELS_MAX_VERSION,
        ):
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": "deepgemm_incompatible_kernels_version",
            }
        elif dependency_versions.get("kernels") != PINNED_DEPENDENCY_VERSIONS[
            "kernels"
        ]:
            availability["deepgemm"] = {
                "status": "blocked",
                "reason": "deepgemm_kernels_version_pin_mismatch",
            }

    missing_sonic_modules = sorted({"kernels", "cutlass"} - available_modules)
    if missing_sonic_modules:
        availability["sonicmoe"] = {
            "status": "blocked",
            "reason": "sonicmoe_missing_modules:" + ",".join(missing_sonic_modules),
        }
    elif not _version_in_half_open_range(
        dependency_versions.get("kernels"),
        KERNELS_MIN_VERSION,
        KERNELS_MAX_VERSION,
    ):
        availability["sonicmoe"] = {
            "status": "blocked",
            "reason": "sonicmoe_incompatible_kernels_version",
        }
    elif dependency_versions.get("kernels") != PINNED_DEPENDENCY_VERSIONS[
        "kernels"
    ]:
        availability["sonicmoe"] = {
            "status": "blocked",
            "reason": "sonicmoe_kernels_version_pin_mismatch",
        }
    elif dependency_versions.get("nvidia-cutlass-dsl") != PINNED_DEPENDENCY_VERSIONS[
        "nvidia-cutlass-dsl"
    ]:
        availability["sonicmoe"] = {
            "status": "blocked",
            "reason": "sonicmoe_nvidia_cutlass_dsl_version_mismatch",
        }
    return availability


def validate_loaded_model_revision(model: Any) -> str:
    loaded_revision = getattr(model.config, "_commit_hash", None)
    if loaded_revision != MODEL_REVISION:
        raise RuntimeError(
            f"model revision mismatch: expected {MODEL_REVISION}, observed {loaded_revision}"
        )
    return str(loaded_revision)


def stage_experts_backends(phase: str, resolved_backend: str) -> Dict[str, str]:
    timed_backend = (
        "batched_mm"
        if phase == "decode" and resolved_backend == "grouped_mm"
        else resolved_backend
    )
    return {"setup": resolved_backend, "timed": timed_backend}


def correctness_metrics(candidate: Any, reference: Any) -> Dict[str, Any]:
    torch = _torch()
    candidate_float = candidate.detach().to(dtype=torch.float32, device="cpu")
    reference_float = reference.detach().to(dtype=torch.float32, device="cpu")
    if tuple(candidate_float.shape) != tuple(reference_float.shape):
        return {
            "status": "failed",
            "reference": "eager__uncompiled",
            "max_abs_error": None,
            "relative_l2_error": None,
            "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
            "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
            "reason": "output_shape_mismatch",
        }
    difference = candidate_float - reference_float
    max_abs_error = float(torch.max(torch.abs(difference)).item())
    reference_norm = float(torch.linalg.vector_norm(reference_float).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    relative_l2_error = difference_norm / max(reference_norm, sys.float_info.epsilon)
    passed = (
        math.isfinite(max_abs_error)
        and math.isfinite(relative_l2_error)
        and max_abs_error <= CORRECTNESS_MAX_ABS_TOLERANCE
        and relative_l2_error <= CORRECTNESS_RELATIVE_L2_TOLERANCE
    )
    return {
        "status": "passed" if passed else "failed",
        "reference": "eager__uncompiled",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "reason": None if passed else "correctness_tolerance_exceeded",
    }


class RealConfigurationRunner:
    """Load one stock configuration at a time and reuse an eager reference."""

    def __init__(
        self,
        *,
        warmup_repetitions: int,
        measured_repetitions: int,
    ) -> None:
        self.warmup_repetitions = warmup_repetitions
        self.measured_repetitions = measured_repetitions
        self._reference_rows: Sequence[Mapping[str, Any]] | None = None
        self._reference_outputs: Dict[str, Any] | None = None

    def __call__(
        self,
        configuration: Mapping[str, Any],
        regimes: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        self._ensure_reference(regimes)
        if configuration["configuration_id"] == "eager__uncompiled":
            assert self._reference_rows is not None
            return self._reference_rows
        assert self._reference_outputs is not None
        rows, _ = self._run_loaded_configuration(
            configuration,
            regimes,
            reference_outputs=self._reference_outputs,
        )
        return rows

    def _ensure_reference(self, regimes: Sequence[Mapping[str, Any]]) -> None:
        if self._reference_rows is not None and self._reference_outputs is not None:
            return
        reference_configuration = next(
            configuration
            for configuration in configuration_inventory(["default"])
            if configuration["configuration_id"] == "eager__uncompiled"
        )
        rows, outputs = self._run_loaded_configuration(
            reference_configuration,
            regimes,
            reference_outputs=None,
        )
        self._reference_rows = rows
        self._reference_outputs = outputs

    def _run_loaded_configuration(
        self,
        configuration: Mapping[str, Any],
        regimes: Sequence[Mapping[str, Any]],
        *,
        reference_outputs: Mapping[str, Any] | None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        torch = _torch()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - real preflight prevents this.
            raise RuntimeError("the pinned Transformers checkout is required") from exc

        _reset_compiler_and_allocator(torch)
        model = None
        try:
            load_start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                **model_load_kwargs(configuration, torch),
            )
            model.eval()
            loaded_model_revision = validate_loaded_model_revision(model)
            torch.cuda.synchronize()
            model_load_seconds = time.perf_counter() - load_start
            resolved_backend = _resolved_experts_backend(model, configuration)

            compile_wrapper_seconds = 0.0
            if configuration["compiled"]:
                compile_start = time.perf_counter()
                model.forward = torch.compile(
                    model.forward,
                    **compile_kwargs(configuration, resolved_backend),
                )
                compile_wrapper_seconds = time.perf_counter() - compile_start

            rows: List[Dict[str, Any]] = []
            outputs: Dict[str, Any] = {}
            for regime in regimes:
                measurement, output = _measure_regime(
                    model,
                    regime,
                    torch,
                    resolved_backend=resolved_backend,
                    warmup_repetitions=self.warmup_repetitions,
                    measured_repetitions=self.measured_repetitions,
                )
                measurement["resolved_experts_backend"] = resolved_backend
                measurement["configuration_setup"] = {
                    "model_load_seconds": model_load_seconds,
                    "compile_wrapper_seconds": compile_wrapper_seconds,
                    "loaded_model_revision": loaded_model_revision,
                }
                if reference_outputs is None:
                    measurement["correctness"] = {
                        "status": "passed",
                        "reference": "eager__uncompiled",
                        "max_abs_error": 0.0,
                        "relative_l2_error": 0.0,
                        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
                        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
                        "reason": None,
                    }
                else:
                    measurement["correctness"] = correctness_metrics(
                        output,
                        reference_outputs[regime["regime_id"]],
                    )
                rows.append(measurement)
                outputs[regime["regime_id"]] = output
            return rows, outputs
        finally:
            if model is not None:
                del model
            gc.collect()
            _reset_compiler_and_allocator(torch)


def _resolved_experts_backend(model: Any, configuration: Mapping[str, Any]) -> str:
    getter = getattr(model, "get_experts_implementation", None)
    if getter is None:
        return str(configuration["experts_backend"])
    resolved = getter()
    if isinstance(resolved, Mapping):
        return str(resolved.get("", configuration["experts_backend"]))
    return str(resolved)


def _measure_regime(
    model: Any,
    regime: Mapping[str, Any],
    torch: Any,
    *,
    resolved_backend: str,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> tuple[Dict[str, Any], Any]:
    stage_backends = stage_experts_backends(str(regime["phase"]), resolved_backend)
    phase = str(regime["phase"])
    input_start = time.perf_counter()
    input_ids = _deterministic_tokens(
        torch,
        batch_size=int(regime["batch_size"]),
        sequence_length=int(regime["sequence_length"]),
        vocab_size=int(model.config.vocab_size),
        salt=0,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda:0")
    decode_token = None
    decode_attention_mask = None
    if phase == "decode":
        decode_token = _deterministic_tokens(
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
    input_preparation_seconds = time.perf_counter() - input_start

    warmup_start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warmup_repetitions):
            if phase == "prefill":
                warmup_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
                torch.cuda.synchronize()
                del warmup_output
            else:
                if stage_backends["setup"] != stage_backends["timed"]:
                    _set_model_experts_backend(model, stage_backends["setup"])
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
                    _set_model_experts_backend(model, stage_backends["timed"])
                warmup_output = model(
                    input_ids=decode_token,
                    attention_mask=decode_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                torch.cuda.synchronize()
                del warmup_output
                del past_key_values
            gc.collect()
    warmup_seconds = time.perf_counter() - warmup_start

    cuda_event_seconds: List[float] = []
    wall_seconds: List[float] = []
    decode_prefill_setup_seconds: List[float] = []
    cleanup_seconds: List[float] = []
    allocated_before_bytes: List[int] = []
    peak_allocated_bytes: List[int] = []
    cache_resident_allocated_bytes: List[int] = []
    last_token_logits = None
    with torch.inference_mode():
        for _ in range(measured_repetitions):
            if phase == "decode":
                if stage_backends["setup"] != stage_backends["timed"]:
                    _set_model_experts_backend(model, stage_backends["setup"])
                torch.cuda.synchronize()
                setup_start = time.perf_counter()
                prompt_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    logits_to_keep=1,
                )
                torch.cuda.synchronize()
                decode_prefill_setup_seconds.append(time.perf_counter() - setup_start)
                past_key_values = prompt_output.past_key_values
                del prompt_output
                if stage_backends["setup"] != stage_backends["timed"]:
                    _set_model_experts_backend(model, stage_backends["timed"])
                torch.cuda.synchronize()
                cache_resident = int(torch.cuda.memory_allocated(0))
                cache_resident_allocated_bytes.append(cache_resident)
                torch.cuda.reset_peak_memory_stats(0)
                allocated_before_bytes.append(cache_resident)
                output, cuda_seconds, elapsed_wall_seconds = _timed_cuda_forward(
                    torch,
                    lambda: model(
                        input_ids=decode_token,
                        attention_mask=decode_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    ),
                )
            else:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                allocated_before_bytes.append(int(torch.cuda.memory_allocated(0)))
                output, cuda_seconds, elapsed_wall_seconds = _timed_cuda_forward(
                    torch,
                    lambda: model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        logits_to_keep=1,
                    ),
                )

            cuda_event_seconds.append(cuda_seconds)
            wall_seconds.append(elapsed_wall_seconds)
            peak_allocated_bytes.append(int(torch.cuda.max_memory_allocated(0)))
            last_token_logits = output.logits.detach().to(dtype=torch.float32, device="cpu")
            cleanup_start = time.perf_counter()
            del output
            if phase == "decode":
                del past_key_values
            gc.collect()
            cleanup_seconds.append(time.perf_counter() - cleanup_start)

    assert last_token_logits is not None
    cuda_median = float(statistics.median(cuda_event_seconds))
    wall_median = float(statistics.median(wall_seconds))
    tokens_per_forward = int(regime["tokens_per_timed_forward"])
    throughput = tokens_per_forward / cuda_median if cuda_median > 0.0 else None
    peak_increments = [
        max(0, peak - allocated)
        for peak, allocated in zip(peak_allocated_bytes, allocated_before_bytes)
    ]
    return (
        {
            "regime_id": regime["regime_id"],
            "status": "ok",
            "reason": None,
            "stage_experts_backends": stage_backends,
            "timing": {
                "cuda_event_median_seconds": cuda_median,
                "wall_median_seconds": wall_median,
                "cuda_event_seconds": cuda_event_seconds,
                "wall_seconds": wall_seconds,
                "warmup_repetitions": warmup_repetitions,
                "measured_repetitions": measured_repetitions,
            },
            "throughput_tokens_per_second": throughput,
            "preprocessing": {
                "status": "measured",
                "input_preparation_seconds": input_preparation_seconds,
                "warmup_seconds": warmup_seconds,
                "decode_prompt_prefill_median_seconds": (
                    float(statistics.median(decode_prefill_setup_seconds))
                    if phase == "decode"
                    else None
                ),
                "inter_iteration_cleanup_median_seconds": float(
                    statistics.median(cleanup_seconds)
                ),
                "included_in_timed_region": False,
            },
            "routing_overhead": {
                "status": "requires_profiled_target_validation",
                "median_seconds": None,
            },
            "allocator": {
                "status": "measured_cuda_allocator",
                "scope": "timed_phase_forward_only",
                "allocated_before_bytes": int(statistics.median(allocated_before_bytes)),
                "allocated_before_bytes_samples": allocated_before_bytes,
                "peak_allocated_bytes": max(peak_allocated_bytes),
                "peak_allocated_bytes_samples": peak_allocated_bytes,
                "peak_increment_bytes": max(peak_increments),
                "peak_increment_bytes_samples": peak_increments,
                "cache_construction_in_timed_region": phase == "prefill",
                "decode_prompt_setup_excluded": phase == "decode",
                "cache_resident_allocated_bytes": (
                    cache_resident_allocated_bytes if phase == "decode" else None
                ),
            },
        },
        last_token_logits,
    )


def _timed_cuda_forward(
    torch: Any,
    call: Callable[[], Any],
) -> tuple[Any, float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    output = call()
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    cuda_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    return output, cuda_seconds, wall_seconds


def _set_model_experts_backend(model: Any, backend: str) -> None:
    setter = getattr(model, "set_experts_implementation", None)
    if setter is None:
        raise RuntimeError("model does not expose stock experts backend switching")
    setter(backend)


def _deterministic_tokens(
    torch: Any,
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    salt: int,
) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED + batch_size * 10_000 + sequence_length * 10 + salt)
    return torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
        device="cpu",
    ).to(device="cuda:0")


def _reset_compiler_and_allocator(torch: Any) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json_artifact(
    path: str | os.PathLike[str],
    **collect_kwargs: Any,
) -> Dict[str, Any]:
    artifact = collect_results(**collect_kwargs)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def probe_environment() -> Dict[str, Any]:
    blockers: List[str] = []
    try:
        import torch
    except ImportError:
        return {
            "preflight_status": "blocked",
            "readiness_blockers": ["torch_unavailable"],
            "cuda_available": False,
            "platform": platform.platform(),
        }

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        blockers.append("cuda_unavailable")

    transformers_version = None
    transformers_revision = None
    try:
        transformers_version = importlib.metadata.version("transformers")
        transformers_revision = _installed_vcs_revision("transformers")
    except importlib.metadata.PackageNotFoundError:
        blockers.append("transformers_unavailable")
    if transformers_revision != TRANSFORMERS_REVISION:
        blockers.append("transformers_revision_unverified")

    device_name = None
    compute_capability = None
    gpu_total_memory_bytes = None
    gpu_multiprocessor_count = None
    if cuda_available:
        device_properties = torch.cuda.get_device_properties(0)
        device_name = device_properties.name
        compute_capability = [device_properties.major, device_properties.minor]
        gpu_total_memory_bytes = int(device_properties.total_memory)
        gpu_multiprocessor_count = int(device_properties.multi_processor_count)

    available_modules = {
        module_name
        for module_name in ("kernels", "cutlass")
        if importlib.util.find_spec(module_name) is not None
    }
    dependencies = {
        distribution_name: _installed_distribution_version(distribution_name)
        for distribution_name in (
            "torch",
            "transformers",
            "accelerate",
            "safetensors",
            "huggingface-hub",
            "kernels",
            "nvidia-cutlass-dsl",
        )
    }
    blockers.extend(dependency_readiness_blockers(dependencies))
    cuda_runtime = getattr(torch.version, "cuda", None)
    cuda_toolkit = (
        probe_cuda_toolkit_metadata()
        if cuda_available
        else {
            "status": "unavailable",
            "reason": "cuda_unavailable",
            "nvcc_path": None,
            "cuda_toolkit_version": None,
        }
    )
    nvidia_smi = (
        probe_nvidia_smi_metadata()
        if cuda_available
        else {
            "status": "unavailable",
            "reason": "cuda_unavailable",
            "gpu_uuid": None,
            "driver_version": None,
        }
    )
    if cuda_available and nvidia_smi["status"] != "measured":
        blockers.append("nvidia_driver_unavailable")

    return {
        "preflight_status": "ready" if not blockers else "blocked",
        "readiness_blockers": blockers,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": cuda_available,
        "cuda_device_name": device_name,
        "cuda_compute_capability": compute_capability,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_multiprocessor_count": gpu_multiprocessor_count,
        "gpu_uuid": nvidia_smi["gpu_uuid"],
        "nvidia_driver_version": nvidia_smi["driver_version"],
        "nvidia_smi_status": nvidia_smi["status"],
        "nvidia_smi_reason": nvidia_smi.get("reason"),
        "cuda_runtime": cuda_runtime,
        "cuda_toolkit_status": cuda_toolkit["status"],
        "cuda_toolkit_reason": cuda_toolkit.get("reason"),
        "cuda_toolkit_version": cuda_toolkit["cuda_toolkit_version"],
        "nvcc_path": cuda_toolkit["nvcc_path"],
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "transformers_revision": transformers_revision,
        "model_revision": MODEL_REVISION,
        "dtype": DTYPE,
        "dependency_versions": dependencies,
        "backend_availability": backend_availability(
            cuda_available=cuda_available,
            compute_capability=compute_capability,
            cuda_runtime=cuda_runtime,
            available_modules=available_modules,
            cuda_toolkit_version=cuda_toolkit["cuda_toolkit_version"],
            nvcc_available=cuda_toolkit["status"] == "measured",
            dependency_versions=dependencies,
        ),
        "compile_mode_availability": _compile_mode_availability(torch),
    }


def dependency_readiness_blockers(
    dependency_versions: Mapping[str, str | None],
) -> List[str]:
    blockers = []
    for dependency in sorted(CORE_DEPENDENCIES):
        observed = dependency_versions.get(dependency)
        expected = PINNED_DEPENDENCY_VERSIONS[dependency]
        blocker_name = dependency.replace("-", "_")
        if observed is None:
            blockers.append(f"{blocker_name}_unavailable")
        elif observed != expected:
            blockers.append(f"{blocker_name}_version_mismatch")
    return blockers


def probe_cuda_toolkit_metadata(
    *,
    environ: Mapping[str, str] | None = None,
    run_command: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    environment = os.environ if environ is None else environ
    cuda_home = environment.get("CUDA_HOME") or environment.get("CUDA_PATH")
    nvcc_path = None
    if cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(candidate):
            nvcc_path = candidate
        else:
            return {
                "status": "unavailable",
                "reason": "nvcc_missing_from_cuda_home",
                "nvcc_path": candidate,
                "cuda_toolkit_version": None,
            }
    else:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None and os.path.isdir("/usr/local/cuda"):
            candidate = "/usr/local/cuda/bin/nvcc"
            if os.path.isfile(candidate):
                nvcc_path = candidate
    if nvcc_path is None:
        return {
            "status": "unavailable",
            "reason": "nvcc_unavailable",
            "nvcc_path": None,
            "cuda_toolkit_version": None,
        }

    runner = run_command or subprocess.run
    try:
        result = runner(
            [nvcc_path, "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {
            "status": "unavailable",
            "reason": f"nvcc_failed:{type(exc).__name__}",
            "nvcc_path": nvcc_path,
            "cuda_toolkit_version": None,
        }
    if result.returncode != 0:
        return {
            "status": "unavailable",
            "reason": f"nvcc_exit_{result.returncode}",
            "nvcc_path": nvcc_path,
            "cuda_toolkit_version": None,
        }
    match = re.search(r"\brelease\s+(\d+\.\d+)\b", result.stdout)
    if match is None:
        return {
            "status": "unavailable",
            "reason": "nvcc_unparseable_output",
            "nvcc_path": nvcc_path,
            "cuda_toolkit_version": None,
        }
    return {
        "status": "measured",
        "reason": None,
        "nvcc_path": nvcc_path,
        "cuda_toolkit_version": match.group(1),
    }


def probe_nvidia_smi_metadata(
    *,
    run_command: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    runner = run_command or subprocess.run
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {
            "status": "unavailable",
            "reason": f"nvidia_smi_failed:{type(exc).__name__}",
            "gpu_uuid": None,
            "driver_version": None,
        }
    if result.returncode != 0:
        return {
            "status": "unavailable",
            "reason": f"nvidia_smi_exit_{result.returncode}",
            "gpu_uuid": None,
            "driver_version": None,
        }
    fields = [field.strip() for field in result.stdout.strip().split(",", maxsplit=1)]
    if len(fields) != 2 or not all(fields):
        return {
            "status": "unavailable",
            "reason": "nvidia_smi_unparseable_output",
            "gpu_uuid": None,
            "driver_version": None,
        }
    return {
        "status": "measured",
        "reason": None,
        "gpu_uuid": fields[0],
        "driver_version": fields[1],
    }


def _installed_vcs_revision(distribution_name: str) -> str | None:
    direct_url_text = importlib.metadata.distribution(distribution_name).read_text("direct_url.json")
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    return direct_url.get("vcs_info", {}).get("commit_id")


def _installed_distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _compile_mode_availability(torch: Any) -> Dict[str, Dict[str, str | None]]:
    available_modes = {"default"}
    inductor = getattr(torch, "_inductor", None)
    list_options = getattr(inductor, "list_mode_options", None)
    if list_options is not None:
        try:
            available_modes.update(str(mode) for mode in list_options())
        except Exception:  # pragma: no cover - depends on the pinned torch build.
            pass
    return {
        mode: (
            {"status": "available", "reason": None}
            if mode in available_modes
            else {
                "status": "not_applicable",
                "reason": "compile_mode_not_offered_by_pinned_torch",
            }
        )
        for mode in DEFAULT_COMPILE_MODES
    }


def _smoke_environment() -> Dict[str, Any]:
    return {
        "preflight_status": "blocked",
        "readiness_blockers": ["contract_smoke_not_performance_evidence"],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": False,
        "cuda_device_name": None,
        "cuda_compute_capability": None,
        "cuda_runtime": None,
        "torch_version": None,
        "transformers_version": None,
        "transformers_revision": None,
        "model_revision": MODEL_REVISION,
        "dtype": DTYPE,
    }


def _preflight_row_reason(mode: str, environment: Mapping[str, Any]) -> str:
    if mode == "contract-smoke":
        return "contract_smoke_not_performance_evidence"
    blockers = list(environment.get("readiness_blockers", []))
    return str(blockers[0]) if blockers else "environment_preflight_blocked"


def _runtime_configuration_status(
    configuration: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> tuple[str, str | None]:
    backend = str(configuration["experts_backend"])
    backend_availability = environment.get("backend_availability", {})
    backend_status = backend_availability.get(backend, {})
    status = backend_status.get("status")
    if status in {"not_applicable", "blocked"}:
        return str(status), str(backend_status.get("reason") or f"{backend}_{status}")

    if configuration["compiled"]:
        mode_availability = environment.get("compile_mode_availability", {})
        mode_status = mode_availability.get(configuration["compile_mode"], {})
        status = mode_status.get("status")
        if status in {"not_applicable", "blocked"}:
            return str(status), str(
                mode_status.get("reason")
                or f"compile_mode_{configuration['compile_mode']}_{status}"
            )
    return "applicable", None


def _execute_configuration(
    executor: RunConfiguration,
    configuration: Mapping[str, Any],
    regimes: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    try:
        rows = executor(configuration, regimes)
    except (ImportError, ModuleNotFoundError) as exc:
        return [
            {
                "regime_id": regime["regime_id"],
                "status": "failed",
                "reason": f"configuration_dependency_failed:{type(exc).__name__}:{exc}",
            }
            for regime in regimes
        ]
    except Exception as exc:  # pragma: no cover - exercised by real stock failures.
        return [
            {
                "regime_id": regime["regime_id"],
                "status": "failed",
                "reason": f"configuration_executor_failed:{type(exc).__name__}:{exc}",
            }
            for regime in regimes
        ]
    return list(rows)


def _is_incomplete_failure_reason(reason: Any) -> bool:
    text = str(reason)
    return text in {
        "executor_missing_required_regime",
        "real_executor_not_configured",
    } or text.startswith("configuration_dependency_failed:")


def _unconfigured_real_executor(
    _configuration: Mapping[str, Any],
    regimes: Sequence[Mapping[str, Any]],
) -> Sequence[Mapping[str, Any]]:
    return [
        {
            "regime_id": regime["regime_id"],
            "status": "failed",
            "reason": "real_executor_not_configured",
        }
        for regime in regimes
    ]


def _normalize_measured_row(
    regime: Mapping[str, Any],
    configuration: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> Dict[str, Any]:
    row = _empty_result_row(
        regime,
        configuration,
        status=str(measurement.get("status", "failed")),
        reason=measurement.get("reason"),
    )
    for field in (
        "resolved_experts_backend",
        "stage_experts_backends",
        "configuration_setup",
        "correctness",
        "timing",
        "throughput_tokens_per_second",
        "preprocessing",
        "routing_overhead",
        "allocator",
    ):
        if field in measurement:
            row[field] = measurement[field]
    return row


def _empty_result_row(
    regime: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    status: str,
    reason: Any,
    runtime_applicability: str = "applicable",
) -> Dict[str, Any]:
    return {
        "regime_id": regime["regime_id"],
        "phase": regime["phase"],
        "batch_size": regime["batch_size"],
        "sequence_length": regime["sequence_length"],
        "configuration_id": configuration["configuration_id"],
        "experts_backend": configuration["experts_backend"],
        "compiled": configuration["compiled"],
        "compile_mode": configuration["compile_mode"],
        "fullgraph": configuration["fullgraph"],
        "configuration_eligibility": configuration["eligibility"],
        "runtime_applicability": runtime_applicability,
        "resolved_experts_backend": None,
        "stage_experts_backends": None,
        "configuration_setup": {
            "model_load_seconds": None,
            "compile_wrapper_seconds": None,
        },
        "status": status,
        "reason": reason,
        "correctness": {
            "status": "not_measured",
            "reference": "eager__uncompiled",
            "max_abs_error": None,
            "relative_l2_error": None,
            "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
            "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        },
        "timing": {
            "cuda_event_median_seconds": None,
            "wall_median_seconds": None,
            "warmup_repetitions": None,
            "measured_repetitions": None,
        },
        "throughput_tokens_per_second": None,
        "preprocessing": {"status": "not_measured", "median_seconds": None},
        "routing_overhead": {"status": "not_measured", "median_seconds": None},
        "allocator": {"status": "not_measured", "peak_allocated_bytes": None},
    }


def _row_inventory_complete(
    results: Sequence[Mapping[str, Any]],
    regimes: Sequence[Mapping[str, Any]],
    configurations: Sequence[Mapping[str, Any]],
) -> bool:
    expected = {
        (regime["regime_id"], configuration["configuration_id"])
        for regime in regimes
        for configuration in configurations
    }
    observed = {
        (row.get("regime_id"), row.get("configuration_id"))
        for row in results
    }
    return observed == expected and len(results) == len(expected)


def _validate_compile_modes(modes: Sequence[str]) -> List[str]:
    normalized = []
    for mode in modes:
        value = str(mode).strip()
        if not value:
            raise ValueError("compile modes must be non-empty")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("at least one compile mode is required")
    return normalized


def _version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    components = []
    for component in str(version).split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def _version_in_half_open_range(
    version: str | None,
    minimum: str,
    maximum: str,
) -> bool:
    parsed = _version_tuple(version)
    return _version_tuple(minimum) <= parsed < _version_tuple(maximum)


def _deduplicate(values: Sequence[Any]) -> List[str]:
    deduplicated: List[str] = []
    for value in values:
        text = str(value)
        if text not in deduplicated:
            deduplicated.append(text)
    return deduplicated


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="write a contract-only smoke artifact")
    mode.add_argument("--real", action="store_true", help="run the real pinned CUDA cohort")
    parser.add_argument("--compile-mode", action="append", dest="compile_modes")
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=DEFAULT_WARMUP_REPETITIONS,
    )
    parser.add_argument(
        "--measured-repetitions",
        type=int,
        default=DEFAULT_MEASURED_REPETITIONS,
    )
    parser.add_argument("--json-output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "contract-smoke" if args.smoke else "real"
    artifact = write_json_artifact(
        args.json_output,
        mode=mode,
        compile_modes=args.compile_modes or DEFAULT_COMPILE_MODES,
        warmup_repetitions=args.warmup_repetitions,
        measured_repetitions=args.measured_repetitions,
        command=sys.argv if argv is None else [sys.argv[0], *argv],
    )
    print(
        f"row_inventory_complete={str(artifact['summary']['row_inventory_complete']).lower()} "
        f"cohort_complete={str(artifact['summary']['cohort_complete']).lower()} "
        f"blockers={','.join(artifact['summary']['readiness_blockers']) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
