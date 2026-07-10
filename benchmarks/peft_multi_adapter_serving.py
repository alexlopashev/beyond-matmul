#!/usr/bin/env python3
"""PEFT multi-adapter serving benchmark harness."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Sequence


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BENCHMARK = "peft_multi_adapter_serving"
CONTRACT_PATH = "docs/peft_multi_adapter_serving_benchmark_contract.md"
HARDWARE_BENCHMARK = "hardware_backed_peft_multi_adapter_serving"
HARDWARE_CONTRACT_PATH = "docs/hardware_backed_production_benchmark_contract.md"
BASE_MODEL = "facebook/opt-125m"
BASE_MODEL_REVISION = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"
MODEL_CONTEXT_LIMIT = 2048
INPUT_SEED = 20260708
DEFAULT_SEQUENCE_LENGTHS = [16, 64, 128]
DEFAULT_BATCH_SIZES = [1, 2]
DEFAULT_WARMUP_REPETITIONS = 10
DEFAULT_MEASURED_REPETITIONS = 50
HARDWARE_FORWARD_WARMUP_REPETITIONS = 25
HARDWARE_FORWARD_MEASURED_REPETITIONS = 100
HARDWARE_SWITCH_WARMUP_REPETITIONS = 25
HARDWARE_SWITCH_MEASURED_REPETITIONS = 100
DEFAULT_UPSTREAM_REF = "main"
DEFAULT_FORK_REF = "beyond-matmul/provenance-lora-inference"
HARDWARE_UPSTREAM_REF = "1598ecb8fc504bfcb08b9b232b295414a729d7ed"
HARDWARE_FORK_REF = "7ac8d57b100846837c5a3b76c65e1e1954ccc3c8"
UPSTREAM_REPOSITORY = "huggingface/peft"
FORK_REPOSITORY = "alexlopashev/peft"
CORRECTNESS_MAX_ABS_TOLERANCE = 1e-4
CORRECTNESS_RELATIVE_L2_TOLERANCE = 1e-5
BASE_MODEL_FP32_BYTES_APPROX = 125_000_000 * 4
STRUCTURED_LOW_RANK_DEVICE = "cpu"
STRUCTURED_LOW_RANK_DTYPE = "torch.float32"
BASELINES = [
    "upstream_peft_unmerged",
    "upstream_peft_merged_dense_cache",
    "upstream_peft_repeated_merge_unmerge",
    "beyond_matmul_factor_provenance",
]
HARDWARE_BASELINES = [
    "upstream_peft_unmerged",
    "upstream_peft_merged_dense_cache",
    "upstream_peft_repeated_merge_unmerge",
    "beyond_matmul_structured_low_rank",
]


class AdapterSpec(NamedTuple):
    name: str
    repository: str
    revision: str
    payload_file: str
    payload_bytes: int


ADAPTERS = [
    AdapterSpec(
        name="merchant",
        repository="choyiny/opt-125m-lora-merchant-finetune",
        revision="c25d7ba3a15502b4dcbd609758caec8b2ce78eb4",
        payload_file="adapter_model.safetensors",
        payload_bytes=2_365_968,
    ),
    AdapterSpec(
        name="gaisb",
        repository="guyk1971/gaisb",
        revision="cdad7e89c32a940aa1269dddbfcf29e7c9cdda37",
        payload_file="adapter_model.bin",
        payload_bytes=2_376_641,
    ),
]

TimeForward = Callable[[str, str, Dict[str, Any], int, int], Sequence[float]]
TimeSwitch = Callable[[str, str, int, int], Sequence[float]]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local CI pins torch.
        raise RuntimeError("PyTorch is required for the PEFT multi-adapter benchmark harness") from exc
    return torch


def _time_forward(
    _baseline: str,
    _adapter: str,
    inputs: Dict[str, Any],
    warmup_repetitions: int,
    measured_repetitions: int,
) -> List[float]:
    model = inputs["model"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    torch = _torch()
    with torch.inference_mode():
        for _ in range(warmup_repetitions):
            model(input_ids, attention_mask)
        latencies = []
        for _ in range(measured_repetitions):
            start = time.perf_counter()
            model(input_ids, attention_mask)
            latencies.append(time.perf_counter() - start)
    return latencies


def _time_switch(_baseline: str, _adapter: str, warmup_repetitions: int, measured_repetitions: int) -> List[float]:
    for _ in range(warmup_repetitions):
        pass
    latencies = []
    for _ in range(measured_repetitions):
        start = time.perf_counter()
        latencies.append(time.perf_counter() - start)
    return latencies


class _SyntheticAdapterModel:
    def __init__(self, baseline: str, adapter: AdapterSpec, vocab_size: int = 16) -> None:
        self.baseline = baseline
        self.adapter = adapter
        self.vocab_size = vocab_size

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        torch = _torch()
        ids = input_ids.to(dtype=torch.float32)
        mask = attention_mask.to(dtype=torch.float32)
        features = torch.arange(self.vocab_size, dtype=torch.float32, device=ids.device)
        adapter_offset = 0.125 if self.adapter.name == "merchant" else 0.25
        logits = torch.sin((ids.unsqueeze(-1) + features) * 0.03125 + adapter_offset)
        logits = logits + mask.unsqueeze(-1) * 0.0625
        if self.baseline in {"upstream_peft_merged_dense_cache", "beyond_matmul_factor_provenance"}:
            logits = logits + 1e-8
        return logits


def collect_results(
    *,
    adapters: Sequence[AdapterSpec] | None = None,
    sequence_lengths: Sequence[int] | None = None,
    batch_sizes: Sequence[int] | None = None,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    mode: str = "real",
    device: str = "cpu",
    model_context_limit: int | None = None,
    upstream_peft_path: str | None = None,
    fork_peft_path: str | None = None,
    upstream_peft_ref: str = DEFAULT_UPSTREAM_REF,
    fork_peft_ref: str = DEFAULT_FORK_REF,
    checkout_dir: str | os.PathLike[str] | None = None,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
    time_forward: TimeForward = _time_forward,
    time_switch: TimeSwitch = _time_switch,
) -> Dict[str, Any]:
    adapters = list(adapters or ADAPTERS)
    sequence_lengths = list(sequence_lengths or DEFAULT_SEQUENCE_LENGTHS)
    batch_sizes = list(batch_sizes or DEFAULT_BATCH_SIZES)
    _validate_positive_ints("sequence length", sequence_lengths)
    _validate_positive_ints("batch size", batch_sizes)
    if not adapters:
        raise ValueError("at least one adapter is required")
    if warmup_repetitions < 0:
        raise ValueError("warmup repetitions must be non-negative")
    if measured_repetitions <= 0:
        raise ValueError("measured repetitions must be positive")
    if mode not in {"synthetic-smoke", "real"}:
        raise ValueError(f"unsupported mode: {mode}")
    if device != "cpu":
        raise ValueError("issue #98 only defines the CPU fp32 contract")

    resolved_context_limit = model_context_limit
    if resolved_context_limit is None:
        resolved_context_limit = _resolve_model_context_limit(mode)

    resolved_upstream_path = upstream_peft_path
    resolved_fork_path = fork_peft_path
    if mode == "real":
        resolved_upstream_path = _resolve_checkout(
            repository=UPSTREAM_REPOSITORY,
            ref=upstream_peft_ref,
            path=upstream_peft_path,
            checkout_dir=checkout_dir,
        )
        resolved_fork_path = _resolve_checkout(
            repository=FORK_REPOSITORY,
            ref=fork_peft_ref,
            path=fork_peft_path,
            checkout_dir=checkout_dir,
        )

    results = []
    for adapter in adapters:
        for sequence_length in sequence_lengths:
            for batch_size in batch_sizes:
                if resolved_context_limit is not None and sequence_length > resolved_context_limit:
                    results.extend(
                        _blocked_context_rows(
                            adapter=adapter,
                            sequence_length=sequence_length,
                            batch_size=batch_size,
                            model_context_limit=resolved_context_limit,
                        )
                    )
                elif mode == "synthetic-smoke":
                    results.extend(
                        _collect_synthetic_case(
                            adapter,
                            sequence_length,
                            batch_size,
                            warmup_repetitions,
                            measured_repetitions,
                            time_forward,
                            time_switch,
                        )
                    )
                else:
                    results.extend(
                        _collect_real_case(
                            adapter,
                            sequence_length,
                            batch_size,
                            warmup_repetitions,
                            measured_repetitions,
                            str(resolved_upstream_path),
                            str(resolved_fork_path),
                        )
                    )

    return {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "contract": CONTRACT_PATH,
        "mode": mode,
        "workload": {
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
            "model_context_limit": resolved_context_limit,
            "adapters": [_adapter_metadata(adapter) for adapter in adapters],
            "task": "causal_lm_prefill_logits",
            "dtype": "float32",
            "device": device,
            "sequence_lengths": sequence_lengths,
            "batch_sizes": batch_sizes,
            "input_seed": INPUT_SEED,
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
        },
        "dependencies": _dependency_metadata(
            mode=mode,
            upstream_peft_path=resolved_upstream_path,
            fork_peft_path=resolved_fork_path,
            upstream_peft_ref=upstream_peft_ref,
            fork_peft_ref=fork_peft_ref,
        ),
        "environment": _environment_metadata(),
        "run": _run_metadata(command=command, generated_at_utc=generated_at_utc, mode=mode),
        "results": results,
        "summary": _artifact_summary(
            results,
            adapters=adapters,
            sequence_lengths=sequence_lengths,
            batch_sizes=batch_sizes,
            mode=mode,
            context_limit=resolved_context_limit,
        ),
    }


def collect_hardware_contract_results(
    *,
    adapters: Sequence[AdapterSpec] | None = None,
    sequence_lengths: Sequence[int] | None = None,
    batch_sizes: Sequence[int] | None = None,
    mode: str = "synthetic-smoke",
    cuda_available: bool | None = None,
    model_context_limit: int | None = None,
    hardware_metadata: Dict[str, Any] | None = None,
    upstream_peft_path: str | None = None,
    checkout_dir: str | os.PathLike[str] | None = None,
    hardware_case_runner: Callable[..., Dict[str, Any]] | None = None,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
) -> Dict[str, Any]:
    adapters = list(adapters or ADAPTERS)
    sequence_lengths = list(sequence_lengths or DEFAULT_SEQUENCE_LENGTHS)
    batch_sizes = list(batch_sizes or DEFAULT_BATCH_SIZES)
    _validate_positive_ints("sequence length", sequence_lengths)
    _validate_positive_ints("batch size", batch_sizes)
    if not adapters:
        raise ValueError("at least one adapter is required")
    if mode not in {"synthetic-smoke", "fake-backend", "real"}:
        raise ValueError(f"unsupported hardware contract mode: {mode}")

    resolved_context_limit = MODEL_CONTEXT_LIMIT if model_context_limit is None else model_context_limit
    resolved_cuda_available = _cuda_available() if cuda_available is None else bool(cuda_available)
    resolved_hardware = _hardware_contract_metadata(
        cuda_available=resolved_cuda_available,
        override=hardware_metadata,
    )
    preflight_blockers = _hardware_preflight_blockers(
        adapters=adapters,
        sequence_lengths=sequence_lengths,
        model_context_limit=resolved_context_limit,
        cuda_available=resolved_cuda_available,
        hardware=resolved_hardware,
        mode=mode,
    )

    resolved_upstream_path = upstream_peft_path
    if mode == "real" and not _hardware_measurement_preflight_blockers(preflight_blockers):
        resolved_upstream_path = _resolve_checkout(
            repository=UPSTREAM_REPOSITORY,
            ref=HARDWARE_UPSTREAM_REF,
            path=upstream_peft_path,
            checkout_dir=checkout_dir,
        )

    if mode == "synthetic-smoke" or _hardware_measurement_preflight_blockers(preflight_blockers):
        row_blockers = _ordered_unique_blockers([*preflight_blockers, "cuda_measurements_missing"])
        results = [
            _hardware_contract_row(
                adapter=adapter,
                baseline=baseline,
                sequence_length=sequence_length,
                batch_size=batch_size,
                readiness_blockers=row_blockers,
            )
            for adapter in adapters
            for sequence_length in sequence_lengths
            for batch_size in batch_sizes
            for baseline in HARDWARE_BASELINES
        ]
    else:
        runner = hardware_case_runner or _run_hardware_contract_worker
        results = _collect_hardware_measured_rows(
            adapters=adapters,
            sequence_lengths=sequence_lengths,
            batch_sizes=batch_sizes,
            runner=runner,
            mode=mode,
            upstream_peft_path=resolved_upstream_path,
        )

    return {
        "schema_version": 1,
        "benchmark": HARDWARE_BENCHMARK,
        "contract": HARDWARE_CONTRACT_PATH,
        "mode": mode,
        "workload": {
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
            "model_context_limit": resolved_context_limit,
            "adapters": [_adapter_metadata(adapter) for adapter in adapters],
            "task": "causal_lm_prefill_logits",
            "dtype": "float32",
            "device": "cuda",
            "sequence_lengths": sequence_lengths,
            "batch_sizes": batch_sizes,
            "input_seed": INPUT_SEED,
            "forward_warmup_repetitions": HARDWARE_FORWARD_WARMUP_REPETITIONS,
            "forward_measured_repetitions": HARDWARE_FORWARD_MEASURED_REPETITIONS,
            "switch_warmup_repetitions": HARDWARE_SWITCH_WARMUP_REPETITIONS,
            "switch_measured_repetitions": HARDWARE_SWITCH_MEASURED_REPETITIONS,
        },
        "dependencies": _hardware_dependency_metadata(),
        "hardware": resolved_hardware,
        "environment": _environment_metadata(),
        "run": _run_metadata(command=command, generated_at_utc=generated_at_utc, mode=mode),
        "results": results,
        "summary": _hardware_contract_summary(
            results,
            adapters=adapters,
            sequence_lengths=sequence_lengths,
            batch_sizes=batch_sizes,
            readiness_blockers=preflight_blockers,
        ),
    }


def _cuda_available() -> bool:
    torch = _torch()
    return bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())


def _hardware_preflight_blockers(
    *,
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    model_context_limit: int | None,
    cuda_available: bool,
    hardware: Dict[str, Any],
    mode: str,
) -> List[str]:
    blockers = []
    if mode != "real":
        blockers.append("synthetic_smoke_not_benchmark_evidence")
    if not cuda_available:
        blockers.append("missing_cuda")
    if model_context_limit is None:
        blockers.append("model_context_limit_unresolved")
    elif any(sequence_length > model_context_limit for sequence_length in sequence_lengths):
        blockers.append("context_limit_exceeded")
    if _hardware_contract_pin_mismatches(adapters):
        blockers.append("contract_pin_mismatch")
    if cuda_available and _incomplete_hardware_metadata(hardware):
        blockers.append("incomplete_hardware_metadata")
    return _ordered_unique_blockers(blockers)


def _hardware_measurement_preflight_blockers(blockers: Sequence[str]) -> List[str]:
    non_measurement_blockers = {"synthetic_smoke_not_benchmark_evidence"}
    return [blocker for blocker in blockers if blocker not in non_measurement_blockers]


def _hardware_contract_pin_mismatches(adapters: Sequence[AdapterSpec]) -> List[str]:
    expected = {adapter.name: adapter for adapter in ADAPTERS}
    mismatches = []
    for adapter in adapters:
        contract = expected.get(adapter.name)
        if contract is None:
            mismatches.append(adapter.name)
            continue
        if (
            adapter.repository != contract.repository
            or adapter.revision != contract.revision
            or adapter.payload_file != contract.payload_file
        ):
            mismatches.append(adapter.name)
    return mismatches


def _incomplete_hardware_metadata(hardware: Dict[str, Any]) -> bool:
    required_fields = ("gpu", "cuda_device", "compute_capability", "total_memory_bytes", "driver")
    return any(hardware.get(field) is None for field in required_fields)


def _ordered_unique_blockers(blockers: Sequence[str]) -> List[str]:
    order = [
        "synthetic_smoke_not_benchmark_evidence",
        "missing_cuda",
        "model_context_limit_unresolved",
        "context_limit_exceeded",
        "contract_pin_mismatch",
        "incomplete_hardware_metadata",
        "cuda_measurements_missing",
        "structured_path_blocked_milestone_2",
        "required_cases_missing",
        "correctness_checks_failed",
        "cuda_memory_fields_unmeasured",
        "switching_cases_unmeasured",
        "dense_fallback_cases_not_explicit",
    ]
    seen = set()
    unique = []
    for blocker in order:
        if blocker in blockers and blocker not in seen:
            unique.append(blocker)
            seen.add(blocker)
    for blocker in blockers:
        if blocker not in seen:
            unique.append(blocker)
            seen.add(blocker)
    return unique


def _hardware_contract_row(
    *,
    adapter: AdapterSpec,
    baseline: str,
    sequence_length: int,
    batch_size: int,
    readiness_blockers: Sequence[str],
) -> Dict[str, Any]:
    status = "blocked" if readiness_blockers else "not_measured"
    structured_blocked = baseline == "beyond_matmul_structured_low_rank"
    return {
        "case": _case_name(adapter.name, sequence_length, batch_size),
        "adapter": adapter.name,
        "baseline": baseline,
        "status": status,
        "readiness_blockers": list(readiness_blockers),
        "reason": ", ".join(readiness_blockers) if readiness_blockers else "hardware measurement not implemented",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "forward_latency_seconds": None,
        "forward_latency_wall_seconds": None,
        "forward_repetitions": {
            "warmup": HARDWARE_FORWARD_WARMUP_REPETITIONS,
            "measured": HARDWARE_FORWARD_MEASURED_REPETITIONS,
        },
        "adapter_switch_seconds": None,
        "adapter_switch_wall_seconds": None,
        "adapter_switch_status": "not_measured_blocked" if readiness_blockers else "not_measured",
        "adapter_switch_repetitions": {
            "warmup": HARDWARE_SWITCH_WARMUP_REPETITIONS,
            "measured": HARDWARE_SWITCH_MEASURED_REPETITIONS,
        },
        "preprocessing_seconds": {
            "model_load": None,
            "hub_download": None,
            "tokenization": None,
            "dense_cache_build": None,
            "structured_factor_pack": None,
            "compilation_or_graph_capture": None,
        },
        "cuda_memory": {
            "setup_peak_allocated_bytes": None,
            "setup_peak_reserved_bytes": None,
            "steady_peak_allocated_bytes": None,
            "steady_peak_reserved_bytes": None,
            "post_setup_allocated_bytes": None,
            "post_setup_reserved_bytes": None,
            "post_loop_allocated_bytes": None,
            "post_loop_reserved_bytes": None,
        },
        "measurement": {
            "isolated_process": False,
            "process_id": None,
            "mode": "blocked" if readiness_blockers else "not_measured",
        },
        "timing_protocol": {
            "cuda_events": False,
            "synchronized_before_after": False,
            "allocator_reset": False,
            "setup_excluded_from_steady_state": False,
        },
        "storage": _hardware_storage_metadata(baseline, adapter),
        "correctness": _hardware_empty_correctness(passed=False),
        "lowering": _hardware_lowering_metadata(
            baseline,
            adapter.name,
            blocked=bool(readiness_blockers) or structured_blocked,
        ),
    }


def _collect_hardware_measured_rows(
    *,
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    runner: Callable[..., Dict[str, Any]],
    mode: str,
    upstream_peft_path: str | None,
) -> List[Dict[str, Any]]:
    rows = []
    for adapter in adapters:
        for sequence_length in sequence_lengths:
            for batch_size in batch_sizes:
                reference_logits = None
                for baseline in HARDWARE_BASELINES:
                    if baseline == "beyond_matmul_structured_low_rank":
                        rows.append(
                            _hardware_contract_row(
                                adapter=adapter,
                                baseline=baseline,
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                readiness_blockers=["structured_path_blocked_milestone_2"],
                            )
                        )
                        continue
                    try:
                        payload = runner(
                            adapter=adapter,
                            baseline=baseline,
                            sequence_length=sequence_length,
                            batch_size=batch_size,
                            forward_warmup_repetitions=HARDWARE_FORWARD_WARMUP_REPETITIONS,
                            forward_measured_repetitions=HARDWARE_FORWARD_MEASURED_REPETITIONS,
                            switch_warmup_repetitions=HARDWARE_SWITCH_WARMUP_REPETITIONS,
                            switch_measured_repetitions=HARDWARE_SWITCH_MEASURED_REPETITIONS,
                            mode=mode,
                            upstream_peft_path=upstream_peft_path,
                        )
                    except Exception as exc:
                        payload = {
                            "status": "failed",
                            "reason": str(exc),
                            "forward_latencies_seconds": None,
                            "forward_wall_seconds": None,
                            "switch_latencies_seconds": None,
                            "switch_wall_seconds": None,
                            "preprocessing_seconds": {},
                            "cuda_memory": {},
                            "correctness": {"passed": False},
                        }
                    if baseline == "upstream_peft_unmerged" and payload.get("logits") is not None:
                        reference_logits = payload["logits"]
                    rows.append(
                        _hardware_payload_to_row(
                            adapter=adapter,
                            baseline=baseline,
                            sequence_length=sequence_length,
                            batch_size=batch_size,
                            payload=payload,
                            reference_logits=reference_logits,
                        )
                    )
    return rows


def _hardware_payload_to_row(
    *,
    adapter: AdapterSpec,
    baseline: str,
    sequence_length: int,
    batch_size: int,
    payload: Dict[str, Any],
    reference_logits: Any,
) -> Dict[str, Any]:
    status = payload.get("status") or "failed"
    correctness = _hardware_correctness_from_payload(payload, reference_logits)
    reason = payload.get("reason")
    if status == "ok" and not correctness["passed"]:
        status = "failed_correctness"
        reason = reason or "correctness tolerance failed"
    forward_latencies = payload.get("forward_latencies_seconds")
    forward_wall = payload.get("forward_wall_seconds")
    switch_latencies = payload.get("switch_latencies_seconds")
    switch_wall = payload.get("switch_wall_seconds")
    storage = _hardware_storage_metadata(baseline, adapter)
    storage.update(payload.get("storage") or {})
    return {
        "case": _case_name(adapter.name, sequence_length, batch_size),
        "adapter": adapter.name,
        "baseline": baseline,
        "status": status,
        "readiness_blockers": [],
        "reason": reason,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "forward_latency_seconds": _latency_stats_or_none(forward_latencies),
        "forward_latency_wall_seconds": _latency_stats_or_none(forward_wall),
        "forward_repetitions": {
            "warmup": HARDWARE_FORWARD_WARMUP_REPETITIONS,
            "measured": HARDWARE_FORWARD_MEASURED_REPETITIONS,
        },
        "adapter_switch_seconds": _latency_stats_or_none(switch_latencies),
        "adapter_switch_wall_seconds": _latency_stats_or_none(switch_wall),
        "adapter_switch_status": _hardware_adapter_switch_status(
            baseline,
            switch_latencies=switch_latencies,
            status=status,
        ),
        "adapter_switch_repetitions": {
            "warmup": HARDWARE_SWITCH_WARMUP_REPETITIONS,
            "measured": HARDWARE_SWITCH_MEASURED_REPETITIONS,
        },
        "preprocessing_seconds": _hardware_preprocessing_seconds(payload.get("preprocessing_seconds")),
        "cuda_memory": _hardware_cuda_memory(payload.get("cuda_memory")),
        "measurement": {
            "isolated_process": bool(payload.get("isolated_process")),
            "process_id": payload.get("process_id"),
            "mode": "measured" if status in {"ok", "failed_correctness"} else status,
        },
        "timing_protocol": _hardware_timing_protocol(payload.get("timing_protocol")),
        "storage": storage,
        "correctness": correctness,
        "lowering": _hardware_lowering_metadata(baseline, adapter.name, blocked=False),
    }


def _hardware_correctness_from_payload(payload: Dict[str, Any], reference_logits: Any) -> Dict[str, Any]:
    correctness = dict(payload.get("correctness") or {})
    if "passed" not in correctness and payload.get("logits") is not None and reference_logits is not None:
        correctness = _correctness_metrics(payload["logits"], reference_logits)
    return {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": correctness.get("max_abs_error"),
        "relative_l2_error": correctness.get("relative_l2_error"),
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "tolerance_profile": "cuda_fp32",
        "passed": bool(correctness.get("passed")),
    }


def _latency_stats_or_none(latencies: Sequence[float] | None) -> Dict[str, float] | None:
    if latencies is None:
        return None
    values = list(latencies)
    if not values:
        return None
    return _latency_stats(values)


def _hardware_preprocessing_seconds(preprocessing: Dict[str, Any] | None) -> Dict[str, float | None]:
    values = dict(preprocessing or {})
    return {
        "model_load": values.get("model_load"),
        "hub_download": values.get("hub_download"),
        "tokenization": values.get("tokenization"),
        "dense_cache_build": values.get("dense_cache_build"),
        "structured_factor_pack": values.get("structured_factor_pack"),
        "compilation_or_graph_capture": values.get("compilation_or_graph_capture"),
    }


def _hardware_cuda_memory(memory: Dict[str, Any] | None) -> Dict[str, int | None]:
    values = dict(memory or {})
    fields = (
        "setup_peak_allocated_bytes",
        "setup_peak_reserved_bytes",
        "steady_peak_allocated_bytes",
        "steady_peak_reserved_bytes",
        "post_setup_allocated_bytes",
        "post_setup_reserved_bytes",
        "post_loop_allocated_bytes",
        "post_loop_reserved_bytes",
    )
    return {
        field: int(values[field]) if isinstance(values.get(field), int) and values[field] >= 0 else None
        for field in fields
    }


def _hardware_timing_protocol(protocol: Dict[str, Any] | None) -> Dict[str, bool]:
    values = dict(protocol or {})
    return {
        "cuda_events": bool(values.get("cuda_events")),
        "synchronized_before_after": bool(values.get("synchronized_before_after")),
        "allocator_reset": bool(values.get("allocator_reset")),
        "setup_excluded_from_steady_state": bool(values.get("setup_excluded_from_steady_state")),
    }


def _hardware_adapter_switch_status(
    baseline: str,
    *,
    switch_latencies: Sequence[float] | None,
    status: str,
) -> str:
    if status in {"blocked", "failed", "not_applicable"}:
        return f"not_measured_{status}"
    if switch_latencies is None:
        return "not_measured_unavailable"
    if baseline == "upstream_peft_merged_dense_cache":
        return "measured_cuda_dense_cache_pointer_swap"
    if baseline == "upstream_peft_repeated_merge_unmerge":
        return "measured_cuda_merge_unmerge"
    return "measured_cuda_loaded_adapters"


def _hardware_empty_correctness(*, passed: bool) -> Dict[str, Any]:
    return {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": None,
        "relative_l2_error": None,
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "tolerance_profile": "cuda_fp32",
        "passed": passed,
    }


def _hardware_storage_metadata(baseline: str, adapter: AdapterSpec) -> Dict[str, Any]:
    return {
        "adapter_payload_bytes": adapter.payload_bytes,
        "dense_cache_bytes_per_adapter": BASE_MODEL_FP32_BYTES_APPROX
        if baseline == "upstream_peft_merged_dense_cache"
        else None,
        "structured_factor_bytes": None,
        "provenance_metadata_bytes": None,
        "dense_fallback_cache_bytes": BASE_MODEL_FP32_BYTES_APPROX
        if baseline in {"upstream_peft_merged_dense_cache", "upstream_peft_repeated_merge_unmerge"}
        else None,
    }


def _hardware_lowering_metadata(baseline: str, adapter_name: str, *, blocked: bool) -> Dict[str, Any]:
    if baseline == "beyond_matmul_structured_low_rank":
        return {
            "kind": "structured_low_rank",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": blocked,
            "fallback_reason": "preflight_blocked" if blocked else None,
        }
    return {
        "kind": _hardware_baseline_kind(baseline),
        "active_adapter": adapter_name,
        "dense_fallback_available": True,
        "dense_fallback_used": baseline != "upstream_peft_unmerged",
        "fallback_reason": None,
    }


def _hardware_baseline_kind(baseline: str) -> str:
    return {
        "upstream_peft_unmerged": "peft_unmerged_adapter",
        "upstream_peft_merged_dense_cache": "peft_merged_dense_cache",
        "upstream_peft_repeated_merge_unmerge": "peft_repeated_merge_unmerge",
    }.get(baseline, baseline)


def _hardware_contract_summary(
    results: Sequence[Dict[str, Any]],
    *,
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    readiness_blockers: Sequence[str],
) -> Dict[str, Any]:
    all_required_cases_present = _all_hardware_required_cases_present(results, adapters, sequence_lengths, batch_sizes)
    all_correctness_checks_passed = all(row["status"] == "ok" and row["correctness"]["passed"] for row in results)
    all_memory_fields_measured = all(_hardware_memory_measured(row) for row in results)
    all_switching_cases_present = all(row.get("adapter_switch_seconds") is not None for row in results)
    all_fallback_cases_explicit = all(
        "dense_fallback_available" in row["lowering"] and "dense_fallback_used" in row["lowering"]
        for row in results
    )
    blockers = list(readiness_blockers)
    for row in results:
        blockers.extend(row.get("readiness_blockers", []))
    if not all_required_cases_present:
        blockers.append("required_cases_missing")
    if not all_correctness_checks_passed:
        blockers.append("correctness_checks_failed")
    if not all_memory_fields_measured:
        blockers.append("cuda_memory_fields_unmeasured")
    if not all_switching_cases_present:
        blockers.append("switching_cases_unmeasured")
    if not all_fallback_cases_explicit:
        blockers.append("dense_fallback_cases_not_explicit")
    blockers = _ordered_unique_blockers(blockers)
    ready = not blockers
    return {
        "all_required_cases_present": all_required_cases_present,
        "all_correctness_checks_passed": all_correctness_checks_passed,
        "all_memory_fields_measured": all_memory_fields_measured,
        "all_switching_cases_present": all_switching_cases_present,
        "all_fallback_cases_explicit": all_fallback_cases_explicit,
        "production_contract_ready": ready,
        "performance_fields_interpretable": ready,
        "memory_fields_interpretable": ready,
        "readiness_blockers": blockers,
        "primary_performance_question": "structured_low_rank_switch_or_memory_without_forward_regression",
        "performance_claim": "none",
        "memory_or_control_claim": "none",
        "claim_summary_rows": _hardware_claim_summary_rows(results),
    }


def _all_hardware_required_cases_present(
    results: Sequence[Dict[str, Any]],
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
) -> bool:
    present = {
        (row["adapter"], row["baseline"], row["sequence_length"], row["batch_size"])
        for row in results
    }
    required = {
        (adapter.name, baseline, sequence_length, batch_size)
        for adapter in adapters
        for sequence_length in sequence_lengths
        for batch_size in batch_sizes
        for baseline in HARDWARE_BASELINES
    }
    return required.issubset(present)


def _hardware_memory_measured(row: Dict[str, Any]) -> bool:
    memory = row.get("cuda_memory")
    if not isinstance(memory, dict):
        return False
    required_fields = (
        "setup_peak_allocated_bytes",
        "setup_peak_reserved_bytes",
        "steady_peak_allocated_bytes",
        "steady_peak_reserved_bytes",
        "post_setup_allocated_bytes",
        "post_setup_reserved_bytes",
        "post_loop_allocated_bytes",
        "post_loop_reserved_bytes",
    )
    return all(isinstance(memory.get(field), int) and memory[field] >= 0 for field in required_fields)


def _hardware_claim_summary_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in results:
        if row["status"] != "ok" or not row["correctness"]["passed"]:
            continue
        if row["baseline"] == "beyond_matmul_structured_low_rank":
            continue
        rows.append(
            {
                "case": row["case"],
                "adapter": row["adapter"],
                "baseline": row["baseline"],
                "forward_latency_median_seconds": (
                    row["forward_latency_seconds"]["median"]
                    if row.get("forward_latency_seconds")
                    else None
                ),
                "adapter_switch_median_seconds": (
                    row["adapter_switch_seconds"]["median"]
                    if row.get("adapter_switch_seconds")
                    else None
                ),
                "correctness_passed": True,
            }
        )
    return rows


def _hardware_dependency_metadata() -> Dict[str, Any]:
    torch = _torch()
    return {
        "python": platform.python_version(),
        "torch": {"version": torch.__version__, "cuda": getattr(torch.version, "cuda", None)},
        "transformers": _module_metadata("transformers", "hardware-contract"),
        "peft_upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "version": None,
            "revision": HARDWARE_UPSTREAM_REF,
        },
        "peft_fork": {
            "repository": FORK_REPOSITORY,
            "version": None,
            "revision": HARDWARE_FORK_REF,
        },
        "beyond_matmul": {
            "repository": "alexlopashev/beyond-matmul",
            "revision": _git_revision(Path(ROOT)),
        },
        "huggingface_hub": _module_metadata("huggingface_hub", "hardware-contract"),
        "accelerate": _module_metadata("accelerate", "hardware-contract"),
        "safetensors": _module_metadata("safetensors", "hardware-contract"),
    }


def _hardware_contract_metadata(
    *, cuda_available: bool, override: Dict[str, Any] | None
) -> Dict[str, Any]:
    torch = _torch()
    if cuda_available and getattr(torch, "cuda", None) is not None:
        cuda_device = 0
        try:
            properties = torch.cuda.get_device_properties(cuda_device)
            gpu = properties.name
            compute_capability = f"{properties.major}.{properties.minor}"
            total_memory_bytes = int(properties.total_memory)
        except Exception:
            gpu = None
            compute_capability = None
            total_memory_bytes = None
    else:
        cuda_device = 0
        gpu = None
        compute_capability = None
        total_memory_bytes = None

    metadata = {
        "gpu": gpu,
        "cuda_device": cuda_device,
        "compute_capability": compute_capability,
        "total_memory_bytes": total_memory_bytes,
        "driver": _cuda_driver_version(torch) if cuda_available else None,
        "mig_partition": None,
        "torch_backends": _torch_backend_flags(torch),
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "env": {
            name: os.environ[name]
            for name in sorted(["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS"])
            if name in os.environ
        },
    }
    if override:
        metadata.update(override)
        metadata.setdefault("torch_backends", _torch_backend_flags(torch))
    return metadata


def _cuda_driver_version(torch: Any) -> str | None:
    driver_version = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    if not callable(driver_version):
        return None
    try:
        value = driver_version()
    except Exception:
        return None
    return str(value) if value else None


def _torch_backend_flags(torch: Any) -> Dict[str, Any]:
    cuda_matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    cudnn = getattr(torch.backends, "cudnn", None)
    return {
        "cuda_matmul_allow_tf32": getattr(cuda_matmul, "allow_tf32", None),
        "cudnn_allow_tf32": getattr(cudnn, "allow_tf32", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
    }


def write_hardware_contract_json_artifact(output_path: str | os.PathLike[str], **kwargs: Any) -> Dict[str, Any]:
    artifact = collect_hardware_contract_results(**kwargs)
    _write_json(output_path, artifact)
    return artifact


def _validate_positive_ints(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"at least one {name} is required")
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be positive: {value}")


def _adapter_metadata(adapter: AdapterSpec) -> Dict[str, Any]:
    return {
        "name": adapter.name,
        "repository": adapter.repository,
        "revision": adapter.revision,
        "payload_file": adapter.payload_file,
        "payload_bytes": adapter.payload_bytes,
    }


def _resolve_model_context_limit(mode: str) -> int | None:
    if mode != "real":
        return MODEL_CONTEXT_LIMIT
    try:
        from transformers import AutoConfig
    except ImportError:
        return None
    try:
        config = AutoConfig.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    except Exception:
        return None
    for attribute in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _collect_synthetic_case(
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    time_forward: TimeForward,
    time_switch: TimeSwitch,
) -> List[Dict[str, Any]]:
    inputs = _synthetic_inputs(sequence_length, batch_size, vocab_size=16)
    reference_logits = None
    rows = []
    for baseline in BASELINES:
        model = _SyntheticAdapterModel(baseline, adapter)
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        if reference_logits is None:
            reference_logits = logits
        timing_inputs = {**inputs, "model": model}
        latencies = list(time_forward(baseline, adapter.name, timing_inputs, warmup_repetitions, measured_repetitions))
        switch_latencies = list(time_switch(baseline, adapter.name, warmup_repetitions, measured_repetitions))
        peft_provenance_events = (
            _synthetic_structured_low_rank_events(adapter, sequence_length, batch_size)
            if baseline == "beyond_matmul_factor_provenance"
            else None
        )
        rows.append(
            _result_row(
                baseline=baseline,
                adapter=adapter,
                sequence_length=sequence_length,
                batch_size=batch_size,
                latencies=latencies,
                switch_latencies=switch_latencies,
                logits=logits,
                reference_logits=reference_logits,
                peft_provenance_events=peft_provenance_events,
            )
        )
    return rows


def _collect_real_case(
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    upstream_peft_path: str,
    fork_peft_path: str,
) -> List[Dict[str, Any]]:
    reference = _run_real_worker(
        "upstream_peft_unmerged",
        adapter,
        sequence_length,
        batch_size,
        warmup_repetitions,
        measured_repetitions,
        upstream_peft_path,
    )
    rows = [_worker_payload_to_row(reference, reference.get("logits"))]
    for baseline, path in [
        ("upstream_peft_merged_dense_cache", upstream_peft_path),
        ("upstream_peft_repeated_merge_unmerge", upstream_peft_path),
        ("beyond_matmul_factor_provenance", fork_peft_path),
    ]:
        payload = _run_real_worker(
            baseline,
            adapter,
            sequence_length,
            batch_size,
            warmup_repetitions,
            measured_repetitions,
            path,
        )
        rows.append(_worker_payload_to_row(payload, reference.get("logits")))
    return rows


def _synthetic_inputs(sequence_length: int, batch_size: int, vocab_size: int) -> Dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED + sequence_length * 1000 + batch_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), generator=generator, dtype=torch.long)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _synthetic_structured_low_rank_events(
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "kind": "beyond_matmul_lora_provenance",
            "path": "structured_low_rank",
            "adapter": adapter.name,
            "module_path": "synthetic.lora_projection",
            "base_module": "Linear",
            "rank": 2,
            "in_features": 16,
            "out_features": 16,
            "input_shape": [batch_size, sequence_length, 16],
            "a_shape": [2, 16],
            "b_shape": [16, 2],
            "scaling": 1.0,
            "dtype": STRUCTURED_LOW_RANK_DTYPE,
            "device": STRUCTURED_LOW_RANK_DEVICE,
            "a_dtype": STRUCTURED_LOW_RANK_DTYPE,
            "a_device": STRUCTURED_LOW_RANK_DEVICE,
            "b_dtype": STRUCTURED_LOW_RANK_DTYPE,
            "b_device": STRUCTURED_LOW_RANK_DEVICE,
            "fan_in_fan_out": False,
            "lora_bias": False,
            "use_dora": False,
            "lora_variant": None,
            "dense_fallback_available": True,
            "dense_fallback_used": False,
            "fallback_reason": None,
            "schema_version": 1,
        }
    ]


def _result_row(
    *,
    baseline: str,
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    latencies: Sequence[float] | None,
    switch_latencies: Sequence[float] | None,
    logits: Any,
    reference_logits: Any,
    status: str = "ok",
    reason: str | None = None,
    storage: Dict[str, Any] | None = None,
    peft_provenance_events: Sequence[Dict[str, Any]] | None = None,
    peak_memory_bytes: int | None = None,
    peak_memory_status: str = "not_measured_synthetic_smoke",
) -> Dict[str, Any]:
    correctness = _correctness_metrics(logits, reference_logits)
    if status == "ok" and not correctness["passed"]:
        status = "failed_correctness"
        reason = reason or "correctness tolerance failed"
    structured_execution_allowed = status == "ok" and correctness["passed"]
    row = {
        "case": _case_name(adapter.name, sequence_length, batch_size),
        "adapter": adapter.name,
        "baseline": baseline,
        "status": status,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "latency_seconds": _latency_stats(latencies) if latencies is not None else None,
        "adapter_switch_seconds": (
            _latency_stats(switch_latencies) if switch_latencies is not None else None
        ),
        "adapter_switch_status": _adapter_switch_status(baseline, switch_latencies=switch_latencies, status=status),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_status": peak_memory_status,
        "storage": _storage_metadata(baseline, adapter, storage),
        "correctness": correctness,
        "lowering": _lowering_metadata(
            baseline,
            adapter.name,
            peft_provenance_events=peft_provenance_events,
            structured_execution_allowed=structured_execution_allowed,
        ),
    }
    if peft_provenance_events is not None:
        row["peft_provenance_events"] = list(peft_provenance_events)
    if reason:
        row["reason"] = reason
    return row


def _blocked_context_rows(
    *,
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    model_context_limit: int,
) -> List[Dict[str, Any]]:
    reason = f"context limit exceeded: sequence_length {sequence_length} > model context limit {model_context_limit}"
    return [
        _blocked_result_row(
            baseline=baseline,
            adapter=adapter,
            sequence_length=sequence_length,
            batch_size=batch_size,
            reason=reason,
        )
        for baseline in BASELINES
    ]


def _blocked_result_row(
    *,
    baseline: str,
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    reason: str,
) -> Dict[str, Any]:
    return {
        "case": _case_name(adapter.name, sequence_length, batch_size),
        "adapter": adapter.name,
        "baseline": baseline,
        "status": "blocked",
        "reason": reason,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "latency_seconds": None,
        "adapter_switch_seconds": None,
        "adapter_switch_status": "not_measured_blocked",
        "peak_memory_bytes": None,
        "peak_memory_status": "not_measured_blocked",
        "storage": _storage_metadata(baseline, adapter, None),
        "correctness": _empty_correctness(passed=False),
        "lowering": _lowering_metadata(baseline, adapter.name),
    }


def _case_name(adapter_name: str, sequence_length: int, batch_size: int) -> str:
    return f"{adapter_name}_seq{sequence_length}_batch{batch_size}"


def _latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        raise ValueError("at least one latency sample is required")
    values = sorted(float(value) for value in latencies)
    for value in values:
        if value < 0.0:
            raise ValueError(f"latency must be non-negative: {value}")
    return {
        "median": _stable_float(statistics.median(values)),
        "mean": _stable_float(statistics.fmean(values)),
        "stdev": _stable_float(statistics.pstdev(values)),
        "p50": _stable_float(_percentile(values, 50)),
        "p90": _stable_float(_percentile(values, 90)),
        "p95": _stable_float(_percentile(values, 95)),
        "p99": _stable_float(_percentile(values, 99)),
        "min": _stable_float(values[0]),
        "max": _stable_float(values[-1]),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def _correctness_metrics(logits: Any, reference_logits: Any) -> Dict[str, Any]:
    if logits is None or reference_logits is None:
        return _empty_correctness(passed=False)
    torch = _torch()
    candidate = torch.as_tensor(logits, dtype=torch.float32)
    reference = torch.as_tensor(reference_logits, dtype=torch.float32)
    finite = bool(torch.isfinite(candidate).all().item())
    diff = candidate - reference
    max_abs_error = float(torch.max(torch.abs(diff)).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    relative_l2_error = 0.0 if reference_norm == 0.0 else diff_norm / reference_norm
    passed = (
        finite
        and max_abs_error <= CORRECTNESS_MAX_ABS_TOLERANCE
        and relative_l2_error <= CORRECTNESS_RELATIVE_L2_TOLERANCE
    )
    return {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "tolerance_profile": "cpu_fp32",
        "passed": passed,
    }


def _empty_correctness(*, passed: bool) -> Dict[str, Any]:
    return {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": None,
        "relative_l2_error": None,
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "tolerance_profile": "cpu_fp32",
        "passed": passed,
    }


def _artifact_summary(
    results: Sequence[Dict[str, Any]],
    *,
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    mode: str,
    context_limit: int | None,
) -> Dict[str, Any]:
    all_required_cases_present = _all_requested_cases_present(results, adapters, sequence_lengths, batch_sizes)
    all_correctness_checks_passed = all(
        row["status"] in {"ok", "not_applicable"} and row["correctness"]["passed"] for row in results
    )
    all_switching_cases_present = _all_switching_cases_present(results)
    all_dense_fallback_cases_explicit = _all_dense_fallback_cases_explicit(results)
    all_peak_memory_cases_measured = _all_peak_memory_cases_measured(results)
    all_adapter_switch_cases_measured = _all_adapter_switch_cases_measured(results)
    readiness_blockers = _readiness_blockers(
        mode=mode,
        context_limit=context_limit,
        results=results,
        all_required_cases_present=all_required_cases_present,
        all_correctness_checks_passed=all_correctness_checks_passed,
        all_switching_cases_present=all_switching_cases_present,
        all_dense_fallback_cases_explicit=all_dense_fallback_cases_explicit,
    )
    memory_control_readiness_blockers = _memory_control_readiness_blockers(
        mode=mode,
        context_limit=context_limit,
        results=results,
        all_required_cases_present=all_required_cases_present,
        all_correctness_checks_passed=all_correctness_checks_passed,
        all_peak_memory_cases_measured=all_peak_memory_cases_measured,
        all_adapter_switch_cases_measured=all_adapter_switch_cases_measured,
    )
    return {
        "all_required_cases_present": all_required_cases_present,
        "all_correctness_checks_passed": all_correctness_checks_passed,
        "all_switching_cases_present": all_switching_cases_present,
        "all_dense_fallback_cases_explicit": all_dense_fallback_cases_explicit,
        "all_peak_memory_cases_measured": all_peak_memory_cases_measured,
        "all_adapter_switch_cases_measured": all_adapter_switch_cases_measured,
        "memory_control_claim_ready": not memory_control_readiness_blockers,
        "memory_control_readiness_blockers": memory_control_readiness_blockers,
        "benchmark_ready": not readiness_blockers,
        "readiness_blockers": readiness_blockers,
        "max_abs_error": _max_correctness_metric(results, "max_abs_error"),
        "max_relative_l2_error": _max_correctness_metric(results, "relative_l2_error"),
        "fallback_cases": _fallback_cases(results),
        "structured_low_rank_cases": _structured_low_rank_cases(results),
        "negative_cases": _negative_cases(results),
        "memory_or_control_claim": "none",
        "performance_claim": "none",
    }


def _readiness_blockers(
    *,
    mode: str,
    context_limit: int | None,
    results: Sequence[Dict[str, Any]],
    all_required_cases_present: bool,
    all_correctness_checks_passed: bool,
    all_switching_cases_present: bool,
    all_dense_fallback_cases_explicit: bool,
) -> List[str]:
    blockers = []
    if mode != "real":
        blockers.append("synthetic_smoke_not_benchmark_evidence")
    if context_limit is None:
        blockers.append("model_context_limit_unresolved")
    if any(row["status"] == "blocked" and "context limit" in row.get("reason", "") for row in results):
        blockers.append("context_limit_exceeded")
    if not all_required_cases_present:
        blockers.append("required_cases_missing")
    if not all_correctness_checks_passed:
        blockers.append("correctness_checks_failed")
    if not all_switching_cases_present:
        blockers.append("switching_cases_missing")
    if not all_dense_fallback_cases_explicit:
        blockers.append("dense_fallback_cases_not_explicit")
    return blockers


def _memory_control_readiness_blockers(
    *,
    mode: str,
    context_limit: int | None,
    results: Sequence[Dict[str, Any]],
    all_required_cases_present: bool,
    all_correctness_checks_passed: bool,
    all_peak_memory_cases_measured: bool,
    all_adapter_switch_cases_measured: bool,
) -> List[str]:
    blockers = []
    if mode != "real":
        blockers.append("synthetic_smoke_not_benchmark_evidence")
    if context_limit is None:
        blockers.append("model_context_limit_unresolved")
    if any(row["status"] == "blocked" and "context limit" in row.get("reason", "") for row in results):
        blockers.append("context_limit_exceeded")
    if not all_required_cases_present:
        blockers.append("required_cases_missing")
    if not all_correctness_checks_passed:
        blockers.append("correctness_checks_failed")
    if not all_peak_memory_cases_measured:
        blockers.append("peak_memory_cases_unavailable")
    if not all_adapter_switch_cases_measured:
        blockers.append("adapter_switch_cases_unavailable")
    return blockers


def _max_correctness_metric(results: Sequence[Dict[str, Any]], metric: str) -> float | None:
    values = [
        float(row["correctness"][metric])
        for row in results
        if row["correctness"].get(metric) is not None
    ]
    if not values:
        return None
    return max(values)


def _fallback_cases(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for row in results:
        if row["baseline"] != "beyond_matmul_factor_provenance":
            continue
        lowering = row["lowering"]
        fallback_reasons = list(lowering.get("fallback_reasons", []))
        if not (lowering.get("dense_fallback_used") or fallback_reasons):
            continue
        cases.append(
            {
                "case": row["case"],
                "adapter": row["adapter"],
                "baseline": row["baseline"],
                "status": row["status"],
                "kind": lowering["kind"],
                "fallback_reasons": fallback_reasons,
                "correctness_passed": row["correctness"]["passed"],
            }
        )
    return cases


def _structured_low_rank_cases(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for row in results:
        if row["baseline"] != "beyond_matmul_factor_provenance":
            continue
        lowering = row["lowering"]
        if lowering.get("execution_path") != "structured_low_rank":
            continue
        cases.append(
            {
                "case": row["case"],
                "adapter": row["adapter"],
                "baseline": row["baseline"],
                "status": row["status"],
                "kind": lowering["kind"],
                "execution_path": lowering["execution_path"],
                "correctness_passed": row["correctness"]["passed"],
            }
        )
    return cases


def _negative_cases(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    for row in results:
        if row["status"] == "ok" and row["correctness"]["passed"]:
            continue
        cases.append(
            {
                "case": row["case"],
                "adapter": row["adapter"],
                "baseline": row["baseline"],
                "status": row["status"],
                "reason": row.get("reason") or row["status"],
                "correctness_passed": row["correctness"]["passed"],
            }
        )
    return cases


def _all_requested_cases_present(
    results: Sequence[Dict[str, Any]],
    adapters: Sequence[AdapterSpec],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
) -> bool:
    present = {(row["adapter"], row["case"], row["baseline"]) for row in results}
    required = {
        (adapter.name, _case_name(adapter.name, sequence_length, batch_size), baseline)
        for adapter in adapters
        for sequence_length in sequence_lengths
        for batch_size in batch_sizes
        for baseline in BASELINES
    }
    return required.issubset(present)


def _all_switching_cases_present(results: Sequence[Dict[str, Any]]) -> bool:
    for row in results:
        if row["status"] == "ok" and row["adapter_switch_seconds"] is None:
            return False
        if row["adapter_switch_status"] is None:
            return False
    return True


def _all_peak_memory_cases_measured(results: Sequence[Dict[str, Any]]) -> bool:
    for row in results:
        if row["status"] in {"blocked", "not_applicable", "failed"}:
            continue
        if row.get("peak_memory_status") != "measured_process_maxrss":
            return False
        if not isinstance(row.get("peak_memory_bytes"), int) or row["peak_memory_bytes"] < 0:
            return False
    return True


def _all_adapter_switch_cases_measured(results: Sequence[Dict[str, Any]]) -> bool:
    for row in results:
        if row["status"] in {"blocked", "not_applicable", "failed"}:
            continue
        if row.get("adapter_switch_seconds") is None:
            return False
        if not str(row.get("adapter_switch_status", "")).startswith("measured_"):
            return False
    return True


def _all_dense_fallback_cases_explicit(results: Sequence[Dict[str, Any]]) -> bool:
    for row in results:
        if row["baseline"] != "beyond_matmul_factor_provenance":
            continue
        lowering = row["lowering"]
        if "dense_fallback_available" not in lowering or "dense_fallback_used" not in lowering:
            return False
        if lowering.get("dense_fallback_used") and not lowering.get("fallback_reasons"):
            return False
    return True


def _adapter_switch_status(
    baseline: str,
    *,
    switch_latencies: Sequence[float] | None,
    status: str,
) -> str:
    if status == "blocked":
        return "not_measured_blocked"
    if switch_latencies is None:
        return "not_measured_not_applicable"
    if baseline == "upstream_peft_merged_dense_cache":
        return "measured_dense_cache_pointer_swap"
    return "measured_loaded_adapters"


def _storage_metadata(baseline: str, adapter: AdapterSpec, storage: Dict[str, Any] | None) -> Dict[str, Any]:
    config_bytes = None if storage is None else storage.get("adapter_config_bytes")
    resident_bytes = None if storage is None else storage.get("resident_adapter_bytes")
    if resident_bytes is None:
        if baseline == "upstream_peft_merged_dense_cache":
            resident_bytes = BASE_MODEL_FP32_BYTES_APPROX
        elif baseline in {"upstream_peft_unmerged", "beyond_matmul_factor_provenance"}:
            resident_bytes = adapter.payload_bytes
    return {
        "adapter_payload_bytes": adapter.payload_bytes,
        "adapter_config_bytes": config_bytes,
        "dense_cache_bytes_per_adapter": BASE_MODEL_FP32_BYTES_APPROX
        if baseline == "upstream_peft_merged_dense_cache"
        else 0,
        "resident_adapter_bytes": resident_bytes,
    }


def _lowering_metadata(
    baseline: str,
    adapter_name: str,
    *,
    peft_provenance_events: Sequence[Dict[str, Any]] | None = None,
    structured_execution_allowed: bool = True,
) -> Dict[str, Any]:
    if baseline == "upstream_peft_unmerged":
        return {
            "kind": "peft_unmerged_adapter",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": False,
        }
    if baseline == "upstream_peft_merged_dense_cache":
        return {
            "kind": "peft_merged_dense_cache",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": True,
        }
    if baseline == "upstream_peft_repeated_merge_unmerge":
        return {
            "kind": "peft_repeated_merge_unmerge",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": True,
        }
    if peft_provenance_events is not None:
        return _provenance_lowering_metadata_from_events(
            adapter_name,
            peft_provenance_events,
            structured_execution_allowed=structured_execution_allowed,
        )
    return {
        "kind": "peft_dense_fallback",
        "active_adapter": adapter_name,
        "dense_fallback_available": True,
        "dense_fallback_used": True,
        "execution_path": "dense_fallback",
        "fallback_reasons": ["no_fork_provenance_events"],
    }


def _provenance_lowering_metadata_from_events(
    adapter_name: str,
    peft_provenance_events: Sequence[Dict[str, Any]],
    *,
    structured_execution_allowed: bool,
) -> Dict[str, Any]:
    events = list(peft_provenance_events)
    structured_events = [
        event for event in events if event.get("path") == "structured_low_rank"
    ]
    fallback_reasons = _event_fallback_reasons(events)
    for event in structured_events:
        fallback_reasons.extend(_structured_low_rank_contract_rejections(event))
    if not structured_execution_allowed:
        fallback_reasons.append("correctness_failed")
    if not events:
        fallback_reasons.append("no_fork_provenance_events")
    if structured_events and len(structured_events) == len(events) and not fallback_reasons:
        lowering = {
            "kind": "provenance_lora_factors",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": False,
            "execution_path": "structured_low_rank",
        }
    else:
        lowering = {
            "kind": "peft_dense_fallback",
            "active_adapter": adapter_name,
            "dense_fallback_available": True,
            "dense_fallback_used": True,
            "execution_path": "dense_fallback",
        }
        if not fallback_reasons:
            fallback_reasons = ["non_structured_provenance_event"]
    if fallback_reasons:
        lowering["fallback_reasons"] = sorted(set(fallback_reasons))
    return lowering


def _event_fallback_reasons(events: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        event["fallback_reason"]
        for event in events
        if event.get("fallback_reason") is not None
    ]


def _structured_low_rank_contract_rejections(event: Dict[str, Any]) -> List[str]:
    reasons = []
    dtype_fields = ("dtype", "a_dtype", "b_dtype")
    device_fields = ("device", "a_device", "b_device")
    if any(event.get(field) != STRUCTURED_LOW_RANK_DTYPE for field in dtype_fields):
        reasons.append("non_fp32_dtype")
    if any(event.get(field) != STRUCTURED_LOW_RANK_DEVICE for field in device_fields):
        reasons.append("non_cpu_device")
    if event.get("base_module") != "Linear":
        reasons.append("unsupported_base_module")
    if event.get("fan_in_fan_out"):
        reasons.append("unsupported_layout")
    if event.get("lora_bias"):
        reasons.append("lora_bias")
    if event.get("use_dora"):
        reasons.append("use_dora")
    if event.get("lora_variant") is not None:
        reasons.append("unsupported_lora_variant")
    return reasons


def _dependency_metadata(
    *,
    mode: str,
    upstream_peft_path: str | os.PathLike[str] | None,
    fork_peft_path: str | os.PathLike[str] | None,
    upstream_peft_ref: str,
    fork_peft_ref: str,
) -> Dict[str, Any]:
    torch = _torch()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _module_metadata("transformers", mode),
        "peft_upstream": _peft_metadata(
            repository=UPSTREAM_REPOSITORY,
            path=upstream_peft_path,
            requested_ref=upstream_peft_ref,
            mode=mode,
        ),
        "peft_fork": _peft_metadata(
            repository=FORK_REPOSITORY,
            path=fork_peft_path,
            requested_ref=fork_peft_ref,
            mode=mode,
        ),
        "huggingface_hub": _module_metadata("huggingface_hub", mode),
        "beyond_matmul": {
            "repository": "alexlopashev/beyond-matmul",
            "revision": _git_revision(Path(ROOT)),
        },
    }


def _module_metadata(module_name: str, mode: str) -> Dict[str, Any]:
    try:
        module = __import__(module_name)
    except ImportError:
        return {"version": None, "revision": None, "status": f"not_imported_{mode}"}
    return {"version": getattr(module, "__version__", None), "revision": None}


def _peft_metadata(
    *,
    repository: str,
    path: str | os.PathLike[str] | None,
    requested_ref: str,
    mode: str,
) -> Dict[str, Any]:
    metadata = {
        "repository": repository,
        "version": None,
        "revision": _git_revision(Path(path)) if path else None,
        "requested_ref": requested_ref,
        "path": str(path) if path else None,
    }
    if metadata["revision"] is None:
        metadata["revision"] = f"not_resolved_{mode}"
    return metadata


def _environment_metadata() -> Dict[str, Any]:
    torch = _torch()
    return {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine() or None,
        "accelerator": None,
        "torch_num_threads": torch.get_num_threads(),
        "env": {
            name: os.environ[name]
            for name in sorted(["OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"])
            if name in os.environ
        },
    }


def _run_metadata(
    *, command: Sequence[str] | None, generated_at_utc: str | None, mode: str
) -> Dict[str, Any]:
    command_list = list(command) if command is not None else None
    return {
        "mode": mode,
        "command": command_list,
        "command_text": shlex.join(command_list) if command_list is not None else None,
        "generated_at_utc": generated_at_utc,
    }


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _resolve_checkout(
    *,
    repository: str,
    ref: str,
    path: str | None,
    checkout_dir: str | os.PathLike[str] | None,
) -> str:
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(f"PEFT checkout path does not exist: {path}")
        return path
    base_dir = Path(checkout_dir) if checkout_dir else Path.cwd() / ".peft_benchmark_checkouts"
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / f"{repository.replace('/', '-')}-{_safe_ref(ref)}"
    repo_url = f"https://github.com/{repository}.git"
    if not target.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", repo_url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "FETCH_HEAD"], check=True)
    return str(target)


def _peft_import_paths(path: str | os.PathLike[str]) -> List[str]:
    checkout = Path(path)
    if not checkout.exists():
        raise FileNotFoundError(f"PEFT checkout path does not exist: {checkout}")
    src = checkout / "src"
    paths: List[Path] = []
    if (src / "peft").is_dir():
        paths.extend([src, checkout])
    elif (checkout / "peft").is_dir():
        paths.append(checkout)
    else:
        raise ImportError(f"PEFT checkout path is not importable; expected peft/ or src/peft under {checkout}")
    return [str(path) for path in paths]


def _safe_ref(ref: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in ref)


def _run_real_worker(
    baseline: str,
    adapter: AdapterSpec,
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    peft_path: str,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json", delete=False) as output:
        output_path = output.name
    command = [
        sys.executable,
        __file__,
        "--_worker-json-output",
        output_path,
        "--_worker-baseline",
        baseline,
        "--_worker-adapter-name",
        adapter.name,
        "--_worker-sequence-length",
        str(sequence_length),
        "--_worker-batch-size",
        str(batch_size),
        "--_worker-warmup",
        str(warmup_repetitions),
        "--_worker-repetitions",
        str(measured_repetitions),
        "--_worker-peft-path",
        peft_path,
    ]
    try:
        subprocess.run(command, check=True)
        with open(output_path, encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _run_hardware_contract_worker(
    *,
    adapter: AdapterSpec,
    baseline: str,
    sequence_length: int,
    batch_size: int,
    forward_warmup_repetitions: int,
    forward_measured_repetitions: int,
    switch_warmup_repetitions: int,
    switch_measured_repetitions: int,
    mode: str,
    upstream_peft_path: str | None,
) -> Dict[str, Any]:
    if mode != "real":
        raise RuntimeError(f"default hardware worker only supports real mode, got {mode}")
    if upstream_peft_path is None:
        raise RuntimeError("real hardware contract measurement requires an upstream PEFT checkout")
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".json", delete=False) as output:
        output_path = output.name
    command = [
        sys.executable,
        __file__,
        "--_hardware-worker-json-output",
        output_path,
        "--_hardware-worker-baseline",
        baseline,
        "--_hardware-worker-adapter-name",
        adapter.name,
        "--_hardware-worker-sequence-length",
        str(sequence_length),
        "--_hardware-worker-batch-size",
        str(batch_size),
        "--_hardware-worker-forward-warmup",
        str(forward_warmup_repetitions),
        "--_hardware-worker-forward-repetitions",
        str(forward_measured_repetitions),
        "--_hardware-worker-switch-warmup",
        str(switch_warmup_repetitions),
        "--_hardware-worker-switch-repetitions",
        str(switch_measured_repetitions),
        "--_hardware-worker-peft-path",
        upstream_peft_path,
    ]
    try:
        subprocess.run(command, check=True)
        with open(output_path, encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


class _CudaMeasurementBackend:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.cuda = torch_module.cuda

    def synchronize(self) -> None:
        self.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        self.cuda.reset_peak_memory_stats()

    def allocator_values(self) -> Dict[str, int]:
        return {
            "allocated_bytes": int(self.cuda.memory_allocated()),
            "reserved_bytes": int(self.cuda.memory_reserved()),
            "peak_allocated_bytes": int(self.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(self.cuda.max_memory_reserved()),
        }

    def time_cuda_region(self, callback: Callable[[], None]) -> Dict[str, float]:
        start_event = self.cuda.Event(enable_timing=True)
        end_event = self.cuda.Event(enable_timing=True)
        self.synchronize()
        start_event.record()
        wall_start = time.perf_counter()
        callback()
        end_event.record()
        self.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        return {
            "cuda_seconds": float(start_event.elapsed_time(end_event)) / 1000.0,
            "wall_seconds": wall_seconds,
        }


def _hardware_real_worker(args: argparse.Namespace) -> None:
    adapter = _adapter_by_name(args._hardware_worker_adapter_name)
    payload: Dict[str, Any]
    try:
        _prepend_peft_import_paths(args._hardware_worker_peft_path)
        torch = _torch()
        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in hardware worker")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        backend = _CudaMeasurementBackend(torch)
        backend.synchronize()
        backend.reset_peak_memory_stats()
        model, preprocessing_seconds = _load_hardware_worker_model(
            args,
            AutoModelForCausalLM,
            PeftModel,
            adapter,
        )
        tokenization_start = time.perf_counter()
        inputs = _worker_inputs(args._hardware_worker_sequence_length, args._hardware_worker_batch_size, model.config.vocab_size)
        inputs = {name: value.to("cuda") for name, value in inputs.items()}
        preprocessing_seconds["tokenization"] = time.perf_counter() - tokenization_start
        backend.synchronize()
        setup_memory = backend.allocator_values()
        backend.reset_peak_memory_stats()
        switch_times = _measure_hardware_worker_switch(args, model, adapter, backend)
        forward_times = _measure_hardware_worker_forward(args, model, inputs, backend)
        backend.synchronize()
        steady_memory = backend.allocator_values()
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        payload = {
            "status": "ok",
            "reason": None,
            "isolated_process": True,
            "process_id": os.getpid(),
            "forward_latencies_seconds": forward_times["cuda_seconds"],
            "forward_wall_seconds": forward_times["wall_seconds"],
            "switch_latencies_seconds": switch_times["cuda_seconds"],
            "switch_wall_seconds": switch_times["wall_seconds"],
            "preprocessing_seconds": preprocessing_seconds,
            "cuda_memory": {
                "setup_peak_allocated_bytes": setup_memory["peak_allocated_bytes"],
                "setup_peak_reserved_bytes": setup_memory["peak_reserved_bytes"],
                "steady_peak_allocated_bytes": steady_memory["peak_allocated_bytes"],
                "steady_peak_reserved_bytes": steady_memory["peak_reserved_bytes"],
                "post_setup_allocated_bytes": setup_memory["allocated_bytes"],
                "post_setup_reserved_bytes": setup_memory["reserved_bytes"],
                "post_loop_allocated_bytes": steady_memory["allocated_bytes"],
                "post_loop_reserved_bytes": steady_memory["reserved_bytes"],
            },
            "logits": logits.detach().cpu().tolist(),
            "storage": _worker_storage(model, adapter, args._hardware_worker_baseline),
            "timing_protocol": {
                "cuda_events": True,
                "synchronized_before_after": True,
                "allocator_reset": True,
                "setup_excluded_from_steady_state": True,
            },
        }
    except Exception as exc:  # pragma: no cover - depends on CUDA and optional dependencies.
        payload = {
            "status": "failed",
            "reason": str(exc),
            "isolated_process": True,
            "process_id": os.getpid(),
            "forward_latencies_seconds": None,
            "forward_wall_seconds": None,
            "switch_latencies_seconds": None,
            "switch_wall_seconds": None,
            "preprocessing_seconds": {},
            "cuda_memory": {},
            "correctness": {"passed": False},
            "storage": _worker_storage(None, adapter, args._hardware_worker_baseline),
            "timing_protocol": {
                "cuda_events": False,
                "synchronized_before_after": False,
                "allocator_reset": False,
                "setup_excluded_from_steady_state": False,
            },
        }
    _write_json(args._hardware_worker_json_output, payload)


def _load_hardware_worker_model(
    args: argparse.Namespace,
    model_class: Any,
    peft_model_class: Any,
    adapter: AdapterSpec,
) -> tuple[Any, Dict[str, float | None]]:
    preprocessing_seconds: Dict[str, float | None] = {
        "model_load": None,
        "hub_download": None,
        "tokenization": None,
        "dense_cache_build": None,
        "structured_factor_pack": None,
        "compilation_or_graph_capture": None,
    }
    model_start = time.perf_counter()
    if args._hardware_worker_baseline == "upstream_peft_merged_dense_cache":
        cache_start = time.perf_counter()
        dense_cache = {}
        for cached_adapter in ADAPTERS:
            base_model = _load_base_worker_model(model_class)
            model = _load_primary_worker_adapter(
                peft_model_class,
                base_model,
                cached_adapter,
                args._hardware_worker_baseline,
            )
            if hasattr(model, "set_adapter"):
                model.set_adapter(cached_adapter.name)
            merged = model.merge_and_unload()
            dense_cache[cached_adapter.name] = merged.to("cuda").eval()
        preprocessing_seconds["dense_cache_build"] = time.perf_counter() - cache_start
        preprocessing_seconds["model_load"] = time.perf_counter() - model_start
        return dense_cache[adapter.name], preprocessing_seconds
    base_model = _load_base_worker_model(model_class)
    model = _load_worker_model(peft_model_class, base_model, adapter, args._hardware_worker_baseline)
    model = model.to("cuda").eval()
    preprocessing_seconds["model_load"] = time.perf_counter() - model_start
    return model, preprocessing_seconds


def _measure_hardware_worker_forward(
    args: argparse.Namespace,
    model: Any,
    inputs: Dict[str, Any],
    backend: _CudaMeasurementBackend,
) -> Dict[str, List[float]]:
    with backend.torch.inference_mode():
        for _ in range(args._hardware_worker_forward_warmup):
            model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        cuda_seconds = []
        wall_seconds = []
        for _ in range(args._hardware_worker_forward_repetitions):
            timing = backend.time_cuda_region(
                lambda: model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            )
            cuda_seconds.append(timing["cuda_seconds"])
            wall_seconds.append(timing["wall_seconds"])
    return {"cuda_seconds": cuda_seconds, "wall_seconds": wall_seconds}


def _measure_hardware_worker_switch(
    args: argparse.Namespace,
    model: Any,
    adapter: AdapterSpec,
    backend: _CudaMeasurementBackend,
) -> Dict[str, List[float]]:
    baseline = args._hardware_worker_baseline
    other_adapter = _other_adapter(adapter)
    cuda_seconds = []
    wall_seconds = []
    if baseline == "upstream_peft_merged_dense_cache":
        for _ in range(args._hardware_worker_switch_warmup):
            _ = model
        for _ in range(args._hardware_worker_switch_repetitions):
            timing = backend.time_cuda_region(lambda: None)
            cuda_seconds.append(timing["cuda_seconds"])
            wall_seconds.append(timing["wall_seconds"])
        return {"cuda_seconds": cuda_seconds, "wall_seconds": wall_seconds}
    if not hasattr(model, "set_adapter"):
        return {"cuda_seconds": cuda_seconds, "wall_seconds": wall_seconds}
    for _ in range(args._hardware_worker_switch_warmup):
        model.set_adapter(other_adapter.name)
        model.set_adapter(adapter.name)
    for _ in range(args._hardware_worker_switch_repetitions):
        def switch() -> None:
            model.set_adapter(other_adapter.name)
            model.set_adapter(adapter.name)
            if baseline == "upstream_peft_repeated_merge_unmerge":
                model.merge_adapter()
                model.unmerge_adapter()

        timing = backend.time_cuda_region(switch)
        cuda_seconds.append(timing["cuda_seconds"])
        wall_seconds.append(timing["wall_seconds"])
    return {"cuda_seconds": cuda_seconds, "wall_seconds": wall_seconds}


def _worker_payload_to_row(payload: Dict[str, Any], reference_logits: Any) -> Dict[str, Any]:
    adapter = _adapter_by_name(payload["adapter"])
    if payload["status"] != "ok":
        return _non_ok_result_row(payload, adapter)
    peak_memory = _peak_memory_from_payload(payload, default_status="unavailable_worker_payload")
    return _result_row(
        baseline=payload["baseline"],
        adapter=adapter,
        sequence_length=payload["sequence_length"],
        batch_size=payload["batch_size"],
        latencies=payload["latencies"],
        switch_latencies=payload["switch_latencies"],
        logits=payload["logits"],
        reference_logits=reference_logits,
        status=payload["status"],
        reason=payload.get("reason"),
        storage=payload.get("storage"),
        peft_provenance_events=payload.get("peft_provenance_events"),
        peak_memory_bytes=peak_memory["peak_memory_bytes"],
        peak_memory_status=peak_memory["peak_memory_status"],
    )


def _non_ok_result_row(payload: Dict[str, Any], adapter: AdapterSpec) -> Dict[str, Any]:
    passed = payload["status"] == "not_applicable"
    status = payload["status"]
    peak_memory = _peak_memory_from_payload(payload, default_status=_default_peak_memory_status(status))
    switch_latencies = payload.get("switch_latencies")
    row = {
        "case": _case_name(adapter.name, payload["sequence_length"], payload["batch_size"]),
        "adapter": adapter.name,
        "baseline": payload["baseline"],
        "status": status,
        "sequence_length": payload["sequence_length"],
        "batch_size": payload["batch_size"],
        "latency_seconds": _latency_stats(payload["latencies"]) if payload.get("latencies") is not None else None,
        "adapter_switch_seconds": _latency_stats(switch_latencies) if switch_latencies is not None else None,
        "adapter_switch_status": _adapter_switch_status(
            payload["baseline"],
            switch_latencies=switch_latencies,
            status=status,
        ),
        "peak_memory_bytes": peak_memory["peak_memory_bytes"],
        "peak_memory_status": peak_memory["peak_memory_status"],
        "storage": _storage_metadata(payload["baseline"], adapter, payload.get("storage")),
        "correctness": _empty_correctness(passed=passed),
        "lowering": _lowering_metadata(
            payload["baseline"],
            adapter.name,
            peft_provenance_events=payload.get("peft_provenance_events"),
            structured_execution_allowed=False,
        ),
        "reason": payload.get("reason") or payload["status"],
    }
    if "peft_provenance_events" in payload:
        row["peft_provenance_events"] = payload["peft_provenance_events"]
    return row


def _peak_memory_from_payload(payload: Dict[str, Any], *, default_status: str) -> Dict[str, Any]:
    peak_memory_bytes = payload.get("peak_memory_bytes")
    peak_memory_status = payload.get("peak_memory_status") or default_status
    if peak_memory_status == "measured_process_maxrss" and (
        not isinstance(peak_memory_bytes, int) or peak_memory_bytes < 0
    ):
        return {"peak_memory_bytes": None, "peak_memory_status": "unavailable_invalid_measurement"}
    if peak_memory_status != "measured_process_maxrss":
        peak_memory_bytes = None
    return {"peak_memory_bytes": peak_memory_bytes, "peak_memory_status": peak_memory_status}


def _default_peak_memory_status(status: str) -> str:
    if status == "blocked":
        return "not_measured_blocked"
    if status == "not_applicable":
        return "not_measured_not_applicable"
    if status == "failed":
        return "not_measured_failed"
    return "unavailable_worker_payload"


def _process_peak_memory_sample(
    resource_module: Any | None = None,
    *,
    platform_name: str | None = None,
) -> Dict[str, Any]:
    if resource_module is None:
        try:
            import resource as resource_module
        except ImportError:
            return {"peak_memory_bytes": None, "peak_memory_status": "unavailable_platform_api"}
    try:
        usage = resource_module.getrusage(resource_module.RUSAGE_SELF)
        raw_peak = getattr(usage, "ru_maxrss", None)
    except (AttributeError, OSError, ValueError):
        return {"peak_memory_bytes": None, "peak_memory_status": "unavailable_platform_api"}
    if not isinstance(raw_peak, int) or raw_peak <= 0:
        return {"peak_memory_bytes": None, "peak_memory_status": "unavailable_platform_api"}
    resolved_platform = platform_name or sys.platform
    peak_memory_bytes = raw_peak if resolved_platform == "darwin" else raw_peak * 1024
    return {"peak_memory_bytes": int(peak_memory_bytes), "peak_memory_status": "measured_process_maxrss"}


def _adapter_by_name(name: str) -> AdapterSpec:
    for adapter in ADAPTERS:
        if adapter.name == name:
            return adapter
    raise KeyError(f"unknown adapter: {name}")


def _prepend_peft_import_paths(path: str | os.PathLike[str]) -> None:
    for import_path in reversed(_peft_import_paths(path)):
        if import_path in sys.path:
            sys.path.remove(import_path)
        sys.path.insert(0, import_path)


def _real_worker(args: argparse.Namespace) -> None:
    adapter = _adapter_by_name(args._worker_adapter_name)
    try:
        _prepend_peft_import_paths(args._worker_peft_path)
        torch = _torch()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if args._worker_baseline == "upstream_peft_merged_dense_cache":
            payload = _run_dense_cache_worker(args, AutoModelForCausalLM, PeftModel, adapter)
            _write_json(args._worker_json_output, payload)
            return

        base_model = _load_base_worker_model(AutoModelForCausalLM)
        model = _load_worker_model(PeftModel, base_model, adapter, args._worker_baseline)
        model.eval()
        inputs = _worker_inputs(args._worker_sequence_length, args._worker_batch_size, model.config.vocab_size)
        if args._worker_baseline == "upstream_peft_repeated_merge_unmerge":
            payload = _run_repeated_merge_worker(args, model, inputs, adapter)
            _write_json(args._worker_json_output, payload)
            return

        switch_latencies = _measure_worker_switch(args, model, adapter, args._worker_baseline)
        latencies = _time_forward(
            args._worker_baseline,
            adapter.name,
            {"model": lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, **inputs},
            args._worker_warmup,
            args._worker_repetitions,
        )
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        payload = {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "ok",
            "reason": None,
            "latencies": latencies,
            "switch_latencies": switch_latencies,
            "logits": logits.detach().cpu().tolist(),
            "peft_provenance_events": _collect_peft_provenance_events(model),
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }
    except Exception as exc:  # pragma: no cover - depends on optional external dependencies.
        payload = {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "failed",
            "reason": str(exc),
            "latencies": None,
            "switch_latencies": None,
            "logits": None,
            "storage": _worker_storage(None, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }
    _write_json(args._worker_json_output, payload)


def _load_worker_model(peft_model_class: Any, base_model: Any, adapter: AdapterSpec, baseline: str) -> Any:
    model = _load_primary_worker_adapter(peft_model_class, base_model, adapter, baseline)
    for other_adapter in ADAPTERS:
        if other_adapter.name == adapter.name:
            continue
        _load_additional_worker_adapter(model, other_adapter, baseline)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter.name)
    return model


def _load_base_worker_model(model_class: Any) -> Any:
    torch = _torch()
    return model_class.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        dtype=torch.float32,
    )


def _load_primary_worker_adapter(peft_model_class: Any, base_model: Any, adapter: AdapterSpec, baseline: str) -> Any:
    config = _worker_lora_config(adapter, baseline)
    kwargs = {"adapter_name": adapter.name, "revision": adapter.revision}
    if config is not None:
        kwargs["config"] = config
    try:
        return peft_model_class.from_pretrained(base_model, adapter.repository, **kwargs)
    except TypeError:
        if "config" in kwargs:
            kwargs.pop("config")
            try:
                return peft_model_class.from_pretrained(base_model, adapter.repository, **kwargs)
            except TypeError:
                pass
        return peft_model_class.from_pretrained(base_model, adapter.repository, revision=adapter.revision)


def _load_additional_worker_adapter(model: Any, adapter: AdapterSpec, baseline: str) -> None:
    if not hasattr(model, "load_adapter"):
        raise RuntimeError("PEFT model does not expose load_adapter for multi-adapter serving")
    config = _worker_lora_config(adapter, baseline)
    kwargs = {"adapter_name": adapter.name, "revision": adapter.revision}
    if config is not None:
        kwargs["config"] = config
    try:
        model.load_adapter(adapter.repository, **kwargs)
    except TypeError:
        if "config" in kwargs:
            kwargs.pop("config")
            try:
                model.load_adapter(adapter.repository, **kwargs)
                return
            except TypeError:
                pass
        model.load_adapter(adapter.repository, adapter_name=adapter.name, revision=adapter.revision)


def _worker_lora_config(adapter: AdapterSpec, baseline: str) -> Any:
    if baseline == "beyond_matmul_factor_provenance":
        try:
            from peft import LoraConfig

            peft_config = LoraConfig.from_pretrained(adapter.repository, revision=adapter.revision)
            if hasattr(peft_config, "runtime_config"):
                peft_config.runtime_config.beyond_matmul_provenance = True
            return peft_config
        except Exception:
            return None
    return None


def _run_dense_cache_worker(
    args: argparse.Namespace,
    model_class: Any,
    peft_model_class: Any,
    adapter: AdapterSpec,
) -> Dict[str, Any]:
    try:
        dense_cache = {}
        for cached_adapter in ADAPTERS:
            base_model = _load_base_worker_model(model_class)
            model = _load_primary_worker_adapter(
                peft_model_class,
                base_model,
                cached_adapter,
                args._worker_baseline,
            )
            if hasattr(model, "set_adapter"):
                model.set_adapter(cached_adapter.name)
            merged = _merge_dense_or_not_applicable(args, model, cached_adapter)
            if isinstance(merged, dict):
                merged["adapter"] = adapter.name
                return merged
            dense_cache[cached_adapter.name] = merged
        model = dense_cache[adapter.name]
        model.eval()
        inputs = _worker_inputs(args._worker_sequence_length, args._worker_batch_size, model.config.vocab_size)
        switch_latencies = _measure_worker_switch(args, dense_cache, adapter, args._worker_baseline)
        latencies = _time_forward(
            args._worker_baseline,
            adapter.name,
            {"model": lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, **inputs},
            args._worker_warmup,
            args._worker_repetitions,
        )
        torch = _torch()
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "ok",
            "reason": None,
            "latencies": latencies,
            "switch_latencies": switch_latencies,
            "logits": logits.detach().cpu().tolist(),
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }
    except Exception as exc:  # pragma: no cover - external PEFT behavior.
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "failed",
            "reason": str(exc),
            "latencies": None,
            "switch_latencies": None,
            "logits": None,
            "storage": _worker_storage(None, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }


def _merge_dense_or_not_applicable(args: argparse.Namespace, model: Any, adapter: AdapterSpec) -> Any:
    try:
        merged = model.merge_and_unload()
        merged.eval()
        return merged
    except Exception as exc:  # pragma: no cover - external PEFT behavior.
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "not_applicable",
            "reason": f"merge_and_unload failed: {exc}",
            "latencies": None,
            "switch_latencies": None,
            "logits": None,
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }


def _run_repeated_merge_worker(args: argparse.Namespace, model: Any, inputs: Dict[str, Any], adapter: AdapterSpec) -> Dict[str, Any]:
    if not hasattr(model, "merge_adapter") or not hasattr(model, "unmerge_adapter") or not hasattr(model, "set_adapter"):
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "not_applicable",
            "reason": "installed PEFT model does not expose set_adapter/merge_adapter/unmerge_adapter",
            "latencies": None,
            "switch_latencies": None,
            "logits": None,
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }
    other_adapter = _other_adapter(adapter)
    try:
        for _ in range(args._worker_warmup):
            model.set_adapter(other_adapter.name)
            model.unmerge_adapter()
            model.set_adapter(adapter.name)
            model.merge_adapter()
            model.unmerge_adapter()
        switch_latencies = []
        for _ in range(args._worker_repetitions):
            model.set_adapter(other_adapter.name)
            model.unmerge_adapter()
            start = time.perf_counter()
            model.set_adapter(adapter.name)
            model.merge_adapter()
            model.unmerge_adapter()
            switch_latencies.append(time.perf_counter() - start)
        model.set_adapter(adapter.name)
        latencies = _time_forward(
            args._worker_baseline,
            adapter.name,
            {"model": lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, **inputs},
            args._worker_warmup,
            args._worker_repetitions,
        )
        torch = _torch()
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "ok",
            "reason": None,
            "latencies": latencies,
            "switch_latencies": switch_latencies,
            "logits": logits.detach().cpu().tolist(),
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }
    except Exception as exc:  # pragma: no cover - external PEFT behavior.
        return {
            "baseline": args._worker_baseline,
            "adapter": adapter.name,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "not_applicable",
            "reason": f"merge/unmerge transition failed: {exc}",
            "latencies": None,
            "switch_latencies": None,
            "logits": None,
            "storage": _worker_storage(model, adapter, args._worker_baseline),
            **_process_peak_memory_sample(),
        }


def _measure_worker_switch(args: argparse.Namespace, model: Any, adapter: AdapterSpec, baseline: str) -> List[float]:
    if baseline == "upstream_peft_merged_dense_cache":
        if isinstance(model, dict) and adapter.name in model:
            other_adapter = _other_adapter(adapter)
            for _ in range(args._worker_warmup):
                _ = model[other_adapter.name]
                _ = model[adapter.name]
            latencies = []
            for _ in range(args._worker_repetitions):
                _ = model[other_adapter.name]
                start = time.perf_counter()
                _ = model[adapter.name]
                latencies.append(time.perf_counter() - start)
            return latencies
        return _time_switch(baseline, adapter.name, args._worker_warmup, args._worker_repetitions)
    if hasattr(model, "set_adapter"):
        other_adapter = _other_adapter(adapter)
        for _ in range(args._worker_warmup):
            model.set_adapter(other_adapter.name)
            model.set_adapter(adapter.name)
        latencies = []
        for _ in range(args._worker_repetitions):
            model.set_adapter(other_adapter.name)
            start = time.perf_counter()
            model.set_adapter(adapter.name)
            latencies.append(time.perf_counter() - start)
        return latencies
    return _time_switch(baseline, adapter.name, args._worker_warmup, args._worker_repetitions)


def _other_adapter(adapter: AdapterSpec) -> AdapterSpec:
    for other_adapter in ADAPTERS:
        if other_adapter.name != adapter.name:
            return other_adapter
    raise ValueError("multi-adapter switching requires at least two adapters")


def _worker_inputs(sequence_length: int, batch_size: int, vocab_size: int) -> Dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED + sequence_length * 1000 + batch_size)
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, sequence_length), generator=generator, dtype=torch.long),
        "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.long),
    }


def _collect_peft_provenance_events(model: Any) -> List[Dict[str, Any]]:
    if not hasattr(model, "named_modules"):
        return []
    events = []
    for module_name, module in model.named_modules():
        event = getattr(module, "beyond_matmul_last_forward_provenance", None)
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "beyond_matmul_lora_provenance":
            continue
        event_copy = json.loads(json.dumps(event))
        event_copy.setdefault("module_name", module_name)
        events.append(event_copy)
    events.sort(key=lambda event: (event.get("module_path") or "", event.get("adapter") or ""))
    return events


def _worker_storage(model: Any, adapter: AdapterSpec, baseline: str) -> Dict[str, Any]:
    resident_adapter_bytes = None
    if baseline == "upstream_peft_merged_dense_cache":
        resident_adapter_bytes = _model_parameter_bytes(model) or BASE_MODEL_FP32_BYTES_APPROX
    elif baseline in {"upstream_peft_unmerged", "beyond_matmul_factor_provenance"}:
        resident_adapter_bytes = adapter.payload_bytes
    return {
        "adapter_config_bytes": None,
        "resident_adapter_bytes": resident_adapter_bytes,
    }


def _model_parameter_bytes(model: Any) -> int | None:
    if model is None or not hasattr(model, "parameters"):
        return None
    total = 0
    try:
        for parameter in model.parameters():
            total += int(parameter.numel()) * int(parameter.element_size())
    except Exception:
        return None
    return total


def write_json_artifact(output_path: str | os.PathLike[str], **kwargs: Any) -> Dict[str, Any]:
    artifact = collect_results(**kwargs)
    _write_json(output_path, artifact)
    return artifact


def _write_json(output_path: str | os.PathLike[str], artifact: Dict[str, Any]) -> None:
    path = os.fspath(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(artifact, output, indent=2, sort_keys=True)
        output.write("\n")


def _parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item]


def _print_table(artifact: Dict[str, Any]) -> None:
    print("case                  baseline                              status          median_s  switch_s  correct")
    print("--------------------  ------------------------------------  --------------  --------  --------  -------")
    for row in artifact["results"]:
        latency = row["latency_seconds"]["median"] if row["latency_seconds"] else None
        switch = row["adapter_switch_seconds"]["median"] if row["adapter_switch_seconds"] else None
        latency_text = f"{latency:.5f}" if latency is not None else "null"
        switch_text = f"{switch:.5f}" if switch is not None else "null"
        print(
            f"{row['case']:<20}  {row['baseline']:<36}  {row['status']:<14}  "
            f"{latency_text:>8}  {switch_text:>8}  {str(row['correctness']['passed']).lower()}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable benchmark results to this JSON path")
    parser.add_argument("--smoke", action="store_true", help="run the torch-only CI smoke workload")
    parser.add_argument(
        "--hardware-contract-smoke",
        action="store_true",
        help="emit the CI-safe hardware-backed production contract readiness artifact",
    )
    parser.add_argument("--sequence-lengths", default=",".join(str(value) for value in DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup-repetitions", type=int, default=DEFAULT_WARMUP_REPETITIONS)
    parser.add_argument("--measured-repetitions", type=int, default=DEFAULT_MEASURED_REPETITIONS)
    parser.add_argument("--upstream-peft-path")
    parser.add_argument("--fork-peft-path")
    parser.add_argument("--upstream-peft-ref", default=DEFAULT_UPSTREAM_REF)
    parser.add_argument("--fork-peft-ref", default=DEFAULT_FORK_REF)
    parser.add_argument("--checkout-dir")
    parser.add_argument("--_worker-json-output", dest="_worker_json_output", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-baseline", dest="_worker_baseline", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-adapter-name", dest="_worker_adapter_name", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-sequence-length", dest="_worker_sequence_length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-batch-size", dest="_worker_batch_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-warmup", dest="_worker_warmup", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-repetitions", dest="_worker_repetitions", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-peft-path", dest="_worker_peft_path", help=argparse.SUPPRESS)
    parser.add_argument("--_hardware-worker-json-output", dest="_hardware_worker_json_output", help=argparse.SUPPRESS)
    parser.add_argument("--_hardware-worker-baseline", dest="_hardware_worker_baseline", help=argparse.SUPPRESS)
    parser.add_argument("--_hardware-worker-adapter-name", dest="_hardware_worker_adapter_name", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_hardware-worker-sequence-length",
        dest="_hardware_worker_sequence_length",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_hardware-worker-batch-size",
        dest="_hardware_worker_batch_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_hardware-worker-forward-warmup",
        dest="_hardware_worker_forward_warmup",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_hardware-worker-forward-repetitions",
        dest="_hardware_worker_forward_repetitions",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_hardware-worker-switch-warmup",
        dest="_hardware_worker_switch_warmup",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_hardware-worker-switch-repetitions",
        dest="_hardware_worker_switch_repetitions",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_hardware-worker-peft-path", dest="_hardware_worker_peft_path", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args._hardware_worker_json_output:
        _hardware_real_worker(args)
        return
    if args._worker_json_output:
        _real_worker(args)
        return
    command = [sys.executable, __file__, *sys.argv[1:]]
    generated_at_utc = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if args.hardware_contract_smoke:
        artifact = collect_hardware_contract_results(
            mode="synthetic-smoke",
            command=command,
            generated_at_utc=generated_at_utc,
        )
        if args.json_output:
            _write_json(args.json_output, artifact)
        print(
            "hardware_contract_ready="
            f"{str(artifact['summary']['production_contract_ready']).lower()} "
            f"blockers={','.join(artifact['summary']['readiness_blockers'])}"
        )
        return
    sequence_lengths = [4] if args.smoke else _parse_int_list(args.sequence_lengths)
    batch_sizes = [1] if args.smoke else _parse_int_list(args.batch_sizes)
    artifact = collect_results(
        sequence_lengths=sequence_lengths,
        batch_sizes=batch_sizes,
        warmup_repetitions=1 if args.smoke else args.warmup_repetitions,
        measured_repetitions=2 if args.smoke else args.measured_repetitions,
        mode="synthetic-smoke" if args.smoke else "real",
        upstream_peft_path=args.upstream_peft_path,
        fork_peft_path=args.fork_peft_path,
        upstream_peft_ref=args.upstream_peft_ref,
        fork_peft_ref=args.fork_peft_ref,
        checkout_dir=args.checkout_dir,
        command=command,
        generated_at_utc=generated_at_utc,
    )
    if args.json_output:
        _write_json(args.json_output, artifact)
    _print_table(artifact)


if __name__ == "__main__":
    main()
