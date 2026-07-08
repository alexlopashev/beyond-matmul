#!/usr/bin/env python3
"""Live Whisper Conv1d dense-vs-direct benchmark harness."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Sequence


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BENCHMARK = "live_conv1d_whisper_dense_vs_direct"
CONTRACT_PATH = "docs/live_conv1d_benchmark_contract.md"
MODEL = "openai/whisper-tiny"
MODEL_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
MODEL_LICENSE = "apache-2.0"
LAYER_PATH = "model.encoder.conv1"
INPUT_URL = "https://cdn-media.huggingface.co/speech_samples/sample1.flac"
INPUT_SHA256 = "cb5c48a2d1d6f7dedd0330f088a4cbe76de1a86e6a6109c06d255bb1ca2f7542"
DEFAULT_PREFIX_FRAMES = [8, 16, 32]
DEFAULT_WARMUP_REPETITIONS = 10
DEFAULT_MEASURED_REPETITIONS = 50
CORRECTNESS_MAX_ABS_TOLERANCE = 1e-4
CORRECTNESS_RELATIVE_L2_TOLERANCE = 1e-5
BASELINES = ["direct_conv1d", "dense_materialized_toeplitz"]

TimeDirect = Callable[[Any, Any, int, int], Sequence[float]]
TimeDenseApply = Callable[[Any, Any, Any, int, int, int], Sequence[float]]
TimeMaterialization = Callable[[Any, int], Any]


class DenseConv1dMaterialization(NamedTuple):
    matrix: Any
    bias: Any
    output_frames: int
    entries: int
    bytes_float32: int
    toeplitz_nonzero_coefficients: int
    density: float


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local CI pins torch.
        raise RuntimeError("PyTorch is required for the live Conv1d benchmark harness") from exc
    return torch


def collect_results(
    *,
    mode: str = "real",
    prefix_frames: Sequence[int] | None = None,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    device: str = "cpu",
    cache_dir: str | os.PathLike[str] | None = None,
    command: Sequence[str] | None = None,
    generated_at_utc: str | None = None,
    time_direct: TimeDirect | None = None,
    time_dense_apply: TimeDenseApply | None = None,
    time_materialization: TimeMaterialization | None = None,
) -> Dict[str, Any]:
    prefix_frames = list(prefix_frames or DEFAULT_PREFIX_FRAMES)
    _validate_positive_ints("prefix frames", prefix_frames)
    if warmup_repetitions < 0:
        raise ValueError("warmup repetitions must be non-negative")
    if measured_repetitions <= 0:
        raise ValueError("measured repetitions must be positive")
    if mode not in {"real", "synthetic-smoke"}:
        raise ValueError(f"unsupported mode: {mode}")
    if device != "cpu":
        raise ValueError("issue #95 only defines the CPU fp32 contract")

    layer, full_inputs, input_metadata = _load_workload(mode=mode, cache_dir=cache_dir)
    _validate_contract_layer(layer)

    results = []
    for frames in prefix_frames:
        if full_inputs.shape[2] < frames:
            raise RuntimeError(
                f"readiness blocker: input trace produced {full_inputs.shape[2]} frames, "
                f"fewer than required prefix {frames}"
            )
        inputs = full_inputs[:, :, :frames].contiguous()
        results.extend(
            _collect_case(
                layer=layer,
                inputs=inputs,
                prefix_frames=frames,
                warmup_repetitions=warmup_repetitions,
                measured_repetitions=measured_repetitions,
                time_direct=time_direct or _time_direct,
                time_dense_apply=time_dense_apply or _time_dense_apply,
                time_materialization=time_materialization,
            )
        )

    return {
        "schema_version": 1,
        "benchmark": BENCHMARK,
        "contract": CONTRACT_PATH,
        "mode": mode,
        "workload": _workload_metadata(
            prefix_frames=prefix_frames,
            warmup_repetitions=warmup_repetitions,
            measured_repetitions=measured_repetitions,
            input_metadata=input_metadata,
        ),
        "dependencies": _dependency_metadata(mode),
        "environment": _environment_metadata(),
        "run": _run_metadata(command=command, generated_at_utc=generated_at_utc, mode=mode),
        "results": results,
        "summary": _artifact_summary(results, mode=mode),
    }


def _collect_case(
    *,
    layer: Any,
    inputs: Any,
    prefix_frames: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    time_direct: TimeDirect,
    time_dense_apply: TimeDenseApply,
    time_materialization: TimeMaterialization | None,
) -> List[Dict[str, Any]]:
    torch = _torch()
    with torch.inference_mode():
        reference = layer(inputs)

    direct_latencies = list(time_direct(layer, inputs, warmup_repetitions, measured_repetitions))
    dense, materialization_seconds = _materialize_with_timer(layer, prefix_frames, time_materialization)
    dense_output = apply_dense_materialized_conv1d(inputs, dense.matrix, dense.bias, dense.output_frames)
    dense_latencies = list(
        time_dense_apply(inputs, dense.matrix, dense.bias, dense.output_frames, warmup_repetitions, measured_repetitions)
    )

    return [
        _result_row(
            case=f"frames{prefix_frames}_batch1",
            baseline="direct_conv1d",
            inputs=inputs,
            output=reference,
            reference=reference,
            dense=dense,
            latencies=direct_latencies,
            materialization_seconds=None,
            materialization_status="not_applicable_direct_conv1d",
        ),
        _result_row(
            case=f"frames{prefix_frames}_batch1",
            baseline="dense_materialized_toeplitz",
            inputs=inputs,
            output=dense_output,
            reference=reference,
            dense=dense,
            latencies=dense_latencies,
            materialization_seconds=materialization_seconds,
            materialization_status="ok",
        ),
    ]


def materialize_conv1d_to_dense(layer: Any, input_frames: int) -> DenseConv1dMaterialization:
    torch = _torch()
    if input_frames <= 0:
        raise ValueError(f"input frames must be positive: {input_frames}")
    if not isinstance(layer, torch.nn.Conv1d):
        raise TypeError("layer must be torch.nn.Conv1d")

    stride = _single_int(layer.stride, "stride")
    padding = _single_int(layer.padding, "padding")
    dilation = _single_int(layer.dilation, "dilation")
    kernel_size = _single_int(layer.kernel_size, "kernel_size")
    output_frames = ((input_frames + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    if output_frames <= 0:
        raise ValueError(
            "Conv1d parameters produce a non-positive output length: "
            f"input_frames={input_frames}, output_frames={output_frames}"
        )

    weight = layer.weight.detach().to(device="cpu", dtype=torch.float32)
    bias = None
    if layer.bias is not None:
        bias = layer.bias.detach().to(device="cpu", dtype=torch.float32).repeat_interleave(output_frames)

    out_channels = int(layer.out_channels)
    in_channels = int(layer.in_channels)
    groups = int(layer.groups)
    in_channels_per_group = in_channels // groups
    matrix = torch.zeros((out_channels * output_frames, in_channels * input_frames), dtype=torch.float32)
    nonzero_coefficients = 0

    for out_channel in range(out_channels):
        group = out_channel // (out_channels // groups)
        input_channel_start = group * in_channels_per_group
        for out_position in range(output_frames):
            row = out_channel * output_frames + out_position
            base_input_position = out_position * stride - padding
            for kernel_position in range(kernel_size):
                input_position = base_input_position + kernel_position * dilation
                if input_position < 0 or input_position >= input_frames:
                    continue
                nonzero_coefficients += in_channels_per_group
                for relative_input_channel in range(in_channels_per_group):
                    input_channel = input_channel_start + relative_input_channel
                    column = input_channel * input_frames + input_position
                    matrix[row, column] = weight[out_channel, relative_input_channel, kernel_position]

    entries = int(matrix.numel())
    return DenseConv1dMaterialization(
        matrix=matrix,
        bias=bias,
        output_frames=output_frames,
        entries=entries,
        bytes_float32=entries * 4,
        toeplitz_nonzero_coefficients=nonzero_coefficients,
        density=0.0 if entries == 0 else nonzero_coefficients / entries,
    )


def apply_dense_materialized_conv1d(inputs: Any, matrix: Any, bias: Any, output_frames: int) -> Any:
    batch_size, _in_channels, _input_frames = inputs.shape
    flat_inputs = inputs.detach().to(device="cpu", dtype=matrix.dtype).reshape(batch_size, -1)
    flat_output = flat_inputs @ matrix.t()
    if bias is not None:
        flat_output = flat_output + bias
    output_channels = matrix.shape[0] // output_frames
    return flat_output.reshape(batch_size, output_channels, output_frames)


def _materialize_with_timer(
    layer: Any,
    input_frames: int,
    time_materialization: TimeMaterialization | None,
) -> tuple[DenseConv1dMaterialization, float]:
    if time_materialization is not None:
        result = time_materialization(layer, input_frames)
        if isinstance(result, DenseConv1dMaterialization):
            return result, 0.0
        if isinstance(result, tuple):
            dense, seconds = result
            return dense, _stable_float(seconds)
        return result, 0.0

    start = time.perf_counter()
    dense = materialize_conv1d_to_dense(layer, input_frames)
    seconds = time.perf_counter() - start
    return dense, _stable_float(seconds)


def _time_direct(layer: Any, inputs: Any, warmup_repetitions: int, measured_repetitions: int) -> List[float]:
    torch = _torch()
    with torch.inference_mode():
        for _ in range(warmup_repetitions):
            layer(inputs)
        latencies = []
        for _ in range(measured_repetitions):
            start = time.perf_counter()
            layer(inputs)
            latencies.append(time.perf_counter() - start)
    return latencies


def _time_dense_apply(
    inputs: Any,
    matrix: Any,
    bias: Any,
    output_frames: int,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> List[float]:
    for _ in range(warmup_repetitions):
        apply_dense_materialized_conv1d(inputs, matrix, bias, output_frames)
    latencies = []
    for _ in range(measured_repetitions):
        start = time.perf_counter()
        apply_dense_materialized_conv1d(inputs, matrix, bias, output_frames)
        latencies.append(time.perf_counter() - start)
    return latencies


def _result_row(
    *,
    case: str,
    baseline: str,
    inputs: Any,
    output: Any,
    reference: Any,
    dense: DenseConv1dMaterialization,
    latencies: Sequence[float],
    materialization_seconds: float | None,
    materialization_status: str,
) -> Dict[str, Any]:
    correctness = _correctness_metrics(output, reference)
    status = "ok" if correctness["passed"] else "failed_correctness"
    row = {
        "case": case,
        "baseline": baseline,
        "status": status,
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "output_shape": list(reference.shape),
        "latency_seconds": _latency_stats(latencies) if status == "ok" else None,
        "materialization_seconds": materialization_seconds,
        "materialization_status": materialization_status,
        "peak_memory_bytes": None,
        "peak_memory_status": "not_measured",
        "dense_matrix": _dense_matrix_metadata(dense),
        "correctness": correctness,
    }
    if status != "ok":
        row["reason"] = "correctness tolerance failed"
    return row


def _dense_matrix_metadata(dense: DenseConv1dMaterialization) -> Dict[str, Any]:
    return {
        "shape": list(dense.matrix.shape),
        "entries": dense.entries,
        "bytes_float32": dense.bytes_float32,
        "toeplitz_nonzero_coefficients": dense.toeplitz_nonzero_coefficients,
        "density": _stable_float(dense.density),
    }


def _correctness_metrics(output: Any, reference_output: Any) -> Dict[str, Any]:
    torch = _torch()
    candidate = torch.as_tensor(output, dtype=torch.float32)
    reference = torch.as_tensor(reference_output, dtype=torch.float32)
    finite = bool(torch.isfinite(candidate).all().item() and torch.isfinite(reference).all().item())
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
        "reference_baseline": "direct_conv1d",
        "max_abs_error": _stable_float(max_abs_error),
        "relative_l2_error": _stable_float(relative_l2_error),
        "max_abs_tolerance": CORRECTNESS_MAX_ABS_TOLERANCE,
        "relative_l2_tolerance": CORRECTNESS_RELATIVE_L2_TOLERANCE,
        "tolerance_profile": "cpu_fp32",
        "passed": passed,
    }


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


def _artifact_summary(results: Sequence[Dict[str, Any]], *, mode: str) -> Dict[str, Any]:
    all_required_cases_present = _all_required_cases_present(results)
    all_correctness_checks_passed = all(row["status"] == "ok" and row["correctness"]["passed"] for row in results)
    readiness_blockers = _readiness_blockers(
        mode=mode,
        all_required_cases_present=all_required_cases_present,
        all_correctness_checks_passed=all_correctness_checks_passed,
    )
    return {
        "all_required_cases_present": all_required_cases_present,
        "all_correctness_checks_passed": all_correctness_checks_passed,
        "benchmark_ready": not readiness_blockers,
        "readiness_blockers": readiness_blockers,
        "max_abs_error": _max_correctness_metric(results, "max_abs_error"),
        "max_relative_l2_error": _max_correctness_metric(results, "relative_l2_error"),
        "performance_claim": "none",
    }


def _readiness_blockers(
    *,
    mode: str,
    all_required_cases_present: bool,
    all_correctness_checks_passed: bool,
) -> List[str]:
    blockers = []
    if mode != "real":
        blockers.append("synthetic_smoke_not_benchmark_evidence")
    if not all_required_cases_present:
        blockers.append("required_cases_missing")
    if not all_correctness_checks_passed:
        blockers.append("correctness_checks_failed")
    return blockers


def _all_required_cases_present(results: Sequence[Dict[str, Any]]) -> bool:
    present = {(row["case"], row["baseline"]) for row in results}
    required = {
        (f"frames{frames}_batch1", baseline)
        for frames in DEFAULT_PREFIX_FRAMES
        for baseline in BASELINES
    }
    return required.issubset(present)


def _max_correctness_metric(results: Sequence[Dict[str, Any]], metric: str) -> float | None:
    values = [
        float(row["correctness"][metric])
        for row in results
        if row["correctness"].get(metric) is not None
    ]
    if not values:
        return None
    return _stable_float(max(values))


def _load_workload(*, mode: str, cache_dir: str | os.PathLike[str] | None) -> tuple[Any, Any, Dict[str, Any]]:
    if mode == "synthetic-smoke":
        return _synthetic_workload()
    return _real_workload(cache_dir=cache_dir)


def _synthetic_workload() -> tuple[Any, Any, Dict[str, Any]]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260708)
    layer = torch.nn.Conv1d(80, 384, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator, dtype=torch.float32) * 0.01)
        layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator, dtype=torch.float32) * 0.01)
    layer.eval()
    inputs = torch.randn((1, 80, max(DEFAULT_PREFIX_FRAMES)), generator=generator, dtype=torch.float32)
    return layer, inputs, {"mode": "synthetic-smoke", "full_feature_shape": list(inputs.shape)}


def _real_workload(*, cache_dir: str | os.PathLike[str] | None) -> tuple[Any, Any, Dict[str, Any]]:
    torch = _torch()
    try:
        import soundfile as sf
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError(
            "real mode requires optional dependencies; run with "
            "`uv run --with transformers --with librosa --with soundfile --with safetensors "
            "--with huggingface_hub python benchmarks/live_conv1d_whisper.py ...`"
        ) from exc

    audio_path = _cached_audio_path(cache_dir)
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    processor = WhisperProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)
    features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features.to(dtype=torch.float32)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, revision=MODEL_REVISION)
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    layer = model.model.encoder.conv1
    layer.eval()
    return (
        layer,
        features.contiguous(),
        {
            "mode": "real",
            "full_feature_shape": list(features.shape),
            "sample_rate": int(sample_rate),
            "audio_path": str(audio_path),
        },
    )


def _cached_audio_path(cache_dir: str | os.PathLike[str] | None) -> Path:
    base_dir = Path(cache_dir) if cache_dir else Path(ROOT) / ".cache" / "live_conv1d_whisper"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "sample1.flac"
    if not path.exists():
        urllib.request.urlretrieve(INPUT_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != INPUT_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded audio SHA-256 mismatch for {INPUT_URL}: expected {INPUT_SHA256}, got {digest}"
        )
    return path


def _validate_contract_layer(layer: Any) -> None:
    expected = {
        "in_channels": 80,
        "out_channels": 384,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
    }
    observed = {
        "in_channels": int(layer.in_channels),
        "out_channels": int(layer.out_channels),
        "kernel_size": _single_int(layer.kernel_size, "kernel_size"),
        "stride": _single_int(layer.stride, "stride"),
        "padding": _single_int(layer.padding, "padding"),
        "dilation": _single_int(layer.dilation, "dilation"),
        "groups": int(layer.groups),
    }
    if observed != expected:
        raise RuntimeError(f"selected layer no longer matches the contract: expected {expected}, got {observed}")
    if layer.bias is None:
        raise RuntimeError("selected layer no longer matches the contract: expected bias=True")


def _workload_metadata(
    *,
    prefix_frames: Sequence[int],
    warmup_repetitions: int,
    measured_repetitions: int,
    input_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "layer_path": LAYER_PATH,
        "layer_type": "torch.nn.Conv1d",
        "layer": {
            "in_channels": 80,
            "out_channels": 384,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "bias": True,
        },
        "input": {
            "source_url": INPUT_URL,
            "sha256": INPUT_SHA256,
            "preprocessor": "WhisperProcessor",
            "dtype": "float32",
            "device": "cpu",
            "prefix_frames": list(prefix_frames),
            **input_metadata,
        },
        "warmup_repetitions": warmup_repetitions,
        "measured_repetitions": measured_repetitions,
    }


def _dependency_metadata(mode: str) -> Dict[str, Any]:
    torch = _torch()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _module_metadata("transformers", mode),
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
        return {"version": None, "status": f"not_imported_{mode}"}
    return {"version": getattr(module, "__version__", None)}


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


def _validate_positive_ints(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"at least one {name} is required")
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be positive: {value}")


def _single_int(value: Any, name: str) -> int:
    if isinstance(value, int):
        return value
    if len(value) != 1:
        raise ValueError(f"{name} must be one-dimensional for this benchmark: {value}")
    return int(value[0])


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def write_json_artifact(output_path: str | os.PathLike[str], **kwargs: Any) -> Dict[str, Any]:
    artifact = collect_results(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _print_table(artifact: Dict[str, Any]) -> None:
    print("case             baseline                      median_s  materialize_s  dense_mb  status")
    print("---------------- ----------------------------- --------- ------------- -------- ------")
    for row in artifact["results"]:
        latency = row["latency_seconds"]
        median = latency["median"] if latency is not None else None
        materialization = row["materialization_seconds"]
        dense_mb = row["dense_matrix"]["bytes_float32"] / (1024 * 1024)
        print(
            f"{row['case']:<16} {row['baseline']:<29} "
            f"{_format_optional_float(median):>9} "
            f"{_format_optional_float(materialization):>13} "
            f"{dense_mb:8.2f} "
            f"{row['status']}"
        )
    summary = artifact["summary"]
    print(
        "\n"
        f"benchmark_ready={summary['benchmark_ready']} "
        f"performance_claim={summary['performance_claim']} "
        f"max_abs_error={summary['max_abs_error']} "
        f"max_relative_l2_error={summary['max_relative_l2_error']}"
    )
    if summary["readiness_blockers"]:
        print("readiness_blockers=" + ",".join(summary["readiness_blockers"]))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable benchmark results to this JSON path")
    parser.add_argument("--smoke", action="store_true", help="run deterministic torch-only smoke mode")
    parser.add_argument("--prefix-frames", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-repetitions", type=int, default=DEFAULT_WARMUP_REPETITIONS)
    parser.add_argument("--measured-repetitions", type=int, default=DEFAULT_MEASURED_REPETITIONS)
    parser.add_argument("--cache-dir", help="directory for the downloaded audio trace")
    args = parser.parse_args()

    generated_at_utc = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    artifact = collect_results(
        mode="synthetic-smoke" if args.smoke else "real",
        prefix_frames=args.prefix_frames,
        warmup_repetitions=args.warmup_repetitions,
        measured_repetitions=args.measured_repetitions,
        cache_dir=args.cache_dir,
        command=sys.argv,
        generated_at_utc=generated_at_utc,
    )
    _print_table(artifact)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
