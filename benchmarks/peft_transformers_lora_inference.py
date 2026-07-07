#!/usr/bin/env python3
"""TorchBench-style PEFT plus Transformers upstream-vs-fork benchmark harness."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE_MODEL = "hf-internal-testing/tiny-random-OPTForCausalLM"
ADAPTER = "peft-internal-testing/tiny-OPTForCausalLM-lora"
INPUT_SEED = 20260707
DEFAULT_SEQUENCE_LENGTHS = [16, 64, 128]
DEFAULT_BATCH_SIZES = [1, 4]
DEFAULT_WARMUP_REPETITIONS = 10
DEFAULT_MEASURED_REPETITIONS = 50
CONTRACT_PATH = "docs/peft_capstone_benchmark_contract.md"
UPSTREAM_REPOSITORY = "huggingface/peft"
FORK_REPOSITORY = "alexlopashev/peft"
DEFAULT_UPSTREAM_REF = "main"
DEFAULT_FORK_REF = "beyond-matmul/provenance-lora-inference"
BASELINES = [
    "upstream_peft_unmerged",
    "upstream_peft_merged_dense",
    "beyond_matmul_peft_fork",
]

TimeForward = Callable[[str, Dict[str, Any], int, int], Sequence[float]]


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local CI pins torch.
        raise RuntimeError("PyTorch is required for the PEFT benchmark harness") from exc
    return torch


def _time_forward(_baseline: str, inputs: Dict[str, Any], warmup_repetitions: int, measured_repetitions: int) -> List[float]:
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


class _SyntheticLoraModel:
    def __init__(self, baseline: str, vocab_size: int = 16) -> None:
        self.baseline = baseline
        self.vocab_size = vocab_size

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        torch = _torch()
        ids = input_ids.to(dtype=torch.float32)
        mask = attention_mask.to(dtype=torch.float32)
        features = torch.arange(self.vocab_size, dtype=torch.float32, device=ids.device)
        logits = torch.sin((ids.unsqueeze(-1) + features) * 0.03125)
        logits = logits + mask.unsqueeze(-1) * 0.125
        if self.baseline == "upstream_peft_merged_dense":
            logits = logits + 1e-8
        return logits


def collect_results(
    *,
    sequence_lengths: Sequence[int] | None = None,
    batch_sizes: Sequence[int] | None = None,
    warmup_repetitions: int = DEFAULT_WARMUP_REPETITIONS,
    measured_repetitions: int = DEFAULT_MEASURED_REPETITIONS,
    mode: str = "real",
    device: str = "cpu",
    upstream_peft_path: str | None = None,
    fork_peft_path: str | None = None,
    upstream_peft_ref: str = DEFAULT_UPSTREAM_REF,
    fork_peft_ref: str = DEFAULT_FORK_REF,
    checkout_dir: str | os.PathLike[str] | None = None,
    time_forward: TimeForward = _time_forward,
) -> Dict[str, Any]:
    sequence_lengths = list(sequence_lengths or DEFAULT_SEQUENCE_LENGTHS)
    batch_sizes = list(batch_sizes or DEFAULT_BATCH_SIZES)
    _validate_positive_ints("sequence length", sequence_lengths)
    _validate_positive_ints("batch size", batch_sizes)
    if warmup_repetitions < 0:
        raise ValueError("warmup repetitions must be non-negative")
    if measured_repetitions <= 0:
        raise ValueError("measured repetitions must be positive")
    if mode not in {"synthetic-smoke", "real"}:
        raise ValueError(f"unsupported mode: {mode}")
    if device != "cpu":
        raise ValueError("issue #76 only defines the CPU fp32 contract")

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
    for sequence_length in sequence_lengths:
        for batch_size in batch_sizes:
            if mode == "synthetic-smoke":
                results.extend(
                    _collect_synthetic_case(
                        sequence_length,
                        batch_size,
                        warmup_repetitions,
                        measured_repetitions,
                        time_forward,
                    )
                )
            else:
                results.extend(
                    _collect_real_case(
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
        "benchmark": "peft_transformers_lora_inference",
        "contract": CONTRACT_PATH,
        "mode": mode,
        "workload": {
            "base_model": BASE_MODEL,
            "base_model_revision": _revision_label(BASE_MODEL, mode),
            "adapter": ADAPTER,
            "adapter_revision": _revision_label(ADAPTER, mode),
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
        "results": results,
        "summary": {
            "all_required_cases_present": _all_requested_cases_present(results, sequence_lengths, batch_sizes),
            "all_correctness_checks_passed": all(
                row["status"] in {"ok", "not_applicable"} and row["correctness"]["passed"] for row in results
            ),
            "performance_claim": "none",
        },
    }


def _validate_positive_ints(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"at least one {name} is required")
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be positive: {value}")


def _collect_synthetic_case(
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    time_forward: TimeForward,
) -> List[Dict[str, Any]]:
    inputs = _synthetic_inputs(sequence_length, batch_size)
    reference_logits = None
    rows = []
    for baseline in BASELINES:
        model = _SyntheticLoraModel(baseline)
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        if reference_logits is None:
            reference_logits = logits
        timing_inputs = {**inputs, "model": model}
        latencies = list(time_forward(baseline, timing_inputs, warmup_repetitions, measured_repetitions))
        rows.append(
            _result_row(
                baseline=baseline,
                sequence_length=sequence_length,
                batch_size=batch_size,
                latencies=latencies,
                logits=logits,
                reference_logits=reference_logits,
            )
        )
    return rows


def _collect_real_case(
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    upstream_peft_path: str,
    fork_peft_path: str,
) -> List[Dict[str, Any]]:
    reference = _run_real_worker(
        "upstream_peft_unmerged",
        sequence_length,
        batch_size,
        warmup_repetitions,
        measured_repetitions,
        upstream_peft_path,
        merge_dense=False,
    )
    rows = [_worker_payload_to_row(reference, reference["logits"])]
    for baseline, path, merge_dense in [
        ("upstream_peft_merged_dense", upstream_peft_path, True),
        ("beyond_matmul_peft_fork", fork_peft_path, False),
    ]:
        payload = _run_real_worker(
            baseline,
            sequence_length,
            batch_size,
            warmup_repetitions,
            measured_repetitions,
            path,
            merge_dense=merge_dense,
        )
        rows.append(_worker_payload_to_row(payload, reference["logits"]))
    return rows


def _synthetic_inputs(sequence_length: int, batch_size: int) -> Dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED + sequence_length * 1000 + batch_size)
    input_ids = torch.randint(0, 16, (batch_size, sequence_length), generator=generator, dtype=torch.long)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _result_row(
    *,
    baseline: str,
    sequence_length: int,
    batch_size: int,
    latencies: Sequence[float] | None,
    logits: Any,
    reference_logits: Any,
    status: str = "ok",
    reason: str | None = None,
) -> Dict[str, Any]:
    correctness = _correctness_metrics(logits, reference_logits)
    if status == "ok" and not correctness["passed"]:
        status = "failed_correctness"
        reason = reason or "correctness tolerance failed"
    row = {
        "case": f"seq{sequence_length}_batch{batch_size}",
        "baseline": baseline,
        "status": status,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "latency_seconds": _latency_stats(latencies) if latencies is not None and status == "ok" else None,
        "peak_memory_bytes": None,
        "peak_memory_status": "not_measurable_on_cpu",
        "adapter_switch_seconds": None,
        "adapter_switch_status": "not_measured_single_adapter",
        "correctness": correctness,
        "lowering": _lowering_metadata(baseline),
    }
    if reason:
        row["reason"] = reason
    return row


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


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def _correctness_metrics(logits: Any, reference_logits: Any) -> Dict[str, Any]:
    torch = _torch()
    candidate = torch.as_tensor(logits, dtype=torch.float32)
    reference = torch.as_tensor(reference_logits, dtype=torch.float32)
    finite = bool(torch.isfinite(candidate).all().item())
    diff = candidate - reference
    max_abs_error = float(torch.max(torch.abs(diff)).item())
    reference_norm = float(torch.linalg.vector_norm(reference).item())
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    relative_l2_error = 0.0 if reference_norm == 0.0 else diff_norm / reference_norm
    passed = finite and max_abs_error <= 1e-4 and relative_l2_error <= 1e-5
    return {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": max_abs_error,
        "relative_l2_error": relative_l2_error,
        "tolerance_profile": "cpu_fp32",
        "passed": passed,
    }


def _lowering_metadata(baseline: str) -> Dict[str, Any]:
    if baseline == "upstream_peft_unmerged":
        return {
            "kind": "peft_unmerged_adapter",
            "dense_fallback_available": True,
            "dense_fallback_used": False,
        }
    if baseline == "upstream_peft_merged_dense":
        return {
            "kind": "peft_merged_dense",
            "dense_fallback_available": True,
            "dense_fallback_used": True,
        }
    return {
        "kind": "provenance_lora_fork",
        "dense_fallback_available": True,
        "dense_fallback_used": False,
    }


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


def _safe_ref(ref: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in ref)


def _revision_label(identifier: str, mode: str) -> str:
    return f"not_resolved_{mode}:{identifier}"


def _all_requested_cases_present(
    results: Sequence[Dict[str, Any]],
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
) -> bool:
    present = {(row["case"], row["baseline"]) for row in results}
    required = {
        (f"seq{sequence_length}_batch{batch_size}", baseline)
        for sequence_length in sequence_lengths
        for batch_size in batch_sizes
        for baseline in BASELINES
    }
    return required.issubset(present)


def _run_real_worker(
    baseline: str,
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    peft_path: str,
    *,
    merge_dense: bool,
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
    if merge_dense:
        command.append("--_worker-merge-dense")
    try:
        subprocess.run(command, check=True)
        with open(output_path, encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _worker_payload_to_row(payload: Dict[str, Any], reference_logits: Any) -> Dict[str, Any]:
    if payload["status"] != "ok":
        return _non_ok_result_row(payload)
    return _result_row(
        baseline=payload["baseline"],
        sequence_length=payload["sequence_length"],
        batch_size=payload["batch_size"],
        latencies=payload["latencies"],
        logits=payload["logits"],
        reference_logits=reference_logits,
        status=payload["status"],
        reason=payload.get("reason"),
    )


def _non_ok_result_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    passed = payload["status"] == "not_applicable"
    return {
        "case": f"seq{payload['sequence_length']}_batch{payload['batch_size']}",
        "baseline": payload["baseline"],
        "status": payload["status"],
        "sequence_length": payload["sequence_length"],
        "batch_size": payload["batch_size"],
        "latency_seconds": None,
        "peak_memory_bytes": None,
        "peak_memory_status": "not_measurable_on_cpu",
        "adapter_switch_seconds": None,
        "adapter_switch_status": "not_measured_single_adapter",
        "correctness": {
            "reference_baseline": "upstream_peft_unmerged",
            "max_abs_error": None,
            "relative_l2_error": None,
            "tolerance_profile": "cpu_fp32",
            "passed": passed,
        },
        "lowering": _lowering_metadata(payload["baseline"]),
        "reason": payload.get("reason") or payload["status"],
    }


def _real_worker(args: argparse.Namespace) -> None:
    if args._worker_peft_path:
        sys.path.insert(0, args._worker_peft_path)
    torch = _torch()
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        if args._worker_merge_dense:
            try:
                model = model.merge_and_unload()
                model.eval()
            except Exception as exc:  # pragma: no cover - depends on external PEFT behavior.
                _write_json(
                    args._worker_json_output,
                    {
                        "baseline": args._worker_baseline,
                        "sequence_length": args._worker_sequence_length,
                        "batch_size": args._worker_batch_size,
                        "status": "not_applicable",
                        "reason": f"merge_and_unload failed: {exc}",
                        "logits": None,
                        "latencies": None,
                    },
                )
                return
        inputs = _worker_inputs(args._worker_sequence_length, args._worker_batch_size, model.config.vocab_size)
        latencies = _time_forward(
            args._worker_baseline,
            {"model": lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, **inputs},
            args._worker_warmup,
            args._worker_repetitions,
        )
        with torch.inference_mode():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        payload = {
            "baseline": args._worker_baseline,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "ok",
            "reason": None,
            "latencies": latencies,
            "logits": logits.detach().cpu().tolist(),
        }
    except Exception as exc:  # pragma: no cover - depends on optional external dependencies.
        payload = {
            "baseline": args._worker_baseline,
            "sequence_length": args._worker_sequence_length,
            "batch_size": args._worker_batch_size,
            "status": "failed",
            "reason": str(exc),
            "latencies": None,
            "logits": None,
        }
    _write_json(args._worker_json_output, payload)


def _worker_inputs(sequence_length: int, batch_size: int, vocab_size: int) -> Dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INPUT_SEED + sequence_length * 1000 + batch_size)
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, sequence_length), generator=generator, dtype=torch.long),
        "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.long),
    }


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
    print("case           baseline                    status     median_s  correct")
    print("-------------  --------------------------  ---------  --------  -------")
    for row in artifact["results"]:
        latency = row["latency_seconds"]["median"] if row["latency_seconds"] else None
        latency_text = f"{latency:.5f}" if latency is not None else "null"
        print(
            f"{row['case']:<13}  {row['baseline']:<26}  {row['status']:<9}  "
            f"{latency_text:>8}  {str(row['correctness']['passed']).lower()}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="write machine-readable benchmark results to this JSON path")
    parser.add_argument("--smoke", action="store_true", help="run the torch-only CI smoke workload")
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
    parser.add_argument("--_worker-sequence-length", dest="_worker_sequence_length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-batch-size", dest="_worker_batch_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-warmup", dest="_worker_warmup", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-repetitions", dest="_worker_repetitions", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-peft-path", dest="_worker_peft_path", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-merge-dense", dest="_worker_merge_dense", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args._worker_json_output:
        _real_worker(args)
        return
    sequence_lengths = [4] if args.smoke else _parse_int_list(args.sequence_lengths)
    batch_sizes = [1] if args.smoke else _parse_int_list(args.batch_sizes)
    artifact = collect_results(
        sequence_lengths=sequence_lengths,
        batch_sizes=batch_sizes,
        warmup_repetitions=1 if args.smoke else args.warmup_repetitions,
        measured_repetitions=1 if args.smoke else args.measured_repetitions,
        mode="synthetic-smoke" if args.smoke else "real",
        upstream_peft_path=args.upstream_peft_path,
        fork_peft_path=args.fork_peft_path,
        upstream_peft_ref=args.upstream_peft_ref,
        fork_peft_ref=args.fork_peft_ref,
        checkout_dir=args.checkout_dir,
    )
    _print_table(artifact)
    if args.json_output:
        _write_json(args.json_output, artifact)
        print(f"\nwrote JSON artifact: {args.json_output}")


if __name__ == "__main__":
    main()
