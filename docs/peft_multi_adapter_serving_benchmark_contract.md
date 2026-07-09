# PEFT Multi-Adapter Serving Benchmark Contract

This contract defines the second external PEFT plus Transformers proof for
Beyond Matmul. It is a benchmark design target, not benchmark evidence. The
follow-up implementation issue should use this base model, adapter set,
baseline set, grid, and artifact shape unless a material blocker is recorded on
that issue.

The proof asks whether retaining LoRA factor provenance helps a fixed-weight
serving workload that switches among multiple adapters. It does not assume that
a provenance-preserving path is faster for a single adapter forward pass.

## Workload

The benchmark target is prefill-only causal language model inference with one
public OPT base model and two public LoRA adapters that share the same base
model and attention-projection target modules:

- base model: `facebook/opt-125m`
- immutable base model revision:
  `27dcfa74d334bc871f3234de431e71c6eeba5dd6`
- model context limit from `config.json`: `max_position_embeddings=2048`
- library target: `transformers.AutoModelForCausalLM`
- adapter `merchant`: `choyiny/opt-125m-lora-merchant-finetune`
- immutable `merchant` adapter revision:
  `c25d7ba3a15502b4dcbd609758caec8b2ce78eb4`
- `merchant` adapter payload: `adapter_model.safetensors`, observed
  `2365968` bytes at contract time
- adapter `gaisb`: `guyk1971/gaisb`
- immutable `gaisb` adapter revision:
  `cdad7e89c32a940aa1269dddbfcf29e7c9cdda37`
- `gaisb` adapter payload: `adapter_model.bin`, observed `2376641` bytes at
  contract time
- shared adapter config: `peft_type=LORA`, `task_type=CAUSAL_LM`, `r=16`,
  `lora_alpha=32`, `fan_in_fan_out=false`, and target modules `q_proj` and
  `v_proj`
- task: one forward pass over `input_ids` and `attention_mask`, returning
  logits
- execution mode: `model.eval()` with `torch.inference_mode()`
- primary dtype and device: `torch.float32` on CPU

The required CPU fp32 grid is sequence lengths `16`, `64`, and `128` with batch
sizes `1` and `2`. All sequence lengths are below the model-native 2048-token
context limit. The implementation must still preflight the resolved model
config and fail readiness with a clear context-limit blocker if the model
revision changes or if tokenization metadata imposes a tighter effective limit.

Input generation is deterministic synthetic token IDs with seed `20260708`,
values in `[0, model.config.vocab_size)`, and attention masks of ones. The
required timing protocol is `10` warmup forwards and `50` measured repetitions
per adapter, baseline, and shape.

The real worker must force the pinned base model to `torch.float32` when loading
Transformers weights. Relying on auto dtype selection can silently materialize
the OPT base as fp16 on CPU, which makes PEFT dense merge paths fail this
contract's fp32 correctness tolerance.

## Regeneration Command

The implementation issue should add the harness at
`benchmarks/peft_multi_adapter_serving.py` and regenerate the real artifact
with:

```bash
mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_multi_adapter_serving.py --json-output docs/results/peft_multi_adapter_serving.json
```

The harness may provide a small smoke mode for CI, but the smoke artifact must
set `summary.benchmark_ready=false` and `summary.performance_claim="none"`
unless it uses the real pinned model, both pinned adapters, the required grid,
and all required baselines below.

## Baselines

Each required shape and adapter must report these baselines:

1. `upstream_peft_unmerged`: stock `huggingface/peft` with both adapters
   loaded into one base model and the current adapter selected with
   `set_adapter()` or PEFT's equivalent active-adapter mechanism.
2. `upstream_peft_merged_dense_cache`: stock PEFT with one loaded base model
   per adapter after `merge_and_unload()` or the equivalent supported merge
   path. This baseline represents serving from a dense merged cache; its
   per-adapter storage and resident model footprint must include the duplicated
   dense base weights attributable to each cached adapter.
3. `upstream_peft_repeated_merge_unmerge`: stock PEFT repeatedly switching the
   active adapter by merging the requested adapter and unmerging or restoring
   the prior state without reloading model weights. If the installed PEFT
   revision cannot perform this transition safely, every row must remain
   present with status `not_applicable` and a reason.
4. `beyond_matmul_factor_provenance`: `alexlopashev/peft` on branch
   `beyond-matmul/provenance-lora-inference`, preserving LoRA factor
   provenance for the active adapter while retaining an explicit dense fallback
   path.

The comparison must not time unrelated model-loading overhead as adapter
switching. Load the base model and adapter artifacts before the measured
switching loop. Adapter-switch measurements may include PEFT's active-adapter
selection, merge/unmerge work, dense-cache pointer change, provenance metadata
selection, and any required synchronization, but not Hub download, tokenizer
load, base model construction, or process startup.

The dense-cache baseline is not a failure case. It is the serving strategy that
materializes one dense merged model per adapter. The factor-provenance row must
report whether dense fallback is available and whether it was used.

## Correctness

Correctness is checked against `upstream_peft_unmerged` for the same adapter,
shape, dtype, device, model revision, adapter revision, and input tensor. The
required CPU fp32 tolerance is:

- `max_abs_error <= 1e-4`
- `relative_l2_error <= 1e-5`
- no NaN or infinite logits

Rows that fail correctness remain in the artifact with status
`failed_correctness`; they must not be dropped from summaries. If an optional
GPU or reduced-precision run is added later, it must use a separate tolerance
profile and cannot replace the CPU fp32 contract.

## Measurements

Each measured row must include:

- latency in seconds per forward pass: median, mean, standard deviation, p50,
  p90, p95, and p99
- adapter-switch cost in seconds for the transition into the row's adapter:
  median, mean, standard deviation, p50, p90, p95, and p99, or `null` with a
  reason for rows where switching is not applicable
- peak memory when measurable on the device, otherwise `null` with reason
- per-adapter storage bytes for the adapter payload and config
- per-adapter resident bytes attributable to dense merged model copies,
  unmerged factors, provenance metadata, or dense fallback caches
- output-equivalence metrics from the correctness section
- whether the run used unmerged adapter factors, a dense merged cache,
  repeated merge/unmerge, a provenance-preserving factor path, or dense
  fallback
- dependency and environment metadata: operating system, Python version,
  PyTorch version, Transformers version, PEFT version, Hugging Face Hub
  revision, CPU model, accelerator model if any, thread settings, and relevant
  environment variables

Memory and storage summaries must distinguish adapter payload storage from
resident serving footprint. A dense merged cache should account for one dense
base-model copy per cached adapter, even if the process allocator makes peak
memory difficult to observe directly on CPU.

## JSON Artifact Schema

The benchmark artifact is a single JSON object:

```json
{
  "schema_version": 1,
  "benchmark": "peft_multi_adapter_serving",
  "contract": "docs/peft_multi_adapter_serving_benchmark_contract.md",
  "workload": {
    "base_model": "facebook/opt-125m",
    "base_model_revision": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    "model_context_limit": 2048,
    "adapters": [
      {
        "name": "merchant",
        "repository": "choyiny/opt-125m-lora-merchant-finetune",
        "revision": "c25d7ba3a15502b4dcbd609758caec8b2ce78eb4",
        "payload_file": "adapter_model.safetensors",
        "payload_bytes": 2365968
      },
      {
        "name": "gaisb",
        "repository": "guyk1971/gaisb",
        "revision": "cdad7e89c32a940aa1269dddbfcf29e7c9cdda37",
        "payload_file": "adapter_model.bin",
        "payload_bytes": 2376641
      }
    ],
    "task": "causal_lm_prefill_logits",
    "dtype": "float32",
    "device": "cpu",
    "sequence_lengths": [16, 64, 128],
    "batch_sizes": [1, 2],
    "input_seed": 20260708,
    "warmup_repetitions": 10,
    "measured_repetitions": 50
  },
  "dependencies": {
    "python": "<version>",
    "torch": "<version>",
    "transformers": {"version": "<version>", "revision": "<sha-or-null>"},
    "peft_upstream": {"version": "<version>", "revision": "<sha>"},
    "peft_fork": {
      "repository": "alexlopashev/peft",
      "revision": "<sha-or-null>"
    },
    "huggingface_hub": {"version": "<version-or-null>"},
    "beyond_matmul": {
      "repository": "alexlopashev/beyond-matmul",
      "revision": "<sha>"
    }
  },
  "environment": {
    "platform": "<platform string>",
    "cpu": "<cpu model>",
    "accelerator": null,
    "torch_num_threads": "<int-or-null>",
    "env": {"<name>": "<value>"}
  },
  "results": [
    {
      "case": "merchant_seq16_batch1",
      "adapter": "merchant",
      "baseline": "upstream_peft_unmerged",
      "status": "ok",
      "sequence_length": 16,
      "batch_size": 1,
      "latency_seconds": {
        "median": 0.0,
        "mean": 0.0,
        "stdev": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0
      },
      "adapter_switch_seconds": {
        "median": 0.0,
        "mean": 0.0,
        "stdev": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0
      },
      "adapter_switch_status": "measured_loaded_adapters",
      "peak_memory_bytes": null,
      "peak_memory_status": "not_measurable_on_cpu",
      "storage": {
        "adapter_payload_bytes": 2365968,
        "adapter_config_bytes": "<int-or-null>",
        "dense_cache_bytes_per_adapter": 0,
        "resident_adapter_bytes": "<int-or-null>"
      },
      "correctness": {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
        "max_abs_tolerance": 0.0001,
        "relative_l2_tolerance": 0.00001,
        "tolerance_profile": "cpu_fp32",
        "passed": true
      },
      "lowering": {
        "kind": "peft_unmerged_adapter",
        "dense_fallback_available": true,
        "dense_fallback_used": false
      }
    }
  ],
  "summary": {
    "all_required_cases_present": true,
    "all_correctness_checks_passed": true,
    "all_switching_cases_present": true,
    "all_dense_fallback_cases_explicit": true,
    "benchmark_ready": true,
    "readiness_blockers": [],
    "max_abs_error": 0.0,
    "max_relative_l2_error": 0.0,
    "fallback_cases": [],
    "negative_cases": [],
    "memory_or_control_claim": "none",
    "performance_claim": "none"
  }
}
```

Rows may add fields, but they must not remove or rename the required fields
above without a schema-version bump. Unsupported, skipped, not-applicable, or
failed rows must stay in `results` with a status, measured fields set to
`null` where appropriate, and a human-readable reason.

## Success Thresholds

The first implementation succeeds as a benchmark artifact if:

- every required adapter, baseline, and shape row is present;
- the JSON follows the schema above;
- model, adapter, dependency, and hardware metadata are present;
- the resolved model context limit is recorded and every required shape is
  preflighted within that limit;
- all non-`not_applicable` rows pass the correctness tolerance;
- dense fallback availability is explicit for every provenance-preserving row;
- adapter-switch cost excludes model-loading, Hub-download, and process-startup
  overhead;
- memory and storage fields distinguish adapter payloads from dense merged
  cache footprint.

The artifact must also report whether it is ready to support benchmark
comparisons. `summary.benchmark_ready` is true only for real PEFT runs, not
synthetic smoke artifacts, and only when the required grid is present,
correctness checks pass, context preflight passes, and any fallback or
not-applicable cases are explicit.

A latency claim requires more than successful execution. The artifact may claim
a measured latency win only if `beyond_matmul_factor_provenance` improves
median forward latency or adapter-switch latency by at least `10%` against the
applicable upstream baseline for at least one required shape, does not regress
median forward latency by more than `5%` on any other required shape, and
passes correctness everywhere.

A memory or control win may be claimed even if single-adapter forward latency is
neutral, but only if the artifact shows at least one of:

- per-adapter resident bytes at least `10%` lower than
  `upstream_peft_merged_dense_cache` while correctness passes;
- adapter-switch median latency at least `10%` lower than
  `upstream_peft_repeated_merge_unmerge` for the same already-loaded adapters;
- explicit row-level control metadata showing the active LoRA factor identity
  and dense fallback status for both adapters without duplicating dense merged
  model copies.

If these thresholds are not met, the result is still useful and must be
reported as negative or neutral.

## Claim Boundary

This benchmark covers fixed-weight inference for two pinned LoRA adapters on
one pinned OPT causal-LM base model. It is serving-strategy evidence for
already-available adapters, not a training, fine-tuning, adapter-search,
quality, perplexity, RLHF, generation-loop, KV-cache, batching, GPU-kernel, or
upstreaming claim.

Dense merged serving remains a valid fallback and comparison strategy. The
project claim is not that dense GEMM is invalid; it is that retaining adapter
factor provenance can expose memory, switching, and control tradeoffs while
still making dense materialization available for correctness, portability, and
negative results.
