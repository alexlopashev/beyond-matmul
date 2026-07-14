# Hardware-Backed Production Performance Benchmark Contract

> **Roadmap status (2026-07-14): paused.** This PEFT contract remains a valid
> historical design artifact, but it is not the active project finish line.
> Issue #129 reopens target selection around an external LLM tensor contraction.
> Work on issues #123 through #126 should not resume until that decision either
> reselects PEFT or records why this contract is still the highest-leverage path.

This contract defines the first hardware-backed production/performance target
for Beyond Matmul. It is a benchmark design target, not benchmark evidence. It
exists so follow-up implementation work can extend the current CPU process
memory/control instrumentation to CUDA allocator measurements and structured
PEFT serving paths without inventing claim rules after seeing a result.

The target workload is the existing PEFT multi-adapter serving case promoted to
a measured accelerator contract. The primary performance question is:

Can a provenance-preserving structured LoRA serving path reduce already-loaded
adapter switching cost or per-adapter resident memory against dense merged
serving strategies on one pinned CUDA backend while preserving upstream-output
correctness and avoiding material forward-latency regression?

The answer may be negative. Dense GEMM and dense merged serving remain valid
fallbacks and comparison strategies.

## Workload

The benchmark target is prefill-only causal language model inference with the
same public model and adapters as the current multi-adapter correctness
artifact:

- base model: `facebook/opt-125m`
- immutable base model revision:
  `27dcfa74d334bc871f3234de431e71c6eeba5dd6`
- model context limit from `config.json`: `max_position_embeddings=2048`
- library target: `transformers.AutoModelForCausalLM`
- adapter `merchant`: `choyiny/opt-125m-lora-merchant-finetune`
- immutable `merchant` adapter revision:
  `c25d7ba3a15502b4dcbd609758caec8b2ce78eb4`
- adapter `gaisb`: `guyk1971/gaisb`
- immutable `gaisb` adapter revision:
  `cdad7e89c32a940aa1269dddbfcf29e7c9cdda37`
- shared adapter contract: `peft_type=LORA`, `task_type=CAUSAL_LM`, `r=16`,
  `lora_alpha=32`, `fan_in_fan_out=false`, and target modules `q_proj` and
  `v_proj`
- task: one prefill forward pass over `input_ids` and `attention_mask`,
  returning logits
- execution mode: `model.eval()` with `torch.inference_mode()`
- primary device/backend: one NVIDIA CUDA GPU through PyTorch CUDA

The required shape grid is sequence lengths `16`, `64`, and `128` with batch
sizes `1` and `2`. Input generation is deterministic synthetic token IDs with
seed `20260708`, values in `[0, model.config.vocab_size)`, and attention masks
of ones.

The implementation must preflight the resolved model config and fail readiness
with a clear blocker if any selected shape exceeds the model context limit or
if adapter metadata differs from the contract above.

## Dependency And Hardware Pins

The artifact must record exact dependency and hardware metadata before any
performance field is interpreted:

- Beyond Matmul repository revision
- upstream PEFT revision used for baseline rows
- Beyond Matmul PEFT fork revision used for structured rows
- Python, PyTorch, CUDA runtime, CUDA driver, Transformers, PEFT, Accelerate,
  Safetensors, and Hugging Face Hub versions
- GPU model, total memory, compute capability, driver version, CUDA device
  ordinal, MIG partition status if applicable, power-management mode when
  visible, and relevant environment variables
- `torch.backends` flags that affect matmul, TF32, deterministic behavior, and
  cuDNN benchmarking
- `torch.get_num_threads()`, process affinity if set, and host platform

The initial dependency pins are:

- Python `3.14.6`
- PyTorch `2.12.1` with a CUDA-enabled build
- Transformers `5.13.0`
- Hugging Face Hub `1.23.0`
- upstream PEFT repository `huggingface/peft` at
  `1598ecb8fc504bfcb08b9b232b295414a729d7ed`
- Beyond Matmul PEFT fork `alexlopashev/peft` branch
  `beyond-matmul/provenance-lora-inference` at
  `7ac8d57b100846837c5a3b76c65e1e1954ccc3c8`
- Beyond Matmul repository revision for the benchmark harness under test

Accelerate and Safetensors must be pinned by exact installed version in the
future artifact. If the implementation needs to change any model, adapter,
library, fork, or backend pin, it must update this contract in the same PR.
Changed pins create a new comparison cohort; the artifact must keep old and
new cohorts separate instead of pooling results.

## Baselines

Each required adapter and shape must report these baselines:

1. `upstream_peft_unmerged`: stock PEFT with both adapters loaded into one base
   model and the current adapter selected through PEFT's active-adapter
   mechanism.
2. `upstream_peft_merged_dense_cache`: stock PEFT with one dense merged model
   cache per adapter after `merge_and_unload()` or an equivalent supported
   merge path.
3. `upstream_peft_repeated_merge_unmerge`: stock PEFT repeatedly switching the
   active adapter by merge/unmerge or an equivalent supported state transition
   without Hub download or model construction inside the measured loop.
4. `beyond_matmul_structured_low_rank`: the Beyond Matmul PEFT fork selecting a
   structured LoRA factor path when the dtype, device, layout, and correctness
   contracts allow it, with explicit dense fallback metadata when they do not.

Unsupported or not-applicable rows must stay in the artifact with a status and
reason. Dropping a hard row makes the benchmark not ready.

## Timing Protocol

Model and adapter downloads, process startup, tokenizer loading, and base model
construction are excluded from forward-latency and adapter-switch measurements.
They may be reported as setup metadata, but they must not be pooled with steady
state serving measurements.

For each row:

- perform at least `25` warmup forwards before timing forward latency;
- record at least `100` measured forward repetitions;
- perform at least `25` warmup adapter switches before timing switching;
- record at least `100` measured adapter-switch repetitions where switching is
  applicable;
- call `torch.cuda.synchronize()` before starting and after ending each timed
  region;
- use CUDA event timing for GPU elapsed time and record wall-clock timing as
  secondary metadata;
- report median, mean, standard deviation, p50, p90, p95, p99, minimum, and
  maximum seconds.

Forward latency, adapter-switch latency, and preprocessing/materialization time
are separate fields. Dense cache creation, structured-factor packing, graph
capture, compilation, or other one-time preparation must be reported as
preprocessing and cannot be hidden inside amortized forward latency. If a row
reports amortized totals, it must also declare the reuse count.

## Memory Measurement

GPU memory is measured in an isolated process per baseline cohort unless a
follow-up issue documents why isolation is impossible. Each measured row must:

- call `torch.cuda.empty_cache()` before setup where doing so does not change
  the serving strategy being measured;
- reset peak memory stats immediately before the measured setup or steady-state
  region being reported;
- report `torch.cuda.max_memory_allocated()` and
  `torch.cuda.max_memory_reserved()` for setup/preprocessing and steady-state
  serving regions separately;
- record `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
  after setup and after the measured loop;
- distinguish adapter payload bytes, dense merged cache bytes per adapter,
  structured factor bytes, provenance metadata bytes, and dense fallback cache
  bytes;
- mark host RSS, allocator snapshots, or platform tools as optional secondary
  measurements unless the implementation issue makes them required.

Dense model byte estimates are useful metadata, but they are not measured peak
memory. A memory claim requires measured CUDA allocator fields and row-complete
status, not dense byte proxies alone.

## Correctness Gates

Correctness is checked before performance fields are interpreted. The reference
is `upstream_peft_unmerged` for the same adapter, shape, dtype, device, model
revision, adapter revision, and input tensor.

The primary tolerance profile is CUDA fp32:

- `max_abs_error <= 1e-4`
- `relative_l2_error <= 1e-5`
- no NaN or infinite logits

Optional fp16, bf16, TF32, or quantized rows may use separate tolerance
profiles, but they cannot replace the fp32 readiness grid. Any row that fails
correctness must remain in the artifact with status `failed_correctness`, and
its latency, memory, switching, and control fields must be excluded from claim
summaries.

## JSON Artifact Schema

The future benchmark artifact is a single JSON object:

```json
{
  "schema_version": 1,
  "benchmark": "hardware_backed_peft_multi_adapter_serving",
  "contract": "docs/hardware_backed_production_benchmark_contract.md",
  "mode": "real",
  "workload": {
    "base_model": "facebook/opt-125m",
    "base_model_revision": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    "adapters": [
      {
        "name": "merchant",
        "repository": "choyiny/opt-125m-lora-merchant-finetune",
        "revision": "c25d7ba3a15502b4dcbd609758caec8b2ce78eb4"
      },
      {
        "name": "gaisb",
        "repository": "guyk1971/gaisb",
        "revision": "cdad7e89c32a940aa1269dddbfcf29e7c9cdda37"
      }
    ],
    "task": "causal_lm_prefill_logits",
    "dtype": "float32",
    "device": "cuda",
    "sequence_lengths": [16, 64, 128],
    "batch_sizes": [1, 2],
    "input_seed": 20260708,
    "forward_warmup_repetitions": 25,
    "forward_measured_repetitions": 100,
    "switch_warmup_repetitions": 25,
    "switch_measured_repetitions": 100
  },
  "dependencies": {
    "python": "<version>",
    "torch": {"version": "<version>", "cuda": "<version-or-null>"},
    "transformers": {"version": "<version>", "revision": "<sha-or-null>"},
    "peft_upstream": {"version": "<version>", "revision": "<sha>"},
    "peft_fork": {
      "repository": "alexlopashev/peft",
      "revision": "<sha>"
    },
    "beyond_matmul": {
      "repository": "alexlopashev/beyond-matmul",
      "revision": "<sha>"
    }
  },
  "hardware": {
    "gpu": "<model>",
    "cuda_device": 0,
    "compute_capability": "<major.minor>",
    "total_memory_bytes": 0,
    "driver": "<version>",
    "mig_partition": null
  },
  "results": [
    {
      "case": "merchant_seq16_batch1",
      "adapter": "merchant",
      "baseline": "beyond_matmul_structured_low_rank",
      "status": "ok",
      "sequence_length": 16,
      "batch_size": 1,
      "forward_latency_seconds": {
        "median": 0.0,
        "mean": 0.0,
        "stdev": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "min": 0.0,
        "max": 0.0
      },
      "adapter_switch_seconds": {
        "median": 0.0,
        "mean": 0.0,
        "stdev": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "min": 0.0,
        "max": 0.0
      },
      "preprocessing_seconds": {
        "dense_cache_build": null,
        "structured_factor_pack": 0.0,
        "compilation_or_graph_capture": null
      },
      "cuda_memory": {
        "setup_peak_allocated_bytes": 0,
        "setup_peak_reserved_bytes": 0,
        "steady_peak_allocated_bytes": 0,
        "steady_peak_reserved_bytes": 0,
        "post_setup_allocated_bytes": 0,
        "post_loop_allocated_bytes": 0
      },
      "storage": {
        "adapter_payload_bytes": 0,
        "dense_cache_bytes_per_adapter": 0,
        "structured_factor_bytes": 0,
        "provenance_metadata_bytes": 0,
        "dense_fallback_cache_bytes": 0
      },
      "correctness": {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
        "max_abs_tolerance": 0.0001,
        "relative_l2_tolerance": 0.00001,
        "tolerance_profile": "cuda_fp32",
        "passed": true
      },
      "lowering": {
        "kind": "structured_low_rank",
        "dense_fallback_available": true,
        "dense_fallback_used": false,
        "fallback_reason": null
      }
    }
  ],
  "summary": {
    "all_required_cases_present": true,
    "all_correctness_checks_passed": true,
    "all_memory_fields_measured": true,
    "all_switching_cases_present": true,
    "all_fallback_cases_explicit": true,
    "production_contract_ready": true,
    "performance_fields_interpretable": true,
    "memory_fields_interpretable": true,
    "readiness_blockers": [],
    "primary_performance_question": "structured_low_rank_switch_or_memory_without_forward_regression",
    "performance_claim": "none",
    "memory_or_control_claim": "none"
  }
}
```

Rows may add fields, but they must not remove or rename required fields without
a schema-version bump. Smoke artifacts, CPU artifacts, local dry runs, and rows
with missing CUDA memory fields must set `production_contract_ready=false`.
Until the Milestone 2 structured CUDA execution path exists,
`beyond_matmul_structured_low_rank` rows remain present but blocked with
`structured_path_blocked_milestone_2`; artifacts with that blocker must keep
`production_contract_ready=false` even when upstream baseline rows have CUDA
timing and allocator measurements.

## Readiness And Claim Thresholds

The artifact is production-contract ready only when:

- every required adapter, baseline, shape, and dtype row is present;
- the primary CUDA backend metadata is present;
- every non-optional row passes correctness;
- forward-latency, adapter-switch, preprocessing, and memory fields are
  present with measured values or explicit not-applicable statuses;
- all dense fallback use is explicit and justified;
- model loading, Hub download, process startup, and tokenizer work are excluded
  from steady-state timing;
- summary fields state whether latency, memory, and control claims are
  interpretable.

A latency win may be claimed only if the Beyond Matmul structured row improves
median adapter-switch latency or median forward latency by at least `10%`
against the relevant upstream baseline for at least one required shape, passes
correctness everywhere, and does not regress median forward latency by more
than `5%` on any required shape.

A memory or control win may be claimed only if correctness passes everywhere
and at least one of these is true:

- measured steady-state CUDA allocated bytes are at least `10%` lower than the
  dense merged cache baseline for the same adapter and shape;
- measured adapter-switch median latency is at least `10%` lower than repeated
  merge/unmerge for already-loaded adapters;
- structured rows expose active adapter identity, factor identity, and dense
  fallback status without duplicating dense merged model copies.

If these thresholds are not met, the result remains useful as negative evidence
and must keep `performance_claim` and `memory_or_control_claim` at `none`.

## Claim Boundary

This contract is future work. It defines what would be required to interpret a
hardware-backed PEFT performance artifact, but it is not itself evidence for
GPU speedups, memory savings, production kernels, upstream PEFT readiness,
generation throughput, KV-cache behavior, training, broad adapter coverage, or
universal Transformer acceleration.

The existing CPU PEFT artifacts remain correctness and provenance evidence.
Follow-up issues may extend the current CPU process instrumentation and CPU
fp32 structured execution path to CUDA-backed production-kernel measurements,
but no whitepaper claim should move from future work to current evidence until
a generated artifact satisfies the readiness gates above.
