# PEFT Capstone Benchmark Contract

This contract defines the first external PEFT plus Transformers benchmark for
the Beyond Matmul capstone. It is a benchmark design target, not benchmark
evidence. Downstream implementation issues should follow this target instead of
choosing a different model, adapter, input grid, baseline set, or artifact
shape.

## Workload

The first workload is prefill-only causal language model inference with a fixed
LoRA adapter:

- base model: `hf-internal-testing/tiny-random-OPTForCausalLM`
- adapter: `peft-internal-testing/tiny-OPTForCausalLM-lora`
- task: single forward pass over `input_ids` and `attention_mask`, returning
  logits
- execution mode: `model.eval()` with `torch.inference_mode()`
- primary dtype and device: `torch.float32` on CPU
- required shape grid: sequence lengths `16`, `64`, and `128`; batch sizes `1`
  and `4`
- input generation: deterministic synthetic token IDs, generated with seed
  `20260707`, values in `[0, model.config.vocab_size)`, and attention masks of
  ones
- required warmup and repetition count: `10` warmup forwards and `50` measured
  repetitions per baseline and shape

The artifact must record the resolved model and adapter revisions. Floating
tags or branch names may be used for local development only; benchmark artifacts
must include immutable commit SHAs or revision identifiers for the base model,
adapter, Transformers dependency, PEFT dependency, Beyond Matmul commit, and
the PEFT fork commit when the fork path is measured.

## Baselines

Each shape in the required grid must report these baselines:

1. `upstream_peft_unmerged`: stock `huggingface/peft` with the adapter active
   and unmerged.
2. `upstream_peft_merged_dense`: stock PEFT after `merge_and_unload()` or the
   equivalent supported merge path. If PEFT cannot merge this adapter, the row
   must be present with status `not_applicable` and an explanatory reason.
3. `beyond_matmul_peft_fork`: `alexlopashev/peft` on branch
   `beyond-matmul/provenance-lora-inference`, preserving LoRA factor
   provenance for planning while retaining a dense fallback path.

The unmerged upstream PEFT row is the correctness reference. The merged dense
row is the dense-materialized baseline, not a failure case. The fork row must
report which lowering it used and whether dense fallback was available for the
same input and adapter.

## Correctness

Correctness is checked against `upstream_peft_unmerged` logits for the same
shape, dtype, device, model revision, adapter revision, and input tensor. The
required CPU fp32 tolerance is:

- `max_abs_error <= 1e-4`
- `relative_l2_error <= 1e-5`
- no NaN or infinite logits

If an optional GPU or reduced-precision run is added later, it must use a
separate tolerance profile and cannot replace the CPU fp32 contract. Any row
that fails correctness remains in the artifact with status `failed_correctness`;
it must not be dropped from summaries.

## Measurements

Each measured row must include:

- latency in seconds per forward pass: median, mean, standard deviation, p50,
  p90, p95, and p99
- peak memory when measurable on the device, otherwise `null` with reason
- adapter-switching cost when measurable, otherwise `null` with reason
- output-equivalence metrics from the correctness section
- whether the run used an unmerged adapter, merged dense weights, a
  provenance-preserving fork lowering, or dense fallback
- environment metadata: operating system, Python version, PyTorch version,
  Transformers version, PEFT version, CPU model, accelerator model if any,
  thread settings, and relevant environment variables

Adapter switching is measurable only when the implementation can load two
named adapters for the same base model and time `set_adapter()` or the
equivalent transition without reloading the base model. If only one adapter is
loaded, record `adapter_switch_seconds: null` and
`adapter_switch_status: "not_measured_single_adapter"`.

## JSON Artifact Schema

The benchmark artifact is a single JSON object:

```json
{
  "schema_version": 1,
  "benchmark": "peft_transformers_lora_inference",
  "contract": "docs/peft_capstone_benchmark_contract.md",
  "workload": {
    "base_model": "hf-internal-testing/tiny-random-OPTForCausalLM",
    "base_model_revision": "<immutable revision>",
    "adapter": "peft-internal-testing/tiny-OPTForCausalLM-lora",
    "adapter_revision": "<immutable revision>",
    "task": "causal_lm_prefill_logits",
    "dtype": "float32",
    "device": "cpu",
    "sequence_lengths": [16, 64, 128],
    "batch_sizes": [1, 4],
    "input_seed": 20260707,
    "warmup_repetitions": 10,
    "measured_repetitions": 50
  },
  "dependencies": {
    "python": "<version>",
    "torch": "<version>",
    "transformers": {"version": "<version>", "revision": "<sha-or-null>"},
    "peft_upstream": {"version": "<version>", "revision": "<sha>"},
    "peft_fork": {"repository": "alexlopashev/peft", "revision": "<sha-or-null>"},
    "beyond_matmul": {"repository": "alexlopashev/beyond-matmul", "revision": "<sha>"}
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
      "case": "seq16_batch1",
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
      "peak_memory_bytes": null,
      "peak_memory_status": "not_measurable_on_cpu",
      "adapter_switch_seconds": null,
      "adapter_switch_status": "not_measured_single_adapter",
      "correctness": {
        "reference_baseline": "upstream_peft_unmerged",
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
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
    "performance_claim": "none"
  }
}
```

Rows may add fields, but they must not remove or rename the required fields
above without a schema-version bump. A row that is unsupported, not applicable,
or failed must stay in `results` with `status`, measured fields set to `null`
where appropriate, and a human-readable `reason`.

## Success Thresholds

The first implementation succeeds as a benchmark artifact if:

- every required baseline and shape row is present;
- the JSON follows the schema above;
- dependency revisions and hardware metadata are present;
- all non-`not_applicable` rows pass the correctness tolerance;
- the fork row explicitly records dense fallback availability.

A performance claim requires more than successful execution. The capstone may
claim a measured win only if the fork improves median latency or peak memory by
at least `10%` against both upstream baselines for at least one required shape,
does not regress median latency by more than `5%` on any other required shape,
and passes correctness everywhere. Adapter-switching wins require at least a
`10%` measured improvement against an upstream switching baseline. If these
thresholds are not met, the result is still useful and must be reported as a
negative or neutral result.

## Negative-Result Reporting

Negative results must be explicit. Reports should state whether the fork is
slower, memory-neutral, unable to measure adapter switching, blocked by missing
PEFT functionality, or equivalent to dense fallback. Summaries must include the
full required grid and may not cherry-pick only favorable shapes.

## Inference Boundary

This benchmark covers fixed-weight inference only. It excludes training,
fine-tuning, gradient checkpointing, optimizer state, adapter rank search,
adapter composition during training, quantization-aware training, generation
quality, and perplexity. It also excludes dynamic decoding loops, KV-cache
management, speculative decoding, attention-mask semantics, and GPU kernel
implementation claims for the first benchmark.

Dense fallback is preserved as a valid path throughout the benchmark. The
project claim is not that dense GEMM is invalid; it is that retaining adapter
provenance can expose additional fixed-weight inference choices while keeping
dense materialization available for correctness, portability, and negative
results.
