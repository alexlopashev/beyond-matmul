# PEFT Multi-Adapter Dense-Merge Investigation

Issue #110 investigated why `docs/results/peft_multi_adapter_serving.json`
reported correctness failures for every `upstream_peft_merged_dense_cache` and
`upstream_peft_repeated_merge_unmerge` row.

## Root Cause

The benchmark contract is CPU fp32, but the real PEFT worker loaded
`facebook/opt-125m` through Transformers without forcing the base-model dtype.
With the measured Transformers/PEFT stack, that resolved the OPT base weights
to `torch.float16` on CPU while the LoRA adapter factors were `torch.float32`.
PEFT's dense merge paths then materialized adapter deltas into fp16 base
weights, producing logit differences around `1e-2` against the unmerged PEFT
reference and failing the benchmark's fp32 tolerance.

The harness also failed to call `set_adapter(<cached adapter>)` before
`merge_and_unload()` while constructing the dense-cache models. That was a
harness sequencing bug, but it was not sufficient to explain the committed
artifact: after fixing adapter activation alone, a targeted real probe still
failed until the base model was loaded with explicit fp32 dtype.

## Affected Rows

The stale artifact failed exactly 24 rows:

- `upstream_peft_merged_dense_cache` for adapters `merchant` and `gaisb`,
  sequence lengths `16`, `64`, and `128`, and batch sizes `1` and `2`;
- `upstream_peft_repeated_merge_unmerge` for the same two adapters and shape
  grid.

`upstream_peft_unmerged` and `beyond_matmul_factor_provenance` rows passed in
the stale artifact. The Beyond Matmul rows fell back because the worker was not
actually running the CPU fp32 contract.

## Resolution

`benchmarks/peft_multi_adapter_serving.py` now routes all real worker base-model
loads through a shared helper that passes `dtype=torch.float32`, and the
dense-cache worker activates each cached adapter before merging it. Unit tests
pin both behaviors.

The regenerated measured artifact at
`docs/results/peft_multi_adapter_serving.json` has all 48 required rows present,
all correctness checks passing, no negative cases, no fallback cases, and
`summary.benchmark_ready=true`. It still records
`summary.performance_claim="none"` and `summary.memory_or_control_claim="none"`;
passing correctness makes the artifact benchmark-ready, but it does not create
a latency, memory, or adapter-switching win.
