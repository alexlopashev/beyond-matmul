# Benchmark Artifacts

## Evidence Matrix

| Artifact | Regeneration command | Meaning | Boundary |
| --- | --- | --- | --- |
| `docs/results/workload_case_studies.json` | `mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json` | Captured adapter, Conv1d, fixed-mask, and quantized-linear workload provenance, dense fallback comparison, selected lowering, output error, cost proxy, memory proxy, and timing/proxy boundary. | Case-study evidence only; planner cost and memory proxies are recorded, but no benchmark timings or production integer-kernel performance are measured. |
| `docs/results/fixed_weight.json` | `mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json` | Synthetic fixed-weight benchmark rows for structured lowerings versus dense fallback. | Pure-Python latency proxies, not hardware-calibrated production performance. |
| `docs/results/approximation_error_ablation.json` | `mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json` | Deterministic matrix-reconstruction-error versus output-error candidate table. | One bounded synthetic case, not a broad approximation-quality benchmark. |
| `docs/results/planner_contract_ablation.json` | `mise exec -- uv run python benchmarks/planner_contract_ablation.py --json-output docs/results/planner_contract_ablation.json` | Deterministic exactness, bounded-error, reuse, backend-support, and dense fallback planner checks. | Contract coverage only; costs are planner estimates, not runtime measurements. |
| `docs/results/peft_transformers_lora_inference_smoke.json` | `mise exec -- uv run python benchmarks/peft_transformers_lora_inference.py --smoke --json-output docs/results/peft_transformers_lora_inference_smoke.json` | Contract-shaped PEFT plus Transformers LoRA upstream-vs-fork benchmark smoke artifact. | CI smoke uses a tiny torch-only synthetic path for schema, timing, and correctness checks; real PEFT plus Transformers runs require explicit checkouts and optional dependencies on suitable hardware. |

## Fixed-Weight Benchmark

`benchmarks/fixed_weight.py` can emit a machine-readable JSON artifact while
preserving the human-readable smoke table used by local CI. CI uploads
`docs/results/fixed_weight.json` as the workflow artifact
`fixed-weight-benchmark-json`.

Regenerate the fixed-weight benchmark artifact with:

```bash
mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json
```

The JSON schema is versioned with `schema_version: 1`. Each case records:

- case name and selected lowering
- validity, exactness, and relative output error
- planner cost, memory, preprocessing, and requested-call proxies
- dense and selected-lowering seconds per apply
- Python and platform metadata for the run

The controlled case set includes diagonal, sparse, low-rank, codebook, dense,
single-channel valid Conv1d, and multi-channel valid Conv1d operators. Conv1d
rows are expected to select the direct Conv1d lowerings when the benchmark
request and backend contract allow them.

The timings are pure-Python latency proxies for research triage and figure
generation. They are not hardware-calibrated production performance claims.

## Workload Case Studies

`examples/case_study_artifacts.py` reuses the adapter and Conv1d demo logic and
adds deterministic fixed-mask and quantized-linear rows to emit a
machine-readable artifact while the demo scripts keep their human-readable
local output. CI uploads `docs/results/workload_case_studies.json` as the
workflow artifact `workload-case-studies-json`.

Regenerate the workload case-study artifact with:

```bash
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
```

The JSON schema is versioned with `schema_version: 1`. Each case records:

- captured operator name, kind, linear kind, shape, lowerings, and provenance
- provenance notes from the capture pass
- selected lowering with output-relative error against the reference workload
- dense fallback selected lowering and the same planner proxy fields
- cost and memory proxies from the planner
- an explicit timing/proxy boundary stating that the case-study artifact does
  not measure timings

The controlled case set currently includes a merged LoRA-style adapter,
`nn.Conv1d` and functional `F.conv1d` rows, grouped/depthwise Conv1d rows, a
fixed causal-band mask applied as a sparse linear map, and a fixed per-tensor
affine quantized linear module row. These rows are case-study evidence for
provenance preservation and dense fallback comparison; they are not benchmark
timing evidence. The fixed-mask row does not claim full masked attention
support, and the quantized row does not claim production integer-kernel
performance.

## Approximation Error Ablation

`benchmarks/approximation_error_ablation.py` emits a deterministic JSON table
source for the current matrix-reconstruction-error versus output-error
ablation. Regenerate it with:

```bash
mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json
```

CI uploads `docs/results/approximation_error_ablation.json` as
`approximation-error-ablation-json`.

The JSON schema is versioned with `schema_version: 1`. Each candidate row
records:

- candidate kind and parameters
- matrix reconstruction error
- output-relative error on the deterministic sample inputs
- matrix-threshold and output-threshold decisions
- candidate lowering, planner selection flag, and acceptance or rejection
  reason

The controlled case currently covers one dense matrix with a dominant feature
that the sample inputs do not exercise. In this bounded case, low-rank and
sparse top-k candidates pass the matrix-relative threshold but fail the
output-relative threshold; codebook and bitpacked candidates fail both. This is
paper-supporting evidence for output-aware acceptance, not a general benchmark
of approximation quality.

## Planner Contract Ablation

`benchmarks/planner_contract_ablation.py` emits a deterministic JSON table
source for current planner contract behavior. Regenerate it with:

```bash
mise exec -- uv run python benchmarks/planner_contract_ablation.py --json-output docs/results/planner_contract_ablation.json
```

CI uploads `docs/results/planner_contract_ablation.json` as
`planner-contract-ablation-json`.

The JSON schema is versioned with `schema_version: 1`. The scenario rows record:

- selected lowering, output-relative error, and dense fallback validity
- exact-only versus bounded-error planning on the same fixed-weight case
- reuse sensitivity before and at a preprocessing amortization threshold
- backend support rejection for an unsupported specialized lowering

The controlled cases are small deterministic planner checks. They support
claims about contract enforcement and dense fallback availability, but they do
not measure backend runtime or establish broad performance conclusions.

## PEFT Transformers LoRA Inference Smoke

`benchmarks/peft_transformers_lora_inference.py` implements the first
TorchBench-style harness for the PEFT capstone benchmark contract. CI runs the
torch-only smoke path:

```bash
mise exec -- uv run python benchmarks/peft_transformers_lora_inference.py --smoke --json-output docs/results/peft_transformers_lora_inference_smoke.json
```

The smoke artifact uses the schema from
`docs/peft_capstone_benchmark_contract.md`, including warmup, repetitions,
timing summaries, correctness metrics, dependency target metadata, device
metadata, the three required baselines, and summary-level readiness blockers.
It is a contract and CI health check, not external PEFT performance evidence,
so `summary.benchmark_ready` remains false for smoke artifacts even when the
schema and torch-only correctness checks pass.

For manual full runs, provide PEFT checkouts by path or allow the harness to
resolve the configured git refs:

```bash
mise exec -- uv run python benchmarks/peft_transformers_lora_inference.py \
  --upstream-peft-path /path/to/huggingface-peft \
  --fork-peft-path /path/to/alexlopashev-peft \
  --json-output docs/results/peft_transformers_lora_inference.json
```

The default refs are upstream `huggingface/peft@main` and
`alexlopashev/peft@beyond-matmul/provenance-lora-inference`. The manual run
expects compatible `transformers` and `peft` dependencies and remains outside
the local CI dependency set.
