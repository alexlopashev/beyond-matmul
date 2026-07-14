# Benchmark Artifacts

## Evidence Matrix

| Artifact | Regeneration command | Meaning | Boundary |
| --- | --- | --- | --- |
| `docs/results/workload_case_studies.json` | `mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json` | Captured adapter, Conv1d, fixed-mask, and quantized-linear workload provenance, dense fallback comparison, selected lowering, output error, cost proxy, memory proxy, and timing/proxy boundary. | Case-study evidence only; planner cost and memory proxies are recorded, but no benchmark timings or production integer-kernel performance are measured. |
| `docs/results/fixed_weight.json` | `mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json` | Synthetic fixed-weight benchmark rows for structured lowerings versus dense fallback. | Pure-Python latency proxies, not hardware-calibrated production performance. |
| `docs/results/approximation_error_ablation.json` | `mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json` | Deterministic matrix-reconstruction-error versus output-error candidate table. | One bounded synthetic case, not a broad approximation-quality benchmark. |
| `docs/results/planner_contract_ablation.json` | `mise exec -- uv run python benchmarks/planner_contract_ablation.py --json-output docs/results/planner_contract_ablation.json` | Deterministic exactness, bounded-error, reuse, backend-support, and dense fallback planner checks. | Contract coverage only; costs are planner estimates, not runtime measurements. |
| `docs/results/peft_transformers_lora_inference_smoke.json` | `mise exec -- uv run python benchmarks/peft_transformers_lora_inference.py --smoke --json-output docs/results/peft_transformers_lora_inference_smoke.json` | Contract-shaped PEFT plus Transformers LoRA upstream-vs-fork benchmark smoke artifact. | CI smoke uses a tiny torch-only synthetic path for schema, timing, and correctness checks; real PEFT plus Transformers runs require explicit checkouts and optional dependencies on suitable hardware. |
| `docs/results/peft_transformers_lora_inference.json` | `mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_transformers_lora_inference.py --json-output docs/results/peft_transformers_lora_inference.json` | Real PEFT plus Transformers LoRA upstream-vs-fork capstone matrix for the contract-defined model, adapter, shapes, and baselines. | Measured local CPU run, not CI smoke. The refreshed committed run uses valid seq16, seq64, and seq100 rows; all required correctness checks pass and `summary.benchmark_ready` is true, but `summary.performance_claim` remains `none`. |
| `docs/results/peft_multi_adapter_serving_smoke.json` | `mise exec -- uv run python benchmarks/peft_multi_adapter_serving.py --smoke --json-output docs/results/peft_multi_adapter_serving_smoke.json` | Contract-shaped PEFT multi-adapter serving smoke artifact for schema, switching metadata, fallback metadata, and correctness summaries. | CI smoke uses a tiny torch-only synthetic path; it is not external PEFT performance evidence and keeps `summary.benchmark_ready=false`. |
| `docs/results/peft_multi_adapter_serving.json` | `mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_multi_adapter_serving.py --json-output docs/results/peft_multi_adapter_serving.json` | Real PEFT multi-adapter serving matrix for the contract-defined OPT-125M base model, two LoRA adapters, switching baselines, dense merged cache, and provenance-preserving factor path. | Measured local CPU run, not CI smoke. The committed run is benchmark-ready correctness evidence: all required rows are present, all baselines pass correctness, 12 Beyond Matmul rows report `execution_path=structured_low_rank`, and no latency, memory, or control win is claimed. |
| `docs/results/live_conv1d_whisper.json` | `mise exec -- uv run --with transformers --with librosa --with soundfile --with safetensors --with huggingface_hub python benchmarks/live_conv1d_whisper.py --json-output docs/results/live_conv1d_whisper.json` | Real Whisper encoder Conv1d dense-vs-direct layer benchmark for the contract-defined model revision, audio trace, prefixes, and exact dense Toeplitz fallback. | Measured local CPU layer run, not CI smoke or end-to-end ASR. Correctness passes for all required rows; dense matrix byte counts expose materialized fallback footprint, not measured peak memory. The dense materialized fallback is slower on this run and `summary.performance_claim` is `none`. |
| `docs/results/olmoe_stock_baseline_smoke.json` | `mise exec -- uv run python benchmarks/olmoe_stock_baseline.py --smoke --json-output docs/results/olmoe_stock_baseline_smoke.json` | Contract-shaped OLMoE stock-backend inventory covering every required prefill/decode regime and explicit compilation exclusion. | CI smoke performs no OLMoE inference, keeps every measurement empty, and records `cohort_complete=false`, `target_decision_ready=false`, and `performance_claim=none`. |
| `docs/results/olmoe_stock_baseline.json` | See the pinned CUDA command below. | Future hardware-pinned stock-only OLMoE cohort used to choose the best successful upstream configuration per regime. | Not yet generated or committed. Even a row-complete stock cohort is target-validation input, not a Beyond Matmul speedup or an accept decision by itself. |

## OLMoE Stock-Baseline Harness

`benchmarks/olmoe_stock_baseline.py` implements the baseline-only measurement
surface for issues #132 and #133 under the gate in
`docs/olmoe_tensor_contraction_capstone.md`. It pins
`allenai/OLMoE-1B-7B-0924` at
`bd1c52f59153f724c1ad11ca1791edc77bab3806` and Transformers at
`a6895655b289cc3fdd29afec36904e0b8545ef92`.

The required grid has four BF16 full-model prefill regimes—batch 1 and 4 at
sequence lengths 128 and 512—and four one-token decode regimes—batch 1 and 8
after prompt lengths 128 and 512. Prompt prefill is setup for decode and is not
included in the per-token timed region. A stock grouped/default decode row uses
grouped prompt prefill and the same grouped-to-batched stage switch that
Transformers generation applies; both effective backends are recorded. Inputs
are deterministic token IDs. Full-model prefill uses `use_cache=true`; KV-cache
construction is part of the prefill timed region. Correctness compares
last-token logits with stock uncompiled eager using fixed maximum-absolute
`0.125` and relative-L2 `0.01` tolerances.

For every regime, the inventory contains stock default, eager, `batched_mm`,
`grouped_mm`, `deepgemm`, and `sonicmoe`. It also contains the runtime-supported
`torch.compile` modes for default, eager, and batched execution, the audited
`grouped_mm` modes, and explicit excluded rows for compilation modes or external
kernels that do not apply. GPU UUID, device properties, NVIDIA driver, CUDA
runtime, pinned Transformers source revision, model revision, compilation mode,
and external-kernel dependencies are checked or recorded rather than silently
skipped. Real mode refuses a reduced compile-mode list, so a caller cannot
silently turn a partial search into a complete cohort.

Run the schema-only CI smoke with:

```bash
mise exec -- uv run python benchmarks/olmoe_stock_baseline.py --smoke --json-output docs/results/olmoe_stock_baseline_smoke.json
```

On the CUDA machine selected by issue #133, run the real stock cohort with the
pinned source and external-kernel dependencies:

```bash
mise exec -- uv run \
  --with 'transformers @ git+https://github.com/huggingface/transformers.git@a6895655b289cc3fdd29afec36904e0b8545ef92' \
  --with accelerate==1.14.0 \
  --with safetensors==0.8.0 \
  --with huggingface-hub==1.23.0 \
  --with kernels==0.15.2 \
  --with nvidia-cutlass-dsl==4.6.0 \
  python benchmarks/olmoe_stock_baseline.py \
  --real \
  --warmup-repetitions 5 \
  --measured-repetitions 20 \
  --json-output docs/results/olmoe_stock_baseline.json
```

Each successful row records eager-relative correctness, CUDA-event and wall
samples and medians, token throughput, setup/warmup timing, decode prompt
prefill timing, CUDA allocator state, and the resolved experts backend. Decode
allocator peaks reset only after prompt prefill and backend switching, record
the resident KV-cache baseline, and exclude prompt setup from the one-token
peak. DeepGEMM preflight separately records the `nvcc` path and full CUDA
toolkit version because the pinned Transformers integration has no NVRTC
fallback; Hopper requires toolkit 12.3 or newer and Blackwell requires 12.9 or
newer. Routing overhead remains explicitly
`requires_profiled_target_validation` until issue #133 adds the profiler
attribution required for the accept-or-reject decision. Failed, blocked,
hardware-inapplicable, and contract-excluded rows remain in the artifact with
reasons.

The harness contains no candidate implementation or candidate field. A smoke
artifact is not performance evidence, and a future real artifact may set
`cohort_complete=true` only after every applicable stock configuration has an
explicit terminal attempt and every regime has at least one correct successful
stock row. Interpretable upstream configuration failures remain as warnings and
are never eligible for best-stock selection; blocked, missing, or nonterminal
attempts keep the cohort incomplete. The artifact always keeps
`target_decision_ready=false` and `performance_claim=none`; issue #133 owns
profiling, best-successful-stock interpretation, and the binary OLMoE
accept-or-reject decision.

## Live Conv1d Whisper Dense-vs-Direct Benchmark

`benchmarks/live_conv1d_whisper.py` implements the benchmark target defined by
`docs/live_conv1d_benchmark_contract.md`. It loads `openai/whisper-tiny` at
revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, selects
`model.encoder.conv1`, verifies the Hugging Face Whisper widget LibriSpeech
sample SHA-256, and compares direct `torch.nn.Conv1d` with an exact dense
materialized Toeplitz fallback for prefixes of 8, 16, and 32 log-Mel frames.

Regenerate the measured artifact with:

```bash
mise exec -- uv run --with transformers --with librosa --with soundfile --with safetensors --with huggingface_hub python benchmarks/live_conv1d_whisper.py --json-output docs/results/live_conv1d_whisper.json
```

The committed run records mode `real`, generated time `2026-07-08T10:12:00Z`,
macOS `26.5.1` on arm64 CPU, Python `3.14.6`, PyTorch `2.12.1`,
Transformers `5.13.0`, and Hugging Face Hub `1.22.0`.

The artifact includes all required rows. Correctness passes for every dense
fallback row with maximum absolute error `6.198883e-06` and maximum relative
L2 error `3.84896e-07`, within the CPU fp32 contract tolerances. The dense
materialized matrices have 1,966,080, 7,864,320, and 31,457,280 float32 entries
for the 8-, 16-, and 32-frame prefixes respectively, or 7,864,320,
31,457,280, and 125,829,120 float32 bytes. These bytes are the explicit dense
matrix footprint that a Toeplitz fallback must materialize for the layer, not
measured process peak memory; each row keeps `peak_memory_status` at
`not_measured`.

No performance win is claimed. In this measured CPU run, direct Conv1d median
latency is faster than dense application for every required prefix, and dense
matrix construction takes measurable one-time time per shape. The artifact
therefore keeps `summary.performance_claim` set to `none`. This supports only
the layer-level claim that preserved Conv1d provenance exposes the direct
channel-aware path while retaining exact dense fallback evidence; it does not
support decoder generation, word-error-rate, streaming ASR, GPU, Conv2d,
quantized convolution, or broader CNN-block performance claims.

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

## PEFT Transformers LoRA Inference Measured Run

`docs/results/peft_transformers_lora_inference.json` is the measured artifact
for issue #80, refreshed by issue #109 to replace invalid seq128 rows with
seq100. It was regenerated with:

```bash
mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_transformers_lora_inference.py --json-output docs/results/peft_transformers_lora_inference.json
```

The committed run records mode `real`, generated time
`2026-07-09T14:06:28Z`, macOS `26.5.1` on arm64 CPU, Python `3.14.6`, PyTorch
`2.12.1`, Transformers `5.13.0`, base model revision
`0abea37ca0a786ba455967e799b7b3d67f86541f`, adapter revision
`14e64b8ba522284138bfc22e76002ab6c0ce31e2`, upstream PEFT revision
`1598ecb8fc504bfcb08b9b232b295414a729d7ed`, fork PEFT revision
`7ac8d57b100846837c5a3b76c65e1e1954ccc3c8`, and Beyond Matmul revision
`7aa1568c51815ce89850b8dd630b6d57b2290602`.

The artifact includes all required baseline and shape rows. Correctness passes
for seq16, seq64, and seq100 rows across upstream unmerged, upstream merged
dense, and the Beyond Matmul PEFT fork. `summary.benchmark_ready` is true with
no readiness blockers.

No performance win is claimed. The fork does not satisfy the capstone
performance threshold across the required grid, CPU peak memory remains
unmeasured, and adapter switching remains unmeasured for the single-adapter
workload; the JSON records `adapter_switch_status:
"not_measured_single_adapter"` and `peak_memory_status:
"not_measurable_on_cpu"` row by row.

## PEFT Multi-Adapter Serving Smoke

`benchmarks/peft_multi_adapter_serving.py` implements the PEFT serving
benchmark target defined by
`docs/peft_multi_adapter_serving_benchmark_contract.md`. CI runs the torch-only
smoke path:

```bash
mise exec -- uv run python benchmarks/peft_multi_adapter_serving.py --smoke --json-output docs/results/peft_multi_adapter_serving_smoke.json
```

The smoke artifact exercises the schema, adapter list, required baselines,
switching metadata, storage metadata, memory/control readiness fields,
correctness summaries, structured-low-rank execution-path reporting, and
fallback reporting without loading external PEFT or Transformers dependencies.
It is a contract health check only:
`summary.benchmark_ready=false`, `summary.memory_control_claim_ready=false`,
and row-level `peak_memory_status="not_measured_synthetic_smoke"` keep smoke
memory fields unavailable.

## PEFT Multi-Adapter Serving Measured Run

`docs/results/peft_multi_adapter_serving.json` is the measured artifact for
issue #98. The contract selects
`facebook/opt-125m` at revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`
with two public LoRA adapters,
`choyiny/opt-125m-lora-merchant-finetune` at
`c25d7ba3a15502b4dcbd609758caec8b2ce78eb4` and `guyk1971/gaisb` at
`cdad7e89c32a940aa1269dddbfcf29e7c9cdda37`. The required grid uses sequence
lengths 16, 64, and 128 with batch sizes 1 and 2, inside the model's
2048-token context limit.

Regenerate the real artifact with:

```bash
mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_multi_adapter_serving.py --json-output docs/results/peft_multi_adapter_serving.json
```

The committed run records mode `real`, generated time
`2026-07-09T16:54:16Z`, macOS `26.5.1` on arm64 CPU, Python `3.14.6`, PyTorch
`2.12.1`, Transformers `5.13.0`, Hugging Face Hub `1.23.0`, upstream PEFT
revision `1598ecb8fc504bfcb08b9b232b295414a729d7ed`, fork PEFT revision
`7ac8d57b100846837c5a3b76c65e1e1954ccc3c8`, and the required 10 warmup and
50 measured repetitions.

The artifact includes all 48 required adapter, shape, and baseline rows. The
`upstream_peft_unmerged`, `upstream_peft_merged_dense_cache`,
`upstream_peft_repeated_merge_unmerge`, and
`beyond_matmul_factor_provenance` rows pass correctness for both adapters and
all shapes. The 12 `beyond_matmul_factor_provenance` rows report
`lowering.execution_path="structured_low_rank"` and appear in
`summary.structured_low_rank_cases`; rows that violate correctness or the CPU
fp32 structured-event contract would instead report
`lowering.execution_path="dense_fallback"` with explicit fallback reasons. The
artifact is benchmark-ready correctness evidence with
`summary.benchmark_ready=true`, no fallback cases, no negative cases, and no
readiness blockers. The real worker subprocesses also record process max RSS
through `resource.getrusage(...).ru_maxrss` where supported, so
`summary.all_peak_memory_cases_measured=true`,
`summary.all_adapter_switch_cases_measured=true`, and
`summary.memory_control_claim_ready=true` mean the memory/control fields are
ready to interpret after correctness. These are measured CPU process high-water
marks and adapter-switch timings, not CUDA allocator measurements or a memory
win by themselves; row-level process max RSS ranges from `1271021568` to
`2113093632` bytes in this run. The largest observed error is
`summary.max_abs_error=0.0000553131103515625` and
`summary.max_relative_l2_error=0.0000013214124852245438`.

This fixes the stale dense-merge failures investigated in
`docs/peft_multi_adapter_dense_merge_investigation.md`. The result still keeps
`summary.performance_claim` and `summary.memory_or_control_claim` set to
`none`; passing correctness and measuring process memory/control overhead do
not by themselves establish a latency, memory, or adapter-switching win.
