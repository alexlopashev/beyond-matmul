# Completion Audit

Date: 2026-07-09

This note records the final first-artifact audit for the Beyond Matmul
whitepaper after the July 8 live Conv1d and PEFT multi-adapter evidence
refresh plus the July 9 PEFT capstone seq100 correction. It is a compact
companion to `whitepaper/main.tex`,
`docs/evidence_matrix.md`, and `docs/benchmark_artifacts.md`, not a
replacement for those sources of truth.

## Final Draft Status

`whitepaper/main.tex` now integrates the project motivation, fixed-weight
scope, provenance-aware IR, frontend capture, recovery analyzer, approximation
builders, planner contracts, controlled benchmark results, workload
case-study evaluation, live Conv1d layer evidence, external PEFT evidence,
recovery and approximation evaluation, limitations, related-work boundaries,
completion criteria, capstone boundaries, and conclusion.

The related-work section intentionally ships as a scoped map rather than a
formal bibliography. The first public artifact names the relevant research
areas and records how this project differs from them; canonical citation
curation is paper-polish work that does not change the executable evidence.

## Claim-To-Evidence Audit

| Claim area | Evidence anchor | Audit result |
| --- | --- | --- |
| Fixed-weight scope and dense fallback | `README.md`, `whitepaper/main.tex`, `docs/research_outline.md`, `docs/evidence_matrix.md` | Supported; training-time mutation, production kernels, and hardware speedups remain out of scope. |
| Torch frontend coverage | `docs/torch_frontend_coverage.md`, `tests/test_frontend.py`, Torch demos, `docs/evidence_matrix.md` | Supported for listed fixed-weight linear, adapter, embedding-projection, Conv1d, matmul/mm, addmm, fixed per-tensor affine quantized `nn.Linear`, and exported graph rows; unsupported rows are explicit. |
| IR operator families | `docs/ir_spec.md`, `beyond_matmul/ir.py`, `tests/test_ir_planner.py`, `docs/evidence_matrix.md` | Supported for implemented exact and approximate operator families with dense fallback preserved. |
| Recovery after lost provenance | `beyond_matmul/analyzer.py`, `tests/test_analyzer.py`, `examples/fixed_weight_inference_demo.py`, `docs/evidence_matrix.md` | Supported as heuristic recovery plus sample validation; not calibrated provenance proof. |
| Planner exactness and fallback | `beyond_matmul/planner.py`, `tests/test_ir_planner.py`, `benchmarks/planner_contract_ablation.py`, `docs/benchmark_artifacts.md` | Supported as deterministic contract evidence; planner costs are estimates unless separately benchmarked. |
| Approximation and error contracts | `beyond_matmul/approximations.py`, `tests/test_ir_planner.py`, `benchmarks/approximation_error_ablation.py`, `docs/benchmark_artifacts.md` | Supported for the bounded output-aware acceptance claim, not broad model-quality conclusions. |
| Benchmark and cost claims | `benchmarks/fixed_weight.py`, `docs/results/fixed_weight.json`, `docs/results/live_conv1d_whisper.json`, `docs/results/peft_transformers_lora_inference.json`, `docs/results/peft_multi_adapter_serving.json`, `tests/test_benchmark_artifacts.py`, `tests/test_live_conv1d_whisper.py`, `tests/test_peft_transformers_lora_inference.py`, `tests/test_peft_multi_adapter_serving.py`, `scripts/ci_local`, `docs/benchmark_artifacts.md` | Supported as generated research artifacts, pure-Python proxies, live layer-level Conv1d evidence, and bounded PEFT capstone/serving evidence; not production performance evidence. |
| External PEFT provenance | `docs/results/peft_transformers_lora_inference.json`, `docs/results/peft_multi_adapter_serving.json`, `docs/benchmark_artifacts.md`, `docs/evidence_matrix.md`, `whitepaper/main.tex` | Supported for metadata-level LoRA provenance and dense-fallback visibility on the measured CPU fp32 workloads; not production kernels, memory savings, adapter-switching gains, broader PEFT coverage, or universal Transformer speedups. |
| Workload narratives | Torch examples, `examples/case_study_artifacts.py`, `docs/results/workload_case_studies.json`, `tests/test_case_study_artifacts.py` | Supported for adapter, Conv1d, grouped/depthwise Conv1d, fixed-mask, and per-tensor affine quantized-linear rows; broader workloads remain future work. |

## Open Blocker Audit

Live GitHub issue state on 2026-07-08 showed no unresolved priority-zero or
priority-one blocker against the first-artifact thesis. On 2026-07-09, #109
corrected the PEFT capstone shape grid from invalid seq128 rows to valid
seq100 rows and refreshed the measured artifact. The newly opened
production/performance roadmap issues (#110 through #114) are follow-on work
for stronger future claims; they do not invalidate the bounded first-artifact
completion state.

After the final-draft work merged, the first-artifact completion state became
historical context rather than an active blocker:

- #40 was the final-draft issue for the first public artifact.
- #41 was the roadmap tracker for that first artifact and should be treated as
  closed historical coordination rather than the current roadmap.
- #30, #31, and #52 are completed: the current artifact includes fixed
  per-tensor affine quantized `nn.Linear` frontend capture, packed affine
  quantized IR evidence, and a quantized-linear workload row.
- Quantized convolution, per-axis/per-channel or dynamic quantization,
  production integer kernels, and hardware-calibrated speedups remain outside
  the first public artifact unless separate issues add executable evidence.
- #73 and #82 are closed: the PEFT plus Transformers capstone is no longer the
  next roadmap target. The retrospective decision was to close it as a bounded
  provenance proof, not to pursue PEFT upstreaming, broader adapter coverage,
  or TorchBench integration from the current evidence.
- #96, #98, #99, #105, and PR #106 refreshed the July 8 Conv1d, PEFT
  multi-adapter, whitepaper-boundary, evidence-matrix, benchmark-index, and wiki
  evidence. Those updates narrow claims rather than creating a new expansion
  roadmap.

## July 8 Evidence Refresh

The live Conv1d benchmark adds measured layer-level evidence for the
`openai/whisper-tiny` encoder `model.encoder.conv1` layer. The measured
artifact is `docs/results/live_conv1d_whisper.json`, produced by
`benchmarks/live_conv1d_whisper.py` for the contract in
`docs/live_conv1d_benchmark_contract.md` and summarized in
`docs/benchmark_artifacts.md`, `docs/evidence_matrix.md`, and
`whitepaper/main.tex`.

The result supports a narrow Conv1d provenance claim: direct Conv1d and the
exact dense Toeplitz fallback match within the CPU fp32 tolerance for the
required 8-, 16-, and 32-frame prefixes, while the artifact records dense
materialized fallback byte counts and materialization time. It does not support
speedup, GPU, peak-memory, end-to-end ASR, Conv2d, quantized-convolution, or
broader CNN-block claims; `summary.performance_claim` remains `none`, and dense
bytes are fallback footprint metadata rather than measured peak memory.

## PEFT Capstone And Serving Boundaries

The external PEFT plus Transformers capstone remains closed as bounded
evidence, not as a new expansion roadmap. The measured artifact is
`docs/results/peft_transformers_lora_inference.json`, produced for the contract
in `docs/peft_capstone_benchmark_contract.md` and summarized in
`docs/evidence_matrix.md` and `whitepaper/main.tex`. The project fork was
`alexlopashev/peft`, and the measured integration branch was
`beyond-matmul/provenance-lora-inference`.

The refreshed result supports a narrow provenance claim: successful seq16,
seq64, and seq100 fork rows expose structured LoRA provenance while keeping
dense fallback available and matching upstream outputs. It is benchmark-ready
correctness evidence, but still not performance evidence:
`summary.benchmark_ready` is true, `summary.performance_claim` is `none`, CPU
peak memory is not measurable in the run, and adapter switching is not measured
for the single-adapter workload. The #82 retrospective created no upstreaming
or broader PEFT expansion issue because the measured result supports claim
narrowing and closure, not larger PEFT implementation work.

The PEFT multi-adapter serving follow-up extends that boundary with a
row-complete two-adapter artifact, not a stronger performance claim. The
measured artifact is `docs/results/peft_multi_adapter_serving.json`, produced
for `docs/peft_multi_adapter_serving_benchmark_contract.md` and summarized in
`docs/benchmark_artifacts.md`, `docs/evidence_matrix.md`, and
`whitepaper/main.tex`.

The result supports metadata-level serving evidence: all 48 required rows are
present; upstream unmerged PEFT, dense-cache, repeated merge/unmerge, and
Beyond Matmul factor-provenance rows pass correctness; Beyond Matmul rows
expose structured factor provenance without dense fallback; and adapter,
shape, correctness, storage, latency, and switching metadata are recorded. The
stale dense-merge failures were traced in
`docs/peft_multi_adapter_dense_merge_investigation.md` to harness dtype
mismatch against the CPU fp32 contract plus dense-cache adapter activation. The
refreshed result is benchmark-ready correctness evidence, but it does not
support a memory, latency, or adapter-switching gain: CPU peak memory is not
measured, no threshold win is claimed, and
`summary.performance_claim` plus `summary.memory_or_control_claim` remain
`none`.

## Reader Pointers

- `README.md` points to the final whitepaper draft, evidence matrix, benchmark
  artifacts, and this completion audit.
- The GitHub wiki points readers to the north star, artifact map,
  `whitepaper/main.tex`, `docs/evidence_matrix.md`, and generated benchmark
  artifacts without duplicating the paper.
- `docs/research_outline.md` remains the compact research plan and now points
  to this audit for the final status.
- PEFT reader pointers are historical capstone and serving evidence, not a next
  capstone target.

## Validation Commands

For first-artifact reproducibility, use:

```bash
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json
mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json
mise exec -- uv run python benchmarks/planner_contract_ablation.py --json-output docs/results/planner_contract_ablation.json
mise exec -- uv run --with transformers --with librosa --with soundfile --with safetensors --with huggingface_hub python benchmarks/live_conv1d_whisper.py --json-output docs/results/live_conv1d_whisper.json
mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_transformers_lora_inference.py --json-output docs/results/peft_transformers_lora_inference.json
mise exec -- uv run --with transformers --with accelerate --with safetensors --with huggingface_hub python benchmarks/peft_multi_adapter_serving.py --json-output docs/results/peft_multi_adapter_serving.json
scripts/ci_local
```

## Residual Risks

- The related-work section has scoped literature areas but no formal
  bibliography.
- Benchmark timings remain pure-Python proxies.
- Live Conv1d and external PEFT runs are measured local CPU artifacts, not
  production performance evidence.
- Recovery confidence remains heuristic and sample-limited.
- Quantized convolution, per-axis/per-channel or dynamic quantization, full
  masked attention, Conv2d, broader CNN blocks, production integer kernels, and
  hardware-calibrated speedups remain future work unless separate issues add
  executable evidence.
