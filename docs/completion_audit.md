# Completion Audit

Last updated: 2026-08-08

This note records the final first-artifact audit for the Beyond Matmul
whitepaper after the July 8 live Conv1d and PEFT multi-adapter evidence
refresh plus the July 9 PEFT capstone seq100 correction. It is a compact
companion to `whitepaper/main.tex`,
`docs/evidence_matrix.md`, and `docs/benchmark_artifacts.md`, not a
replacement for those sources of truth.

## August 8 OLMoE Capstone Result

The phrase "final first artifact" is historical, not the current project-level
completion state. Merged issue #129 and PR #131 restore the stronger finish
line: preserved
tensor-contraction provenance must cause an attributable inference performance
improvement in an external open-source project. Matrix multiplication is the
rank-2 case, and the implemented matrix IR remains bounded prior work.

`allenai/OLMoE-1B-7B-0924` through Transformers is the accepted experimental target. Its
routed expert computation is a tensor program composed of expert-indexed
gate/up and down contractions, nonlinear gating, dynamic selection, and
aggregation over token, selected-expert, expert, hidden, intermediate, and
output axes. Current Transformers already provides provenance-aware grouped,
batched, and optimized expert backends, so reproducing an existing
eager-versus-optimized win cannot satisfy the new capstone.
`docs/olmoe_tensor_contraction_capstone.md` defines the best-stock baseline,
10% win, 5% regression, correctness, external-review, and rejection gates.

Issue #133 accepted one scoped stable
route-plan experiment, not a general tensor IR: preserve eager token/expert and
accumulation order while constructing route membership and offsets once for the
expert program. Issue #139 now implements that narrow path, its explicit eager
fallback and audit metadata, a deterministic exactness demo, and the frozen
candidate artifact surface. Its committed real H100 artifact passes correctness
in all eight regimes, records 4,800 stable candidate calls and zero fallbacks,
and improves the frozen best-correct-stock CUDA-event medians by 19.26% to
63.30%. It therefore clears the fixed 10% win and 5% regression gates with
`performance_claim=qualified_candidate_speedup`. Independent review and merge,
followed by demo packaging, remain before project-level completion. The PEFT
CUDA roadmap in issues #123 through #126 remains paused.

Merged issue #132/PR #134 adds the baseline-only harness
`benchmarks/olmoe_stock_baseline.py`. It pins the eight required full-model
prefill/decode regimes, the stock backend and `torch.compile` inventory,
explicit exclusion/failure rows, correctness-first interpretation, and the
machine-readable timing schema. Issue #136 adds
`benchmarks/olmoe_stock_profile.py`, which binds one full-model profile to the
best stock row in every regime and defines a real-activation sparse-layer-8
diagnostic for `prefill_b1_s512`. Their CI smokes perform no model execution.
Issue #133 supplies the committed real H100 baseline and profile artifacts plus
the binary `accept` decision. The 288-row baseline is cohort-complete; only the
eight uncompiled eager rows pass the fixed correctness tolerance. All eight
bound CUPTI profiles and the exact diagnostic replay are complete. Both real
stock artifacts contain no candidate result and retain `performance_claim=none`.
The issue #139 candidate CI smoke also performs no model execution and local
FP32/BF16 ordering checks are semantic evidence only; the separate source-bound
real candidate artifact is the performance evidence.

## Historical Final Draft Status

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
| Benchmark and cost claims | `benchmarks/fixed_weight.py`, `docs/results/fixed_weight.json`, `docs/results/live_conv1d_whisper.json`, `docs/results/peft_transformers_lora_inference.json`, `docs/results/peft_multi_adapter_serving.json`, `docs/results/olmoe_stock_baseline.json`, `docs/results/olmoe_stock_profile.json`, `docs/results/olmoe_stable_route_candidate.json`, `benchmarks/olmoe_stock_baseline.py`, `benchmarks/olmoe_stock_profile.py`, `benchmarks/olmoe_stable_route_candidate.py`, `tests/test_benchmark_artifacts.py`, `tests/test_live_conv1d_whisper.py`, `tests/test_peft_transformers_lora_inference.py`, `tests/test_peft_multi_adapter_serving.py`, `tests/test_olmoe_measured_artifacts.py`, `scripts/ci_local`, `docs/benchmark_artifacts.md` | Supported as generated research artifacts, pure-Python proxies, live layer-level Conv1d evidence, bounded PEFT capstone/serving evidence, a real stock-only OLMoE baseline/profile, and a source-bound H100 candidate result. The candidate clears the frozen gate on one cohort; it is not a universal, memory, production, or upstream-acceptance result. |
| OLMoE stock-baseline capability | `benchmarks/olmoe_stock_baseline.py`, `tests/test_olmoe_stock_baseline.py`, `docs/results/olmoe_stock_baseline.json`, `docs/benchmark_artifacts.md`, `scripts/ci_local` | Supported by a frozen H100 cohort with 288 explicit rows, 160 required terminal attempts, fixed correctness gates, and eight best-correct-stock selections. Only uncompiled eager passes in every regime; correctness-failed faster rows remain ineligible. |
| OLMoE profiler capability | `benchmarks/olmoe_stock_profile.py`, `tests/test_olmoe_stock_profile.py`, `docs/results/olmoe_stock_profile.json`, `docs/benchmark_artifacts.md`, `scripts/ci_local` | Supported by eight exact best-stock bindings with CUPTI kernel traces plus a correct real-activation layer diagnostic. The replay attributes remaining work but is never end-to-end candidate evidence. |
| OLMoE stable-route candidate capability | `beyond_matmul/olmoe_route_plan.py`, `benchmarks/olmoe_stable_route_candidate.py`, `docs/results/olmoe_stable_route_candidate.json`, `examples/olmoe_stable_route_demo.py`, `tests/test_olmoe_stable_route.py`, `tests/test_olmoe_stable_route_candidate.py`, `tests/test_olmoe_stable_route_demo.py`, `tests/test_olmoe_measured_artifacts.py`, `scripts/ci_local` | Supported for eager-order route planning, BF16 duplicate accumulation semantics, explicit fallback/audit metadata, the frozen eight-regime contract, and the accepted source-bound H100 result. All rows pass correctness and improve median latency by 19.26% to 63.30%; cross-hardware and production generality remain unmeasured. |
| External PEFT provenance | `docs/results/peft_transformers_lora_inference.json`, `docs/results/peft_multi_adapter_serving.json`, `docs/benchmark_artifacts.md`, `docs/evidence_matrix.md`, `whitepaper/main.tex` | Supported for metadata-level LoRA provenance and dense-fallback visibility on the measured CPU fp32 workloads; not production kernels, memory savings, adapter-switching gains, broader PEFT coverage, or universal Transformer speedups. |
| Workload narratives | Torch examples, `examples/case_study_artifacts.py`, `docs/results/workload_case_studies.json`, `tests/test_case_study_artifacts.py` | Supported for adapter, Conv1d, grouped/depthwise Conv1d, fixed-mask, and per-tensor affine quantized-linear rows; broader workloads remain future work. |

## Open Blocker Audit

Live GitHub issue state on 2026-07-08 showed no unresolved priority-zero or
priority-one blocker against the bounded first-artifact thesis. On 2026-07-09, #109
corrected the PEFT capstone shape grid from invalid seq128 rows to valid
seq100 rows and refreshed the measured artifact. The newly opened
production/performance roadmap issues (#110 through #114) were follow-on work
for stronger claims. Merged issue #129/PR #131 records that the missing external
result does invalidate project-level completion, without changing the accuracy
of the historical first-artifact audit. Issue #132/PR #134 completed the
baseline harness; #136 supplies the profiling prerequisite for #133, and
#130/PR #135 synchronizes the concise wiki. Issue #133 supplies the matching
CUDA/CUPTI artifacts and accepts one scoped experiment; issue #139 now supplies
the measured candidate. The remaining blocker is independent review/merge and
demo packaging, not implementation, target access, or the quantitative gate.

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
listed in `summary.structured_low_rank_cases` report
`lowering.execution_path=structured_low_rank` without dense fallback; and
adapter, shape, correctness, storage, latency, switching metadata, and
platform-supported process max RSS are recorded. The stale dense-merge failures
were traced in
`docs/peft_multi_adapter_dense_merge_investigation.md` to harness dtype
mismatch against the CPU fp32 contract plus dense-cache adapter activation. The
refreshed result is benchmark-ready correctness and memory/control
instrumentation evidence, but it does not support a memory, latency, or
adapter-switching gain: no latency, process-memory, CUDA peak-memory, or
adapter-switch threshold win is claimed, and `summary.performance_claim` plus
`summary.memory_or_control_claim` remain `none`.

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
- `docs/olmoe_tensor_contraction_capstone.md` is the active target decision and
  rejection gate; the executable harness and its evidence boundary are in
  `benchmarks/olmoe_stock_baseline.py`, `benchmarks/olmoe_stock_profile.py`, and
  `docs/benchmark_artifacts.md`.

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
mise exec -- uv run python benchmarks/olmoe_stock_baseline.py --smoke --json-output docs/results/olmoe_stock_baseline_smoke.json
mise exec -- uv run python benchmarks/olmoe_stock_profile.py --smoke --json-output docs/results/olmoe_stock_profile_smoke.json
mise exec -- uv run python benchmarks/olmoe_stable_route_candidate.py --smoke --json-output docs/results/olmoe_stable_route_candidate_smoke.json
scripts/ci_local
```

## Residual Risks

- The related-work section has scoped literature areas but no formal
  bibliography.
- Controlled matrix benchmark timings remain pure-Python proxies; the OLMoE
  candidate artifact is the narrow hardware-backed exception.
- Live Conv1d and external PEFT runs are measured local CPU artifacts, not
  production performance evidence.
- The OLMoE stable route-plan result is measured on one H100 PCIe GPU and one
  pinned software/model cohort. It does not establish cross-hardware,
  cross-model, compile-path, production, memory, or upstream-acceptance claims.
- Current Transformers already has strong routed-expert backends. Every
  optimized executable row failed the immutable parity gate in this cohort;
  future work must preserve correctness rather than reinterpret those timings.
- The real cohort is one H100 PCIe host and one pinned software environment;
  hardware and software generality remain unmeasured.
- Recovery confidence remains heuristic and sample-limited.
- Quantized convolution, per-axis/per-channel or dynamic quantization, full
  masked attention, Conv2d, broader CNN blocks, production integer kernels, and
  hardware-calibrated speedups remain future work unless separate issues add
  executable evidence.
