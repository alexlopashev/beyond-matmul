# Completion Audit

Date: 2026-07-07

This note records the final first-artifact audit for the Beyond Matmul
whitepaper. It is a compact companion to `whitepaper/main.tex` and
`docs/evidence_matrix.md`, not a replacement for either source of truth.

## Final Draft Status

`whitepaper/main.tex` now integrates the project motivation, fixed-weight
scope, provenance-aware IR, frontend capture, recovery analyzer, approximation
builders, planner contracts, controlled benchmark results, workload
case-study evaluation, recovery and approximation evaluation, limitations,
related-work boundaries, completion criteria, and conclusion.

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
| Benchmark and cost claims | `benchmarks/fixed_weight.py`, `docs/results/fixed_weight.json`, `docs/results/peft_transformers_lora_inference.json`, `tests/test_benchmark_artifacts.py`, `tests/test_peft_transformers_lora_inference.py`, `scripts/ci_local`, `docs/benchmark_artifacts.md` | Supported as generated research artifacts, pure-Python proxies, and bounded PEFT capstone evidence; not production performance evidence. |
| Workload narratives | Torch examples, `examples/case_study_artifacts.py`, `docs/results/workload_case_studies.json`, `tests/test_case_study_artifacts.py` | Supported for adapter, Conv1d, grouped/depthwise Conv1d, fixed-mask, and per-tensor affine quantized-linear rows; broader workloads remain future work. |

## Open Blocker Audit

Live GitHub issue state on 2026-07-07 showed no unresolved priority-zero or
priority-one blocker against the current artifact thesis, aside from this
documentation refresh while it is in flight. After the final-draft work merged,
the first-artifact completion state became historical context rather than an
active blocker:

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

## Closed PEFT Capstone

The external PEFT plus Transformers capstone has closed as bounded evidence,
not as a new expansion roadmap. The measured artifact is
`docs/results/peft_transformers_lora_inference.json`, produced for the contract
in `docs/peft_capstone_benchmark_contract.md` and summarized in
`docs/evidence_matrix.md` and `whitepaper/main.tex`. The project fork was
`alexlopashev/peft`, and the measured integration branch was
`beyond-matmul/provenance-lora-inference`.

The result supports a narrow provenance claim: successful seq16 and seq64 fork
rows expose structured LoRA provenance while keeping dense fallback available
and matching upstream outputs. It is negative performance-readiness evidence:
seq128 fails across all baselines, `summary.benchmark_ready` is false,
`summary.performance_claim` is `none`, CPU peak memory is not measurable in the
run, and adapter switching is not measured for the single-adapter workload. The
#82 retrospective created no upstreaming or broader PEFT expansion issue
because the measured result supports claim narrowing and closure, not larger
PEFT implementation work.

## Reader Pointers

- `README.md` points to the final whitepaper draft, evidence matrix, benchmark
  artifacts, and this completion audit.
- The GitHub wiki points readers to the north star, artifact map,
  `whitepaper/main.tex`, `docs/evidence_matrix.md`, and generated benchmark
  artifacts without duplicating the paper.
- `docs/research_outline.md` remains the compact research plan and now points
  to this audit for the final status.
- PEFT reader pointers are historical capstone evidence, not a next capstone
  target.

## Validation Commands

For first-artifact reproducibility, use:

```bash
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json
mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json
mise exec -- uv run python benchmarks/planner_contract_ablation.py --json-output docs/results/planner_contract_ablation.json
scripts/ci_local
```

## Residual Risks

- The related-work section has scoped literature areas but no formal
  bibliography.
- Benchmark timings remain pure-Python proxies.
- Recovery confidence remains heuristic and sample-limited.
- Quantized convolution, per-axis/per-channel or dynamic quantization, full
  masked attention, Conv2d, broader CNN blocks, production integer kernels, and
  hardware-calibrated speedups remain future work unless separate issues add
  executable evidence.
