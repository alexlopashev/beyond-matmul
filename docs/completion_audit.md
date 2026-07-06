# Completion Audit

Date: 2026-07-06

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
| Benchmark and cost claims | `benchmarks/fixed_weight.py`, `docs/results/fixed_weight.json`, `tests/test_benchmark_artifacts.py`, `scripts/ci_local`, `docs/benchmark_artifacts.md` | Supported as generated research artifacts and pure-Python proxies; not production performance evidence. |
| Workload narratives | Torch examples, `examples/case_study_artifacts.py`, `docs/results/workload_case_studies.json`, `tests/test_case_study_artifacts.py` | Supported for adapter, Conv1d, grouped/depthwise Conv1d, fixed-mask, and per-tensor affine quantized-linear rows; broader workloads remain future work. |

## Open Blocker Audit

Live GitHub issue state on 2026-07-06 showed no unresolved priority-zero or
priority-one blocker against the first-artifact thesis:

- #40 is this final-draft issue.
- #41 is a roadmap tracker blocked by #40 and should close or be replaced after
  this final-draft PR is reviewed and merged.
- #30, #31, and #52 are completed: the current artifact includes fixed
  per-tensor affine quantized `nn.Linear` frontend capture, packed affine
  quantized IR evidence, and a quantized-linear workload row.
- Quantized convolution, per-axis/per-channel or dynamic quantization,
  production integer kernels, and hardware-calibrated speedups remain outside
  the first public artifact unless separate issues add executable evidence.

## Reader Pointers

- `README.md` points to the final whitepaper draft, evidence matrix, benchmark
  artifacts, and this completion audit.
- The GitHub wiki points readers to the north star, artifact map,
  `whitepaper/main.tex`, `docs/evidence_matrix.md`, and generated benchmark
  artifacts without duplicating the paper.
- `docs/research_outline.md` remains the compact research plan and now points
  to this audit for the final status.

## Validation Commands

The final-draft PR should record the exact validation results for:

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
