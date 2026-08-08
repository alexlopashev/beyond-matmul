# Next Layer Handoff

Date: 2026-07-05

Status update: 2026-08-08

The cumulative research draft is tracked in `whitepaper/main.tex`,
`docs/evidence_matrix.md`, and `docs/completion_audit.md`. This handoff remains
useful for historical engineering context, but the OLMoE capstone document is
the current roadmap source and the completion audit distinguishes first-artifact
claim support from project-level completion.

This file's original recommended frontend/Conv1d work is historical and has
already landed. Merged issue #129/PR #131 established the external OLMoE target
gate in `docs/olmoe_tensor_contraction_capstone.md`. Merged issue #132/PR #134
provides the stock-only baseline harness. Issue #136 adds the bound profiler
and real-activation diagnostic prerequisite. Issue #133 now supplies the
complete real H100 baseline/profile cohort and accepts one scoped stable
route-plan experiment. Do not use the old suggested PR at the end of this file
as current roadmap state.

## Current State

The repository now has a runnable fixed-weight inference research artifact:

- Provenance-aware linear and affine operator IR.
- Exact operators for dense, diagonal, sparse COO, fixed masks, low-rank,
  affine, conv1d, codebook, and bitpacked binary weights.
- Cheap dense recovery probes.
- Product-aware approximation scoring.
- Fixed-weight lowering planner with exactness, error, reuse, backend contracts,
  and per-option cost breakdowns.
- Pure-Python demos and benchmark.
- Torch FX demo that captures a nested low-rank linear pattern before
  densification.
- Torch FX frontend capture for nested `F.linear`/`nn.Linear`, biased affine
  linears, named adapter factors, merged-weight hints, and embedding-projection
  patterns over one-hot inputs.
- Tiny PyTorch adapter workload demo and machine-readable adapter, Conv1d, and
  fixed-mask case-study artifact.
- Reproducible tooling through mise, uv, and `uv.lock`.

For the current merged evidence surface, prefer `docs/evidence_matrix.md` over
historical PR lists.

## How To Reproduce

Bootstrap tools:

```bash
sh scripts/bootstrap
```

Install dependencies through the pinned toolchain:

```bash
mise exec -- uv sync
```

Run validation:

```bash
mise exec -- uv run python -m unittest discover -s tests
mise exec -- uv run python examples/fixed_weight_inference_demo.py
mise exec -- uv run python examples/torch_fx_frontend_demo.py
mise exec -- uv run python examples/adapter_workload_demo.py
mise exec -- uv run python examples/conv1d_workload_demo.py
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
mise exec -- uv run python benchmarks/fixed_weight.py
```

Last known real-dependency validation used:

```bash
mise exec -- uv sync
mise exec -- uv run python -m unittest discover -s tests
mise exec -- uv run python -m py_compile beyond_matmul/*.py examples/*.py tests/*.py
mise exec -- uv run python examples/fixed_weight_inference_demo.py
mise exec -- uv run python examples/torch_fx_frontend_demo.py
mise exec -- uv run python examples/adapter_workload_demo.py
mise exec -- uv run python examples/conv1d_workload_demo.py
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
mise exec -- uv run python benchmarks/fixed_weight.py
```

The Torch FX demo captured `linear_1` as a rank-2 `LowRankOperator`, selected
`low_rank_product` when provenance was preserved, and fell back to `dense_gemm`
after dense materialization.

## Important Design Decisions

- Fixed-weight inference is the first scope. This keeps preprocessing
  amortization explicit and avoids training-time mutation concerns.
- Dense GEMM is represented as a valid fallback, not as the default semantic IR.
- Planner costs now expose operation count, memory movement, cache footprint,
  preprocessing cost, and call-count amortization. They are still estimates, not
  hardware-calibrated conclusions.
- Torch is now a required dependency because the next research layer needs real
  framework capture, not only fake graph tests.
- NumPy is required because Torch expects it for clean tensor interop.
- `uv.lock` is committed for reproducible binary dependency resolution.
- Explicit binary fixed masks are supported only as sparse linear maps over
  values or features; full masked attention remains future work.

## Known Gaps

- CI publishes the fixed-weight benchmark JSON as `fixed-weight-benchmark-json`.
- No GPU or production kernels exist.
- Benchmarks are pure-Python latency proxies, not serious performance evidence.
- Recovery probes are cheap heuristics and do not yet emit calibrated confidence
  intervals.
- Approximation search is basic and not learned or hardware aware.

## Recommended Next Layer

Superseded by issues #129, #132, #136, and #133. The frozen H100 cohort and
target decision are complete. The current order is:

1. Review and merge the issue #133 stock artifacts and `accept` decision.
2. Open one implementation issue for stable route-plan expert execution only:
   construct route membership/offsets once while preserving eager token/expert
   order and accumulation semantics.
3. Measure the changed full-model path against all eight selected best-correct
   stock rows; require the fixed 10% win, 5% regression, correctness, fallback,
   and external-review gates.
4. Reject OLMoE if that scoped experiment fails. Generalize the local IR only
   from concepts required by a successful result.

The recommendations below are retained as historical context.

1. Harden Torch FX capture further.

   Extend frontend capture beyond the current low-rank and embedding patterns to:

   - exported or compiled graphs where module names have been erased.
   - matmul/addmm patterns not expressed through `linear`.
   - convolution modules and quantized modules.

2. Broaden captured-operator examples as tests.

   Keep dependency-free fake FX tests, and keep adding real Torch tests for small
   modules now that Torch is required. These should assert the captured IR, dense
   equivalence, planner selection, and output error.

3. Add a second workload case study.

   The next best case is convolution, because dense lowering loses a very
   different kind of structure than adapters do.

4. Expand result artifacts.

   Add any next benchmark artifacts needed for paper figures without parsing
   console text.

## Suggested Next PR

Title:

```text
[codex] Capture convolutional Torch modules
```

Scope:

- Add real Torch `nn.Conv1d` capture.
- Compare direct convolution IR against dense Toeplitz materialization.
- Add a convolution workload demo.
- Include the demo in CI.

Success criteria:

- CI is green.
- The convolution path is exact against PyTorch for a small fixed-weight module.
- Planner comparison shows `conv1d_direct` versus dense materialization.
