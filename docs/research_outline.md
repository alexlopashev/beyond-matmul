# Research Outline

## Thesis

Many matmuls in ML systems are semantically structured computations that were
densified for convenience. A provenance-aware linear-operator IR can preserve or
recover that structure and allow a planner to choose exact or bounded-error
lowerings that are cheaper than dense GEMM for fixed-weight inference.

## Minimum Viable Artifact

1. Provenance-aware `LinearOperator` IR.
2. Planner for fixed-weight inference with amortized preprocessing.
3. Exact lowerings: dense, diagonal, sparse COO, low-rank product, direct conv1d.
4. Approximate lowerings: low-rank sketch, top-k sparse, codebook, bitpacked.
5. Benchmarks comparing latency proxy, memory proxy, preprocessing, and output
   error against dense fallback, with JSON artifacts documented in
   `docs/benchmark_artifacts.md`.
6. A small approximation-error ablation that compares matrix reconstruction
   error with output-relative error for low-rank, sparse top-k, codebook, and
   bitpacked candidates.

## Prototype Modules

- `ir.py`: operator schema and executable semantics.
- `frontend.py`: structured capture before densification.
- `analyzer.py`: cheap recovery after provenance is lost.
- `approximations.py`: product-aware approximation builders and metrics.
- `planner.py`: valid lowering selection.
- `benchmarks/fixed_weight.py`: synthetic fixed-weight benchmark with optional
  machine-readable JSON output.
- `benchmarks/approximation_error_ablation.py`: deterministic table source for
  the current matrix-error versus output-error ablation.

## Evaluation Plan

Synthetic controlled cases:

- exact diagonal
- exact sparse
- exact low-rank
- exact small-codebook
- random dense fallback

Real workload case studies to add next:

- Conv/linear lowering from a CNN block.
- Attention projection or masked attention block.
- Embedding plus projection in a language model.
- LoRA/adapters merged into a dense weight.

Metrics:

- wall-clock latency
- operation-count proxy
- memory footprint proxy
- preprocessing cost
- required calls to amortize preprocessing
- matrix reconstruction error
- product/output error on representative inputs

Current bounded evidence: `benchmarks/approximation_error_ablation.py` emits a
deterministic JSON artifact for one synthetic dense matrix and one fixed sample
input set. In that case, low-rank and sparse top-k candidates pass a
matrix-relative threshold but fail the same output-relative threshold, while
codebook and bitpacked candidates fail both. This supports using output-aware
scoring for planner acceptance, but it is not a broad model-quality study.

## Ablations

- provenance-preserved operator vs dense recovery
- recovery analyzer candidates vs dense fallback
- matrix reconstruction error vs product/output error
- exact-only planner vs bounded-error planner
- backend support constraints enabled vs ignored
