# Research Outline

## Thesis

Many matmuls in ML systems are semantically structured computations that were
densified for convenience. A provenance-aware linear-operator IR can preserve or
recover that structure and allow a planner to choose exact or bounded-error
lowerings that are cheaper than dense GEMM for fixed-weight inference.

## Minimum Viable Artifact

1. Provenance-aware `LinearOperator` IR.
2. Planner for fixed-weight inference with amortized preprocessing.
3. Exact lowerings: dense, diagonal, sparse COO, fixed-mask sparse, low-rank
   product, direct conv1d.
4. Approximate lowerings: low-rank sketch, top-k sparse, codebook, bitpacked.
5. Benchmarks comparing latency proxy, memory proxy, preprocessing, and output
   error against dense fallback, with JSON artifacts documented in
   `docs/benchmark_artifacts.md`.
6. A small approximation-error ablation that compares matrix reconstruction
   error with output-relative error for low-rank, sparse top-k, codebook, and
   bitpacked candidates.
7. A small planner-contract ablation that compares exact-only versus
   bounded-error planning, reuse amortization, backend support, and dense
   fallback validity on deterministic fixed-weight cases.
8. A concise evidence matrix in `docs/evidence_matrix.md` that maps major
   whitepaper claims to tests, demos, benchmark artifacts, or future-work
   boundaries.

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
- `benchmarks/planner_contract_ablation.py`: deterministic table source for
  exactness, bounded-error, reuse, backend, and dense fallback planner checks.
- `examples/case_study_artifacts.py`: machine-readable adapter, Conv1d, and
  fixed-mask workload case-study evidence with dense fallback comparisons.

## Recovery Confidence Semantics

Analyzer confidence values are bounded heuristic scores in `[0, 1]` for ranking
candidate structures from a single dense matrix. They are probe-specific:
diagonal confidence follows off-diagonal residual, sparse confidence follows
zero fraction, codebook confidence follows rounded unique-value pressure,
low-rank confidence follows matrix-relative reconstruction error against a
small "good error" threshold, and reuse confidence follows observed repeated
fixed-weight sightings. If representative sample inputs are supplied,
executable recovered candidates cap this structural score by a validation
confidence bound derived from output-relative error.

These values are not statistically calibrated probabilities and should not be
read as model-quality guarantees. When representative sample inputs are
available, executable recovered candidates also record validation evidence:
`output_relative_l2`, sample count, whether the candidate was exact on those
samples, and the validation-derived confidence bound. That validation is
evidence for the provided input set only; it does not replace dense fallback or
broader benchmark coverage.

## Evaluation Plan

Synthetic controlled cases:

- exact diagonal
- exact sparse
- exact low-rank
- exact small-codebook
- random dense fallback

Current workload case-study artifacts:

- Conv1d module and functional Conv1d capture before dense materialization.
- LoRA/adapters merged into a dense weight, with nearby adapter factors
  recovered by provenance capture.
- A fixed causal-band mask applied as a sparse linear map over features,
  independent of attention scores.

Real workload case studies still future work:

- Attention projection with fixed structured weights; full masked attention,
  broadcast mask contracts, score masking, softmax, KV-cache layout, and
  dynamic sequence behavior are outside the current fixed-weight linear IR
  boundary.
- Embedding plus projection in a language model.
- Quantized linear or convolutional modules once frontend capture has
  executable packed-payload rules for the documented quantization contracts.

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

Planner contract evidence is similarly bounded:
`benchmarks/planner_contract_ablation.py` emits a deterministic JSON artifact
where an exact-only request keeps dense GEMM, a bounded-error request on the
same fixed-weight case can choose a low-rank product, the same low-rank product
is rejected before its reuse threshold and accepted at the threshold, and a GPU
request rejects an unsupported codebook lowering while preserving dense GEMM as
a valid fallback.

For a reviewer-facing map from claims to executable support, see
`docs/evidence_matrix.md`.

## Ablations

- provenance-preserved operator vs dense recovery
- recovery analyzer candidates vs dense fallback
- matrix reconstruction error vs product/output error
- exact-only planner vs bounded-error planner
- reuse threshold before vs after amortization
- backend support constraints enabled vs ignored
