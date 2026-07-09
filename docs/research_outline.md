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
9. A completion audit in `docs/completion_audit.md` that records final-draft
   claim support, limitations, blocker status, validation commands, and
   optional follow-up issues.

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
- `examples/case_study_artifacts.py`: machine-readable adapter, Conv1d,
  fixed-mask, and quantized-linear workload case-study evidence with dense
  fallback comparisons.

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

- Conv1d module, functional, grouped, and depthwise capture before dense
  materialization.
- LoRA/adapters merged into a dense weight, with nearby adapter factors
  recovered by provenance capture.
- A fixed causal-band mask applied as a sparse linear map over features,
  independent of attention scores.
- A fixed per-tensor affine quantized linear module preserving packed integer
  payload, scale, and zero point before dense dequantized fallback.

Future workload case studies still outside the current artifact:

- Attention projection with fixed structured weights; full masked attention,
  broadcast mask contracts, score masking, softmax, KV-cache layout, and
  dynamic sequence behavior are outside the current fixed-weight linear IR
  boundary.
- A fuller language-model-like embedding plus projection workload artifact;
  the current frontend coverage already includes embedding followed by
  projection over one-hot token inputs.
- Quantized convolutional modules once frontend capture has executable
  packed-payload rules for convolution-specific quantization contracts.

Live Conv1d layer-level benchmark:

- `benchmarks/live_conv1d_whisper.py` implements the contract in
  `docs/live_conv1d_benchmark_contract.md` for `openai/whisper-tiny`
  `model.encoder.conv1`.
- The measured artifact is `docs/results/live_conv1d_whisper.json`: direct
  Conv1d and exact dense materialized Toeplitz fallback match within tolerance
  for 8-, 16-, and 32-frame prefixes of the public LibriSpeech audio trace.
- The result is negative for performance readiness:
  `summary.performance_claim=none`, direct Conv1d is faster than dense
  application in the committed local CPU run, and dense materialization cost is
  recorded separately.

Closed PEFT capstone:

- PEFT plus Transformers inference used a fork of `huggingface/peft` at
  `alexlopashev/peft` on branch
  `beyond-matmul/provenance-lora-inference`.
- The benchmark contract is
  `docs/peft_capstone_benchmark_contract.md`: CPU fp32 prefill-only causal LM
  inference for `hf-internal-testing/tiny-random-OPTForCausalLM` with
  `peft-internal-testing/tiny-OPTForCausalLM-lora`, sequence lengths `16`,
  `64`, and `100`, and batch sizes `1` and `4`. The seq100 upper row matches
  the selected tiny model's positional limit.
- The fork integration design is
  `docs/peft_low_rank_provenance_design.md`. It limits #78 to vanilla PEFT
  LoRA `Linear` inference metadata, dense fallback, and benchmark harness
  reporting before any custom kernel or broad adapter coverage.
- The measured artifact is
  `docs/results/peft_transformers_lora_inference.json`. It supports the bounded
  claim that successful seq16, seq64, and seq100 fork rows expose structured
  LoRA provenance while preserving dense fallback and matching upstream
  outputs.
- The result is benchmark-ready correctness evidence but still negative for
  performance claims: `summary.benchmark_ready=true`,
  `summary.performance_claim=none`, CPU peak memory is not measurable, and
  adapter switching is not measured for the single-adapter workload.
- Training, generation loops, KV-cache behavior, broad PEFT coverage, GPU
  kernel claims, memory savings, adapter-switching gains, and universal
  Transformer speedups remain unsupported by this capstone.
- The #82 retrospective closed the capstone as a bounded proof and created no
  PEFT upstreaming or broader expansion issue.

Closed PEFT multi-adapter serving follow-up:

- The multi-adapter serving contract is
  `docs/peft_multi_adapter_serving_benchmark_contract.md`: CPU fp32
  prefill-only causal LM inference for `facebook/opt-125m` at
  `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, adapters `merchant` and
  `gaisb`, sequence lengths `16`, `64`, and `128`, batch sizes `1` and `2`,
  and four serving baselines.
- The harness is `benchmarks/peft_multi_adapter_serving.py`; CI exercises the
  torch-only smoke artifact at
  `docs/results/peft_multi_adapter_serving_smoke.json`.
- The measured artifact is
  `docs/results/peft_multi_adapter_serving.json`. It includes all 48 required
  rows with the contract timing protocol. Upstream unmerged, dense-cache,
  repeated merge/unmerge, and Beyond Matmul provenance rows pass correctness
  for both adapters and all shapes.
- The result is benchmark-ready correctness evidence: the Beyond Matmul rows
  expose structured factor provenance without dense fallback,
  `summary.benchmark_ready=true`, `summary.performance_claim=none`, and
  `summary.memory_or_control_claim=none`.
- The stale dense-merge failures were traced in
  `docs/peft_multi_adapter_dense_merge_investigation.md` to harness dtype
  mismatch against the CPU fp32 contract plus a dense-cache adapter activation
  bug.
- The artifact supports only the narrower claim that the external PEFT path can
  produce row-complete multi-adapter serving metadata, switching measurements,
  and structured factor provenance. It does not support memory savings,
  adapter-switching gains, training, generation loops, GPU kernels, or
  universal Transformer speedups.

Future hardware-backed production/performance contract:

- The contract is
  `docs/hardware_backed_production_benchmark_contract.md`: a design target for
  interpreting future CUDA-backed PEFT multi-adapter serving measurements.
- It keeps the existing `facebook/opt-125m` two-adapter workload as the
  minimal target and asks whether a structured low-rank serving path can reduce
  already-loaded adapter switching cost or per-adapter resident memory without
  correctness failures or material forward-latency regression.
- The contract distinguishes forward latency, CUDA memory, preprocessing,
  adapter-switching/control, and dense fallback readiness fields before any
  performance claim can be interpreted.
- This is future work only. It is not current hardware evidence and does not
  move GPU speedups, memory savings, production kernels, or universal
  Transformer acceleration into the supported claim set.

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
