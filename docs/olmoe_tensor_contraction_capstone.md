# OLMoE Routed Tensor-Program Capstone

Status: independently reviewed candidate clears the frozen gate

Decision date: 2026-08-08

Tracking issues: #129 (contract), #133 (measured target decision), and #139
(candidate implementation and measurement)

## Decision

Issue #133 accepts OLMoE for exactly one scoped implementation experiment:
stable route-plan expert execution that preserves eager token order, expert
identity, routing weights, and accumulation semantics while removing repeated
per-expert route discovery and materialization. The external patch surface is
the Transformers expert integration in `src/transformers/integrations/moe.py`
plus, only if a reusable kernel is required, the Hugging Face `kernels`
project. Stock eager and generic/dense execution remain explicit fallbacks.

That target-validation decision was not itself a Beyond Matmul performance
result. Issue #139 subsequently implemented and measured the scoped path. The
committed candidate artifact clears the fixed 10% win, 5% cross-regime
regression, correctness, and fallback gates below and records
`performance_claim=qualified_candidate_speedup`. Independent review was the
last gate before merge. PR #140 passed that review and merged without bypass;
the review recomputed every row and binding rather than trusting the summary.

OLMoE remains a useful tensor case because its sparse mixture-of-experts layer
retains token routing, expert identity, routing weights, and 3D expert weights
rather than presenting the computation as one anonymous matrix. Current
Transformers already contains provenance-aware expert backends; their
eager-versus-optimized gains are background evidence and cannot count as a
Beyond Matmul result.

## Immutable References

- model: [`allenai/OLMoE-1B-7B-0924`](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/bd1c52f59153f724c1ad11ca1791edc77bab3806)
- model revision: `bd1c52f59153f724c1ad11ca1791edc77bab3806`
- model license: Apache-2.0
- model architecture: 16 layers, hidden size 2048, intermediate size 1024,
  64 experts, 8 selected experts per token, BF16 weights, and context limit 4096
- model project: [`allenai/OLMoE@357454f`](https://github.com/allenai/OLMoE/tree/357454f4f647385839c0ff6b99a688dc7cd9c13f)
- reference library audit:
  [`huggingface/transformers@a689565`](https://github.com/huggingface/transformers/tree/a6895655b289cc3fdd29afec36904e0b8545ef92)
- reference model implementation:
  `src/transformers/models/olmoe/modeling_olmoe.py`
- reference expert backends:
  `src/transformers/integrations/moe.py` and
  `docs/source/en/experts_interface.md`

Dependency and hardware pins for a measured cohort must be frozen in the
follow-up benchmark contract. A later upstream revision creates a separate
cohort; measurements from different cohorts must not be pooled.

## The Routed Tensor Program

For flattened token index `t`, selected-expert slot `j`, hidden axis `h`,
intermediate axis `i`, and output hidden axis `o`, let:

- `X[t, h]` be the token hidden states;
- `E[t, j]` be the selected expert identities;
- `R[t, j]` be the selected routing weights, globally softmax-normalized before
  top-k selection but not renormalized over the selected experts in the pinned
  configuration;
- `G[e, 2i, h]` be the expert gate/up projection tensor; and
- `D[e, o, i]` be the expert down-projection tensor.

The MoE contribution is a routed tensor program:

```text
Z[t, j, i] = act(sum_h G_gate[E[t, j], i, h] * X[t, h])
              * sum_h G_up[E[t, j], i, h] * X[t, h]
Y[t, o] = sum_j R[t, j] * sum_i D[E[t, j], o, i] * Z[t, j, i]
```

This notation is schematic; the benchmark uses the upstream implementation as
the semantic reference. The complete program is not one tensor contraction: it
combines expert-indexed gate/up and down contractions with a data-dependent
gather, nonlinear activation, elementwise gating, and routed aggregation. The
important fact is that provenance connects those operations across token,
selected-expert, expert, hidden, intermediate, and output axes plus a routing
relation. Lowering the contraction subgraphs into independent GEMMs or an
opaque grouped GEMM can discard facts needed for scheduling, fusion, layout,
reuse, and fallback.

The provenance record needed for target validation includes:

- model, layer, parameter, and input identities;
- named axes and contraction axes;
- token-to-expert assignments and routing weights;
- active-expert counts and tokens per expert;
- expert tensor layout, dtype, device, stride, and transposition;
- gated activation and aggregation semantics;
- batch, sequence, prefill/decode phase, and KV-cache state;
- exactness and output-equivalence contracts;
- compilation, graph-capture, preprocessing, and reuse state;
- available eager, batched, grouped, fused, and dense fallback paths.

This target does not authorize a general-purpose tensor algebra IR. The first
implementation should represent only the fields required by the measured
OLMoE opportunity.

## Current Upstream Baseline

At the audited Transformers revision:

- the eager OLMoE expert path discovers active experts, gathers their tokens,
  applies gate/up and down projections per expert, weights the results, and
  accumulates them;
- `grouped_mm` sorts routed token-expert pairs, preserves per-expert offsets,
  and uses grouped matrix multiplication;
- `batched_mm` duplicates selected expert parameters per routed token and uses
  batched matrix multiplication;
- the default is `grouped_mm` when the model and platform support it, otherwise
  eager;
- generation can switch a grouped backend to `batched_mm` for the low-token
  decode stage;
- an optimized DeepGEMM grouped path and a fused SonicMoE routed path are
  available on supported Hopper or newer NVIDIA hardware; the audited BF16
  DeepGEMM path still dispatches separate up and down grouped multiplications.

These are already provenance-aware optimizations. The OLMoE project README's
statement that the Transformers implementation is slow is useful historical
motivation, but it must be revalidated against this current backend surface.

## Target-Validation Questions

Target validation must answer these questions before an optimization issue is
opened:

1. On one frozen and accessible hardware/dependency cohort, what is the best
   stock Transformers backend for each required OLMoE prefill and decode regime?
2. Is there a material cost not already removed by the best stock backend, such
   as routing sort/permutation, histogram/offset construction, separate gated
   up and down contractions, aggregation, layout conversion, or a coarse backend
   decision?
3. Can preserved route, axis, layout, and phase provenance enable a distinct
   execution—not merely label the existing one—that plausibly removes that
   cost?
4. Is the proposed change externally reviewable in Transformers, OLMoE, or a
   reusable kernel project?

Candidate hypotheses include a fused routed tensor program on hardware not
served by the existing fused backend, or a route-aware lowering that removes
work the best stock backend still performs. These are hypotheses, not claims.

## Measured H100 Target Decision

Issue #133 ran the frozen cohort on one NVIDIA H100 PCIe 80 GB GPU
(`GPU-a044420f-519f-3f44-eb83-76272b8d274e`) with driver `580.159.04`, CUDA
runtime and toolkit `13.0`, PyTorch `2.12.1+cu130`, Python `3.14.6`, the pinned
Transformers revision, and the dependency versions recorded in
`docs/results/olmoe_stock_baseline.json`. The real run used five warmups and 20
measured repetitions. Its 36-configuration inventory contains 20 required and
16 contract-excluded configurations across eight regimes, for 288 explicit
rows.

The artifact is row- and cohort-complete with no readiness blocker. Of the 160
required rows, 96 completed execution and 64 ended in an explicit executor
failure. Only the eight uncompiled eager rows passed both predeclared
correctness tolerances. The other 88 executable rows, including default,
grouped, DeepGEMM, SonicMoE, and compiled variants, failed output parity; their
lower latencies therefore cannot be interpreted as correct stock baselines or
as a Beyond Matmul result. The best correct stock row in every regime is:

| Regime | Configuration | CUDA-event median | Throughput |
| --- | --- | ---: | ---: |
| `prefill_b1_s128` | `eager__uncompiled` | 296.094 ms | 432.30 tokens/s |
| `prefill_b1_s512` | `eager__uncompiled` | 231.673 ms | 2,210.01 tokens/s |
| `prefill_b4_s128` | `eager__uncompiled` | 245.158 ms | 2,088.45 tokens/s |
| `prefill_b4_s512` | `eager__uncompiled` | 264.093 ms | 7,754.85 tokens/s |
| `decode_b1_p128` | `eager__uncompiled` | 49.354 ms | 20.26 tokens/s |
| `decode_b1_p512` | `eager__uncompiled` | 49.372 ms | 20.25 tokens/s |
| `decode_b8_p128` | `eager__uncompiled` | 120.709 ms | 66.28 tokens/s |
| `decode_b8_p512` | `eager__uncompiled` | 116.120 ms | 68.89 tokens/s |

The selected eager rows record timed-forward allocator increments from 21.9 MB
to 351.7 MB for prefill and from 0.68 MB to 18.0 MB for decode after the prompt
cache is resident. These fields preserve the measurement boundary; they do not
establish a memory-saving opportunity by themselves.

`docs/results/olmoe_stock_profile.json` binds a successful CUPTI profile to all
eight selected rows and contains an exact layer-8 diagnostic replay with zero
maximum-absolute and relative-L2 error. Generic full-model matrix operations
remain intentionally unclassified, so the scoped diagnostic is the attribution
evidence. Its device self time is 29.40% sorting/permutation, 28.63% expert
contractions, 16.18% activation/gating, 11.30% unclassified, 8.97%
aggregation/scatter, 4.06% layout/copy conversion, and 1.45% routing/top-k.
Repeated route discovery and permutation therefore cost at least as much as
the expert contractions in this diagnostic.

The four target-validation questions resolve as follows:

1. The best correct stock path is uncompiled eager in all eight regimes under
   the immutable tolerance. Faster correctness-failed rows are not eligible.
2. Stable route construction, permutation, and aggregation remain attributable
   work around the contractions; the diagnostic exposes 38.37% device self time
   in sorting/permutation plus aggregation/scatter.
3. Preserved route and axis provenance enables a distinct stable route plan:
   construct expert offsets and token membership once, preserve eager token and
   selected-expert order, reuse that plan across gate/up, activation, down, and
   weighted aggregation, and retain eager-order accumulation for parity. This
   is changed execution, not metadata attached to an existing backend.
4. The intervention is reviewable as a narrow Transformers expert backend,
   with any reusable kernel isolated in the existing `kernels` integration
   surface and eager/generic fallbacks retained.

The binary decision is therefore **accept for one scoped experiment**. It does
not authorize a broad tensor IR, does not treat the diagnostic as end-to-end
evidence, and does not claim that the intervention will win. No optimization
issue may begin until this decision is reviewed and merged.

## Benchmark Gate

The follow-up benchmark contract must freeze the exact GPU, driver, PyTorch,
Transformers, kernel dependencies, model revision, dtype, backend flags, input
tokens, warmups, and repetitions before the candidate implementation is timed.
It must include:

- full-model BF16 prefill for batch sizes 1 and 4 at sequence lengths 128 and
  512;
- full-model decode for batch sizes 1 and 8 at fixed prompt lengths 128 and
  512, separating prefill from per-token decode;
- a real-activation OLMoE expert-layer diagnostic for attribution, without
  substituting that layer result for end-to-end evidence;
- stock eager, stock default, `grouped_mm`, `batched_mm`, and every optimized or
  fused backend applicable to the frozen hardware;
- both uncompiled and `torch.compile` stock variants, including each supported
  compilation mode applicable to the frozen model and hardware; any exclusion
  must be fixed and justified before candidate measurement, and a candidate's
  compilation settings must also be tested on every capable stock backend;
- the best successful stock configuration per regime as the comparison
  baseline;
- output parity against the stock eager reference, with tolerances fixed by the
  contract before candidate measurements;
- CUDA-event latency, wall time, throughput, preprocessing, routing overhead,
  and allocator measurements with setup separated from steady state.

Issue #132 implements this stock-only surface in
`benchmarks/olmoe_stock_baseline.py`. Its CI smoke enumerates the complete
regime/configuration cross product and explicit exclusions without running the
model. The real path pins the model and Transformers revisions, uses
uncompiled eager last-token logits as the semantic reference, and fixes maximum
absolute `0.125` plus relative-L2 `0.01` tolerances before any candidate exists.
Real mode refuses a reduced compile inventory. Serving prefill constructs the
KV cache inside its timed region; decode constructs the prompt cache outside
the one-token timed and allocator-peak regions while retaining its resident
bytes as the decode baseline. DeepGEMM readiness checks the full `nvcc` toolkit
version required by the pinned integration, not only PyTorch's bundled CUDA
runtime. It records routing attribution as still required rather than
fabricating it.

Issue #136 adds `benchmarks/olmoe_stock_profile.py`. The profiler consumes only
a complete real stock artifact, verifies that it is running on the same frozen
hardware and software cohort, and binds one full-model profile to the selected
best stock configuration in every regime. It requires runtime CUPTI device
kernel events—not merely CUDA memcpy or memset events—and assigns aggregated
frontend-operator self time exactly once across the predeclared categories.
Raw CUDA rows are excluded because their durations are already attached to
frontend operators; every unknown retained event name remains `unclassified`.

The same profiler captures the input and output of zero-based sparse layer 8
during a real `prefill_b1_s512` forward, then replays that exposed router and
expert program for attribution and correctness. This replay is diagnostic only.
It cannot replace any full-model row, and when the selected full-model path is
compiled the diagnostic remains an uncompiled same-backend replay while the
bound full-model profile owns compiler attribution.

Issue #133 supplies the real CUDA baseline/profile artifacts and binary target
decision; neither smoke artifact, a row-complete stock cohort, nor the isolated
layer diagnostic is a performance result. The reviewed `accept` decision, not
artifact completeness alone, is the prerequisite for an optimization issue.

Issue #139 implements that one optimization experiment. The
`beyond_matmul_stable_route` backend flattens router assignments in eager's
selected-expert-slot-then-token order, performs one stable grouping by expert,
and records one reusable plan containing token indices, slot indices, routing
weights, expert offsets, active experts, and sentinel counts. Gate/up
projection, activation and gating, down projection, BF16 weighting, and
`index_add_` aggregation consume the same plan. Unsupported training or expert
layouts use the pinned eager program and expose the fallback reason. The local
demo is bitwise exact in FP32 and BF16 test cases, including duplicate routes;
it is not CUDA or performance evidence.

`benchmarks/olmoe_stable_route_candidate.py` binds the candidate result to the
canonical checksum and environment of `olmoe_stock_baseline.json`, reruns an
eager output reference, requires five warmups and 20 CUDA-event samples for all
eight regimes, recomputes each median, and records an aggregate execution-path
audit. It also records the clean candidate git revision and module checksum and
rejects a dirty source tree. Its CI smoke has eight empty contract rows and
`performance_claim=none`. Only a real artifact that observes the new path with
no fallback and clears every gate below may change that claim.

## Measured H100 Candidate Result

Issue #139 ran the source-bound candidate at revision
`34b6f14967cc5dc80f3d436e75d59c7bfae278f9` on the exact H100 and pinned
software cohort used by the stock artifact. The candidate artifact is bound to
the canonical baseline SHA-256
`6b630ce7a174e0b29e21a3df2ab1358cf3b6c14dcf3d548c171eff228ba8436e`,
contains five warmups and 20 positive CUDA-event samples in every regime, and
recomputes each median before comparison.

| Regime | Best correct stock | Stable route plan | Latency improvement |
| --- | ---: | ---: | ---: |
| `prefill_b1_s128` | 296.094 ms | 108.659 ms | 63.30% |
| `prefill_b1_s512` | 231.673 ms | 114.945 ms | 50.38% |
| `prefill_b4_s128` | 245.158 ms | 179.354 ms | 26.84% |
| `prefill_b4_s512` | 264.093 ms | 125.833 ms | 52.35% |
| `decode_b1_p128` | 49.354 ms | 37.271 ms | 24.48% |
| `decode_b1_p512` | 49.372 ms | 36.853 ms | 25.36% |
| `decode_b8_p128` | 120.709 ms | 97.466 ms | 19.26% |
| `decode_b8_p512` | 116.120 ms | 65.393 ms | 43.69% |

All eight rows pass the predeclared maximum-absolute `0.125` and relative-L2
`0.01` correctness tolerances before their timings are interpreted. Seven rows
match the fresh eager reference exactly in the recorded metrics; the
`decode_b8_p128` row records maximum-absolute error `0.0625` and relative-L2
error `0.001772`, both within contract. The aggregate execution audit observes
4,800 candidate calls, 4,800 stable route-plan calls and plan builds, and zero
eager fallbacks. The minimum improvement is 19.26%, so every row clears the
10% success threshold as well as the 5% regression guard. The artifact decision
is `accept` with `performance_claim=qualified_candidate_speedup`.

This is an attributable, end-to-end result for one pinned OLMoE model, one H100
PCIe GPU, and one software revision. It does not establish upstream acceptance,
production readiness, memory savings, compile-path gains, other GPUs, other MoE
models, or future-Transformers behavior. The optimized stock rows that failed
the immutable correctness gate remain ineligible; their lower raw timings are
not silently reintroduced as baselines.

`examples/olmoe_capstone_demo.py` is the portable delivery surface for this
result. It runs offline, checks the committed artifact and source bindings,
recomputes the eight medians, throughput values, improvements, correctness
decisions, and execution audit, and only then prints the result and boundary. It
does not rerun or simulate the GPU benchmark.

The capstone succeeds only if a distinct provenance-enabled path:

- improves median end-to-end latency or throughput by at least 10% against the
  best applicable stock strategy for at least one required regime;
- regresses median end-to-end latency by no more than 5% on every other required
  regime;
- passes correctness everywhere before performance is interpreted;
- retains explicit stock and dense/generic fallbacks; and
- is delivered as an externally reviewable patch or a reproducible maintained
  fork whose changed execution is visible in the artifact.

A memory-only or metadata/control result is useful secondary evidence but does
not satisfy the performance north star. A win only against eager when an
existing stock optimized backend is faster also does not satisfy it.

## Rejection Criteria

Reject OLMoE as the capstone before broad implementation when any of these is
true:

- no accessible hardware cohort can run the pinned full model honestly;
- the best stock backend already removes the identified cost;
- no distinct provenance-enabled execution can be stated before coding;
- the only apparent gain is against a knowingly weak baseline;
- the effect exists only in a synthetic or isolated layer and does not produce
  a plausible end-to-end path;
- the change would require a broad tensor IR or kernel platform before one
  focused result; or
- correctness or the predefined regression bound cannot be maintained.

A rejection is a successful target-validation result, not permission to weaken
the final success condition. The next issue should select another external
project and contraction with the same gate.

## Relationship To Existing Evidence

The current matrix IR, Conv1d artifact, and PEFT artifacts remain accurate
bounded evidence for semantics, provenance visibility, fallback, and benchmark
discipline. They do not satisfy this capstone because they do not show an
attributable external performance improvement. The accepted OLMoE result
supersedes PEFT CUDA issues #123 through #126 as the project finish line; their
code and contracts remain available if PEFT is later selected again under a new
evidence contract.
