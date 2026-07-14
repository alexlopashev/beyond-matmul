# OLMoE Routed Tensor-Program Capstone

Status: provisional target-validation candidate

Decision date: 2026-07-14

Tracking issue: #129

## Decision

Beyond Matmul will evaluate an open language model before expanding its local
IR. The first candidate is AllenAI's OLMoE-1B-7B model through Hugging Face
Transformers. OLMoE is a useful tensor case because its sparse mixture-of-experts
layer retains token routing, expert identity, routing weights, and 3D expert
weights rather than presenting the computation as one anonymous matrix.

This is a candidate, not a foregone conclusion. Current Transformers already
contains provenance-aware expert backends. An existing upstream
eager-versus-grouped win demonstrates the broader thesis but cannot count as a
Beyond Matmul result. OLMoE must be rejected if target validation cannot find a
remaining, attributable opportunity against the best stock implementation.

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

Issue #133 owns the real CUDA baseline/profile artifacts and binary target
decision; neither smoke artifact, a row-complete stock cohort, nor the isolated
layer diagnostic is a performance result or permission to open an optimization
issue.

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
attributable external performance improvement. The PEFT CUDA issues #123 through
#126 are paused while this target decision is reviewed; their code and contracts
remain available if PEFT is later selected again under the stronger gate.
