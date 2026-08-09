# Beyond Matmul Article Series

This series is the public, engineer-facing narrative for the Beyond Matmul
initiative. It explains the thesis, shows the evidence, and separates what the
project has measured from what it still hopes to generalize.

The intended audience is ML systems engineers, compiler and kernel engineers,
and technically curious model builders. Each article should stand alone while
linking to executable evidence for readers who want to inspect the details.

## Editorial Contract

- Dense GEMM is an excellent lowering and remains a valid fallback.
- A benchmark result is described with its model, hardware, software, sample,
  correctness, and comparison boundaries.
- A bespoke optimization is not presented as an automatic compiler result.
- Proxy measurements, isolated diagnostics, and end-to-end performance evidence
  are named separately.
- Negative results and unsupported claims remain visible.

## Sequence

1. [GEMM Is a Lowering, Not the Program](01-gemm-is-a-lowering-not-the-program.md)
   - Introduces the thesis: preserve computation semantics long enough to choose
     an appropriate execution strategy instead of densifying by default.
2. **One Route Plan, Eight Workloads** *(planned)*
   - Opens the OLMoE case study: baseline selection, profiling, the functional
     change, correctness, measured results, and tail-latency caveats.
3. **Evidence Before Speedups** *(planned)*
   - Explains frozen contracts, source-bound artifacts, correctness-first gates,
     independent review, and why benchmark discipline is part of the system.
4. **From Bespoke Wins to a Provenance-Aware Planner** *(planned)*
   - Separates the implemented prototype from the longer-term goal of capturing
     recurring semantics and selecting lowerings automatically.

Future articles should be added only when their evidence and claim boundary are
clear. This index is an editorial sequence, not a commitment to speculative
implementation work.
