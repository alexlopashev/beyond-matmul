# Beyond Matmul

This repository explores whether tensor contractions that are often lowered to
generic GEMM, batched GEMM, grouped GEMM, or `einsum` can stay cheaper and more
meaningful when their computation provenance is preserved. Matrix
multiplication is the rank-2 case, not the research boundary.

The implemented first artifact remains scoped to matrix-shaped fixed-weight
inference:

- a provenance-aware linear and affine operator IR
- exact operators for dense, diagonal, sparse COO, fixed-mask, low-rank,
  convolutional, codebook, bitpacked, and packed affine quantized weights
- cheap structure recovery from dense matrices
- product-aware approximation scoring
- a planner that chooses a valid lowering under exactness, error, reuse, and
  backend contracts
- a synthetic benchmark suite against dense GEMM-style application

## Quick Start

Install mise, trust the local tool config, and install pinned tools:

```bash
sh scripts/bootstrap
```

Then install dependencies and run the project checks and demos through the pinned
toolchain:

```bash
mise exec -- uv sync
mise exec -- uv run python -m unittest discover -s tests
mise exec -- uv run python examples/fixed_weight_inference_demo.py
mise exec -- uv run python examples/torch_fx_frontend_demo.py
mise exec -- uv run python examples/adapter_workload_demo.py
mise exec -- uv run python examples/conv1d_workload_demo.py
mise exec -- uv run python examples/torch_coverage_demo.py
mise exec -- uv run python examples/case_study_artifacts.py --json-output docs/results/workload_case_studies.json
mise exec -- uv run python benchmarks/fixed_weight.py
```

If your shell has mise activation configured, the shorter forms work too:

```bash
uv sync
uv run python examples/torch_fx_frontend_demo.py
```

## Repository Map

- `beyond_matmul/ir.py`: IR metadata and operator implementations
- `beyond_matmul/analyzer.py`: dense-matrix structure recovery probes
- `beyond_matmul/approximations.py`: low-rank, sparse, codebook, and binary
  approximations with output-level error metrics
- `beyond_matmul/planner.py`: fixed-weight lowering planner
- `beyond_matmul/frontend.py`: prototype capture helpers for preserving
  provenance before densification
- `examples/fixed_weight_inference_demo.py`: end-to-end provenance, recovery,
  planning, error, and timing demo
- `examples/torch_fx_frontend_demo.py`: PyTorch FX demo that captures a LoRA-style
  low-rank linear pattern before densification
- `examples/adapter_workload_demo.py`: tiny PyTorch adapter case study with a
  merged dense weight and recovered low-rank factors
- `examples/conv1d_workload_demo.py`: tiny PyTorch module and functional Conv1d
  case study comparing convolution provenance against dense materialization
- `examples/case_study_artifacts.py`: machine-readable adapter, Conv1d,
  fixed-mask, and quantized-linear case-study evidence while preserving the
  human-readable demo paths
- `examples/torch_coverage_demo.py`: Torch frontend coverage smoke demo for
  supported fixed-weight patterns
- `docs/taxonomy.md`: taxonomy of dense lowerings and lost structure
- `docs/ir_spec.md`: IR schema and examples
- `docs/torch_frontend_coverage.md`: support matrix for Torch FX capture
- `docs/research_outline.md`: paper-shaped outline and evaluation plan
- `docs/benchmark_artifacts.md`: benchmark JSON schemas, regeneration commands,
  and CI artifact names
- `docs/live_conv1d_benchmark_contract.md` and
  `benchmarks/live_conv1d_whisper.py`: live Whisper Conv1d layer-level
  dense-vs-direct benchmark contract and measured artifact harness
- `docs/peft_capstone_benchmark_contract.md`: contract for the first PEFT plus
  Transformers capstone benchmark target and JSON artifact
- `docs/peft_multi_adapter_serving_benchmark_contract.md`: contract for the
  measured PEFT multi-adapter serving benchmark comparing factor provenance
  with dense merged serving strategies
- `docs/hardware_backed_production_benchmark_contract.md`: future-work
  contract for the now-paused PEFT hardware roadmap
- `docs/olmoe_tensor_contraction_capstone.md`: provisional open-LLM target,
  routed tensor-program definition, upstream baseline audit, and rejection
  gate
- `benchmarks/olmoe_stock_baseline.py`: pinned stock-only OLMoE prefill/decode
  harness with explicit backend, compilation, correctness, timing, and
  unavailable-row semantics
- `docs/peft_fork_setup.md`: setup, sync, branch, and issue-mapping rules for
  the PEFT fork integration branch
- `docs/peft_low_rank_provenance_design.md`: first PEFT low-rank provenance
  integration design and #78 handoff checklist
- `docs/evidence_matrix.md`: whitepaper claim-to-evidence map and current
  unsupported-claim boundaries
- `docs/completion_audit.md`: historical first-artifact audit plus the current
  project-level completion correction and residual risks
- `docs/handoff_next_layer.md`: current state and next-layer handoff
- `whitepaper/main.tex`: cumulative research draft and completion criteria
- GitHub wiki: concise north star, operating loop, coverage snapshot, and
  artifact map for humans and agents
- `AGENTS.md`: operating contract for issue-driven, worktree-based agent loops

## Crisp Contribution

The first artifact introduces a provenance-aware linear/affine-operator IR and
planner that preserves or recovers structure behind dense matmuls. The active
research goal is stronger: demonstrate that preserved tensor-contraction
provenance causes an attributable performance improvement in an external
open-source ML project.

## Active North Star: Open LLM Routed Tensor Program

The provisional target is AllenAI's Apache-2.0
`allenai/OLMoE-1B-7B-0924` model through Hugging Face Transformers. Its MoE
layers combine token hidden states, token-to-expert routes, routing weights, and
3D expert-weight tensors. This is a routed tensor program composed of
expert-indexed gate/up and down contractions, nonlinear gating, dynamic
selection, and aggregation. Its token, selected-expert, expert, hidden, and
intermediate axes should remain visible to lowering decisions.

Current Transformers already has eager, batched, grouped, and optimized expert
backends. Reproducing an existing eager-versus-grouped speedup is therefore
background evidence, not Beyond Matmul's result. The target passes only if a
distinct provenance-enabled change beats the best applicable stock strategy by
at least 10% on a predefined end-to-end regime, preserves correctness, and
regresses no required regime by more than 5%. If target validation cannot find
that attributable gap, OLMoE is rejected before implementation expands.

The decision record and benchmark gate are in
`docs/olmoe_tensor_contraction_capstone.md`. No general tensor IR is implied by
this target selection. Merged issue #129/PR #131 establishes that contract;
issue #132 implements the stock-only harness, and issue #133 remains blocked on
the real CUDA cohort and accept-or-reject decision. The harness CI smoke runs no
OLMoE inference and supports no performance claim.

## PEFT Capstone Status

The external PEFT plus Transformers capstone is closed as a bounded proof, not
as the current roadmap target. It reached a real open-source PyTorch inference
stack and produced `docs/results/peft_transformers_lora_inference.json`.

- upstream workload: `huggingface/peft` with Transformers inference
- integration fork: `alexlopashev/peft`
- measured branch: `beyond-matmul/provenance-lora-inference`
- contract: `docs/peft_capstone_benchmark_contract.md`
- measurement style: compared upstream PEFT against the fork under pinned
  model, adapter, prompt, batch, dtype, device, warmup, and repetition settings
- supported result: successful seq16, seq64, and seq100 fork rows expose
  structured LoRA provenance with dense fallback and upstream-output parity;
  `summary.benchmark_ready=true`
- unsupported result: `summary.performance_claim=none`; the refreshed run does
  not establish a latency win, CPU peak memory remains unmeasured, and adapter
  switching is outside the single-adapter workload

The #82 retrospective closed the capstone without creating an upstreaming,
TorchBench-integration, or broader PEFT expansion issue because the artifact
supports claim narrowing rather than a performance-readiness claim.

## PEFT Multi-Adapter Serving Status

The follow-up PEFT serving benchmark for issue #98 is implemented as
`benchmarks/peft_multi_adapter_serving.py`, with a CI smoke artifact and a real
measured artifact at `docs/results/peft_multi_adapter_serving.json`.

- workload: `facebook/opt-125m` with `merchant` and `gaisb` LoRA adapters
- baselines: upstream unmerged, dense merged cache, repeated merge/unmerge, and
  Beyond Matmul factor provenance
- supported result: all required rows are present; upstream unmerged and Beyond
  Matmul rows pass correctness, dense-cache and repeated merge/unmerge rows
  also pass after the dtype fix, and Beyond Matmul rows expose structured
  factor provenance without dense fallback
- measured instrumentation: real rows record adapter-switch timings and
  platform-supported process max RSS; smoke rows and unsupported platforms keep
  explicit unavailable memory statuses
- unsupported result: `summary.performance_claim=none` and
  `summary.memory_or_control_claim=none`; the artifact does not establish a
  latency win, process-memory reduction, CUDA peak-memory reduction, or
  adapter-switching gain

Current completion status: the matrix-focused first artifact is historical and
internally bounded, but the project-level north star is open. Issue #129 and PR
#131 merged the tensor-program correction; issues #132 and #133 now separate
the baseline harness from the measured target decision. The PEFT CUDA roadmap
remains paused. `docs/completion_audit.md` records the distinction.
