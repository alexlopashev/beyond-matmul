# Beyond Matmul

This repository explores whether computations that are often lowered to dense
`A @ B` can stay cheaper and more meaningful as provenance-aware linear
operators.

The current artifact is scoped to fixed-weight inference:

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
- `docs/live_conv1d_benchmark_contract.md`: contract for the live Whisper
  Conv1d layer-level dense-vs-direct benchmark target
- `docs/peft_capstone_benchmark_contract.md`: contract for the first PEFT plus
  Transformers capstone benchmark target and JSON artifact
- `docs/peft_fork_setup.md`: setup, sync, branch, and issue-mapping rules for
  the PEFT fork integration branch
- `docs/peft_low_rank_provenance_design.md`: first PEFT low-rank provenance
  integration design and #78 handoff checklist
- `docs/evidence_matrix.md`: whitepaper claim-to-evidence map and current
  unsupported-claim boundaries
- `docs/completion_audit.md`: final first-artifact audit of claims, evidence,
  limitations, validation commands, and optional follow-ups
- `docs/handoff_next_layer.md`: current state and next-layer handoff
- `whitepaper/main.tex`: final-draft whitepaper source and completion criteria
- GitHub wiki: concise north star, operating loop, coverage snapshot, and
  artifact map for humans and agents
- `AGENTS.md`: operating contract for issue-driven, worktree-based agent loops

## Crisp Contribution

This work introduces a provenance-aware linear/affine-operator IR and planner
that preserves or recovers structure behind dense matmuls, then selects exact or
bounded-error lowerings that reduce inference cost relative to dense GEMM.

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
- supported result: successful seq16 and seq64 fork rows expose structured
  LoRA provenance with dense fallback and upstream-output parity
- unsupported result: seq128 fails across all baselines,
  `summary.benchmark_ready=false`, and `summary.performance_claim=none`

The #82 retrospective closed the capstone without creating an upstreaming,
TorchBench-integration, or broader PEFT expansion issue because the artifact
supports claim narrowing rather than a performance-readiness claim.

Current completion status: no unresolved priority-zero or priority-one blocker
is known against the artifact thesis; `docs/completion_audit.md` records the
latest issue audit and residual risks.
