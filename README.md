# Beyond Matmul

This repository explores whether computations that are often lowered to dense
`A @ B` can stay cheaper and more meaningful as provenance-aware linear
operators.

The current artifact is scoped to fixed-weight inference:

- a provenance-aware linear and affine operator IR
- exact operators for dense, diagonal, sparse COO, fixed-mask, low-rank,
  convolutional, codebook, and bitpacked weights
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
- `examples/case_study_artifacts.py`: machine-readable adapter, Conv1d, and
  fixed-mask case-study evidence while preserving the human-readable demo paths
- `examples/torch_coverage_demo.py`: Torch frontend coverage smoke demo for
  supported fixed-weight patterns
- `docs/taxonomy.md`: taxonomy of dense lowerings and lost structure
- `docs/ir_spec.md`: IR schema and examples
- `docs/torch_frontend_coverage.md`: support matrix for Torch FX capture
- `docs/research_outline.md`: paper-shaped outline and evaluation plan
- `docs/benchmark_artifacts.md`: benchmark JSON schemas, regeneration commands,
  and CI artifact names
- `docs/evidence_matrix.md`: whitepaper claim-to-evidence map and current
  unsupported-claim boundaries
- `docs/handoff_next_layer.md`: current state and next-layer handoff
- `whitepaper/main.tex`: first-draft whitepaper source and completion criteria
- GitHub wiki: concise north star, operating loop, coverage snapshot, and
  artifact map for humans and agents
- `AGENTS.md`: operating contract for issue-driven, worktree-based agent loops

## Crisp Contribution

This work introduces a provenance-aware linear/affine-operator IR and planner
that preserves or recovers structure behind dense matmuls, then selects exact or
bounded-error lowerings that reduce inference cost relative to dense GEMM.
