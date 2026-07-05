# Beyond Matmul

This repository explores whether computations that are often lowered to dense
`A @ B` can stay cheaper and more meaningful as provenance-aware linear
operators.

The current artifact is scoped to fixed-weight inference:

- a provenance-aware linear operator IR
- exact operators for dense, diagonal, sparse COO, low-rank, convolutional,
  codebook, and bitpacked weights
- cheap structure recovery from dense matrices
- product-aware approximation scoring
- a planner that chooses a valid lowering under exactness, error, reuse, and
  backend contracts
- a synthetic benchmark suite against dense GEMM-style application

## Quick Start

```bash
python3 -m unittest discover -s tests
python3 examples/fixed_weight_inference_demo.py
python3 examples/torch_fx_frontend_demo.py
python3 benchmarks/fixed_weight.py
```

Install project dependencies with uv:

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
- `docs/taxonomy.md`: taxonomy of dense lowerings and lost structure
- `docs/ir_spec.md`: IR schema and examples
- `docs/research_outline.md`: paper-shaped outline and evaluation plan

## Crisp Contribution

This work introduces a provenance-aware linear-operator IR and planner that
preserves or recovers structure behind dense matmuls, then selects exact or
bounded-error lowerings that reduce inference cost relative to dense GEMM.
