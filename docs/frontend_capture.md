# Frontend Capture Prototype

The main opportunity is to capture structure before a framework lowers an
operation to dense storage. The prototype exposes a `ProvenanceTracer` that can
be used by framework-specific passes.

```python
from beyond_matmul.frontend import ProvenanceTracer

tracer = ProvenanceTracer(framework="python")
conv = tracer.conv1d("stem.conv", kernel=[1.0, -1.0, 2.0], input_length=128)
diag = tracer.diagonal("gate", [0.5, 1.0, 0.25])
```

For a real compiler pass:

1. Match structured source operations such as conv, masked attention, gather,
   diagonal multiply, LoRA factor application, quantized linear, and reshape.
2. Emit a `LinearOperator` with provenance, structure, layout, reuse, and
   hardware hints.
3. Delay densification until the planner has evaluated exact and approximate
   lowerings.
4. Fall back to `DenseOperator` when provenance is unavailable or a backend lacks
   a valid lowering.

`capture_torch_fx_patterns` is a dependency-free placeholder that uses
`torch.fx` only when PyTorch is installed. It returns coarse events today; the
next step is to attach real payload extraction for `linear`, `matmul`, `conv`,
embedding, and quantized modules.
