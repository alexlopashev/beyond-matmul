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
`torch.fx` only when PyTorch is installed. `capture_torch_fx_operators` extracts
executable IR payloads for:

- nested `F.linear` or `nn.Linear` factors, including bias as an
  `AffineOperator`
- named adapter factor pairs such as `lora_A`/`lora_B`, `down`/`up`, and
  `A`/`B`, including cases where forward uses a nearby merged dense weight
- `Embedding` followed by `Linear`, represented as a low-rank projection over
  one-hot token inputs
- fixed-weight `x @ weight.T`, `torch.matmul`, `torch.mm`, and
  `torch.addmm(bias, x, weight.T)` patterns, represented as `DenseOperator` or
  `AffineOperator(DenseOperator)` when the right-hand weight is explicitly
  transposed
- Torch exported-program graphs for the tested fixed-weight `addmm` and nested
  linear forms, where parameter names are recovered through the export graph
  signature and state dictionary rather than module nesting
- valid `nn.Conv1d` modules and fixed-weight functional `conv1d`, represented
  as `Convolution1DOperator`, `MultiChannelConvolution1DOperator`, or affine
  convolution when an input length is known from `sample_inputs` shape
  propagation or a module input-length hint, including valid grouped/depthwise
  channel partitions

`capture_torch_fx_linear_operators` remains as a backward-compatible alias.

See `docs/torch_frontend_coverage.md` for the current support matrix. Open
frontend targets remain stride/padding/dilation Conv1d variants, quantized
modules with executable packed-payload rules, and broader exported graph
operator coverage.
