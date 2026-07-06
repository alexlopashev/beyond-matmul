# Provenance-Aware Linear Operator IR

The IR represents fixed-weight linear and affine maps with shape
`(out_features, in_features)`. Applying a linear operator to a row-major batch
`x` computes `x @ W.T`; applying an affine operator computes `x @ W.T + b`.

## Schema

```python
OperatorMetadata(
    kind="dense | diagonal | sparse_coo | low_rank | affine | conv1d | conv1d_channel | codebook | bitpacked_binary",
    shape=(out_features, in_features),
    provenance=Provenance(
        source="framework node, compiler pass, analyzer, or fallback",
        framework="torch.fx | jax | tensorflow | python | ...",
        expression="original expression if known",
        inputs=("x", "weight"),
        transform_history=("conv -> im2col",),
        confidence=1.0,
    ),
    structure={...},
    contract=ApproximationContract(
        mode="exact | approximate",
        metric="none | matrix_relative_frobenius | output_relative_l2",
        epsilon=0.0,
        observed_error=None,
        sample_count=0,
    ),
    quantization=QuantizationSpec(...),
    reuse=ReuseBudget(
        fixed_weight=True,
        preprocessing_cost=0.0,
        amortize_over_calls=1,
        cache_bytes=0,
    ),
    layout=LayoutSpec(...),
    hardware=HardwareTarget(...),
    lowerings=("dense_gemm",),
)
```

## Lowering Contract

A lowering is valid when all required contracts hold:

- Exactness: exact operators are always allowed if the backend supports them.
- Error: approximate operators require `observed_error <= epsilon` under the
  requested metric.
- Reuse: preprocessing-heavy lowerings require enough calls to amortize setup.
- Backend: the target must advertise support for the lowering family.
- Layout: physical layout must be compatible or cheaply convertible.

## Quantized Fixed-Weight Contract

Quantized operators are fixed-weight operators whose packed representation is
part of the contract. Dense dequantization is always a semantic fallback for
equivalence checks and portability, but it is not provenance preservation by
itself. A lowering preserves quantized provenance only when the operator keeps
the integer codes, binary signs, codebook, scale, zero point, and packing
metadata needed to apply the weight without first materializing an anonymous
dense matrix.

Exact quantized contracts:

- Codebook weights are exact when every stored code indexes an explicit
  codebook value and those values are the source weight values. The payload is
  `codes` plus `codebook`, with `QuantizationSpec(scheme="codebook", bits=...,
  codebook_size=...)`.
- Bitpacked binary weights are exact when every source weight is representable
  as `scale * sign` for a single tensor-wide scale and `sign in {-1, 1}`. The
  payload is `signs` plus `scale`, with
  `QuantizationSpec(scheme="symmetric_binary", bits=1, scale=...)`.
- Per-tensor affine integer weights are exact only when the IR carries the
  integer payload plus one tensor-wide `scale` and `zero_point`, interpreted as
  `(integer - zero_point) * scale`. `QuantizationSpec` can name that metadata,
  but there is not yet a dedicated packed affine-integer operator, so the Torch
  frontend must not claim this capture today. The missing payload operator is
  tracked as follow-up issue #52.

Approximate quantized contracts:

- A codebook or bitpacked binary operator is approximate when it is produced by
  quantizing an arbitrary dense matrix into a smaller value set. The operator
  must carry an `ApproximationContract` and planner acceptance depends on the
  requested error metric, observed error, and epsilon.
- Dense fallback for an approximate quantized operator applies the dequantized
  dense equivalent of the candidate. It preserves output semantics for the
  candidate, not the original source weights unless the approximation contract
  says the candidate is exact under the requested metric.

Intentionally unsupported today:

- Per-channel, per-axis, per-group, and activation-dependent quantization
  schemes are not captured as quantized fixed-weight operators yet. The
  existing `QuantizationSpec.per_axis` field is descriptive metadata only until
  an operator payload, layout rule, and tests define axis semantics.
- Asymmetric binary, ternary, mixed-precision, and blockwise quantization need
  their own payload and lowering contracts before frontend support can be
  claimed. Ternary weights may be represented as a codebook only when the exact
  codebook contract above holds.

Current IR mapping:

| IR object | Contract status | Preserved payload | Dense fallback boundary |
| --- | --- | --- | --- |
| `QuantizationSpec` | Metadata descriptor | Scheme, bits, optional codebook size, scale, zero point, and descriptive per-axis tag | Does not by itself prove packed provenance; an operator payload must carry the codes or integers. |
| `CodebookOperator` | Exact or approximate depending on `ApproximationContract` and provenance | Integer `codes`, floating `codebook`, codebook size, bit width | `to_dense()` dequantizes by table lookup and loses lookup provenance if lowered to `DenseOperator`. |
| `BitpackedBinaryOperator` | Exact for tensor-wide scaled binary sources; approximate for binary approximation candidates | Sign matrix, tensor-wide scale, one-bit symmetric quantization metadata | `to_dense()` expands signs to floats and loses bitpacked storage provenance if lowered to `DenseOperator`. |

## Examples

### Diagonal

```python
DiagonalOperator([0.5, 1.0, 2.0])
```

Structure: one vector, no off-diagonal arithmetic.
Lowerings: `diagonal_kernel`, `dense_gemm`.

### Low Rank

```python
LowRankOperator(left=U, right=V)  # W = U @ V
```

Structure: rank `r`, factor shapes `(out, r)` and `(r, in)`.
Lowerings: `low_rank_product`, `dense_gemm`.

### Affine

```python
AffineOperator(LowRankOperator(left=U, right=V), bias=b)
```

Structure: any linear operator plus an output bias vector.
Lowerings: fused bias variants such as `low_rank_product_bias`, plus
`dense_gemm_bias`.

### Convolutional

```python
Convolution1DOperator(kernel=[1.0, -1.0, 2.0], input_length=128)
```

Structure: single-channel Toeplitz dense equivalent, local kernel reuse.
Lowerings: `conv1d_direct`, `dense_gemm`.

```python
MultiChannelConvolution1DOperator(
    weight=[
        [[1.0, -1.0, 2.0], [0.5, 0.25, -0.5]],
        [[-0.25, 0.75, 1.5], [1.0, -0.5, 0.25]],
    ],
    input_length=128,
    groups=1,
)
```

Structure: block-Toeplitz dense equivalent for fixed valid Conv1d. Input rows
flatten `(in_channels, input_length)` and output rows flatten
`(out_channels, output_length)` in channel-major order. `groups` partitions
input and output channels with PyTorch-style weight shape
`(out_channels, in_channels / groups, kernel_size)`. Ungrouped, grouped, and
depthwise forms keep explicit metadata for `groups`,
`input_channels_per_group`, and `group_type`.
Lowerings: `conv1d_channel_direct`, `conv1d_grouped_direct`,
`conv1d_depthwise_direct`, `dense_gemm`.

### Sparse

```python
SparseCOOOperator(rows, cols, values, operator_shape=(out, in))
```

Structure: only nonzero entries.
Lowerings: `sparse_kernel`, `dense_gemm`.

### Codebook

```python
CodebookOperator(codes, codebook=[-1.0, 0.0, 0.5, 1.0])
```

Structure: weight values are looked up from a small table. Exact when the
codes and codebook are the source representation; approximate when produced by
quantizing a denser source under an `ApproximationContract`.
Lowerings: `codebook_kernel`, `dense_gemm`.

### Bitpacked

```python
BitpackedBinaryOperator(signs, scale=0.125)
```

Structure: one bit per sign plus one tensor-wide scale. Exact only for
tensor-wide scaled binary source weights; approximate when produced as a binary
candidate for denser weights.
Lowerings: `bitpacked_kernel`, `dense_gemm`.

### Dense Fallback

```python
DenseOperator(matrix)
```

Structure: no known structure beyond dense storage.
Lowerings: `dense_gemm`.
