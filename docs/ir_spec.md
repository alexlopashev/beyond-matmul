# Provenance-Aware Linear Operator IR

The IR represents fixed-weight linear and affine maps with shape
`(out_features, in_features)`. Applying a linear operator to a row-major batch
`x` computes `x @ W.T`; applying an affine operator computes `x @ W.T + b`.

## Schema

```python
OperatorMetadata(
    kind="dense | diagonal | sparse_coo | fixed_mask | low_rank | affine | conv1d | conv1d_channel | codebook | bitpacked_binary",
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
Convolution1DOperator(
    kernel=[1.0, -1.0, 2.0],
    input_length=128,
    stride=2,
    padding=1,
    dilation=1,
)
```

Structure: single-channel Toeplitz dense equivalent, local kernel reuse, and
explicit PyTorch-style `stride`, `padding`, and `dilation` metadata. Padded
positions are implicit zeros in both direct application and dense
materialization.
Lowerings: `conv1d_direct`, `dense_gemm`.

```python
MultiChannelConvolution1DOperator(
    weight=[
        [[1.0, -1.0, 2.0], [0.5, 0.25, -0.5]],
        [[-0.25, 0.75, 1.5], [1.0, -0.5, 0.25]],
    ],
    input_length=128,
    groups=1,
    stride=1,
    padding=0,
    dilation=1,
)
```

Structure: block-Toeplitz dense equivalent for fixed Conv1d with explicit
`stride`, `padding`, and `dilation`. Input rows flatten
`(in_channels, input_length)` and output rows flatten `(out_channels,
output_length)` in channel-major order. `groups` partitions input and output
channels with PyTorch-style weight shape
`(out_channels, in_channels / groups, kernel_size)`. Ungrouped, grouped, and
depthwise forms keep explicit metadata for `groups`,
`input_channels_per_group`, `group_type`, and the derived `output_length`.
Lowerings: `conv1d_channel_direct`, `conv1d_grouped_direct`,
`conv1d_depthwise_direct`, `dense_gemm`.

### Sparse

```python
SparseCOOOperator(rows, cols, values, operator_shape=(out, in))
```

Structure: only nonzero entries.
Lowerings: `sparse_kernel`, `dense_gemm`.

### Fixed Mask

```python
FixedMaskOperator(mask, pattern="causal_band")
```

Structure: a binary, fixed mask applied as an exact sparse linear map over
values or features, independent of input-dependent attention scores.
Lowerings: `fixed_mask_sparse`, `dense_gemm`.

### Codebook

```python
CodebookOperator(codes, codebook=[-1.0, 0.0, 0.5, 1.0])
```

Structure: weight values are looked up from a small table.
Lowerings: `codebook_kernel`, `dense_gemm`.

### Bitpacked

```python
BitpackedBinaryOperator(signs, scale=0.125)
```

Structure: one bit per sign plus a scale.
Lowerings: `bitpacked_kernel`, `dense_gemm`.

### Dense Fallback

```python
DenseOperator(matrix)
```

Structure: no known structure beyond dense storage.
Lowerings: `dense_gemm`.
