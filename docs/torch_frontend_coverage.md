# Torch Frontend Coverage

This matrix tracks exact fixed-weight inference patterns that the Torch FX
frontend can preserve before they collapse into anonymous dense matmul. "Next"
means the pattern is a good near-term target, while "Unsupported" means the
current IR or capture rules need more design before claiming coverage.

| Pattern | Status | Captured IR | Notes |
| --- | --- | --- | --- |
| `nn.Linear` nested under `nn.Linear` | Supported | `LowRankOperator` or `AffineOperator` | Captures two fixed linear factors as one low-rank projection. |
| Nested `F.linear` | Supported | `LowRankOperator` or `AffineOperator` | Fixed right and left factors must resolve from FX attributes. |
| Named adapter pairs | Supported | `LowRankOperator` or `AffineOperator` | Scans pairs such as `down`/`up`, `lora_A`/`lora_B`, and merged-weight hints. |
| `Embedding` followed by projection | Supported | `LowRankOperator` or `AffineOperator` | Represents the embedding table as a projection over one-hot token inputs. |
| Single-channel `nn.Conv1d` | Supported | `Convolution1DOperator` or `AffineOperator` | Runtime activation, fixed weights, scalar stride/padding/dilation, groups 1. |
| Multi-channel `nn.Conv1d` | Supported | `MultiChannelConvolution1DOperator` or `AffineOperator` | Runtime activation, fixed weights, channel-major flattened rows, scalar stride/padding/dilation. |
| Functional `conv1d` | Supported | `Convolution1DOperator`, `MultiChannelConvolution1DOperator`, or `AffineOperator` | Captures runtime activation with fixed `torch.nn.functional.conv1d`/`torch.conv1d` weights, optional fixed bias, scalar stride/padding/dilation, and grouped/depthwise forms. |
| `operator.matmul` / `x @ weight.T` | Supported | `DenseOperator` | Requires a runtime left operand and an explicitly transposed fixed right operand. |
| `torch.matmul` | Supported | `DenseOperator` | Same exact fixed-weight rule as `x @ weight.T`. |
| `torch.mm` | Supported | `DenseOperator` | Supports function and method forms when the right operand is fixed and explicitly transposed. |
| `torch.addmm` | Supported | `AffineOperator(DenseOperator)` | Supports `torch.addmm(bias, x, weight.T)` with fixed bias and default `alpha=1`, `beta=1`. |
| Grouped/depthwise `Conv1d` | Supported | `MultiChannelConvolution1DOperator` or `AffineOperator` | Tested for fixed-weight `nn.Conv1d` modules and functional `conv1d`; preserves explicit `groups`, `group_type`, stride, padding, and dilation metadata. |
| Stride/padding/dilation `Conv1d` variants | Supported | `Convolution1DOperator`, `MultiChannelConvolution1DOperator`, or `AffineOperator` | Scalar 1D parameters only; invalid output lengths and multi-dimensional parameter shapes are rejected. |
| `Conv2d` | Unsupported | Not captured | Needs 2D convolution IR and layout decisions before frontend matching. |
| Quantized `nn.Linear` | Supported | `PackedAffineQuantizedOperator` or `AffineOperator(PackedAffineQuantizedOperator)` | Captures fixed per-tensor affine integer module weights with optional fixed bias, preserving integer payload, scale, zero point, bit width, and integer range. Per-channel/per-axis and dynamic quantized modules are ignored cleanly. |
| Quantized `Conv1d`/`Conv2d` | Unsupported | Not captured | Needs convolution-specific packed payload and layout rules before frontend matching can claim quantized convolution provenance. |
| Exported graph fixed-weight `addmm` and nested linear | Supported | `DenseOperator`, `AffineOperator(DenseOperator)`, `LowRankOperator`, or `AffineOperator(LowRankOperator)` | Recovers fixed parameter/buffer values through graph signature state, with provenance notes marking exported recovery. |
| Dynamic-weight matmul/addmm | Unsupported | Not captured | Fixed-weight reuse is the scope; runtime weights are ignored cleanly. |

## Updating Supported Rows

When a frontend support issue promotes a row to `Supported`, add or update its
entry in `scripts/check_torch_frontend_coverage.py` so the row points to at
least one executable test, demo, or evidence file. The local CI check fails if a
supported row has no mapping, if a mapping no longer corresponds to a supported
row, or if a mapped file/test token disappears.

## Current Capture Rule

Dense matmul capture is intentionally narrow: the frontend accepts only
`x @ weight.T`-style orientation, including equivalent `torch.matmul`,
`torch.mm`, `x.matmul`, `x.mm`, `weight.t()`, and `weight.transpose(0, 1)`
forms. It does not infer orientation from shapes alone.

`torch.addmm` follows the same fixed-weight rule and additionally requires a
fixed one-dimensional bias. Non-default `alpha` or `beta` values are ignored
until scaling is represented explicitly and tested.

Exported-program recovery is explicit rather than shape inferred: placeholders
must map through a graph signature to fixed parameter, buffer, or constant state.
Recovered events record `exported_graph_state` notes and add
`exported_graph_constant_recovery` to provenance history. Shape-only or dynamic
placeholders are ignored cleanly, and untransposed right-hand weights are still
ambiguous.

Conv1d capture is exact but intentionally narrow. Module and functional forms
are supported when weights and optional bias are fixed and stride, padding, and
dilation are scalar 1D parameters with positive stride/dilation and non-negative
padding, and the convolution input is a runtime activation rather than a fixed
buffer. Multi-channel Conv1d rows flatten inputs as
`(in_channels, input_length)` and outputs as `(out_channels, output_length)` in
channel-major order. Grouped and depthwise rows preserve PyTorch-style group
partitions with explicit `groups`, `input_channels_per_group`, and
`group_type` metadata; the real Torch tests cover grouped modules and
grouped/depthwise functional forms against PyTorch outputs. Unsupported
parameter shapes, invalid output lengths, dynamic weights, dynamic bias values,
and fixed-buffer inputs are ignored cleanly rather than captured as structured
Conv1d.

Quantized module capture is intentionally limited to fixed per-tensor affine
`nn.Linear` weights. `CodebookOperator` and `BitpackedBinaryOperator` can
preserve codebook and tensor-wide scaled binary payloads,
`PackedAffineQuantizedOperator` can preserve per-tensor affine integer payloads,
and dense dequantization remains a fallback. The Torch frontend maps supported
quantized linear modules to the packed affine IR without first materializing an
anonymous dense weight. Per-channel, per-axis, dynamic quantized, and quantized
convolution modules remain unsupported until executable payload and layout rules
exist for those contracts. Frontend recovery must not silently dequantize an
unsupported quantized module into `DenseOperator` and call that
provenance-preserving quantized capture.
