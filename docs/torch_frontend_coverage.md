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
| Single-channel `nn.Conv1d` | Supported | `Convolution1DOperator` or `AffineOperator` | Valid convolution only: stride 1, padding 0, dilation 1, groups 1. |
| Multi-channel `nn.Conv1d` | Supported | `MultiChannelConvolution1DOperator` or `AffineOperator` | Fixed weights, channel-major flattened rows, valid convolution only, groups 1. |
| Functional `conv1d` | Supported | `Convolution1DOperator`, `MultiChannelConvolution1DOperator`, or `AffineOperator` | Captures fixed `torch.nn.functional.conv1d`/`torch.conv1d` weights and optional fixed bias for valid convolution. |
| `operator.matmul` / `x @ weight.T` | Supported | `DenseOperator` | Requires a runtime left operand and an explicitly transposed fixed right operand. |
| `torch.matmul` | Supported | `DenseOperator` | Same exact fixed-weight rule as `x @ weight.T`. |
| `torch.mm` | Supported | `DenseOperator` | Supports function and method forms when the right operand is fixed and explicitly transposed. |
| `torch.addmm` | Supported | `AffineOperator(DenseOperator)` | Supports `torch.addmm(bias, x, weight.T)` with fixed bias and default `alpha=1`, `beta=1`. |
| Grouped/depthwise `Conv1d` | Next | Not captured | Needs grouped channel semantics and exact tests before support is claimed. |
| Stride/padding/dilation `Conv1d` variants | Next | Not captured | Current Conv1d IR is valid-mode only. |
| `Conv2d` | Unsupported | Not captured | Needs 2D convolution IR and layout decisions before frontend matching. |
| Quantized linear/conv | Unsupported | Not captured | Needs quantization-aware IR mapping rather than lossy dense recovery. |
| Exported graph fixed-weight `addmm` and nested linear | Supported | `DenseOperator`, `AffineOperator(DenseOperator)`, `LowRankOperator`, or `AffineOperator(LowRankOperator)` | Recovers fixed parameter/buffer values through graph signature state, with provenance notes marking exported recovery. |
| Dynamic-weight matmul/addmm | Unsupported | Not captured | Fixed-weight reuse is the scope; runtime weights are ignored cleanly. |

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
are supported when weights and optional bias are fixed, stride is 1, padding is
0, dilation is 1, and groups is 1. Multi-channel Conv1d rows flatten inputs as
`(in_channels, input_length)` and outputs as `(out_channels, output_length)` in
channel-major order.
