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
| Narrow `nn.Conv1d` | Supported | `Convolution1DOperator` or `AffineOperator` | Single input channel, single output channel, stride 1, padding 0, dilation 1, groups 1. |
| `operator.matmul` / `x @ weight.T` | Supported | `DenseOperator` | Requires a runtime left operand and an explicitly transposed fixed right operand. |
| `torch.matmul` | Supported | `DenseOperator` | Same exact fixed-weight rule as `x @ weight.T`. |
| `torch.mm` | Supported | `DenseOperator` | Supports function and method forms when the right operand is fixed and explicitly transposed. |
| `torch.addmm` | Supported | `AffineOperator(DenseOperator)` | Supports `torch.addmm(bias, x, weight.T)` with fixed bias and default `alpha=1`, `beta=1`. |
| Functional `conv1d` | Next | Not captured | Needs call-function matching and the same shape/bias checks as module `Conv1d`. |
| Multi-channel/grouped `Conv1d` | Next | Not captured | Needs richer channel-aware convolution IR or an explicit composition. |
| `Conv2d` | Unsupported | Not captured | Needs 2D convolution IR and layout decisions before frontend matching. |
| Quantized linear/conv | Unsupported | Not captured | Needs quantization-aware IR mapping rather than lossy dense recovery. |
| Exported graph variants | Next | Not hardened | Parameter names and module structure may be erased; fixed weights need robust recovery. |
| Dynamic-weight matmul/addmm | Unsupported | Not captured | Fixed-weight reuse is the scope; runtime weights are ignored cleanly. |

## Current Capture Rule

Dense matmul capture is intentionally narrow: the frontend accepts only
`x @ weight.T`-style orientation, including equivalent `torch.matmul`,
`torch.mm`, `x.matmul`, `x.mm`, `weight.t()`, and `weight.transpose(0, 1)`
forms. It does not infer orientation from shapes alone.

`torch.addmm` follows the same fixed-weight rule and additionally requires a
fixed one-dimensional bias. Non-default `alpha` or `beta` values are ignored
until scaling is represented explicitly and tested.
