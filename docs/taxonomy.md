# Taxonomy of Lost Structure

Dense GEMM is a good fallback representation, but it can erase useful facts
about where the matrix came from. This table is the initial taxonomy for the
project.

| Source computation | Dense lowering | Lost structure | Possible better lowering |
| --- | --- | --- | --- |
| Convolution | `im2col(x) @ W_col` or Toeplitz matrix | Local receptive fields, weight sharing, stride, padding, channel grouping | Direct convolution, Winograd/FFT where valid, block-sparse Toeplitz operator |
| Attention mask | Dense score/value matmul with mask applied nearby | Causal, banded, block, page, or sliding-window sparsity | Mask-aware block sparse attention, causal kernels, paged attention |
| Graph propagation | Dense or sparse adjacency times features | Graph topology, degree normalization, repeated message pattern | CSR/CSC sparse kernels, fused message passing, sampled neighborhood kernels |
| Embedding plus projection | One-hot dense matmul followed by linear map | Lookup semantics, repeated token ids, vocabulary sparsity | Gather plus projection, fused embedding projection, cached hot-token products |
| Low-rank/adapters | Materialized `U @ V` or merged adapter weight | Rank bound, adapter provenance, separately reusable factors | Two-stage low-rank product, fused base plus adapter |
| Quantized weights | Dequantized dense matrix | Codebook, scale, zero point, packing, per-axis quantization | Packed int kernels, codebook/LUT kernels, bitpacked binary/ternary kernels |
| Permutation/reshape/broadcast | Explicit dense matrix | Index map, repeated rows/cols, no arithmetic | View/index kernel, gather/scatter, broadcast-aware fusion |
| Diagonal/scaling/gating | Dense diagonal matrix | Elementwise multiply semantics | Diagonal kernel or fused scale |
| Block structure/Kronecker | Expanded dense matrix | Repeated tiles, tensor product factors | Block kernel, Kronecker product application, tile cache |
| Fixed-weight repeated inference | Recomputed or rediscovered dense lowering | Stable weights and amortizable preprocessing | Preprocessed sparse/packed/codebook/low-rank lowering with cache contract |

The core research claim is not that dense GEMM is bad. It is that dense GEMM
should be one lowering in a richer operator space, not the default semantic IR.

## Mask-Aware Attention Scope Decision

The first complete artifact should not implement full masked attention. It
should treat masks as evidence for future work while keeping the current
fixed-weight linear boundary clear:

| Scope option | Linear fixed-weight part | Outside first artifact scope | Decision |
| --- | --- | --- | --- |
| Fixed mask as sparse linear operator | A constant causal, banded, block, or page mask can be represented as a fixed sparse linear map when applied to values or features independent of input-dependent scores. | Softmax normalization, score-dependent sparsity, KV-cache policy, and dynamic sequence layout. | Plausible future PR after a sparse-mask IR contract exists. |
| Attention projection with structured weight | Query/key/value/output projections with fixed structured weights are ordinary fixed-weight linear maps, already aligned with dense, low-rank, sparse, or adapter-style IR concepts. | The attention score computation, mask application around softmax, and dynamic token-dependent behavior. | In scope only as a projection case study, not as masked attention support. |
| Full masked attention block | Individual projections may be fixed-weight linear maps. | End-to-end attention combines dynamic activations, softmax, masking semantics, cache layout, and backend kernel policy. | Future work note, not a PR-sized first-artifact implementation. |

Decision: keep the first artifact focused on fixed-weight linear projections
and clearly mark full masked attention as future work. A later PR-sized issue
can add fixed-mask metadata and sparse-mask lowering only after it defines
exact mask shape, broadcast, and dynamic-sequence contracts with tests.
