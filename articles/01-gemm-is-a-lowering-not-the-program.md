# GEMM Is a Lowering, Not the Program

*The next inference wins may come from preserving semantics long enough to avoid
unnecessary dense work.*

Matrix multiplication is not dead. Quite the opposite.

It is so successful that we have started mistaking an excellent implementation
primitive for a complete description of the computation.

Modern accelerators devote specialized silicon to matrix multiply-accumulate.
NVIDIA's [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
includes fourth-generation Tensor Cores and a Transformer Engine built around
fast matrix computation. Libraries then add another layer of engineering:
[CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
maps GEMM through a hierarchy of threadblock, warp, instruction, shared-memory,
and register tiles, while
[cuBLAS and cuBLASLt](https://docs.nvidia.com/cuda/cublas/) select among
specialized algorithms for a particular shape, data type, layout, and epilogue.

That stack is the product of years of hardware and software co-design. For a
large, regular, dense matrix product, the bar for doing better is extraordinarily
high.

There is still room to improve GEMM implementations. New precisions, layouts,
fusion strategies, schedulers, and hardware generations will continue to move
the frontier. But for most teams, "write a better dense matrix-multiplication
kernel" is not the highest-leverage place to begin.

The more interesting question is one level up:

> Should this computation have become an anonymous dense matrix product in the
> first place?

That question is the starting point for Beyond Matmul.

## The matrix is often the end of a story

GEMM computes a relationship of the form

```text
C = A x B
```

It tells the machine how values combine. It usually does not tell the machine
why those values are present.

Before lowering, the same product may have carried much richer facts:

- a convolution has locality, stride, dilation, groups, and shared weights;
- a low-rank adapter has two small factors and a known rank;
- a mixture-of-experts layer has token-to-expert assignments, routing weights,
  active experts, and an aggregation order;
- a quantized weight has a codebook, scale, zero point, and packed layout;
- a masked operation has a causal, block, banded, or page structure;
- a fixed-weight operation may be reused enough times to amortize preprocessing.

All of these can be represented by dense matrices. That does not mean the dense
matrix is the best source representation or the best execution plan.

A convolution can be expanded into a dense Toeplitz matrix. A rank-four update
can be multiplied into a full-rank weight. Routed tokens can be placed into
padded expert batches. Quantized values can be dequantized before application.
Those transformations are mathematically convenient, but they can materialize
zeros, duplicate parameters, erase reuse, increase memory traffic, or force the
runtime to rediscover structure that the model already knew.

Once the provenance is gone, recovering it is harder and less reliable than
preserving it.

## Lower later

The thesis is not "matmul bad."

The thesis is:

> Preserve the semantic structure of a tensor program until the runtime has
> enough information to choose a lowering deliberately.

Sometimes the deliberate choice will still be dense GEMM. A mature system must
keep that path because it is fast, portable, and often exactly right.

Sometimes the choice may be different:

- apply two low-rank factors without materializing their dense product;
- execute a convolution directly instead of building a dense linear map;
- keep codebook or packed-quantized weights in their encoded representation;
- run only the blocks or experts that are actually active;
- reuse routing or indexing metadata across the stages that consume it;
- fuse an exact computation so an intermediate never travels to external
  memory;
- accept an approximation only when an explicit output-error contract allows
  it.

This is semantic lowering: the implementation decision is informed by what the
operation means, not only by the dimensions of a matrix observed after the fact.

The distinction resembles a principle demonstrated in other successful systems.
[Halide](https://people.csail.mit.edu/jrk/halide12/halide12.pdf) separates the
algorithm - what is computed - from the schedule - how it is executed.
[FlashAttention](https://arxiv.org/abs/2205.14135) preserves the exact attention
computation while changing its IO-aware schedule so large intermediates do not
repeatedly travel between high-bandwidth memory and on-chip storage.
[PagedAttention](https://arxiv.org/abs/2309.06180) reorganizes KV cache
management around the actual serving problem.
[MegaBlocks](https://arxiv.org/abs/2211.15841) reformulates dynamic
mixture-of-experts work as block-sparse operations instead of forcing the
routing pattern into padding or token dropping.

These systems are not identical, and they are not implementations of Beyond
Matmul. They illustrate the broader point: important performance gains often
come from representing and scheduling the real computation more faithfully, not
from replacing one excellent dense kernel with a slightly better dense kernel.

## What provenance makes possible

Preserving semantics is useful only if it changes an executable decision.

For Beyond Matmul, a useful representation should answer questions such as:

1. **What operation is this?**
   Is it a generic contraction, a convolution, a routed expert program, a
   factorized update, a lookup, or a fixed sparse map?
2. **What must remain exact?**
   Are different floating-point accumulation orders acceptable? Is bounded
   approximation allowed? Against which outputs is error measured?
3. **What is fixed and what is dynamic?**
   Can weights, routes, masks, or layouts be preprocessed or reused?
4. **What does the backend support?**
   Does the target have an efficient direct kernel, or is dense GEMM the honest
   fallback?
5. **What is the complete cost?**
   Compute is only one term. Data movement, materialization, indexing, launch
   overhead, preprocessing, cache pressure, and aggregation also matter.

The long-term goal is a planner that can preserve these facts, enumerate valid
lowerings, and choose among them under explicit correctness and cost contracts.

We are not claiming to have that general system today.

The current project contains a prototype operator representation, capture and
recovery experiments, exact and approximate lowerings, dense fallback, and a
small cost-aware planner. More importantly, it now contains one end-to-end
external case study showing why the direction can matter.

## A first result: the matmuls stayed the same

Our first measured target is
[`allenai/OLMoE-1B-7B-0924`](https://arxiv.org/abs/2409.02060), an open sparse
mixture-of-experts model running through Hugging Face Transformers.

An OLMoE expert layer is more than a collection of matrix products. A router
selects experts for each token and supplies weights. The runtime gathers token
states, performs expert gate/up and down projections, applies nonlinear gating
and routing weights, and aggregates the results back into token order.

The stock eager path represented routes with a dense one-hot mask and searched
that mask for each active expert. Our candidate preserved the router's expert,
selected-slot, token, and weight relationships in one stable route plan. Each
expert then consumed a contiguous slice of that plan.

The expert matrix multiplications did not change. The improvement came from the
tensor program around them.

On one pinned H100 PCIe GPU, one model revision, and one pinned software cohort,
the route-plan backend reduced median end-to-end CUDA-event latency by
**19.26% to 63.30%** across eight prefill and decode regimes. All eight rows
passed correctness gates. The execution audit observed 4,800 route-plan calls
and zero eager fallbacks.

This is deliberately a narrow claim. The implementation was hand-engineered for
the pinned OLMoE expert interface. It is not a new GEMM kernel, an automatic
compiler discovery, a multi-model result, a memory-savings result, or evidence
of production readiness. The measurements used five warmups and 20 timed
samples per regime on one GPU. Broader hardware behavior and strong
tail-latency conclusions require more experiments.

But the result is enough to establish something useful: a computation can spend
less time end to end when its semantics remain available, even when its central
GEMMs are unchanged.

The next article in this series will open that case study completely: how the
baseline was selected, what profiling showed, the functional code difference,
why accumulation order mattered, what the measurements support, and where the
evidence stops.

## Not a zoo of clever patches

A single bespoke optimization is evidence for a direction, not the destination.

The wrong next step would be to collect unrelated tricks and call the result a
compiler. The useful work is to ask which facts recur across successful
optimizations:

- named axes and their roles;
- fixed versus dynamic inputs;
- factorization, sparsity, grouping, routing, and layout;
- reuse and preprocessing opportunities;
- exactness and output-error contracts;
- backend capabilities and dense fallback conditions.

If those concepts can be represented cleanly, then a growing set of bespoke
observations may become reusable lowering rules. If they cannot, the project
should say so rather than hide the gap behind a broad abstraction.

That is also why evidence discipline is part of the initiative. A proposed path
must be compared with the best applicable stock strategy, not a weak baseline.
Correctness is checked before timing. Smoke tests are not performance results.
Artifacts bind measurements to source and environment. Unsupported cases remain
visible as fallbacks. Claims stay no broader than the experiment.

## The initiative

Beyond Matmul is an attempt to move the optimization boundary upward without
giving up the excellent machinery below it.

We want model and framework semantics to survive long enough for an inference
system to make a better decision. We want dense GEMM to remain available when
it wins. We want exactness, approximation, preprocessing, and fallback to be
explicit contracts. And we want end-to-end evidence before declaring that an
elegant representation is a useful one.

The name is not a rejection of matrix multiplication. It is a reminder:

> A matrix product is often how a computation runs. It is not always what the
> computation is.

The code, benchmark artifacts, correctness contracts, and offline OLMoE demo
are available in the
[`beyond-matmul` repository](https://github.com/alexlopashev/beyond-matmul).

## Sources and further reading

- NVIDIA, [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
- NVIDIA CUTLASS, [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).
- NVIDIA, [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/).
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- Jonathan Ragan-Kelley et al., [Halide: Decoupling Algorithms from Schedules for High-Performance Image Processing](https://people.csail.mit.edu/jrk/halide12/halide12.pdf).
- Trevor Gale et al., [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841).
- Niklas Muennighoff et al., [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060).
- Beyond Matmul, [OLMoE capstone contract and measured result](https://github.com/alexlopashev/beyond-matmul/blob/main/docs/olmoe_tensor_contraction_capstone.md).
- Beyond Matmul, [offline evidence-verifying demo](https://github.com/alexlopashev/beyond-matmul/blob/main/examples/olmoe_capstone_demo.py).
