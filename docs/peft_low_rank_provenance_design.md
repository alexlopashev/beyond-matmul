# PEFT Low-Rank Provenance Design

This note resolves issue #77. It designs the smallest PEFT fork integration
for preserving fixed LoRA adapter provenance during inference so the capstone
benchmark can compare upstream unmerged LoRA, upstream merged dense weights,
and a Beyond Matmul provenance-aware fork without implementing the optimization
in this repository.

The PEFT fork inspection target was
`alexlopashev/peft@b6817852324241e4abe00d24ec096c6435ec0dfd` on branch
`beyond-matmul/provenance-lora-inference`. The relevant benchmark contract is
`docs/peft_capstone_benchmark_contract.md`, and the fork mechanics are in
`docs/peft_fork_setup.md`.

## Decision

The first integration should stay inside PEFT's vanilla LoRA `Linear` runtime:

- `src/peft/tuners/lora/config.py`
- `src/peft/tuners/lora/model.py`
- `src/peft/tuners/lora/layer.py`

The implementation should not add a new Beyond Matmul dependency to PEFT. It
should record a small, JSON-serializable provenance event on eligible PEFT
LoRA layers, and the Beyond Matmul benchmark harness should collect that event
from the loaded fork model.

This keeps the first implementation issue about PEFT inference behavior and
metadata only. Any later conversion from PEFT events into native Beyond Matmul
IR can be a follow-up after the benchmark can prove that the PEFT fork exposes
the low-rank factors and fallback state reliably.

## First Supported Pattern

Support exactly this first pattern:

- PEFT `PeftModel.from_pretrained(base_model, adapter)` causal LM inference.
- `model.eval()` under `torch.inference_mode()`.
- CPU `torch.float32` prefill logits for the contract workload.
- Vanilla LoRA wrapping `torch.nn.Linear` through
  `LoraModel._create_and_replace()` and `LoraModel._create_new_module()`.
- One active adapter per eligible `LoraLayer`.
- Unmerged adapters using the `LoraLayer.forward()` path:
  `base_layer(x) + lora_B(lora_A(dropout(x))) * scaling`.
- `fan_in_fan_out == False`, no LoRA bias, and no LoRA variant for the first
  supported benchmark case.
- Dense fallback remains available through the existing upstream
  `merge_and_unload()`/`merge()` path and the layer's `get_delta_weight()`.

The first benchmarked adapter remains
`peft-internal-testing/tiny-OPTForCausalLM-lora` on
`hf-internal-testing/tiny-random-OPTForCausalLM`, as specified by
`docs/peft_capstone_benchmark_contract.md`.

## PEFT Touch Points

### `src/peft/tuners/lora/config.py`

Extend `LoraRuntimeConfig` with an opt-in flag for the fork, for example
`beyond_matmul_provenance: bool = False`. Runtime config is the right location
because the feature changes inference recording, not adapter weight format.

The flag should not be serialized into adapter configs unless PEFT upstream
requires that for runtime config fields. A user who does not opt in should see
upstream behavior and no extra benchmark metadata.

### `src/peft/tuners/lora/model.py`

Keep module replacement unchanged except for passing enough context into the
layer. `LoraModel._create_and_replace()` already computes `current_key`,
rank, alpha, and `lora_config.runtime_config`; it should ensure the LoRA layer
can retain:

- the module path (`current_key`);
- the adapter name;
- whether provenance recording is enabled;
- the base module type selected by `_create_new_module()`;
- target rank and alpha pattern values after overrides.

Do not broaden dispatcher behavior in `_create_new_module()`. The first
implementation should let existing dispatcher order choose the same PEFT LoRA
layer as upstream and then add metadata only when that selected layer is the
vanilla `Linear` case.

### `src/peft/tuners/lora/layer.py`

Add the first metadata and routing logic to `LoraLayer`, not to the model
generation loop.

`LoraLayer.update_layer()` should store per-adapter provenance fields already
known at construction time:

- adapter name;
- target module path;
- base module class;
- rank `r`, `lora_alpha`, `scaling`, `use_rslora`, and `fan_in_fan_out`;
- shapes, dtypes, and devices of `lora_A[adapter].weight` and
  `lora_B[adapter].weight`;
- whether `lora_bias`, `use_dora`, or another `lora_variant` is active.

`LoraLayer.forward()` should make the eligibility decision immediately before
the existing vanilla LoRA addition. For an eligible layer, record a
JSON-serializable event such as:

```json
{
  "schema_version": 1,
  "kind": "beyond_matmul_lora_provenance",
  "path": "structured_low_rank",
  "module_path": "base_model.model.decoder.layers.0.self_attn.q_proj",
  "adapter": "default",
  "base_module": "Linear",
  "rank": 8,
  "in_features": 32,
  "out_features": 32,
  "input_shape": [1, 16, 32],
  "a_shape": [8, 32],
  "b_shape": [32, 8],
  "scaling": 1.0,
  "dtype": "torch.float32",
  "device": "cpu",
  "dense_fallback_available": true,
  "dense_fallback_used": false,
  "fallback_reason": null
}
```

For an ineligible layer or path, record `path: "dense_fallback"` and a stable
`fallback_reason`, then run the same numeric code path PEFT already runs. The
first implementation should not replace the matmuls with a custom kernel; it
only preserves enough provenance for the benchmark harness to prove the fork
could identify and route the low-rank structure.

`LoraLayer.merge()`, `LoraLayer.unmerge()`, and `LoraLayer.get_delta_weight()`
should remain the dense materialization contract. They should not be bypassed
or weakened. Equivalence tests should compare the provenance-recording path
against these existing PEFT paths.

## Benchmark Harness Contract

The Beyond Matmul harness should collect fork metadata after each measured
forward by scanning `model.named_modules()` for the PEFT fork event attribute.
It should summarize the fork row with:

- `lowering.kind: "provenance_lora_fork"` when at least one eligible LoRA
  layer records `path: "structured_low_rank"`;
- `lowering.dense_fallback_available: true`;
- `lowering.dense_fallback_used: true` only when every inspected LoRA layer
  falls back or the fork reports no structured event;
- stable fallback reasons when the fork path is unsupported.

The benchmark should keep the existing correctness reference:
`upstream_peft_unmerged` logits. Output equivalence remains the contract's CPU
fp32 tolerance:

- `max_abs_error <= 1e-4`;
- `relative_l2_error <= 1e-5`;
- no NaN or infinite logits.

The fork metadata is not a performance claim. It is evidence that PEFT
preserved adapter provenance and exposed whether dense fallback was available
or used.

## Unsupported In The First Integration

The first implementation must fall back cleanly for:

- training, gradients, optimizer state, and autograd-specific behavior;
- generation-loop, KV-cache, and decoding-policy changes;
- `adapter_names` mixed-batch routing through `_mixed_batch_forward()`;
- multiple simultaneously active adapters on one layer;
- merged adapters where `self.merged` is true;
- `disable_adapters` paths;
- DoRA, aLoRA, Arrow, VeLoRA, BD-LoRA, LoRA variants, or custom modules;
- embeddings, convolutions, `MultiheadAttention`, `ParamWrapper`, and
  `target_parameters`;
- quantized backends and dispatcher-specific layers such as bnb, GPTQ, AWQ,
  AQLM, HQQ, INC, TorchAO, Megatron, or Transformer Engine;
- tensor-parallel sharded plans;
- `lora_bias=True`;
- non-CPU or non-fp32 benchmark claims.

Fallback rows must not pretend that provenance was preserved. They should
record `dense_fallback_used: true` and an explicit reason.

## Beyond Matmul Changes Needed

No core IR or planner change is required before the first PEFT fork
implementation. Existing `LowRankOperator` semantics and the current benchmark
artifact schema are sufficient for the first evidence loop.

Issue #78 should change two surfaces:

1. The PEFT fork should record the layer-level provenance event described
   above.
2. The Beyond Matmul benchmark worker should copy fork events into the
   `beyond_matmul_peft_fork` result row and set `lowering` from the fork's
   actual reported path instead of assuming the path from the baseline name.

A later issue can translate PEFT provenance events into native Beyond Matmul
IR operators if the benchmark needs planner-level evidence beyond the JSON
row. That should wait until the fork has a passing equivalence test and a
contract-shaped artifact.

## Issue #78 Handoff

Implementation issue #78 should proceed with this narrow checklist:

- add the runtime opt-in flag to `LoraRuntimeConfig`;
- record per-adapter metadata in `LoraLayer.update_layer()`;
- record a last-forward provenance event in the vanilla `LoraLayer.forward()`
  path for eligible `Linear` layers;
- record explicit fallback reasons for unsupported paths;
- add PEFT fork tests for one eligible `Linear` LoRA inference path and at
  least one fallback path;
- update the Beyond Matmul worker to copy fork metadata into the JSON row;
- verify logits against upstream unmerged PEFT within the benchmark tolerance;
- keep `merge_and_unload()` and `get_delta_weight()` available as dense
  fallback/equivalence paths.

Do not implement custom kernels, broad adapter coverage, training behavior,
or upstream PEFT API commitments in #78.
