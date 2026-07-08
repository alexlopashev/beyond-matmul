# Live Conv1d Benchmark Contract

This contract defines the first live model-layer Conv1d benchmark for Beyond
Matmul. It is a benchmark design target, not benchmark evidence. The follow-up
implementation issue should use this model, layer, input trace, baseline set,
grid, and artifact shape unless a material blocker is recorded on the issue.

## Workload

The benchmark target is the first Whisper encoder convolution from a small
public ASR model:

- model: `openai/whisper-tiny`
- immutable model revision:
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
- access and license: public, ungated Hugging Face Hub model, Apache-2.0
- library target: `transformers.WhisperForConditionalGeneration`
- layer path: `model.encoder.conv1`
- layer type: `torch.nn.Conv1d`
- layer parameters: `in_channels=80`, `out_channels=384`, `kernel_size=3`,
  `stride=1`, `padding=1`, `dilation=1`, `groups=1`, `bias=True`
- task boundary: layer-level forward over log-Mel features, not decoder or
  end-to-end ASR generation
- primary dtype and device: `torch.float32` on CPU

The representative input trace is the Hugging Face Whisper widget LibriSpeech
sample:

- URL: `https://cdn-media.huggingface.co/speech_samples/sample1.flac`
- SHA-256:
  `cb5c48a2d1d6f7dedd0330f088a4cbe76de1a86e6a6109c06d255bb1ca2f7542`
- HTTP content type observed for the contract: `audio/flac`
- preprocessing: `WhisperProcessor.from_pretrained("openai/whisper-tiny",
  revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af")`, resampling through
  the processor as needed, returning `input_features`
- full model-native feature shape: batch `1`, channels `80`, and the
  processor-produced frame count for the downloaded sample

The required layer inputs are deterministic prefixes of that processed feature
tensor. If the downloaded sample produces fewer frames than a required prefix,
the harness must fail preflight with a clear readiness blocker instead of
padding with unrelated data.

## Regeneration Command

The implementation issue should add the harness at
`benchmarks/live_conv1d_whisper.py` and regenerate the real artifact with:

```bash
mise exec -- uv run --with transformers --with librosa --with soundfile --with safetensors --with huggingface_hub python benchmarks/live_conv1d_whisper.py --json-output docs/results/live_conv1d_whisper.json
```

The harness may also provide a small smoke mode for CI, but the smoke artifact
must set `summary.benchmark_ready=false` and
`summary.performance_claim="none"` unless it uses the real model revision,
real audio trace, required grid, and exact dense materialization below.

## Baselines

Each required shape must report these baselines:

1. `direct_conv1d`: run the selected `torch.nn.Conv1d` layer directly on the
   prefix input under `model.eval()` and `torch.inference_mode()`.
2. `dense_materialized_toeplitz`: build the exact dense linear map induced by
   the same Conv1d weight, bias, stride, padding, dilation, and groups for the
   same input length, then apply it to the flattened input.

The dense fallback is not an approximation and not a failure case. For an input
`x` with shape `[1, 80, T]`, build a matrix with shape
`[384 * T_out, 80 * T]`, where `T_out` is the Conv1d output length. Each matrix
entry is populated from the Conv1d kernel when the corresponding output
position reads the corresponding padded input position; all other entries are
zero. Add the Conv1d bias once per output channel and output position after the
matrix product. Reshape the result back to `[1, 384, T_out]` before correctness
comparison.

For the contract-selected layer, `stride=1`, `padding=1`, `dilation=1`, and
`kernel_size=3`, so `T_out=T`. The dense matrix has
`(384 * T) * (80 * T)` float32 entries. The artifact must record both the
materialized dense matrix size and the nonzero Toeplitz coefficient count so
reviewers can distinguish semantic equivalence from storage waste.

## Benchmark Grid

The required CPU fp32 grid is:

| Case | Batch | Input channels | Prefix frames `T` | Output channels | Output frames `T_out` | Dense entries | Dense bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `frames8_batch1` | 1 | 80 | 8 | 384 | 8 | 1,966,080 | 7,864,320 |
| `frames16_batch1` | 1 | 80 | 16 | 384 | 16 | 7,864,320 | 31,457,280 |
| `frames32_batch1` | 1 | 80 | 32 | 384 | 32 | 31,457,280 | 125,829,120 |

The required timing protocol is `10` warmup forwards and `50` measured
repetitions per baseline and shape. The implementation may add optional stress
rows such as `T=64` or `T=128`, but optional rows must be marked separately and
must not replace the required grid. If an optional row exceeds a documented
memory budget, keep the row with status `skipped_memory_budget` and record the
computed dense shape and byte count.

## Correctness

Correctness is checked against `direct_conv1d` for the same model revision,
layer, dtype, device, preprocessing path, and prefix tensor. The required CPU
fp32 tolerance is:

- `max_abs_error <= 1e-4`
- `relative_l2_error <= 1e-5`
- no NaN or infinite outputs

The direct baseline has zero error by definition. A dense row that fails
correctness must remain in the artifact with status `failed_correctness`, the
observed error metrics, and the dense matrix metadata.

## Measurements

Each measured row must include:

- latency in seconds per layer application: median, mean, standard deviation,
  p50, p90, p95, and p99
- dense materialization time for `dense_materialized_toeplitz`, or `null` with
  reason for `direct_conv1d`
- peak memory when measurable on the device, otherwise `null` with reason
- materialized dense matrix shape, entry count, and byte count
- Toeplitz nonzero coefficient count and density
- output-equivalence metrics from the correctness section
- dependency and environment metadata: operating system, Python version,
  PyTorch version, Transformers version, Hugging Face Hub revision, CPU model,
  accelerator model if any, thread settings, and relevant environment variables

Latency summaries for the dense baseline must separate matrix construction
from repeated application. The dense row should report both
`materialization_seconds` and `apply_latency_seconds`; the summary may also
include amortized totals for a declared reuse count, but it must not hide the
one-time materialization cost.

## JSON Artifact Schema

The benchmark artifact is a single JSON object:

```json
{
  "schema_version": 1,
  "benchmark": "live_conv1d_whisper_dense_vs_direct",
  "contract": "docs/live_conv1d_benchmark_contract.md",
  "workload": {
    "model": "openai/whisper-tiny",
    "model_revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "model_license": "apache-2.0",
    "layer_path": "model.encoder.conv1",
    "layer_type": "torch.nn.Conv1d",
    "layer": {
      "in_channels": 80,
      "out_channels": 384,
      "kernel_size": 3,
      "stride": 1,
      "padding": 1,
      "dilation": 1,
      "groups": 1,
      "bias": true
    },
    "input": {
      "source_url": "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
      "sha256": "cb5c48a2d1d6f7dedd0330f088a4cbe76de1a86e6a6109c06d255bb1ca2f7542",
      "preprocessor": "WhisperProcessor",
      "dtype": "float32",
      "device": "cpu",
      "prefix_frames": [8, 16, 32]
    },
    "warmup_repetitions": 10,
    "measured_repetitions": 50
  },
  "dependencies": {
    "python": "<version>",
    "torch": "<version>",
    "transformers": {"version": "<version>"},
    "huggingface_hub": {"version": "<version-or-null>"},
    "beyond_matmul": {
      "repository": "alexlopashev/beyond-matmul",
      "revision": "<sha>"
    }
  },
  "environment": {
    "platform": "<platform string>",
    "cpu": "<cpu model>",
    "accelerator": null,
    "torch_num_threads": "<int-or-null>",
    "env": {"<name>": "<value>"}
  },
  "results": [
    {
      "case": "frames8_batch1",
      "baseline": "direct_conv1d",
      "status": "ok",
      "batch_size": 1,
      "input_shape": [1, 80, 8],
      "output_shape": [1, 384, 8],
      "latency_seconds": {
        "median": 0.0,
        "mean": 0.0,
        "stdev": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0
      },
      "materialization_seconds": null,
      "materialization_status": "not_applicable_direct_conv1d",
      "peak_memory_bytes": null,
      "peak_memory_status": "not_measured",
      "dense_matrix": {
        "shape": [3072, 640],
        "entries": 1966080,
        "bytes_float32": 7864320,
        "toeplitz_nonzero_coefficients": 675840,
        "density": 0.34375
      },
      "correctness": {
        "reference_baseline": "direct_conv1d",
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
        "max_abs_tolerance": 0.0001,
        "relative_l2_tolerance": 0.00001,
        "tolerance_profile": "cpu_fp32",
        "passed": true
      }
    }
  ],
  "summary": {
    "all_required_cases_present": true,
    "all_correctness_checks_passed": true,
    "benchmark_ready": true,
    "readiness_blockers": [],
    "max_abs_error": 0.0,
    "max_relative_l2_error": 0.0,
    "performance_claim": "none"
  }
}
```

Rows may add fields, but they must not remove or rename the required fields
above without a schema-version bump. Unsupported, skipped, or failed rows must
stay in `results` with a status, measured fields set to `null` where
appropriate, and a human-readable reason.

The `toeplitz_nonzero_coefficients` value should count the logical nonzero
kernel placements implied by padding and boundaries, before considering
whether an individual learned weight is numerically zero. For the required
Whisper layer and `T=8`, this is `384 * 80 * (3 * T - 2) = 675840`.

## Success Thresholds

The first implementation succeeds as a benchmark artifact if:

- every required baseline and shape row is present;
- the JSON follows the schema above;
- model, input, dependency, and hardware metadata are present;
- the audio SHA-256 matches the contract before preprocessing;
- all required rows pass the correctness tolerance;
- dense matrix size and Toeplitz sparsity metadata are reported for every row.

A speed or memory claim requires more than successful execution. The artifact
may claim a layer-level measured win only if `direct_conv1d` improves median
apply latency or peak memory by at least `10%` against
`dense_materialized_toeplitz` for at least one required shape, does not regress
median apply latency by more than `5%` on any other required shape, and passes
correctness everywhere. If these thresholds are not met, the result is still
useful and must be reported as negative or neutral.

## Claim Boundary

This benchmark covers one fixed-weight Conv1d layer from one public Whisper
model on one public audio-derived input trace. It is live layer-level evidence,
not an end-to-end ASR speedup claim. It excludes decoder generation, word error
rate, streaming ASR, batching beyond batch size 1, GPU kernels, Conv2d, broader
CNN blocks, quantized convolution, training, and dynamic-weight behavior.

Dense materialization remains the exact semantic fallback throughout the
benchmark. The project claim is not that dense GEMM is invalid; it is that
retaining Conv1d provenance can expose the direct channel-aware path while
still making the dense fallback available for correctness, portability, and
negative results.
