# Benchmark Artifacts

`benchmarks/fixed_weight.py` can emit a machine-readable JSON artifact while
preserving the human-readable smoke table used by local CI. CI uploads
`docs/results/fixed_weight.json` as the workflow artifact
`fixed-weight-benchmark-json`.

Regenerate the fixed-weight benchmark artifact with:

```bash
mise exec -- uv run python benchmarks/fixed_weight.py --json-output docs/results/fixed_weight.json
```

The JSON schema is versioned with `schema_version: 1`. Each case records:

- case name and selected lowering
- validity, exactness, and relative output error
- planner cost, memory, preprocessing, and requested-call proxies
- dense and selected-lowering seconds per apply
- Python and platform metadata for the run

The controlled case set includes diagonal, sparse, low-rank, codebook, dense,
single-channel valid Conv1d, and multi-channel valid Conv1d operators. Conv1d
rows are expected to select the direct Conv1d lowerings when the benchmark
request and backend contract allow them.

The timings are pure-Python latency proxies for research triage and figure
generation. They are not hardware-calibrated production performance claims.

## Approximation Error Ablation

`benchmarks/approximation_error_ablation.py` emits a deterministic JSON table
source for the current matrix-reconstruction-error versus output-error
ablation. Regenerate it with:

```bash
mise exec -- uv run python benchmarks/approximation_error_ablation.py --json-output docs/results/approximation_error_ablation.json
```

CI uploads `docs/results/approximation_error_ablation.json` as
`approximation-error-ablation-json`.

The JSON schema is versioned with `schema_version: 1`. Each candidate row
records:

- candidate kind and parameters
- matrix reconstruction error
- output-relative error on the deterministic sample inputs
- matrix-threshold and output-threshold decisions
- candidate lowering, planner selection flag, and acceptance or rejection
  reason

The controlled case currently covers one dense matrix with a dominant feature
that the sample inputs do not exercise. In this bounded case, low-rank and
sparse top-k candidates pass the matrix-relative threshold but fail the
output-relative threshold; codebook and bitpacked candidates fail both. This is
paper-supporting evidence for output-aware acceptance, not a general benchmark
of approximation quality.
