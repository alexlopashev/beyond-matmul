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

The timings are pure-Python latency proxies for research triage and figure
generation. They are not hardware-calibrated production performance claims.
