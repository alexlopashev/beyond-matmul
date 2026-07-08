import importlib.util
import json
import numbers
import tempfile
import unittest
from pathlib import Path


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "live_conv1d_whisper.py"
    spec = importlib.util.spec_from_file_location("live_conv1d_whisper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LiveConv1dWhisperTests(unittest.TestCase):
    def test_dense_toeplitz_matches_torch_conv1d_with_bias(self):
        benchmark = _load_benchmark_module()
        torch = benchmark._torch()

        layer = torch.nn.Conv1d(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        [[-1.0, 0.5, 1.0], [2.0, -2.0, 0.25]],
                        [[0.0, 1.0, -1.0], [1.5, 0.0, 2.0]],
                    ],
                    dtype=torch.float32,
                )
            )
            layer.bias.copy_(torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32))

        inputs = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        dense = benchmark.materialize_conv1d_to_dense(layer, input_frames=4)
        dense_output = benchmark.apply_dense_materialized_conv1d(inputs, dense.matrix, dense.bias, dense.output_frames)

        self.assertEqual(tuple(dense.matrix.shape), (12, 8))
        self.assertEqual(dense.entries, 96)
        self.assertEqual(dense.bytes_float32, 384)
        self.assertEqual(dense.toeplitz_nonzero_coefficients, 60)
        self.assertAlmostEqual(dense.density, 0.625)
        self.assertTrue(torch.allclose(dense_output, layer(inputs), atol=1e-6, rtol=0.0))

    def test_smoke_artifact_matches_contract_shape(self):
        benchmark = _load_benchmark_module()

        first = benchmark.collect_results(
            mode="synthetic-smoke",
            prefix_frames=[4],
            warmup_repetitions=1,
            measured_repetitions=2,
            time_direct=lambda _layer, _inputs, _warmup, _repetitions: [0.01, 0.02],
            time_dense_apply=lambda _inputs, _matrix, _bias, _out_frames, _warmup, _repetitions: [0.03, 0.05],
            time_materialization=lambda _layer, frames: benchmark.materialize_conv1d_to_dense(_layer, frames),
            generated_at_utc="2026-07-08T00:00:00Z",
        )
        second = benchmark.collect_results(
            mode="synthetic-smoke",
            prefix_frames=[4],
            warmup_repetitions=1,
            measured_repetitions=2,
            time_direct=lambda _layer, _inputs, _warmup, _repetitions: [0.01, 0.02],
            time_dense_apply=lambda _inputs, _matrix, _bias, _out_frames, _warmup, _repetitions: [0.03, 0.05],
            time_materialization=lambda _layer, frames: benchmark.materialize_conv1d_to_dense(_layer, frames),
            generated_at_utc="2026-07-08T00:00:00Z",
        )

        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["benchmark"], "live_conv1d_whisper_dense_vs_direct")
        self.assertEqual(first["contract"], "docs/live_conv1d_benchmark_contract.md")
        self.assertEqual(first["mode"], "synthetic-smoke")
        self.assertEqual(first["workload"]["model"], "openai/whisper-tiny")
        self.assertEqual(
            first["workload"]["model_revision"],
            "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        )
        self.assertEqual(first["workload"]["layer_path"], "model.encoder.conv1")
        self.assertEqual(first["workload"]["input"]["prefix_frames"], [4])
        self.assertEqual(first["workload"]["warmup_repetitions"], 1)
        self.assertEqual(first["workload"]["measured_repetitions"], 2)

        results = first["results"]
        self.assertEqual(len(results), 2)
        self.assertEqual({row["baseline"] for row in results}, {"direct_conv1d", "dense_materialized_toeplitz"})

        required_fields = {
            "case",
            "baseline",
            "status",
            "batch_size",
            "input_shape",
            "output_shape",
            "latency_seconds",
            "materialization_seconds",
            "materialization_status",
            "peak_memory_bytes",
            "peak_memory_status",
            "dense_matrix",
            "correctness",
        }
        for row in results:
            self.assertTrue(required_fields.issubset(row))
            self.assertEqual(row["case"], "frames4_batch1")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["batch_size"], 1)
            self.assertEqual(row["input_shape"], [1, 80, 4])
            self.assertEqual(row["output_shape"], [1, 384, 4])
            self.assertEqual(row["peak_memory_bytes"], None)
            self.assertEqual(row["peak_memory_status"], "not_measured")
            for value in row["latency_seconds"].values():
                self.assertIsInstance(value, numbers.Real)
                self.assertGreaterEqual(value, 0.0)
            self.assertEqual(row["correctness"]["reference_baseline"], "direct_conv1d")
            self.assertEqual(row["correctness"]["tolerance_profile"], "cpu_fp32")
            self.assertTrue(row["correctness"]["passed"])

        direct = next(row for row in results if row["baseline"] == "direct_conv1d")
        dense = next(row for row in results if row["baseline"] == "dense_materialized_toeplitz")
        self.assertEqual(direct["materialization_seconds"], None)
        self.assertEqual(direct["materialization_status"], "not_applicable_direct_conv1d")
        self.assertGreaterEqual(dense["materialization_seconds"], 0.0)
        self.assertEqual(dense["materialization_status"], "ok")
        self.assertEqual(dense["dense_matrix"]["shape"], [1536, 320])
        self.assertEqual(dense["dense_matrix"]["entries"], 491520)
        self.assertEqual(dense["dense_matrix"]["bytes_float32"], 1966080)
        self.assertEqual(dense["dense_matrix"]["toeplitz_nonzero_coefficients"], 307200)

        self.assertFalse(first["summary"]["all_required_cases_present"])
        self.assertTrue(first["summary"]["all_correctness_checks_passed"])
        self.assertFalse(first["summary"]["benchmark_ready"])
        self.assertEqual(
            first["summary"]["readiness_blockers"],
            ["synthetic_smoke_not_benchmark_evidence", "required_cases_missing"],
        )
        self.assertEqual(first["summary"]["performance_claim"], "none")

    def test_writes_json_artifact(self):
        benchmark = _load_benchmark_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "live_conv1d_whisper.json"
            artifact = benchmark.write_json_artifact(
                output_path,
                mode="synthetic-smoke",
                prefix_frames=[4],
                warmup_repetitions=0,
                measured_repetitions=1,
                time_direct=lambda _layer, _inputs, _warmup, _repetitions: [0.01],
                time_dense_apply=lambda _inputs, _matrix, _bias, _out_frames, _warmup, _repetitions: [0.02],
                generated_at_utc="2026-07-08T00:00:00Z",
            )
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(artifact, loaded)


if __name__ == "__main__":
    unittest.main()
