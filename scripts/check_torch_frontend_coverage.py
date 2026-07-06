#!/usr/bin/env python3
"""Check that supported Torch frontend coverage rows cite executable evidence."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


VALID_STATUSES = {"Supported", "Next", "Unsupported"}

EVIDENCE_REFERENCES = {
    "`nn.Linear` nested under `nn.Linear`": (
        "tests/test_frontend.py::test_captures_real_torch_linear_modules",
        "examples/torch_coverage_demo.py::LowRankProjection",
    ),
    "Nested `F.linear`": (
        "tests/test_frontend.py::test_extracts_low_rank_functional_linear_pattern",
        "examples/torch_fx_frontend_demo.py",
    ),
    "Named adapter pairs": (
        "tests/test_frontend.py::test_captures_named_adapter_factors_with_merged_weight_hint",
        "examples/adapter_workload_demo.py",
    ),
    "`Embedding` followed by projection": (
        "tests/test_frontend.py::test_captures_embedding_projection_over_one_hot_inputs",
    ),
    "Single-channel `nn.Conv1d`": (
        "tests/test_frontend.py::test_captures_real_biasless_torch_conv1d_module",
        "examples/torch_coverage_demo.py::TinyConv1d",
    ),
    "Multi-channel `nn.Conv1d`": (
        "tests/test_frontend.py::test_captures_real_multi_channel_torch_conv1d_module",
        "examples/torch_coverage_demo.py::MultiChannelConv1d",
    ),
    "Functional `conv1d`": (
        "tests/test_frontend.py::test_captures_real_functional_torch_conv1d_as_affine",
        "tests/test_frontend.py::test_captures_real_functional_torch_conv1d_without_bias",
        "examples/torch_coverage_demo.py::FunctionalConv1d",
    ),
    "`operator.matmul` / `x @ weight.T`": (
        "tests/test_frontend.py::test_captures_real_torch_matmul_operator_pattern",
        "examples/torch_coverage_demo.py::MatmulProjection",
    ),
    "`torch.matmul`": (
        "tests/test_frontend.py::test_captures_real_torch_matmul_function_pattern",
    ),
    "`torch.mm`": (
        "tests/test_frontend.py::test_captures_real_torch_mm_function_pattern",
    ),
    "`torch.addmm`": (
        "tests/test_frontend.py::test_captures_real_torch_addmm_pattern_as_affine",
        "examples/torch_coverage_demo.py::AddmmProjection",
    ),
    "Grouped/depthwise `Conv1d`": (
        "tests/test_frontend.py::test_captures_real_grouped_torch_conv1d_module",
        "tests/test_frontend.py::test_captures_real_depthwise_torch_conv1d_module",
        "tests/test_frontend.py::test_captures_real_functional_grouped_torch_conv1d_as_affine",
        "tests/test_frontend.py::test_captures_real_functional_depthwise_torch_conv1d",
    ),
    "Stride/padding/dilation `Conv1d` variants": (
        "tests/test_frontend.py::test_captures_real_strided_padded_dilated_torch_conv1d_module",
        "tests/test_frontend.py::test_extracts_fake_functional_strided_padded_dilated_conv1d",
    ),
    "Quantized `nn.Linear`": (
        "tests/test_frontend.py::test_captures_real_quantized_linear_module_as_packed_affine",
        "tests/test_frontend.py::test_extracts_fake_quantized_linear_module_as_packed_affine",
        "tests/test_frontend.py::test_extracts_fake_biased_quantized_linear_module_as_affine",
        "tests/test_frontend.py::test_ignores_unsupported_fake_quantized_linear_variants",
    ),
    "Exported graph fixed-weight `addmm` and nested linear": (
        "tests/test_frontend.py::test_captures_real_torch_exported_addmm_pattern_as_affine",
        "tests/test_frontend.py::test_extracts_exported_nested_linear_from_signature_state_dict",
        "tests/test_frontend.py::test_extracts_exported_addmm_from_signature_state_dict",
    ),
}


@dataclass(frozen=True)
class CoverageRow:
    pattern: str
    status: str
    captured_ir: str
    notes: str
    line_number: int


def _split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_coverage_rows(path: Path) -> list[CoverageRow]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue

        cells = _split_markdown_row(stripped)
        if cells[:4] == ["Pattern", "Status", "Captured IR", "Notes"]:
            continue
        if len(cells) != 4:
            continue

        rows.append(
            CoverageRow(
                pattern=cells[0],
                status=cells[1],
                captured_ir=cells[2],
                notes=cells[3],
                line_number=line_number,
            )
        )
    return rows


def _reference_exists(repo_root: Path, reference: str) -> bool:
    path_text, _, token = reference.partition("::")
    path = repo_root / path_text
    if not path.is_file():
        return False
    if not token:
        return True
    return token in path.read_text(encoding="utf-8")


def validate_rows(rows: list[CoverageRow], repo_root: Path) -> list[str]:
    errors = []
    seen_patterns = set()
    supported_patterns = {row.pattern for row in rows if row.status == "Supported"}

    for row in rows:
        if row.pattern in seen_patterns:
            errors.append(f"duplicate coverage row: {row.pattern}")
        seen_patterns.add(row.pattern)

        if row.status not in VALID_STATUSES:
            errors.append(f"unknown status at line {row.line_number}: {row.status}")
            continue

        if row.status == "Supported" and row.pattern not in EVIDENCE_REFERENCES:
            errors.append(f"supported row has no evidence mapping: {row.pattern}")

    for pattern in sorted(EVIDENCE_REFERENCES):
        if pattern not in supported_patterns:
            errors.append(f"evidence mapping is stale or not supported: {pattern}")
            continue
        for reference in EVIDENCE_REFERENCES[pattern]:
            if not _reference_exists(repo_root, reference):
                errors.append(f"missing evidence reference for {pattern}: {reference}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing docs/torch_frontend_coverage.md.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    coverage_path = repo_root / "docs" / "torch_frontend_coverage.md"
    errors = validate_rows(parse_coverage_rows(coverage_path), repo_root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Torch frontend coverage evidence mapping is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
