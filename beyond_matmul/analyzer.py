"""Cheap structure recovery probes for dense matrices and observed matmul sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from beyond_matmul import _linalg as la
from beyond_matmul.approximations import codebook_quantize, low_rank_approximation


@dataclass(frozen=True)
class StructureCandidate:
    kind: str
    confidence: float
    exact: bool
    cost: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedMatmulSite:
    site_id: str
    left_shape: Tuple[int, int]
    right_shape: Tuple[int, int]
    fixed_left: bool = False
    fixed_right: bool = False
    calls: int = 1


def _safe_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def diagonal_probe(matrix: Sequence[Sequence[float]], tolerance: float = 1e-9) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    offdiag_sq = 0.0
    total_sq = 0.0
    for row in range(rows):
        for col in range(cols):
            value = checked[row][col]
            total_sq += value * value
            if row != col:
                offdiag_sq += value * value
    ratio = (offdiag_sq**0.5) / (total_sq**0.5) if total_sq else 0.0
    exact = rows == cols and ratio <= tolerance
    confidence = 1.0 if exact else _safe_confidence(1.0 - ratio)
    return StructureCandidate(
        kind="diagonal",
        confidence=confidence,
        exact=exact,
        cost=rows * cols,
        evidence={"offdiag_norm_ratio": ratio, "square": rows == cols},
    )


def sparsity_probe(matrix: Sequence[Sequence[float]], tolerance: float = 1e-9) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    nnz = la.count_nonzero(checked, tolerance)
    total = rows * cols
    zero_fraction = 1.0 - (nnz / total)
    return StructureCandidate(
        kind="sparse",
        confidence=_safe_confidence(zero_fraction),
        exact=True,
        cost=total,
        evidence={"nnz": nnz, "zero_fraction": zero_fraction, "tolerance": tolerance},
    )


def block_sparsity_probe(
    matrix: Sequence[Sequence[float]],
    block_shape: Tuple[int, int] = (2, 2),
    tolerance: float = 1e-9,
) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    block_rows, block_cols = block_shape
    total_blocks = 0
    zero_blocks = 0
    for row in range(0, rows, block_rows):
        for col in range(0, cols, block_cols):
            total_blocks += 1
            is_zero = True
            for inner_row in range(row, min(row + block_rows, rows)):
                for inner_col in range(col, min(col + block_cols, cols)):
                    if abs(checked[inner_row][inner_col]) > tolerance:
                        is_zero = False
            if is_zero:
                zero_blocks += 1
    fraction = zero_blocks / total_blocks if total_blocks else 0.0
    return StructureCandidate(
        kind="block_sparse",
        confidence=_safe_confidence(fraction),
        exact=True,
        cost=rows * cols,
        evidence={"block_shape": block_shape, "zero_block_fraction": fraction},
    )


def permutation_probe(matrix: Sequence[Sequence[float]], tolerance: float = 1e-9) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    row_counts = []
    col_counts = [0 for _ in range(cols)]
    values_are_unit = True
    for row in checked:
        count = 0
        for col, value in enumerate(row):
            if abs(value) > tolerance:
                count += 1
                col_counts[col] += 1
                if abs(abs(value) - 1.0) > tolerance:
                    values_are_unit = False
        row_counts.append(count)
    one_per_row = all(count == 1 for count in row_counts)
    at_most_one_per_col = all(count <= 1 for count in col_counts)
    exact = one_per_row and at_most_one_per_col and values_are_unit
    near_score = sum(1 for count in row_counts if count == 1) / rows
    return StructureCandidate(
        kind="permutation_or_signed_permutation",
        confidence=1.0 if exact else _safe_confidence(0.5 * near_score + 0.5 * sum(1 for count in col_counts if count <= 1) / cols),
        exact=exact,
        cost=rows * cols,
        evidence={"one_per_row": one_per_row, "at_most_one_per_col": at_most_one_per_col, "unit_values": values_are_unit},
    )


def codebook_probe(matrix: Sequence[Sequence[float]], max_codebook_size: int = 16, decimals: int = 6) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    unique = la.unique_rounded_values(checked, decimals)
    exact = len(unique) <= max_codebook_size
    confidence = 1.0 if exact else _safe_confidence(max_codebook_size / len(unique))
    return StructureCandidate(
        kind="codebook",
        confidence=confidence,
        exact=exact,
        cost=rows * cols,
        evidence={"unique_values": len(unique), "max_codebook_size": max_codebook_size, "decimals": decimals},
    )


def low_rank_probe(
    matrix: Sequence[Sequence[float]],
    ranks: Sequence[int] = (1, 2, 4),
    good_error: float = 0.05,
) -> List[StructureCandidate]:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    candidates: List[StructureCandidate] = []
    for rank in ranks:
        if rank > min(rows, cols):
            continue
        op = low_rank_approximation(checked, rank=rank, iterations=12)
        error = la.relative_frobenius_error(checked, op.to_dense())
        exact = error <= 1e-8
        confidence = 1.0 if exact else _safe_confidence(1.0 - (error / good_error))
        candidates.append(
            StructureCandidate(
                kind="low_rank",
                confidence=confidence,
                exact=exact,
                cost=rows * cols * rank * 12,
                evidence={"rank": rank, "relative_frobenius_error": error},
            )
        )
    return candidates


def repeated_block_probe(matrix: Sequence[Sequence[float]], tolerance: float = 1e-9) -> StructureCandidate:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    repeated_rows = 0
    for row in range(1, rows):
        if all(abs(checked[row][col] - checked[row - 1][col]) <= tolerance for col in range(cols)):
            repeated_rows += 1
    repeated_cols = 0
    for col in range(1, cols):
        if all(abs(checked[row][col] - checked[row][col - 1]) <= tolerance for row in range(rows)):
            repeated_cols += 1
    row_score = repeated_rows / max(1, rows - 1)
    col_score = repeated_cols / max(1, cols - 1)
    return StructureCandidate(
        kind="repeated_block_or_broadcast",
        confidence=_safe_confidence(max(row_score, col_score)),
        exact=max(row_score, col_score) == 1.0,
        cost=rows * cols,
        evidence={"repeated_adjacent_row_fraction": row_score, "repeated_adjacent_col_fraction": col_score},
    )


def analyze_dense(
    matrix: Sequence[Sequence[float]],
    tolerance: float = 1e-9,
    max_codebook_size: int = 16,
    ranks: Sequence[int] = (1, 2, 4),
) -> List[StructureCandidate]:
    checked = la.as_matrix(matrix)
    candidates: List[StructureCandidate] = [
        diagonal_probe(checked, tolerance=tolerance),
        sparsity_probe(checked, tolerance=tolerance),
        block_sparsity_probe(checked, tolerance=tolerance),
        permutation_probe(checked, tolerance=tolerance),
        codebook_probe(checked, max_codebook_size=max_codebook_size),
        repeated_block_probe(checked, tolerance=tolerance),
    ]
    candidates.extend(low_rank_probe(checked, ranks=ranks))
    return sorted(candidates, key=lambda candidate: (candidate.confidence, candidate.exact), reverse=True)


class ReuseTracker:
    """Tracks repeated fixed weights when provenance was lost."""

    def __init__(self, decimals: int = 6) -> None:
        self.decimals = decimals
        self._counts: Dict[Tuple[Tuple[float, ...], ...], int] = {}

    def observe(self, matrix: Sequence[Sequence[float]]) -> StructureCandidate:
        fingerprint = la.matrix_fingerprint(matrix, self.decimals)
        self._counts[fingerprint] = self._counts.get(fingerprint, 0) + 1
        count = self._counts[fingerprint]
        return StructureCandidate(
            kind="fixed_weight_reuse",
            confidence=_safe_confidence(count / 3.0),
            exact=True,
            cost=len(fingerprint) * len(fingerprint[0]),
            evidence={"observed_calls": count, "decimals": self.decimals},
        )


def candidate_codebook_operator(matrix: Sequence[Sequence[float]], max_codebook_size: int = 16):
    return codebook_quantize(matrix, codebook_size=max_codebook_size)
