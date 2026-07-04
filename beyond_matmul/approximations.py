"""Approximation builders and product-aware error metrics."""

from __future__ import annotations

from typing import List, Optional, Sequence

from beyond_matmul import _linalg as la
from beyond_matmul.ir import (
    ApproximationContract,
    BitpackedBinaryOperator,
    CodebookOperator,
    LowRankOperator,
    OperatorMetadata,
    Provenance,
    ReuseBudget,
    SparseCOOOperator,
)


def sparse_from_dense(
    matrix: Sequence[Sequence[float]],
    tolerance: float = 0.0,
    provenance: Optional[Provenance] = None,
    contract: Optional[ApproximationContract] = None,
) -> SparseCOOOperator:
    checked = la.as_matrix(matrix)
    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    for row_index, row in enumerate(checked):
        for col_index, value in enumerate(row):
            if abs(value) > tolerance:
                rows.append(row_index)
                cols.append(col_index)
                values.append(value)
    metadata = None
    if contract is not None:
        metadata = OperatorMetadata(
            kind="sparse_coo",
            shape=(len(checked), len(checked[0])),
            provenance=provenance or Provenance(source="sparse_from_dense"),
            structure={"nnz": len(values), "format": "coo", "threshold": tolerance},
            contract=contract,
            reuse=ReuseBudget(preprocessing_cost=len(checked) * len(checked[0]), amortize_over_calls=2),
            lowerings=("sparse_kernel", "dense_gemm"),
        )
    return SparseCOOOperator(rows, cols, values, (len(checked), len(checked[0])), provenance=provenance, metadata=metadata)


def sparse_topk_by_density(
    matrix: Sequence[Sequence[float]],
    density: float,
    provenance: Optional[Provenance] = None,
) -> SparseCOOOperator:
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")
    checked = la.as_matrix(matrix)
    rows = len(checked)
    cols = len(checked[0])
    flat = [(abs(value), row, col, value) for row, values in enumerate(checked) for col, value in enumerate(values)]
    keep = max(1, int(round(len(flat) * density)))
    threshold_items = sorted(flat, reverse=True)[:keep]
    keep_positions = {(row, col) for _, row, col, _ in threshold_items}
    sparse_rows: List[int] = []
    sparse_cols: List[int] = []
    sparse_values: List[float] = []
    for row in range(rows):
        for col in range(cols):
            if (row, col) in keep_positions:
                sparse_rows.append(row)
                sparse_cols.append(col)
                sparse_values.append(checked[row][col])
    contract = ApproximationContract(mode="approximate", metric="matrix_relative_frobenius")
    metadata = OperatorMetadata(
        kind="sparse_coo",
        shape=(rows, cols),
        provenance=provenance or Provenance(source="sparse_topk_by_density"),
        structure={"nnz": len(sparse_values), "format": "coo", "density": density},
        contract=contract,
        reuse=ReuseBudget(preprocessing_cost=rows * cols, amortize_over_calls=3),
        lowerings=("sparse_kernel", "dense_gemm"),
    )
    return SparseCOOOperator(sparse_rows, sparse_cols, sparse_values, (rows, cols), metadata=metadata)


def low_rank_approximation(
    matrix: Sequence[Sequence[float]],
    rank: int,
    iterations: int = 20,
    provenance: Optional[Provenance] = None,
) -> LowRankOperator:
    if rank <= 0:
        raise ValueError("rank must be positive")
    residual = la.as_matrix(matrix)
    rows, cols = len(residual), len(residual[0])
    actual_rank = min(rank, rows, cols)
    left_columns: List[List[float]] = []
    right_rows: List[List[float]] = []
    for component in range(actual_rank):
        vector = [1.0 / (cols**0.5) for _ in range(cols)]
        for _ in range(iterations):
            left_vec = la.matvec(residual, vector)
            left_norm = la.vector_norm(left_vec)
            if left_norm == 0.0:
                break
            left_unit = [value / left_norm for value in left_vec]
            right_vec = la.matvec(la.transpose(residual), left_unit)
            right_norm = la.vector_norm(right_vec)
            if right_norm == 0.0:
                break
            vector = [value / right_norm for value in right_vec]
        scaled_left = la.matvec(residual, vector)
        singular_value = la.vector_norm(scaled_left)
        if singular_value == 0.0:
            break
        rank_one = la.outer(scaled_left, vector)
        residual = la.subtract(residual, rank_one)
        left_columns.append(scaled_left)
        right_rows.append(vector)
    if not left_columns:
        left_columns = [[0.0 for _ in range(rows)]]
        right_rows = [[0.0 for _ in range(cols)]]
    left = [[left_columns[col][row] for col in range(len(left_columns))] for row in range(rows)]
    metadata = OperatorMetadata(
        kind="low_rank",
        shape=(rows, cols),
        provenance=provenance or Provenance(source="low_rank_approximation"),
        structure={"rank": len(right_rows), "method": "power_iteration_deflation"},
        contract=ApproximationContract(mode="approximate", metric="matrix_relative_frobenius"),
        reuse=ReuseBudget(preprocessing_cost=rows * cols * max(1, len(right_rows)) * iterations, amortize_over_calls=8),
        lowerings=("low_rank_product", "dense_gemm"),
    )
    return LowRankOperator(left, right_rows, metadata=metadata)


def codebook_quantize(
    matrix: Sequence[Sequence[float]],
    codebook_size: int,
    iterations: int = 8,
    provenance: Optional[Provenance] = None,
) -> CodebookOperator:
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    checked = la.as_matrix(matrix)
    values = la.flatten(checked)
    unique_values = sorted(set(values))
    if len(unique_values) <= codebook_size:
        codebook = unique_values
    else:
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            codebook = [min_value]
        else:
            codebook = [
                min_value + (max_value - min_value) * index / (codebook_size - 1)
                for index in range(codebook_size)
            ]
            for _ in range(iterations):
                buckets = [[] for _ in codebook]
                for value in values:
                    nearest = min(range(len(codebook)), key=lambda idx: abs(value - codebook[idx]))
                    buckets[nearest].append(value)
                for index, bucket in enumerate(buckets):
                    if bucket:
                        codebook[index] = sum(bucket) / len(bucket)
    codes: List[List[int]] = []
    for row in checked:
        code_row: List[int] = []
        for value in row:
            code_row.append(min(range(len(codebook)), key=lambda idx: abs(value - codebook[idx])))
        codes.append(code_row)
    return CodebookOperator(codes, codebook, provenance=provenance or Provenance(source="codebook_quantize"))


def bitpacked_binary_approximation(
    matrix: Sequence[Sequence[float]],
    provenance: Optional[Provenance] = None,
) -> BitpackedBinaryOperator:
    checked = la.as_matrix(matrix)
    scale = la.mean_abs(la.flatten(checked))
    signs = [[1 if value >= 0.0 else -1 for value in row] for row in checked]
    return BitpackedBinaryOperator(signs, scale, provenance=provenance or Provenance(source="bitpacked_binary_approximation"))


def product_relative_error(
    reference_matrix: Sequence[Sequence[float]],
    candidate,
    inputs: Sequence[Sequence[float]],
) -> float:
    exact = la.apply_weight(reference_matrix, inputs)
    observed = candidate.apply(inputs)
    return la.rms_relative_error(exact, observed)
