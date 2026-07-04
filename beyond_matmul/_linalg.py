"""Small pure-Python linear algebra helpers.

The core research artifact intentionally has no mandatory numerical dependency.
These helpers are not meant to beat BLAS; they make the IR, planner, and tests
executable in a fresh environment.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

Matrix = List[List[float]]
Vector = List[float]


def as_matrix(values: Sequence[Sequence[float]]) -> Matrix:
    matrix = [[float(value) for value in row] for row in values]
    if not matrix:
        raise ValueError("matrix must have at least one row")
    width = len(matrix[0])
    if width == 0:
        raise ValueError("matrix must have at least one column")
    for row in matrix:
        if len(row) != width:
            raise ValueError("matrix rows must all have the same length")
    return matrix


def as_vector(values: Sequence[float]) -> Vector:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError("vector must not be empty")
    return vector


def shape(matrix: Sequence[Sequence[float]]) -> Tuple[int, int]:
    checked = as_matrix(matrix)
    return len(checked), len(checked[0])


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    matrix = zeros(size, size)
    for index in range(size):
        matrix[index][index] = 1.0
    return matrix


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    checked = as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    return [[checked[row][col] for row in range(rows)] for col in range(cols)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot product inputs have different lengths")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def normalize(vector: Sequence[float]) -> Vector:
    norm = vector_norm(vector)
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    checked = as_matrix(matrix)
    if len(checked[0]) != len(vector):
        raise ValueError("matrix/vector shape mismatch")
    return [dot(row, vector) for row in checked]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    a = as_matrix(left)
    b = as_matrix(right)
    if len(a[0]) != len(b):
        raise ValueError("matrix multiplication shape mismatch")
    b_t = transpose(b)
    return [[dot(row, col) for col in b_t] for row in a]


def outer(left: Sequence[float], right: Sequence[float]) -> Matrix:
    return [[float(a) * float(b) for b in right] for a in left]


def add(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    a = as_matrix(left)
    b = as_matrix(right)
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        raise ValueError("matrix add shape mismatch")
    return [[a[row][col] + b[row][col] for col in range(len(a[0]))] for row in range(len(a))]


def subtract(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    a = as_matrix(left)
    b = as_matrix(right)
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        raise ValueError("matrix subtract shape mismatch")
    return [[a[row][col] - b[row][col] for col in range(len(a[0]))] for row in range(len(a))]


def scale(matrix: Sequence[Sequence[float]], factor: float) -> Matrix:
    checked = as_matrix(matrix)
    return [[factor * value for value in row] for row in checked]


def frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    checked = as_matrix(matrix)
    return math.sqrt(sum(value * value for row in checked for value in row))


def relative_frobenius_error(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> float:
    denom = frobenius_norm(reference)
    if denom == 0.0:
        return frobenius_norm(candidate)
    return frobenius_norm(subtract(reference, candidate)) / denom


def ensure_batch(inputs: Sequence[Sequence[float]] | Sequence[float]) -> Matrix:
    if not inputs:
        raise ValueError("inputs must not be empty")
    first = inputs[0]  # type: ignore[index]
    if isinstance(first, (int, float)):
        return [[float(value) for value in inputs]]  # type: ignore[arg-type]
    return as_matrix(inputs)  # type: ignore[arg-type]


def apply_weight(weight: Sequence[Sequence[float]], inputs: Sequence[Sequence[float]] | Sequence[float]) -> Matrix:
    """Apply a weight matrix with shape (out_features, in_features).

    Inputs are row-major batches with shape (batch, in_features). The result has
    shape (batch, out_features), matching common inference notation y = x W^T.
    """

    checked_weight = as_matrix(weight)
    batch = ensure_batch(inputs)
    in_features = len(checked_weight[0])
    for row in batch:
        if len(row) != in_features:
            raise ValueError("input batch row width does not match operator input dimension")
    return [[dot(weight_row, input_row) for weight_row in checked_weight] for input_row in batch]


def count_nonzero(matrix: Sequence[Sequence[float]], tolerance: float = 0.0) -> int:
    checked = as_matrix(matrix)
    return sum(1 for row in checked for value in row if abs(value) > tolerance)


def flatten(matrix: Sequence[Sequence[float]]) -> Vector:
    checked = as_matrix(matrix)
    return [value for row in checked for value in row]


def unique_rounded_values(matrix: Sequence[Sequence[float]], decimals: int = 6) -> List[float]:
    values = sorted({round(value, decimals) for value in flatten(matrix)})
    return [float(value) for value in values]


def matrix_fingerprint(matrix: Sequence[Sequence[float]], decimals: int = 6) -> Tuple[Tuple[float, ...], ...]:
    checked = as_matrix(matrix)
    return tuple(tuple(round(value, decimals) for value in row) for row in checked)


def random_matrix(rows: int, cols: int, seed: int = 0, scale_value: float = 1.0) -> Matrix:
    rng = random.Random(seed)
    return [[scale_value * (2.0 * rng.random() - 1.0) for _ in range(cols)] for _ in range(rows)]


def random_batch(batch_size: int, width: int, seed: int = 0, scale_value: float = 1.0) -> Matrix:
    return random_matrix(batch_size, width, seed=seed, scale_value=scale_value)


def rms_relative_error(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    a = as_matrix(reference)
    b = as_matrix(candidate)
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        raise ValueError("RMS error shape mismatch")
    numerator = 0.0
    denominator = 0.0
    for row_index in range(len(a)):
        for col_index in range(len(a[0])):
            delta = a[row_index][col_index] - b[row_index][col_index]
            numerator += delta * delta
            denominator += a[row_index][col_index] * a[row_index][col_index]
    if denominator == 0.0:
        return math.sqrt(numerator / (len(a) * len(a[0])))
    return math.sqrt(numerator / denominator)


def mean_abs(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += abs(float(value))
        count += 1
    return total / count if count else 0.0
