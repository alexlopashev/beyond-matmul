"""Fixed-weight lowering planner with exactness, error, reuse, and backend contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from beyond_matmul import _linalg as la
from beyond_matmul.approximations import (
    bitpacked_binary_approximation,
    codebook_quantize,
    low_rank_approximation,
    sparse_from_dense,
    sparse_topk_by_density,
)
from beyond_matmul.ir import (
    AffineOperator,
    ApproximationContract,
    CodebookOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    LowRankOperator,
    OperatorMetadata,
    Provenance,
    ReuseBudget,
    SparseCOOOperator,
)


BACKEND_SUPPORT = {
    "python": {
        "dense_gemm",
        "diagonal_kernel",
        "sparse_kernel",
        "low_rank_product",
        "codebook_kernel",
        "bitpacked_kernel",
        "conv1d_direct",
        "conv1d_channel_direct",
        "dense_gemm_bias",
        "diagonal_kernel_bias",
        "sparse_kernel_bias",
        "low_rank_product_bias",
        "codebook_kernel_bias",
        "bitpacked_kernel_bias",
        "conv1d_direct_bias",
        "conv1d_channel_direct_bias",
    },
    "cpu": {
        "dense_gemm",
        "diagonal_kernel",
        "sparse_kernel",
        "low_rank_product",
        "codebook_kernel",
        "bitpacked_kernel",
        "conv1d_direct",
        "conv1d_channel_direct",
        "dense_gemm_bias",
        "diagonal_kernel_bias",
        "sparse_kernel_bias",
        "low_rank_product_bias",
        "codebook_kernel_bias",
        "bitpacked_kernel_bias",
        "conv1d_direct_bias",
        "conv1d_channel_direct_bias",
    },
    "gpu": {
        "dense_gemm",
        "sparse_kernel",
        "low_rank_product",
        "bitpacked_kernel",
        "conv1d_direct",
        "conv1d_channel_direct",
        "dense_gemm_bias",
        "sparse_kernel_bias",
        "low_rank_product_bias",
        "bitpacked_kernel_bias",
        "conv1d_direct_bias",
        "conv1d_channel_direct_bias",
    },
}


@dataclass(frozen=True)
class PlanningRequest:
    batch_size: int = 1
    calls: int = 1
    max_relative_error: float = 0.0
    backend: str = "python"
    allow_approximate: bool = False
    sample_inputs: Optional[Sequence[Sequence[float]]] = None
    low_rank_ranks: Tuple[int, ...] = (1, 2, 4)
    sparse_densities: Tuple[float, ...] = (0.1, 0.25)
    codebook_sizes: Tuple[int, ...] = (2, 4, 8)


@dataclass(frozen=True)
class CostBreakdown:
    apply_ops: float
    memory_bytes_read: int
    memory_bytes_written: int
    cache_bytes: int
    preprocessing_ops: float
    calls: int

    @property
    def memory_bytes_moved(self) -> int:
        return self.memory_bytes_read + self.memory_bytes_written

    @property
    def amortized_preprocessing_ops(self) -> float:
        return self.preprocessing_ops / max(1, self.calls)

    @property
    def score(self) -> float:
        float32_memory_ops = self.memory_bytes_moved / 4.0
        return self.apply_ops + float32_memory_ops + self.amortized_preprocessing_ops


@dataclass
class PlanOption:
    name: str
    operator: LinearOperator
    exact: bool
    relative_error: float
    estimated_apply_cost: float
    estimated_preprocessing_cost: float
    estimated_memory_bytes: int
    requested_calls: int
    backend_supported: bool
    reuse_supported: bool
    valid: bool
    cost: CostBreakdown
    reasons: Tuple[str, ...] = ()

    @property
    def amortized_cost(self) -> float:
        return self.cost.score


@dataclass
class LoweringPlan:
    selected: PlanOption
    options: List[PlanOption] = field(default_factory=list)
    request: PlanningRequest = field(default_factory=PlanningRequest)

    def summary(self) -> str:
        return (
            f"{self.selected.name}: cost={self.selected.amortized_cost:.2f}, "
            f"error={self.selected.relative_error:.4g}, exact={self.selected.exact}"
        )


def _lowering_name(operator: LinearOperator) -> str:
    return operator.metadata.lowerings[0]


def _estimate_apply_cost(operator: LinearOperator, batch_size: int, word_bits: int = 64) -> float:
    out_features, in_features = operator.shape
    kind = operator.metadata.kind
    if isinstance(operator, AffineOperator):
        return _estimate_apply_cost(operator.linear, batch_size, word_bits=word_bits) + batch_size * out_features
    if kind == "diagonal":
        return batch_size * in_features
    if kind == "sparse_coo":
        return batch_size * getattr(operator, "nnz")
    if kind == "low_rank":
        rank = getattr(operator, "rank")
        return batch_size * (out_features * rank + rank * in_features)
    if kind == "codebook":
        return batch_size * out_features * in_features * 0.75
    if kind == "bitpacked_binary":
        return batch_size * ((out_features * in_features) / word_bits + out_features)
    if kind == "conv1d":
        kernel_size = int(operator.metadata.structure["kernel_size"])
        return batch_size * out_features * kernel_size
    if kind == "conv1d_channel":
        kernel_size = int(operator.metadata.structure["kernel_size"])
        in_channels = int(operator.metadata.structure["in_channels"])
        return batch_size * out_features * in_channels * kernel_size
    return batch_size * out_features * in_features


def _estimate_memory_bytes(operator: LinearOperator) -> int:
    out_features, in_features = operator.shape
    kind = operator.metadata.kind
    if isinstance(operator, AffineOperator):
        return _estimate_memory_bytes(operator.linear) + out_features * 4
    if kind == "diagonal":
        return in_features * 4
    if kind == "sparse_coo":
        return getattr(operator, "nnz") * 12
    if kind == "low_rank":
        rank = getattr(operator, "rank")
        return (out_features * rank + rank * in_features) * 4
    if kind == "codebook":
        codebook_size = int(operator.metadata.structure["codebook_size"])
        bits = operator.metadata.quantization.bits if operator.metadata.quantization else 8
        code_bytes = (out_features * in_features * bits + 7) // 8
        return codebook_size * 4 + code_bytes
    if kind == "bitpacked_binary":
        return 4 + ((out_features * in_features) + 7) // 8
    if kind == "conv1d":
        kernel_size = int(operator.metadata.structure["kernel_size"])
        return kernel_size * 4
    if kind == "conv1d_channel":
        out_channels = int(operator.metadata.structure["out_channels"])
        in_channels = int(operator.metadata.structure["in_channels"])
        kernel_size = int(operator.metadata.structure["kernel_size"])
        return out_channels * in_channels * kernel_size * 4
    return out_features * in_features * 4


def _estimate_cost(operator: LinearOperator, batch_size: int, calls: int) -> CostBreakdown:
    out_features, in_features = operator.shape
    if isinstance(operator, AffineOperator):
        inner = _estimate_cost(operator.linear, batch_size, calls)
        bias_bytes = out_features * 4
        return CostBreakdown(
            apply_ops=_estimate_apply_cost(operator, batch_size),
            memory_bytes_read=inner.memory_bytes_read + bias_bytes,
            memory_bytes_written=inner.memory_bytes_written,
            cache_bytes=inner.cache_bytes + bias_bytes,
            preprocessing_ops=operator.metadata.reuse.preprocessing_cost,
            calls=calls,
        )

    cache_bytes = _estimate_memory_bytes(operator)
    read_bytes = cache_bytes + batch_size * in_features * 4
    write_bytes = batch_size * out_features * 4
    if operator.metadata.kind == "low_rank":
        rank = getattr(operator, "rank")
        hidden_bytes = batch_size * rank * 4
        read_bytes += hidden_bytes
        write_bytes += hidden_bytes
    return CostBreakdown(
        apply_ops=_estimate_apply_cost(operator, batch_size),
        memory_bytes_read=read_bytes,
        memory_bytes_written=write_bytes,
        cache_bytes=cache_bytes,
        preprocessing_ops=operator.metadata.reuse.preprocessing_cost,
        calls=calls,
    )


def _backend_supported(operator: LinearOperator, backend: str) -> bool:
    supported = BACKEND_SUPPORT.get(backend, BACKEND_SUPPORT["python"])
    return any(lowering in supported for lowering in operator.metadata.lowerings)


def _with_contract(operator: LinearOperator, relative_error: float, metric: str, sample_count: int) -> LinearOperator:
    metadata = operator.metadata
    exact = relative_error <= 1e-9
    contract = ApproximationContract(
        mode="exact" if exact else "approximate",
        metric=metric if not exact else "none",
        epsilon=relative_error,
        observed_error=relative_error,
        sample_count=sample_count,
    )
    updated = OperatorMetadata(
        kind=metadata.kind,
        shape=metadata.shape,
        provenance=metadata.provenance,
        structure=metadata.structure,
        contract=contract,
        quantization=metadata.quantization,
        reuse=metadata.reuse,
        layout=metadata.layout,
        hardware=metadata.hardware,
        lowerings=metadata.lowerings,
    )
    if isinstance(operator, DenseOperator):
        return DenseOperator(operator.matrix, metadata=updated)
    if isinstance(operator, DiagonalOperator):
        return DiagonalOperator(operator.diagonal, metadata=updated)
    if isinstance(operator, SparseCOOOperator):
        return SparseCOOOperator(operator.rows, operator.cols, operator.values, operator.shape, metadata=updated)
    if isinstance(operator, LowRankOperator):
        return LowRankOperator(operator.left, operator.right, metadata=updated)
    if isinstance(operator, AffineOperator):
        return AffineOperator(operator.linear, operator.bias, metadata=updated)
    if isinstance(operator, CodebookOperator):
        return CodebookOperator(operator.codes, operator.codebook, metadata=updated)
    # Bitpacked and convolution operators already expose accurate metadata for planning.
    operator.metadata = updated
    return operator


def _bias_vector(operator: LinearOperator) -> List[float]:
    if isinstance(operator, AffineOperator):
        return list(operator.bias)
    return [0.0 for _ in range(operator.out_features)]


def _matrix_and_bias_relative_error(reference: LinearOperator, candidate: LinearOperator) -> float:
    reference_matrix = reference.to_dense()
    candidate_matrix = candidate.to_dense()
    matrix_delta = la.subtract(reference_matrix, candidate_matrix)
    reference_bias = _bias_vector(reference)
    candidate_bias = _bias_vector(candidate)
    if len(reference_bias) != len(candidate_bias):
        return math.inf
    numerator = la.frobenius_norm(matrix_delta) ** 2
    numerator += sum((a - b) * (a - b) for a, b in zip(reference_bias, candidate_bias))
    denominator = la.frobenius_norm(reference_matrix) ** 2
    denominator += sum(value * value for value in reference_bias)
    if denominator == 0.0:
        return math.sqrt(numerator)
    return math.sqrt(numerator / denominator)


def _relative_error(
    reference: LinearOperator,
    candidate: LinearOperator,
    sample_inputs: Optional[Sequence[Sequence[float]]],
) -> Tuple[float, str, int]:
    if sample_inputs is not None:
        exact = reference.apply(sample_inputs)
        observed = candidate.apply(sample_inputs)
        return la.rms_relative_error(exact, observed), "output_relative_l2", len(sample_inputs)
    if isinstance(reference, AffineOperator) or isinstance(candidate, AffineOperator):
        return _matrix_and_bias_relative_error(reference, candidate), "matrix_bias_relative_l2", 0
    return la.relative_frobenius_error(reference.to_dense(), candidate.to_dense()), "matrix_relative_frobenius", 0


def _diagonal_candidate(matrix: Sequence[Sequence[float]]) -> Optional[DiagonalOperator]:
    checked = la.as_matrix(matrix)
    rows, cols = len(checked), len(checked[0])
    if rows != cols:
        return None
    for row in range(rows):
        for col in range(cols):
            if row != col and abs(checked[row][col]) > 1e-9:
                return None
    return DiagonalOperator([checked[index][index] for index in range(rows)], provenance=Provenance(source="recovered_diagonal"))


def _exact_codebook_candidate(matrix: Sequence[Sequence[float]], max_codebook_size: int) -> Optional[CodebookOperator]:
    unique = la.unique_rounded_values(matrix)
    if len(unique) > max_codebook_size:
        return None
    return codebook_quantize(matrix, codebook_size=max_codebook_size, provenance=Provenance(source="recovered_exact_codebook"))


def _candidate_operators(operator: LinearOperator, request: PlanningRequest) -> List[LinearOperator]:
    if isinstance(operator, AffineOperator):
        return [AffineOperator(candidate, operator.bias) for candidate in _candidate_operators(operator.linear, request)]

    matrix = operator.to_dense()
    candidates: List[LinearOperator] = []

    if operator.metadata.kind != "dense":
        candidates.append(operator)

    dense_fallback = DenseOperator(matrix, provenance=Provenance(source="dense_fallback"))
    candidates.append(dense_fallback)

    diagonal = _diagonal_candidate(matrix)
    if diagonal is not None:
        candidates.append(diagonal)

    exact_sparse = sparse_from_dense(matrix, tolerance=1e-12, provenance=Provenance(source="recovered_sparse_exact"))
    if exact_sparse.nnz < len(matrix) * len(matrix[0]):
        candidates.append(exact_sparse)

    exact_codebook = _exact_codebook_candidate(matrix, max(request.codebook_sizes) if request.codebook_sizes else 16)
    if exact_codebook is not None:
        candidates.append(exact_codebook)

    if request.allow_approximate:
        rows, cols = len(matrix), len(matrix[0])
        for rank in request.low_rank_ranks:
            if rank <= min(rows, cols):
                candidates.append(low_rank_approximation(matrix, rank=rank, provenance=Provenance(source="planner_low_rank")))
        for density in request.sparse_densities:
            candidates.append(sparse_topk_by_density(matrix, density=density, provenance=Provenance(source="planner_sparse_topk")))
        for codebook_size in request.codebook_sizes:
            candidates.append(codebook_quantize(matrix, codebook_size=codebook_size, provenance=Provenance(source="planner_codebook")))
        candidates.append(bitpacked_binary_approximation(matrix, provenance=Provenance(source="planner_bitpacked_binary")))

    return candidates


def _make_plan_option(reference: LinearOperator, candidate: LinearOperator, request: PlanningRequest) -> PlanOption:
    error, metric, sample_count = _relative_error(reference, candidate, request.sample_inputs)
    candidate = _with_contract(candidate, error, metric, sample_count)
    cost = _estimate_cost(candidate, request.batch_size, request.calls)
    exact = candidate.metadata.contract.is_exact
    backend_supported = _backend_supported(candidate, request.backend)
    reuse_supported = request.calls >= candidate.metadata.reuse.amortize_over_calls
    error_supported = exact or (request.allow_approximate and error <= request.max_relative_error)
    reasons: List[str] = []
    if not backend_supported:
        reasons.append("backend does not support lowering")
    if not reuse_supported:
        reasons.append("preprocessing does not amortize over requested calls")
    if not error_supported:
        reasons.append("error contract exceeds request")
    valid = backend_supported and reuse_supported and error_supported
    return PlanOption(
        name=_lowering_name(candidate),
        operator=candidate,
        exact=exact,
        relative_error=error,
        estimated_apply_cost=cost.apply_ops + (cost.memory_bytes_moved / 4.0),
        estimated_preprocessing_cost=cost.preprocessing_ops,
        estimated_memory_bytes=cost.cache_bytes,
        requested_calls=request.calls,
        backend_supported=backend_supported,
        reuse_supported=reuse_supported,
        valid=valid,
        cost=cost,
        reasons=tuple(reasons),
    )


def _dense_fallback_for(reference: LinearOperator) -> LinearOperator:
    dense = DenseOperator(reference.to_dense())
    if isinstance(reference, AffineOperator):
        return AffineOperator(dense, reference.bias)
    return dense


def plan_operator(operator: LinearOperator, request: PlanningRequest) -> LoweringPlan:
    options = [_make_plan_option(operator, candidate, request) for candidate in _candidate_operators(operator, request)]
    valid_options = [option for option in options if option.valid]
    if not valid_options:
        fallback_request = PlanningRequest(
            batch_size=request.batch_size,
            calls=max(request.calls, 1),
            max_relative_error=0.0,
            backend="python",
            allow_approximate=False,
            sample_inputs=request.sample_inputs,
        )
        fallback = _make_plan_option(operator, _dense_fallback_for(operator), fallback_request)
        fallback.valid = True
        options.append(fallback)
        valid_options = [fallback]
    selected = min(valid_options, key=lambda option: (option.amortized_cost, option.estimated_memory_bytes))
    return LoweringPlan(selected=selected, options=options, request=request)


def plan_fixed_weight(
    weight: Sequence[Sequence[float]] | LinearOperator,
    request: Optional[PlanningRequest] = None,
) -> LoweringPlan:
    request = request or PlanningRequest()
    operator = weight if isinstance(weight, LinearOperator) else DenseOperator(weight, provenance=Provenance(source="fixed_weight"))
    return plan_operator(operator, request)


def valid_options(plan: LoweringPlan) -> Iterable[PlanOption]:
    return (option for option in plan.options if option.valid)
