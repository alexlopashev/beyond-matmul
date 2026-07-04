"""Fixed-weight lowering planner with exactness, error, reuse, and backend contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from beyond_matmul import _linalg as la
from beyond_matmul.approximations import (
    bitpacked_binary_approximation,
    codebook_quantize,
    low_rank_approximation,
    product_relative_error,
    sparse_from_dense,
    sparse_topk_by_density,
)
from beyond_matmul.ir import (
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
    },
    "cpu": {
        "dense_gemm",
        "diagonal_kernel",
        "sparse_kernel",
        "low_rank_product",
        "codebook_kernel",
        "bitpacked_kernel",
        "conv1d_direct",
    },
    "gpu": {
        "dense_gemm",
        "sparse_kernel",
        "low_rank_product",
        "bitpacked_kernel",
        "conv1d_direct",
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
    reasons: Tuple[str, ...] = ()

    @property
    def amortized_cost(self) -> float:
        calls = max(1, self.requested_calls)
        return self.estimated_apply_cost + (self.estimated_preprocessing_cost / calls)


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
    return batch_size * out_features * in_features


def _estimate_memory_bytes(operator: LinearOperator) -> int:
    out_features, in_features = operator.shape
    kind = operator.metadata.kind
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
    return out_features * in_features * 4


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
    if isinstance(operator, CodebookOperator):
        return CodebookOperator(operator.codes, operator.codebook, metadata=updated)
    # Bitpacked and convolution operators already expose accurate metadata for planning.
    operator.metadata = updated
    return operator


def _relative_error(
    reference_matrix: Sequence[Sequence[float]],
    candidate: LinearOperator,
    sample_inputs: Optional[Sequence[Sequence[float]]],
) -> Tuple[float, str, int]:
    if sample_inputs is not None:
        return product_relative_error(reference_matrix, candidate, sample_inputs), "output_relative_l2", len(sample_inputs)
    return la.relative_frobenius_error(reference_matrix, candidate.to_dense()), "matrix_relative_frobenius", 0


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


def _make_plan_option(reference_matrix: Sequence[Sequence[float]], candidate: LinearOperator, request: PlanningRequest) -> PlanOption:
    error, metric, sample_count = _relative_error(reference_matrix, candidate, request.sample_inputs)
    candidate = _with_contract(candidate, error, metric, sample_count)
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
        estimated_apply_cost=_estimate_apply_cost(candidate, request.batch_size),
        estimated_preprocessing_cost=candidate.metadata.reuse.preprocessing_cost,
        estimated_memory_bytes=_estimate_memory_bytes(candidate),
        requested_calls=request.calls,
        backend_supported=backend_supported,
        reuse_supported=reuse_supported,
        valid=valid,
        reasons=tuple(reasons),
    )


def plan_operator(operator: LinearOperator, request: PlanningRequest) -> LoweringPlan:
    reference_matrix = operator.to_dense()
    options = [_make_plan_option(reference_matrix, candidate, request) for candidate in _candidate_operators(operator, request)]
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
        fallback = _make_plan_option(reference_matrix, DenseOperator(reference_matrix), fallback_request)
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
