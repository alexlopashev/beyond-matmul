"""Provenance-aware linear operators and lowering planner."""

from beyond_matmul.ir import (
    ApproximationContract,
    BitpackedBinaryOperator,
    CodebookOperator,
    Convolution1DOperator,
    DenseOperator,
    DiagonalOperator,
    HardwareTarget,
    LayoutSpec,
    LinearOperator,
    LowRankOperator,
    OperatorMetadata,
    Provenance,
    QuantizationSpec,
    ReuseBudget,
    SparseCOOOperator,
)
from beyond_matmul.planner import LoweringPlan, PlanningRequest, plan_fixed_weight

__all__ = [
    "ApproximationContract",
    "BitpackedBinaryOperator",
    "CodebookOperator",
    "Convolution1DOperator",
    "DenseOperator",
    "DiagonalOperator",
    "HardwareTarget",
    "LayoutSpec",
    "LinearOperator",
    "LowRankOperator",
    "LoweringPlan",
    "OperatorMetadata",
    "PlanningRequest",
    "Provenance",
    "QuantizationSpec",
    "ReuseBudget",
    "SparseCOOOperator",
    "plan_fixed_weight",
]
