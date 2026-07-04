"""Prototype provenance capture helpers.

This module sketches how a framework/compiler pass can preserve structure before
it becomes an anonymous dense matmul. The pure-Python helpers are deliberately
simple; optional integrations can translate framework graph nodes into the same
operator objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from beyond_matmul.ir import (
    Convolution1DOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    LowRankOperator,
    Provenance,
)


@dataclass(frozen=True)
class TraceEvent:
    name: str
    op_type: str
    provenance: Provenance
    notes: Dict[str, str] = field(default_factory=dict)


class ProvenanceTracer:
    """Records structured linear operators before dense fallback."""

    def __init__(self, framework: str = "python") -> None:
        self.framework = framework
        self.events: List[TraceEvent] = []
        self.operators: Dict[str, LinearOperator] = {}

    def _record(self, name: str, op_type: str, operator: LinearOperator, notes: Optional[Dict[str, str]] = None) -> LinearOperator:
        self.operators[name] = operator
        self.events.append(
            TraceEvent(
                name=name,
                op_type=op_type,
                provenance=operator.metadata.provenance,
                notes=notes or {},
            )
        )
        return operator

    def dense_weight(self, name: str, matrix: Sequence[Sequence[float]], expression: Optional[str] = None) -> DenseOperator:
        provenance = Provenance(source=name, framework=self.framework, expression=expression or "dense_weight")
        return self._record(name, "dense", DenseOperator(matrix, provenance=provenance))  # type: ignore[return-value]

    def diagonal(self, name: str, diagonal: Sequence[float], expression: Optional[str] = None) -> DiagonalOperator:
        provenance = Provenance(source=name, framework=self.framework, expression=expression or "diag(v)")
        return self._record(name, "diagonal", DiagonalOperator(diagonal, provenance=provenance))  # type: ignore[return-value]

    def low_rank(
        self,
        name: str,
        left: Sequence[Sequence[float]],
        right: Sequence[Sequence[float]],
        expression: Optional[str] = None,
    ) -> LowRankOperator:
        provenance = Provenance(source=name, framework=self.framework, expression=expression or "U @ V")
        return self._record(name, "low_rank", LowRankOperator(left, right, provenance=provenance))  # type: ignore[return-value]

    def conv1d(
        self,
        name: str,
        kernel: Sequence[float],
        input_length: int,
        expression: Optional[str] = None,
    ) -> Convolution1DOperator:
        provenance = Provenance(source=name, framework=self.framework, expression=expression or "conv1d(x, kernel)")
        return self._record(name, "conv1d", Convolution1DOperator(kernel, input_length, provenance=provenance))  # type: ignore[return-value]


def capture_torch_fx_patterns(module) -> List[TraceEvent]:
    """Optional placeholder for a torch.fx frontend pass.

    The prototype keeps PyTorch out of the dependency chain. If torch is present,
    this function symbolically traces a module and returns coarse events for
    matmul-like nodes. A full pass would replace these events with structured
    LinearOperator payloads before lowering.
    """

    try:
        import torch.fx as fx  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise RuntimeError("PyTorch is required for capture_torch_fx_patterns") from exc

    graph = fx.symbolic_trace(module)
    events: List[TraceEvent] = []
    for node in graph.graph.nodes:
        target = str(node.target)
        if "matmul" in target or "linear" in target or "conv" in target:
            events.append(
                TraceEvent(
                    name=node.name,
                    op_type=target,
                    provenance=Provenance(source=node.name, framework="torch.fx", expression=target, confidence=0.5),
                    notes={"capture": "coarse_fx_pattern"},
                )
            )
    return events
