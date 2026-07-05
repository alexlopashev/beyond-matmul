"""Prototype provenance capture helpers.

This module sketches how a framework/compiler pass can preserve structure before
it becomes an anonymous dense matmul. The pure-Python helpers are deliberately
simple; optional integrations can translate framework graph nodes into the same
operator objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


@dataclass(frozen=True)
class CapturedOperator:
    name: str
    operator: LinearOperator
    event: TraceEvent


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


def capture_torch_fx_linear_operators(module) -> Dict[str, CapturedOperator]:
    """Trace a PyTorch module and capture structured linear operators.

    The first implemented pattern is a fixed-weight low-rank projection:

        y = linear(linear(x, right), left)

    where `right` has shape `(rank, in_features)` and `left` has shape
    `(out_features, rank)`. In dense form, this is equivalent to
    `linear(x, left @ right)`, but preserving the two factors gives the planner
    an exact low-rank lowering before the computation becomes anonymous dense
    matrix multiplication.
    """

    try:
        import torch.fx as fx  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise RuntimeError("PyTorch is required for capture_torch_fx_linear_operators") from exc

    graph_module = fx.symbolic_trace(module)
    return extract_torch_fx_low_rank_operators(graph_module)


def extract_torch_fx_low_rank_operators(graph_module: Any) -> Dict[str, CapturedOperator]:
    """Extract low-rank linear operators from a traced FX graph-like object.

    The function accepts real `torch.fx.GraphModule` objects and lightweight
    fakes used by tests. It intentionally depends on FX node conventions rather
    than importing PyTorch, which keeps the default test suite dependency-free.
    """

    captured: Dict[str, CapturedOperator] = {}
    nodes = list(_iter_fx_nodes(graph_module))
    for node in nodes:
        if not _is_linear_node(node):
            continue
        args = list(getattr(node, "args", ()) or ())
        if not args:
            continue
        inner = args[0]
        if not _is_linear_node(inner):
            continue
        right = _linear_weight(graph_module, inner)
        left = _linear_weight(graph_module, node)
        if right is None or left is None:
            continue
        try:
            operator = LowRankOperator(
                left,
                right,
                provenance=Provenance(
                    source=str(getattr(node, "name", "torch_fx_low_rank")),
                    framework="torch.fx",
                    expression="linear(linear(x, right), left)",
                    inputs=(str(getattr(inner, "name", "inner")), str(getattr(node, "name", "outer"))),
                    transform_history=("torch.fx symbolic_trace", "low_rank_linear_pattern"),
                    confidence=0.95,
                ),
            )
        except ValueError:
            continue
        event = TraceEvent(
            name=str(getattr(node, "name", "torch_fx_low_rank")),
            op_type="low_rank_linear",
            provenance=operator.metadata.provenance,
            notes={
                "inner_node": str(getattr(inner, "name", "unknown")),
                "outer_node": str(getattr(node, "name", "unknown")),
                "rank": str(operator.rank),
            },
        )
        captured[event.name] = CapturedOperator(name=event.name, operator=operator, event=event)
    return captured


def _iter_fx_nodes(graph_module: Any) -> Iterable[Any]:
    graph = getattr(graph_module, "graph", graph_module)
    return getattr(graph, "nodes", ())


def _is_linear_node(node: Any) -> bool:
    op = getattr(node, "op", None)
    target = getattr(node, "target", None)
    target_text = str(target)
    if op == "call_function":
        return target_text.endswith("linear") or "linear" in target_text
    if op == "call_module":
        return "Linear" in type(target).__name__ or "linear" in target_text.lower()
    return False


def _linear_weight(graph_module: Any, node: Any) -> Optional[List[List[float]]]:
    op = getattr(node, "op", None)
    if op == "call_module":
        module = _resolve_attr(graph_module, str(getattr(node, "target", "")))
        return _to_matrix(getattr(module, "weight", None))

    args = list(getattr(node, "args", ()) or ())
    if len(args) < 2:
        return None
    return _node_value_as_matrix(graph_module, args[1])


def _node_value_as_matrix(graph_module: Any, value: Any) -> Optional[List[List[float]]]:
    if getattr(value, "op", None) == "get_attr":
        return _to_matrix(_resolve_attr(graph_module, str(getattr(value, "target", ""))))
    return _to_matrix(value)


def _resolve_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not part:
            continue
        current = getattr(current, part)
    return current


def _to_matrix(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    matrix: List[List[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            return None
        matrix.append([float(item) for item in row])
    return matrix
