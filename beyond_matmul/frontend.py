"""Prototype provenance capture helpers.

This module sketches how a framework/compiler pass can preserve structure before
it becomes an anonymous dense matmul. The pure-Python helpers are deliberately
simple; optional integrations can translate framework graph nodes into the same
operator objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from beyond_matmul import _linalg as la
from beyond_matmul.ir import (
    AffineOperator,
    Convolution1DOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    LowRankOperator,
    MultiChannelConvolution1DOperator,
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


def capture_torch_fx_linear_operators(module, sample_inputs: Any = None) -> Dict[str, CapturedOperator]:
    """Backward-compatible alias for Torch FX structured-operator capture."""

    return capture_torch_fx_operators(module, sample_inputs=sample_inputs)


def capture_torch_fx_operators(module, sample_inputs: Any = None) -> Dict[str, CapturedOperator]:
    """Trace a PyTorch module and capture structured fixed-weight operators.

    The core pattern is a fixed-weight low-rank or affine low-rank projection:

        y = linear(linear(x, right), left)

    where `right` has shape `(rank, in_features)` and `left` has shape
    `(out_features, rank)`. The broader helper also captures narrow fixed-weight
    `matmul`/`addmm` patterns and `nn.Conv1d` modules when enough shape and
    orientation information is present.
    """

    try:
        import torch.fx as fx  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise RuntimeError("PyTorch is required for capture_torch_fx_operators") from exc

    graph_module = fx.symbolic_trace(module)
    if sample_inputs is not None:
        _propagate_torch_fx_shapes(graph_module, sample_inputs)
    captured = extract_torch_fx_operators(graph_module)
    for name, operator in capture_torch_named_adapter_operators(module).items():
        captured.setdefault(name, operator)
    return captured


def extract_torch_fx_low_rank_operators(graph_module: Any) -> Dict[str, CapturedOperator]:
    """Backward-compatible low-rank-only FX extraction helper."""

    return {
        name: captured
        for name, captured in extract_torch_fx_operators(graph_module).items()
        if captured.event.op_type in {"low_rank_linear", "affine_low_rank_linear"}
    }


def extract_torch_fx_linear_operators(graph_module: Any) -> Dict[str, CapturedOperator]:
    """Backward-compatible alias for Torch FX structured-operator extraction."""

    return extract_torch_fx_operators(graph_module)


def extract_torch_fx_operators(graph_module: Any) -> Dict[str, CapturedOperator]:
    """Extract fixed-weight operators from a traced FX graph-like object.

    The function accepts real `torch.fx.GraphModule` objects and lightweight
    fakes used by tests. It intentionally depends on FX node conventions rather
    than importing PyTorch, which keeps the default test suite dependency-free.
    """

    captured: Dict[str, CapturedOperator] = {}
    nodes = list(_iter_fx_nodes(graph_module))
    for node in nodes:
        if _is_conv1d_node(graph_module, node):
            captured_operator = _capture_conv1d(graph_module, node)
            if captured_operator is not None:
                captured[captured_operator.name] = captured_operator
            continue
        if _is_addmm_node(node):
            captured_operator = _capture_dense_addmm(graph_module, node)
            if captured_operator is not None:
                captured[captured_operator.name] = captured_operator
            continue
        if _is_matmul_node(node):
            captured_operator = _capture_dense_matmul(graph_module, node)
            if captured_operator is not None:
                captured[captured_operator.name] = captured_operator
            continue
        if not _is_linear_node(graph_module, node):
            continue
        args = list(getattr(node, "args", ()) or ())
        if not args:
            continue
        inner = args[0]
        captured_operator = None
        if _is_linear_node(graph_module, inner):
            captured_operator = _capture_nested_linear(graph_module, inner, node)
        elif _is_embedding_node(graph_module, inner):
            captured_operator = _capture_embedding_projection(graph_module, inner, node)
        if captured_operator is not None:
            captured[captured_operator.name] = captured_operator
    return captured


def capture_torch_named_adapter_operators(module: Any) -> Dict[str, CapturedOperator]:
    """Capture named low-rank adapter factors even when forward uses a merged weight."""

    captured: Dict[str, CapturedOperator] = {}
    for prefix, parent in _iter_named_modules(module):
        children = _named_children(parent)
        for right_name, left_name in (
            ("down", "up"),
            ("lora_A", "lora_B"),
            ("adapter_A", "adapter_B"),
            ("A", "B"),
            ("right", "left"),
        ):
            right_module = children.get(right_name)
            left_module = children.get(left_name)
            if not (_is_linear_module(right_module) and _is_linear_module(left_module)):
                continue
            right = _module_weight(right_module)
            left = _module_weight(left_module)
            if right is None or left is None:
                continue
            inner_bias = _module_bias(right_module, expected_length=len(right))
            outer_bias = _module_bias(left_module, expected_length=len(left))
            if inner_bias is None or outer_bias is None:
                continue
            source = ".".join(part for part in (prefix, left_name) if part) or left_name
            provenance = Provenance(
                source=source,
                framework="torch.nn",
                expression=f"{left_name}({right_name}(x))",
                inputs=(right_name, left_name),
                transform_history=("named_module_scan", "low_rank_adapter_factors"),
                confidence=0.9,
            )
            notes = {
                "inner_node": right_name,
                "outer_node": left_name,
                "rank": str(len(right)),
                "capture": "named_adapter_pair",
            }
            if _has_merged_weight_hint(parent):
                notes["merged_weight_hint"] = "true"
            captured_operator = _captured_low_rank_or_affine(source, left, right, inner_bias, outer_bias, provenance, notes)
            if captured_operator is not None:
                captured[captured_operator.name] = captured_operator
    return captured


def _capture_nested_linear(graph_module: Any, inner: Any, outer: Any) -> Optional[CapturedOperator]:
    right = _linear_weight(graph_module, inner)
    left = _linear_weight(graph_module, outer)
    if right is None or left is None:
        return None
    inner_bias = _linear_bias(graph_module, inner, expected_length=len(right))
    outer_bias = _linear_bias(graph_module, outer, expected_length=len(left))
    if inner_bias is None or outer_bias is None:
        return None
    provenance = Provenance(
        source=str(getattr(outer, "name", "torch_fx_low_rank")),
        framework="torch.fx",
        expression="linear(linear(x, right), left)",
        inputs=(str(getattr(inner, "name", "inner")), str(getattr(outer, "name", "outer"))),
        transform_history=("torch.fx symbolic_trace", "low_rank_linear_pattern"),
        confidence=0.95,
    )
    notes = {
        "inner_node": str(getattr(inner, "name", "unknown")),
        "outer_node": str(getattr(outer, "name", "unknown")),
        "rank": str(len(right)),
    }
    return _captured_low_rank_or_affine(str(getattr(outer, "name", "torch_fx_low_rank")), left, right, inner_bias, outer_bias, provenance, notes)


def _capture_embedding_projection(graph_module: Any, embedding: Any, projection: Any) -> Optional[CapturedOperator]:
    embedding_weight = _embedding_weight(graph_module, embedding)
    left = _linear_weight(graph_module, projection)
    if embedding_weight is None or left is None:
        return None
    right = la.transpose(embedding_weight)
    inner_bias = [0.0 for _ in right]
    outer_bias = _linear_bias(graph_module, projection, expected_length=len(left))
    if outer_bias is None:
        return None
    provenance = Provenance(
        source=str(getattr(projection, "name", "torch_fx_embedding_projection")),
        framework="torch.fx",
        expression="linear(embedding(ids), projection) over one_hot(ids)",
        inputs=(str(getattr(embedding, "name", "embedding")), str(getattr(projection, "name", "projection"))),
        transform_history=("torch.fx symbolic_trace", "embedding_projection_pattern"),
        confidence=0.85,
    )
    notes = {
        "inner_node": str(getattr(embedding, "name", "unknown")),
        "outer_node": str(getattr(projection, "name", "unknown")),
        "rank": str(len(right)),
        "input_basis": "one_hot",
    }
    return _captured_low_rank_or_affine(str(getattr(projection, "name", "torch_fx_embedding_projection")), left, right, inner_bias, outer_bias, provenance, notes)


def _captured_low_rank_or_affine(
    name: str,
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    inner_bias: Sequence[float],
    outer_bias: Sequence[float],
    provenance: Provenance,
    notes: Dict[str, str],
) -> Optional[CapturedOperator]:
    try:
        linear = LowRankOperator(left, right, provenance=provenance)
        bias = _compose_linear_bias(left, inner_bias, outer_bias)
        operator: LinearOperator
        if _is_zero_vector(bias):
            operator = linear
            op_type = "low_rank_linear"
        else:
            operator = AffineOperator(linear, bias, provenance=provenance)
            op_type = "affine_low_rank_linear"
            notes = {**notes, "bias": "true"}
    except ValueError:
        return None
    event = TraceEvent(name=name, op_type=op_type, provenance=operator.metadata.provenance, notes=notes)
    return CapturedOperator(name=event.name, operator=operator, event=event)


def _capture_conv1d(graph_module: Any, node: Any) -> Optional[CapturedOperator]:
    op = getattr(node, "op", None)
    if op == "call_module":
        return _capture_conv1d_module(graph_module, node)
    if op == "call_function":
        return _capture_conv1d_function(graph_module, node)
    return None


def _capture_conv1d_module(graph_module: Any, node: Any) -> Optional[CapturedOperator]:
    module = _maybe_resolve_attr(graph_module, str(getattr(node, "target", "")))
    if module is None or not _is_supported_conv1d_module(module):
        return None
    weight = _conv1d_weight(getattr(module, "weight", None))
    if weight is None:
        return None
    module_in_channels = _positive_int(getattr(module, "in_channels", None))
    module_out_channels = _positive_int(getattr(module, "out_channels", None))
    if module_in_channels is not None and module_in_channels != len(weight[0]):
        return None
    if module_out_channels is not None and module_out_channels != len(weight):
        return None
    input_shape = _conv1d_input_shape(graph_module, node, module, weight)
    if input_shape is None:
        return None
    input_channels, input_length = input_shape
    if input_channels != len(weight[0]):
        return None
    name = str(getattr(node, "name", "torch_fx_conv1d"))
    target = str(getattr(node, "target", name))
    provenance = Provenance(
        source=name,
        framework="torch.fx",
        expression="conv1d(x, weight)",
        inputs=(target,),
        transform_history=("torch.fx symbolic_trace", "conv1d_module_pattern"),
        confidence=0.95,
    )
    notes = {
        "module": target,
        "capture": "conv1d_module",
    }
    return _captured_conv1d_operator(name, weight, getattr(module, "bias", None), input_length, provenance, notes)


def _capture_conv1d_function(graph_module: Any, node: Any) -> Optional[CapturedOperator]:
    args = list(getattr(node, "args", ()) or ())
    kwargs = dict(getattr(node, "kwargs", {}) or {})
    if len(args) < 2:
        return None
    activation, weight_operand = args[0], args[1]
    if not _is_runtime_activation_operand(activation):
        return None
    bias_operand = args[2] if len(args) >= 3 else kwargs.get("bias")
    stride = args[3] if len(args) >= 4 else kwargs.get("stride", 1)
    padding = args[4] if len(args) >= 5 else kwargs.get("padding", 0)
    dilation = args[5] if len(args) >= 6 else kwargs.get("dilation", 1)
    groups = args[6] if len(args) >= 7 else kwargs.get("groups", 1)
    if not _conv1d_parameters_supported(stride, padding, dilation, groups):
        return None
    weight = _node_value_as_conv1d_weight(graph_module, weight_operand)
    if weight is None:
        return None
    input_shape = _node_conv1d_input_shape(activation)
    if input_shape is None:
        return None
    input_channels, input_length = input_shape
    if input_channels != len(weight[0]):
        return None

    name = str(getattr(node, "name", "torch_fx_conv1d"))
    weight_source = _source_name(weight_operand) or "fixed_weight"
    bias_source = _source_name(bias_operand) if bias_operand is not None else "none"
    provenance = Provenance(
        source=name,
        framework="torch.fx",
        expression="torch.nn.functional.conv1d(x, weight)",
        inputs=(_source_name(activation) or "activation", weight_source),
        transform_history=("torch.fx symbolic_trace", "conv1d_function_pattern"),
        confidence=0.9,
    )
    notes = {
        "capture": "conv1d_function",
        "weight": weight_source,
        "bias": bias_source or "none",
    }
    return _captured_conv1d_operator(name, weight, _node_value(graph_module, bias_operand), input_length, provenance, notes)


def _captured_conv1d_operator(
    name: str,
    weight: Sequence[Sequence[Sequence[float]]],
    bias_value: Any,
    input_length: int,
    provenance: Provenance,
    notes: Dict[str, str],
) -> Optional[CapturedOperator]:
    out_channels = len(weight)
    in_channels = len(weight[0])
    kernel_size = len(weight[0][0])
    output_length = input_length - kernel_size + 1
    if output_length <= 0:
        return None
    notes = {
        **notes,
        "out_channels": str(out_channels),
        "in_channels": str(in_channels),
        "kernel_size": str(kernel_size),
        "input_length": str(input_length),
        "output_length": str(output_length),
    }
    try:
        if out_channels == 1 and in_channels == 1:
            linear: LinearOperator = Convolution1DOperator(weight[0][0], input_length=input_length, provenance=provenance)
        else:
            linear = MultiChannelConvolution1DOperator(weight, input_length=input_length, provenance=provenance)
        bias = _conv1d_bias(bias_value, out_channels=out_channels, output_length=output_length)
        if bias is None:
            return None
        operator: LinearOperator
        if _is_zero_vector(bias):
            operator = linear
            op_type = linear.metadata.kind
        else:
            operator = AffineOperator(linear, bias, provenance=provenance)
            op_type = f"affine_{linear.metadata.kind}"
            notes = {**notes, "bias": "true"}
    except ValueError:
        return None
    event = TraceEvent(name=name, op_type=op_type, provenance=operator.metadata.provenance, notes=notes)
    return CapturedOperator(name=event.name, operator=operator, event=event)


def _capture_dense_matmul(graph_module: Any, node: Any) -> Optional[CapturedOperator]:
    args = list(getattr(node, "args", ()) or ())
    if len(args) < 2:
        return None
    activation, rhs = args[0], args[1]
    if not _is_runtime_activation_operand(activation):
        return None
    weight = _fixed_weight_from_transposed_operand(graph_module, rhs)
    if weight is None or not _matmul_shapes_compatible(activation, weight):
        return None

    name = str(getattr(node, "name", "torch_fx_matmul"))
    weight_source = _source_name(_transpose_base(rhs)) or "fixed_weight"
    provenance = Provenance(
        source=name,
        framework="torch.fx",
        expression="x @ weight.T",
        inputs=(_source_name(activation) or "activation", weight_source),
        transform_history=("torch.fx symbolic_trace", "dense_matmul_pattern"),
        confidence=0.9,
    )
    notes = {
        "capture": "dense_matmul",
        "rhs": weight_source,
        "rhs_orientation": "transposed_fixed_weight",
        "weight_layout": "out_in",
    }
    try:
        operator = DenseOperator(weight, provenance=provenance)
    except ValueError:
        return None
    event = TraceEvent(name=name, op_type="dense_matmul", provenance=operator.metadata.provenance, notes=notes)
    return CapturedOperator(name=event.name, operator=operator, event=event)


def _capture_dense_addmm(graph_module: Any, node: Any) -> Optional[CapturedOperator]:
    args = list(getattr(node, "args", ()) or ())
    kwargs = dict(getattr(node, "kwargs", {}) or {})
    if len(args) < 3:
        return None
    if not (_is_default_scale(args[3] if len(args) >= 4 else kwargs.get("beta", 1.0))):
        return None
    if not (_is_default_scale(args[4] if len(args) >= 5 else kwargs.get("alpha", 1.0))):
        return None

    bias_operand, activation, rhs = args[0], args[1], args[2]
    if not _is_runtime_activation_operand(activation):
        return None
    weight = _fixed_weight_from_transposed_operand(graph_module, rhs)
    if weight is None or not _matmul_shapes_compatible(activation, weight):
        return None
    bias = _bias_value(_node_value(graph_module, bias_operand), expected_length=len(weight))
    if bias is None:
        return None

    name = str(getattr(node, "name", "torch_fx_addmm"))
    weight_source = _source_name(_transpose_base(rhs)) or "fixed_weight"
    bias_source = _source_name(bias_operand) or "fixed_bias"
    provenance = Provenance(
        source=name,
        framework="torch.fx",
        expression="torch.addmm(bias, x, weight.T)",
        inputs=(bias_source, _source_name(activation) or "activation", weight_source),
        transform_history=("torch.fx symbolic_trace", "dense_addmm_pattern"),
        confidence=0.9,
    )
    notes = {
        "capture": "dense_addmm",
        "bias": bias_source,
        "rhs": weight_source,
        "rhs_orientation": "transposed_fixed_weight",
        "weight_layout": "out_in",
    }
    try:
        linear = DenseOperator(weight, provenance=provenance)
        operator = AffineOperator(linear, bias, provenance=provenance)
    except ValueError:
        return None
    event = TraceEvent(name=name, op_type="affine_dense_matmul", provenance=operator.metadata.provenance, notes=notes)
    return CapturedOperator(name=event.name, operator=operator, event=event)


def _propagate_torch_fx_shapes(graph_module: Any, sample_inputs: Any) -> None:
    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional torch
        raise RuntimeError("Torch FX shape propagation is required for sample_inputs") from exc
    args = sample_inputs if isinstance(sample_inputs, tuple) else (sample_inputs,)
    try:
        ShapeProp(graph_module).propagate(*args)
    except Exception as exc:  # pragma: no cover - depends on optional torch graph execution
        raise RuntimeError("Torch FX shape propagation failed for sample_inputs") from exc


def _iter_fx_nodes(graph_module: Any) -> Iterable[Any]:
    graph = getattr(graph_module, "graph", graph_module)
    return getattr(graph, "nodes", ())


def _is_linear_node(graph_module: Any, node: Any) -> bool:
    op = getattr(node, "op", None)
    target = getattr(node, "target", None)
    target_text = str(target)
    if op == "call_function":
        return target_text.endswith("linear") or "linear" in target_text
    if op == "call_module":
        module = _maybe_resolve_attr(graph_module, target_text)
        if module is not None and type(module).__name__ == "Linear":
            return True
        return "linear" in target_text.lower()
    return False


def _is_matmul_node(node: Any) -> bool:
    op = getattr(node, "op", None)
    if op == "call_function":
        return _target_has_name(getattr(node, "target", None), {"matmul", "mm"})
    if op == "call_method":
        return str(getattr(node, "target", "")) in {"matmul", "mm"}
    return False


def _is_addmm_node(node: Any) -> bool:
    return getattr(node, "op", None) == "call_function" and _target_has_name(getattr(node, "target", None), {"addmm"})


def _is_conv1d_node(graph_module: Any, node: Any) -> bool:
    if getattr(node, "op", None) == "call_function":
        return _target_has_name(getattr(node, "target", None), {"conv1d"})
    if getattr(node, "op", None) != "call_module":
        return False
    target_text = str(getattr(node, "target", ""))
    module = _maybe_resolve_attr(graph_module, target_text)
    if module is not None and type(module).__name__ == "Conv1d":
        return True
    return "conv1d" in target_text.lower()


def _is_supported_conv1d_module(module: Any) -> bool:
    if type(module).__name__ != "Conv1d":
        return False
    return _conv1d_parameters_supported(
        getattr(module, "stride", 1),
        getattr(module, "padding", 0),
        getattr(module, "dilation", 1),
        getattr(module, "groups", 1),
    )


def _conv1d_parameters_supported(stride: Any, padding: Any, dilation: Any, groups: Any) -> bool:
    return _single_int(stride) == 1 and _single_int(padding) == 0 and _single_int(dilation) == 1 and _single_int(groups) == 1


def _conv1d_weight(value: Any) -> Optional[List[List[List[float]]]]:
    value = _to_python_value(value)
    if not isinstance(value, (list, tuple)) or not value:
        return None
    weight: List[List[List[float]]] = []
    expected_in_channels: Optional[int] = None
    expected_kernel_size: Optional[int] = None
    for out_channel in value:
        if not isinstance(out_channel, (list, tuple)) or not out_channel:
            return None
        if expected_in_channels is None:
            expected_in_channels = len(out_channel)
        elif len(out_channel) != expected_in_channels:
            return None
        checked_out: List[List[float]] = []
        for kernel in out_channel:
            if not isinstance(kernel, (list, tuple)) or not kernel:
                return None
            if any(isinstance(item, (list, tuple)) for item in kernel):
                return None
            if expected_kernel_size is None:
                expected_kernel_size = len(kernel)
            elif len(kernel) != expected_kernel_size:
                return None
            checked_out.append([float(item) for item in kernel])
        weight.append(checked_out)
    return weight


def _node_value_as_conv1d_weight(graph_module: Any, value: Any) -> Optional[List[List[List[float]]]]:
    return _conv1d_weight(_node_value(graph_module, value))


def _conv1d_bias(value: Any, out_channels: int, output_length: int) -> Optional[List[float]]:
    if value is None:
        return [0.0 for _ in range(out_channels * output_length)]
    bias = _to_vector(value)
    if bias is None or len(bias) != out_channels:
        return None
    return [bias[channel] for channel in range(out_channels) for _ in range(output_length)]


def _conv1d_input_shape(
    graph_module: Any,
    node: Any,
    module: Any,
    weight: Sequence[Sequence[Sequence[float]]],
) -> Optional[tuple[int, int]]:
    args = list(getattr(node, "args", ()) or ())
    if args:
        shape = _node_conv1d_input_shape(args[0])
        if shape is not None:
            return shape
    for owner in (module, graph_module):
        for attr_name in ("input_length", "sequence_length", "fixed_input_length"):
            length = _positive_int(getattr(owner, attr_name, None))
            if length is not None:
                return len(weight[0]), length
    return None


def _node_conv1d_input_shape(value: Any) -> Optional[tuple[int, int]]:
    shape = _shape_from_value(value)
    if shape is None:
        return None
    if len(shape) < 2:
        return None
    channel_dim = shape[-2]
    sequence_dim = shape[-1]
    if channel_dim is None or sequence_dim is None:
        return None
    if channel_dim <= 0 or sequence_dim <= 0:
        return None
    return channel_dim, sequence_dim


def _node_last_feature_dim(value: Any) -> Optional[int]:
    shape = _shape_from_value(value)
    if shape is None:
        return None
    return shape[-1]


def _shape_from_value(value: Any) -> Optional[tuple[int | None, ...]]:
    direct_shape = _shape_tuple(getattr(value, "shape", None))
    if direct_shape is not None:
        return direct_shape
    meta = getattr(value, "meta", None)
    if not isinstance(meta, dict):
        return None
    for key in ("tensor_meta", "val"):
        candidate = meta.get(key)
        shape = _shape_tuple(getattr(candidate, "shape", None))
        if shape is not None:
            return shape
    return None


def _shape_tuple(shape: Any) -> Optional[tuple[int | None, ...]]:
    if shape is None:
        return None
    try:
        values = tuple(shape)
    except TypeError:
        return None
    converted: List[int | None] = []
    for value in values:
        if value is None:
            converted.append(None)
            continue
        try:
            converted.append(int(value))
        except (TypeError, ValueError):
            converted.append(None)
    if not converted or converted[-1] is None:
        return None
    return tuple(converted)


def _single_int(value: Any) -> Optional[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    return _positive_or_zero_int(value)


def _positive_int(value: Any) -> Optional[int]:
    integer = _positive_or_zero_int(value)
    if integer is None or integer <= 0:
        return None
    return integer


def _positive_or_zero_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer < 0:
        return None
    return integer


def _linear_weight(graph_module: Any, node: Any) -> Optional[List[List[float]]]:
    op = getattr(node, "op", None)
    if op == "call_module":
        module = _maybe_resolve_attr(graph_module, str(getattr(node, "target", "")))
        if module is None:
            return None
        return _to_matrix(getattr(module, "weight", None))

    args = list(getattr(node, "args", ()) or ())
    if len(args) < 2:
        return None
    return _node_value_as_matrix(graph_module, args[1])


def _linear_bias(graph_module: Any, node: Any, expected_length: int) -> Optional[List[float]]:
    op = getattr(node, "op", None)
    if op == "call_module":
        module = _maybe_resolve_attr(graph_module, str(getattr(node, "target", "")))
        if module is None:
            return None
        return _module_bias(module, expected_length)

    args = list(getattr(node, "args", ()) or ())
    kwargs = dict(getattr(node, "kwargs", {}) or {})
    bias = args[2] if len(args) >= 3 else kwargs.get("bias")
    return _bias_value(_node_value(graph_module, bias), expected_length)


def _is_embedding_node(graph_module: Any, node: Any) -> bool:
    op = getattr(node, "op", None)
    target_text = str(getattr(node, "target", ""))
    if op != "call_module":
        return False
    module = _maybe_resolve_attr(graph_module, target_text)
    if module is not None and type(module).__name__ == "Embedding":
        return True
    return "embedding" in target_text.lower()


def _embedding_weight(graph_module: Any, node: Any) -> Optional[List[List[float]]]:
    module = _maybe_resolve_attr(graph_module, str(getattr(node, "target", "")))
    if module is None:
        return None
    return _module_weight(module)


def _iter_named_modules(module: Any) -> Iterable[tuple[str, Any]]:
    named_modules = getattr(module, "named_modules", None)
    if named_modules is None:
        return ()
    return named_modules()


def _named_children(module: Any) -> Dict[str, Any]:
    named_children = getattr(module, "named_children", None)
    if named_children is None:
        return {}
    return dict(named_children())


def _is_linear_module(module: Any) -> bool:
    return module is not None and type(module).__name__ == "Linear"


def _module_weight(module: Any) -> Optional[List[List[float]]]:
    return _to_matrix(getattr(module, "weight", None))


def _module_bias(module: Any, expected_length: int) -> Optional[List[float]]:
    return _bias_value(getattr(module, "bias", None), expected_length)


def _bias_value(value: Any, expected_length: int) -> Optional[List[float]]:
    if value is None:
        return [0.0 for _ in range(expected_length)]
    bias = _to_vector(value)
    if bias is None or len(bias) != expected_length:
        return None
    return bias


def _compose_linear_bias(
    left: Sequence[Sequence[float]],
    inner_bias: Sequence[float],
    outer_bias: Sequence[float],
) -> List[float]:
    propagated = la.apply_weight(left, [inner_bias])[0]
    return [propagated[index] + float(outer_bias[index]) for index in range(len(propagated))]


def _has_merged_weight_hint(module: Any) -> bool:
    return any(hasattr(module, name) for name in ("merged_weight", "merged_linear", "base_layer", "base"))


def _is_runtime_activation_operand(value: Any, seen: Optional[set[int]] = None) -> bool:
    op = getattr(value, "op", None)
    if op == "placeholder":
        return True
    if op not in {"call_function", "call_method", "call_module"}:
        return False
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    args = list(getattr(value, "args", ()) or ())
    return any(_is_runtime_activation_operand(arg, seen) for arg in args)


def _matmul_shapes_compatible(activation: Any, weight: Sequence[Sequence[float]]) -> bool:
    feature_dim = _node_last_feature_dim(activation)
    if feature_dim is None:
        return True
    if not weight:
        return False
    return feature_dim == len(weight[0])


def _fixed_weight_from_transposed_operand(graph_module: Any, value: Any) -> Optional[List[List[float]]]:
    base = _transpose_base(value)
    if base is None:
        return None
    return _node_value_as_matrix(graph_module, base)


def _transpose_base(value: Any) -> Optional[Any]:
    op = getattr(value, "op", None)
    args = list(getattr(value, "args", ()) or ())
    target = getattr(value, "target", None)
    if op == "call_function":
        if _target_has_name(target, {"getattr"}) and len(args) >= 2 and args[1] == "T":
            return args[0]
        if _target_has_name(target, {"transpose"}) and _is_2d_transpose_args(args[1:]):
            return args[0]
    if op == "call_method":
        target_text = str(target)
        if target_text in {"t", "T"} and args:
            return args[0]
        if target_text == "transpose" and _is_2d_transpose_args(args[1:]):
            return args[0]
    return None


def _is_2d_transpose_args(args: Sequence[Any]) -> bool:
    if len(args) < 2:
        return False
    try:
        dims = (int(args[0]), int(args[1]))
    except (TypeError, ValueError):
        return False
    return dims in {(0, 1), (1, 0), (-1, -2), (-2, -1)}


def _is_default_scale(value: Any) -> bool:
    try:
        return abs(float(value) - 1.0) <= 1e-12
    except (TypeError, ValueError):
        return False


def _target_has_name(target: Any, names: set[str]) -> bool:
    direct_name = getattr(target, "__name__", None)
    if direct_name in names:
        return True
    text = str(target)
    return any(text == name or text.endswith(f".{name}") or f" {name}" in text for name in names)


def _source_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    for attr in ("target", "name"):
        source = getattr(value, attr, None)
        if source is not None:
            return str(source)
    return None


def _node_value(graph_module: Any, value: Any) -> Any:
    if getattr(value, "op", None) == "get_attr":
        return _resolve_attr(graph_module, str(getattr(value, "target", "")))
    return value


def _node_value_as_matrix(graph_module: Any, value: Any) -> Optional[List[List[float]]]:
    return _to_matrix(_node_value(graph_module, value))


def _maybe_resolve_attr(obj: Any, path: str) -> Any:
    try:
        return _resolve_attr(obj, path)
    except AttributeError:
        return None


def _resolve_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not part:
            continue
        current = getattr(current, part)
    return current


def _is_zero_value(value: Any, tolerance: float = 1e-12) -> bool:
    if value is None:
        return True
    value = _to_python_value(value)
    flattened = _flatten_numeric(value)
    return flattened is not None and all(abs(item) <= tolerance for item in flattened)


def _is_zero_vector(vector: Sequence[float], tolerance: float = 1e-12) -> bool:
    return all(abs(float(item)) <= tolerance for item in vector)


def _flatten_numeric(value: Any) -> Optional[List[float]]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        return None
    flattened: List[float] = []
    for item in value:
        item_values = _flatten_numeric(item)
        if item_values is None:
            return None
        flattened.extend(item_values)
    return flattened


def _to_python_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _to_vector(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    value = _to_python_value(value)
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(isinstance(item, (list, tuple)) for item in value):
        return None
    return [float(item) for item in value]


def _to_matrix(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    value = _to_python_value(value)
    if not isinstance(value, (list, tuple)) or not value:
        return None
    matrix: List[List[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            return None
        matrix.append([float(item) for item in row])
    return matrix
