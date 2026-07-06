"""Provenance-aware linear operator IR and executable operator classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from beyond_matmul import _linalg as la


@dataclass(frozen=True)
class Provenance:
    source: str
    framework: Optional[str] = None
    expression: Optional[str] = None
    inputs: Tuple[str, ...] = ()
    transform_history: Tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class ApproximationContract:
    mode: str = "exact"
    metric: str = "none"
    epsilon: float = 0.0
    observed_error: Optional[float] = None
    sample_count: int = 0

    @property
    def is_exact(self) -> bool:
        return self.mode == "exact"


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    bits: int
    codebook_size: Optional[int] = None
    scale: Optional[float] = None
    zero_point: Optional[float] = None
    per_axis: Optional[str] = None


@dataclass(frozen=True)
class ReuseBudget:
    fixed_weight: bool = True
    preprocessing_cost: float = 0.0
    amortize_over_calls: int = 1
    cache_bytes: int = 0


@dataclass(frozen=True)
class LayoutSpec:
    logical_layout: str = "out_in"
    physical_layout: str = "row_major"
    block_shape: Optional[Tuple[int, int]] = None
    alignment_bytes: Optional[int] = None


@dataclass(frozen=True)
class HardwareTarget:
    backend: str = "python"
    device: str = "cpu"
    dtype: str = "float32"
    supports: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorMetadata:
    kind: str
    shape: Tuple[int, int]
    provenance: Provenance = field(default_factory=lambda: Provenance(source="unknown"))
    structure: Mapping[str, Any] = field(default_factory=dict)
    contract: ApproximationContract = field(default_factory=ApproximationContract)
    quantization: Optional[QuantizationSpec] = None
    reuse: ReuseBudget = field(default_factory=ReuseBudget)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    hardware: HardwareTarget = field(default_factory=HardwareTarget)
    lowerings: Tuple[str, ...] = ("dense_gemm",)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _metadata(
    kind: str,
    shape: Tuple[int, int],
    metadata: Optional[OperatorMetadata],
    provenance: Optional[Provenance],
    structure: Mapping[str, Any],
    lowerings: Tuple[str, ...],
    contract: Optional[ApproximationContract] = None,
    quantization: Optional[QuantizationSpec] = None,
    reuse: Optional[ReuseBudget] = None,
    layout: Optional[LayoutSpec] = None,
    hardware: Optional[HardwareTarget] = None,
) -> OperatorMetadata:
    if metadata is not None:
        if metadata.shape != shape:
            raise ValueError("operator metadata shape does not match payload shape")
        return metadata
    return OperatorMetadata(
        kind=kind,
        shape=shape,
        provenance=provenance or Provenance(source="constructed"),
        structure=dict(structure),
        contract=contract or ApproximationContract(),
        quantization=quantization,
        reuse=reuse or ReuseBudget(),
        layout=layout or LayoutSpec(),
        hardware=hardware or HardwareTarget(),
        lowerings=lowerings,
    )


class LinearOperator:
    metadata: OperatorMetadata

    @property
    def shape(self) -> Tuple[int, int]:
        return self.metadata.shape

    @property
    def out_features(self) -> int:
        return self.shape[0]

    @property
    def in_features(self) -> int:
        return self.shape[1]

    def to_dense(self) -> la.Matrix:
        raise NotImplementedError

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        return la.apply_weight(self.to_dense(), inputs)

    def ir_dict(self) -> Dict[str, Any]:
        return self.metadata.to_dict()


@dataclass
class DenseOperator(LinearOperator):
    matrix: Sequence[Sequence[float]]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        checked = la.as_matrix(self.matrix)
        self.matrix = checked
        self.metadata = _metadata(
            kind="dense",
            shape=(len(checked), len(checked[0])),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"storage": "dense"},
            lowerings=("dense_gemm",),
        )

    def to_dense(self) -> la.Matrix:
        return [list(row) for row in self.matrix]

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        return la.apply_weight(self.matrix, inputs)


@dataclass
class DiagonalOperator(LinearOperator):
    diagonal: Sequence[float]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        checked = la.as_vector(self.diagonal)
        self.diagonal = checked
        size = len(checked)
        self.metadata = _metadata(
            kind="diagonal",
            shape=(size, size),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"diagonal_length": size},
            lowerings=("diagonal_kernel", "dense_gemm"),
        )

    def to_dense(self) -> la.Matrix:
        dense = la.zeros(len(self.diagonal), len(self.diagonal))
        for index, value in enumerate(self.diagonal):
            dense[index][index] = value
        return dense

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        batch = la.ensure_batch(inputs)
        if any(len(row) != len(self.diagonal) for row in batch):
            raise ValueError("input width does not match diagonal operator")
        return [[value * row[index] for index, value in enumerate(self.diagonal)] for row in batch]


@dataclass
class LowRankOperator(LinearOperator):
    left: Sequence[Sequence[float]]
    right: Sequence[Sequence[float]]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        left = la.as_matrix(self.left)
        right = la.as_matrix(self.right)
        if len(left[0]) != len(right):
            raise ValueError("low-rank factors must have shapes (out, rank) and (rank, in)")
        rank = len(right)
        self.left = left
        self.right = right
        self.metadata = _metadata(
            kind="low_rank",
            shape=(len(left), len(right[0])),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"rank": rank, "factor_shapes": [(len(left), rank), (rank, len(right[0]))]},
            lowerings=("low_rank_product", "dense_gemm"),
        )

    @property
    def rank(self) -> int:
        return len(self.right)

    def to_dense(self) -> la.Matrix:
        return la.matmul(self.left, self.right)

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        hidden = la.apply_weight(self.right, inputs)
        return la.apply_weight(self.left, hidden)


@dataclass
class AffineOperator(LinearOperator):
    linear: LinearOperator
    bias: Sequence[float]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        bias = la.as_vector(self.bias)
        if len(bias) != self.linear.out_features:
            raise ValueError("affine bias length must match operator output dimension")
        self.bias = bias
        primary_lowering = f"{self.linear.metadata.lowerings[0]}_bias"
        self.metadata = _metadata(
            kind="affine",
            shape=self.linear.shape,
            metadata=self.metadata,
            provenance=self.provenance or self.linear.metadata.provenance,
            structure={
                "linear_kind": self.linear.metadata.kind,
                "linear_structure": dict(self.linear.metadata.structure),
                "bias_length": len(bias),
            },
            reuse=self.linear.metadata.reuse,
            layout=self.linear.metadata.layout,
            hardware=self.linear.metadata.hardware,
            lowerings=(primary_lowering, "dense_gemm_bias"),
        )

    def to_dense(self) -> la.Matrix:
        """Return the linear weight component; bias is represented separately."""

        return self.linear.to_dense()

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        outputs = self.linear.apply(inputs)
        return [[value + self.bias[index] for index, value in enumerate(row)] for row in outputs]


@dataclass
class SparseCOOOperator(LinearOperator):
    rows: Sequence[int]
    cols: Sequence[int]
    values: Sequence[float]
    operator_shape: Tuple[int, int]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        if not (len(self.rows) == len(self.cols) == len(self.values)):
            raise ValueError("sparse COO rows, cols, and values must have the same length")
        out_features, in_features = self.operator_shape
        if out_features <= 0 or in_features <= 0:
            raise ValueError("sparse operator shape must be positive")
        checked_rows: List[int] = []
        checked_cols: List[int] = []
        checked_values: List[float] = []
        for row, col, value in zip(self.rows, self.cols, self.values):
            if not (0 <= row < out_features and 0 <= col < in_features):
                raise ValueError("sparse COO index out of bounds")
            checked_rows.append(int(row))
            checked_cols.append(int(col))
            checked_values.append(float(value))
        self.rows = checked_rows
        self.cols = checked_cols
        self.values = checked_values
        self.metadata = _metadata(
            kind="sparse_coo",
            shape=self.operator_shape,
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"nnz": len(checked_values), "format": "coo"},
            lowerings=("sparse_kernel", "dense_gemm"),
        )

    @property
    def nnz(self) -> int:
        return len(self.values)

    def to_dense(self) -> la.Matrix:
        dense = la.zeros(self.out_features, self.in_features)
        for row, col, value in zip(self.rows, self.cols, self.values):
            dense[row][col] += value
        return dense

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        batch = la.ensure_batch(inputs)
        if any(len(row) != self.in_features for row in batch):
            raise ValueError("input width does not match sparse operator")
        outputs = la.zeros(len(batch), self.out_features)
        for row_index, input_row in enumerate(batch):
            for out_index, in_index, value in zip(self.rows, self.cols, self.values):
                outputs[row_index][out_index] += value * input_row[in_index]
        return outputs


@dataclass
class CodebookOperator(LinearOperator):
    codes: Sequence[Sequence[int]]
    codebook: Sequence[float]
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None
    contract: Optional[ApproximationContract] = None

    def __post_init__(self) -> None:
        codebook = [float(value) for value in self.codebook]
        if not codebook:
            raise ValueError("codebook must not be empty")
        codes = [[int(code) for code in row] for row in self.codes]
        if not codes or not codes[0]:
            raise ValueError("codes matrix must not be empty")
        width = len(codes[0])
        for row in codes:
            if len(row) != width:
                raise ValueError("code rows must all have the same length")
            for code in row:
                if code < 0 or code >= len(codebook):
                    raise ValueError("code index out of codebook range")
        self.codes = codes
        self.codebook = codebook
        self.metadata = _metadata(
            kind="codebook",
            shape=(len(codes), width),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"codebook_size": len(codebook), "codes_shape": (len(codes), width)},
            contract=self.contract or ApproximationContract(mode="approximate", metric="matrix_relative_frobenius"),
            quantization=QuantizationSpec(scheme="codebook", bits=max(1, (len(codebook) - 1).bit_length()), codebook_size=len(codebook)),
            reuse=ReuseBudget(preprocessing_cost=len(codes) * width * max(1, len(codebook)), amortize_over_calls=4),
            lowerings=("codebook_kernel", "dense_gemm"),
        )

    def to_dense(self) -> la.Matrix:
        return [[self.codebook[code] for code in row] for row in self.codes]


@dataclass
class BitpackedBinaryOperator(LinearOperator):
    signs: Sequence[Sequence[int]]
    scale: float
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None
    contract: Optional[ApproximationContract] = None

    def __post_init__(self) -> None:
        signs = [[1 if int(value) >= 0 else -1 for value in row] for row in self.signs]
        if not signs or not signs[0]:
            raise ValueError("binary sign matrix must not be empty")
        width = len(signs[0])
        for row in signs:
            if len(row) != width:
                raise ValueError("binary sign rows must all have the same length")
        self.signs = signs
        self.scale = float(self.scale)
        self.metadata = _metadata(
            kind="bitpacked_binary",
            shape=(len(signs), width),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"values": [-1, 1], "scale": self.scale},
            contract=self.contract or ApproximationContract(mode="approximate", metric="matrix_relative_frobenius"),
            quantization=QuantizationSpec(scheme="symmetric_binary", bits=1, scale=self.scale),
            reuse=ReuseBudget(preprocessing_cost=len(signs) * width, amortize_over_calls=4),
            lowerings=("bitpacked_kernel", "dense_gemm"),
        )

    def to_dense(self) -> la.Matrix:
        return [[self.scale * sign for sign in row] for row in self.signs]

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        batch = la.ensure_batch(inputs)
        if any(len(row) != self.in_features for row in batch):
            raise ValueError("input width does not match bitpacked operator")
        outputs = la.zeros(len(batch), self.out_features)
        for batch_index, input_row in enumerate(batch):
            for out_index, sign_row in enumerate(self.signs):
                outputs[batch_index][out_index] = self.scale * sum(sign * value for sign, value in zip(sign_row, input_row))
        return outputs


@dataclass
class Convolution1DOperator(LinearOperator):
    kernel: Sequence[float]
    input_length: int
    mode: str = "valid"
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        kernel = la.as_vector(self.kernel)
        if self.mode != "valid":
            raise ValueError("only valid 1D convolution is implemented in the prototype")
        if self.input_length < len(kernel):
            raise ValueError("input length must be at least kernel length")
        self.kernel = kernel
        output_length = self.input_length - len(kernel) + 1
        self.metadata = _metadata(
            kind="conv1d",
            shape=(output_length, self.input_length),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={"kernel_size": len(kernel), "mode": self.mode, "lowering": "toeplitz"},
            lowerings=("conv1d_direct", "dense_gemm"),
        )

    def to_dense(self) -> la.Matrix:
        dense = la.zeros(self.out_features, self.in_features)
        for out_index in range(self.out_features):
            for kernel_index, value in enumerate(self.kernel):
                dense[out_index][out_index + kernel_index] = value
        return dense

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        batch = la.ensure_batch(inputs)
        if any(len(row) != self.input_length for row in batch):
            raise ValueError("input width does not match convolution operator")
        outputs = la.zeros(len(batch), self.out_features)
        for batch_index, input_row in enumerate(batch):
            for out_index in range(self.out_features):
                total = 0.0
                for kernel_index, value in enumerate(self.kernel):
                    total += value * input_row[out_index + kernel_index]
                outputs[batch_index][out_index] = total
        return outputs


@dataclass
class MultiChannelConvolution1DOperator(LinearOperator):
    weight: Sequence[Sequence[Sequence[float]]]
    input_length: int
    mode: str = "valid"
    groups: int = 1
    provenance: Optional[Provenance] = None
    metadata: Optional[OperatorMetadata] = None

    def __post_init__(self) -> None:
        weight = self._checked_weight(self.weight)
        if not isinstance(self.groups, int) or self.groups < 1:
            raise ValueError("conv1d groups must be a positive integer")
        if self.mode != "valid":
            raise ValueError("only valid multi-channel 1D convolution is implemented in the prototype")
        kernel_size = len(weight[0][0])
        if self.input_length < kernel_size:
            raise ValueError("input length must be at least kernel length")
        self.weight = weight
        groups = self.groups
        out_channels = len(weight)
        if out_channels % groups != 0:
            raise ValueError("conv1d output channels must be divisible by groups")
        input_channels_per_group = len(weight[0])
        output_channels_per_group = out_channels // groups
        in_channels = input_channels_per_group * groups
        output_length = self.input_length - kernel_size + 1
        if groups == 1:
            group_type = "standard"
            primary_lowering = "conv1d_channel_direct"
        elif input_channels_per_group == 1 and groups == in_channels:
            group_type = "depthwise"
            primary_lowering = "conv1d_depthwise_direct"
        else:
            group_type = "grouped"
            primary_lowering = "conv1d_grouped_direct"
        self.metadata = _metadata(
            kind="conv1d_channel",
            shape=(out_channels * output_length, in_channels * self.input_length),
            metadata=self.metadata,
            provenance=self.provenance,
            structure={
                "out_channels": out_channels,
                "in_channels": in_channels,
                "input_channels_per_group": input_channels_per_group,
                "output_channels_per_group": output_channels_per_group,
                "groups": groups,
                "group_type": group_type,
                "kernel_size": kernel_size,
                "input_length": self.input_length,
                "output_length": output_length,
                "mode": self.mode,
                "lowering": "block_toeplitz",
            },
            lowerings=(primary_lowering, "dense_gemm"),
        )

    @staticmethod
    def _checked_weight(values: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
        weight: List[List[List[float]]] = []
        if not values:
            raise ValueError("conv1d weight must have at least one output channel")
        expected_in_channels: Optional[int] = None
        expected_kernel_size: Optional[int] = None
        for out_channel in values:
            if not out_channel:
                raise ValueError("conv1d weight must have at least one input channel")
            checked_out: List[List[float]] = []
            if expected_in_channels is None:
                expected_in_channels = len(out_channel)
            elif len(out_channel) != expected_in_channels:
                raise ValueError("conv1d output channels must share the same input-channel count")
            for kernel in out_channel:
                checked_kernel = la.as_vector(kernel)
                if expected_kernel_size is None:
                    expected_kernel_size = len(checked_kernel)
                elif len(checked_kernel) != expected_kernel_size:
                    raise ValueError("conv1d kernels must all have the same length")
                checked_out.append(checked_kernel)
            weight.append(checked_out)
        return weight

    @property
    def out_channels(self) -> int:
        return int(self.metadata.structure["out_channels"])

    @property
    def in_channels(self) -> int:
        return int(self.metadata.structure["in_channels"])

    @property
    def input_channels_per_group(self) -> int:
        return int(self.metadata.structure["input_channels_per_group"])

    @property
    def output_channels_per_group(self) -> int:
        return int(self.metadata.structure["output_channels_per_group"])

    @property
    def group_type(self) -> str:
        return str(self.metadata.structure["group_type"])

    @property
    def kernel_size(self) -> int:
        return int(self.metadata.structure["kernel_size"])

    @property
    def output_length(self) -> int:
        return int(self.metadata.structure["output_length"])

    def to_dense(self) -> la.Matrix:
        dense = la.zeros(self.out_features, self.in_features)
        for out_channel, channel_weight in enumerate(self.weight):
            group_index = out_channel // self.output_channels_per_group
            input_group_offset = group_index * self.input_channels_per_group
            for out_position in range(self.output_length):
                out_index = out_channel * self.output_length + out_position
                for in_channel, kernel in enumerate(channel_weight):
                    input_offset = (input_group_offset + in_channel) * self.input_length
                    for kernel_index, value in enumerate(kernel):
                        dense[out_index][input_offset + out_position + kernel_index] = value
        return dense

    def apply(self, inputs: Sequence[Sequence[float]] | Sequence[float]) -> la.Matrix:
        batch = la.ensure_batch(inputs)
        if any(len(row) != self.in_features for row in batch):
            raise ValueError("input width does not match multi-channel convolution operator")
        outputs = la.zeros(len(batch), self.out_features)
        for batch_index, input_row in enumerate(batch):
            for out_channel, channel_weight in enumerate(self.weight):
                group_index = out_channel // self.output_channels_per_group
                input_group_offset = group_index * self.input_channels_per_group
                for out_position in range(self.output_length):
                    total = 0.0
                    for in_channel, kernel in enumerate(channel_weight):
                        input_offset = (input_group_offset + in_channel) * self.input_length
                        for kernel_index, value in enumerate(kernel):
                            total += value * input_row[input_offset + out_position + kernel_index]
                    outputs[batch_index][out_channel * self.output_length + out_position] = total
        return outputs
