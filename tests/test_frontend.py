import operator as py_operator
import unittest

from beyond_matmul.frontend import (
    capture_torch_fx_operators,
    capture_torch_fx_linear_operators,
    extract_torch_fx_operators,
    capture_torch_named_adapter_operators,
    extract_torch_fx_low_rank_operators,
)
from beyond_matmul.ir import AffineOperator, Convolution1DOperator, DenseOperator, MultiChannelConvolution1DOperator
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - depends on optional torch install
    torch = None
    nn = None
    F = None


class FakeNode:
    def __init__(self, name, op, target, args=(), kwargs=None):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.meta = {}


class FakeTensorMeta:
    def __init__(self, shape):
        self.shape = shape


class FakeTensorArgument:
    def __init__(self, name):
        self.name = name


class FakeInputSpec:
    def __init__(self, placeholder_name, target):
        self.arg = FakeTensorArgument(placeholder_name)
        self.target = target
        self.kind = "PARAMETER"


class FakeGraphSignature:
    def __init__(self, input_specs):
        self.input_specs = input_specs


class FakeExportedProgram:
    def __init__(self, graph_module, state_dict, input_specs):
        self.graph_module = graph_module
        self.state_dict = state_dict
        self.graph_signature = FakeGraphSignature(input_specs)


class Conv1d:
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, in_channels=1, out_channels=1):
        self.weight = weight
        self.bias = bias
        self.stride = (stride,)
        self.padding = (padding,)
        self.dilation = (dilation,)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class FakeGraphModule:
    def __init__(self, nodes):
        self.graph = FakeGraph(nodes)


class FrontendTests(unittest.TestCase):
    def assertMatrixAlmostEqual(self, left, right, places=6):
        self.assertEqual(len(left), len(right))
        for left_row, right_row in zip(left, right):
            self.assertEqual(len(left_row), len(right_row))
            for left_value, right_value in zip(left_row, right_row):
                self.assertAlmostEqual(left_value, right_value, places=places)

    def test_extracts_low_rank_functional_linear_pattern(self):
        graph_module = FakeGraphModule([])
        graph_module.right = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        graph_module.left = [[0.5, 1.0], [-1.0, 2.0], [3.0, -0.5], [0.25, 0.75]]

        x = FakeNode("x", "placeholder", "x")
        right = FakeNode("right", "get_attr", "right")
        left = FakeNode("left", "get_attr", "left")
        inner = FakeNode("linear", "call_function", "torch.nn.functional.linear", args=(x, right))
        outer = FakeNode("linear_1", "call_function", "torch.nn.functional.linear", args=(inner, left))
        graph_module.graph.nodes = [x, right, inner, left, outer]

        captured = extract_torch_fx_low_rank_operators(graph_module)

        self.assertIn("linear_1", captured)
        operator = captured["linear_1"].operator
        self.assertEqual(operator.shape, (4, 3))
        self.assertEqual(operator.rank, 2)
        self.assertEqual(operator.metadata.provenance.framework, "torch.fx")
        self.assertEqual(operator.apply([[1.0, 0.0, -1.0]]), [[-3.0, -2.0, -5.0, -2.0]])

    def test_ignores_incompatible_linear_factors(self):
        graph_module = FakeGraphModule([])
        graph_module.right = [[1.0, 2.0, 3.0]]
        graph_module.left = [[0.5, 1.0], [-1.0, 2.0]]

        x = FakeNode("x", "placeholder", "x")
        right = FakeNode("right", "get_attr", "right")
        left = FakeNode("left", "get_attr", "left")
        inner = FakeNode("linear", "call_function", "torch.nn.functional.linear", args=(x, right))
        outer = FakeNode("linear_1", "call_function", "torch.nn.functional.linear", args=(inner, left))
        graph_module.graph.nodes = [x, right, inner, left, outer]

        self.assertEqual(extract_torch_fx_low_rank_operators(graph_module), {})

    def test_ignores_unsupported_fake_conv1d_variants(self):
        unsupported_modules = [
            Conv1d([[[1.0, 2.0, 3.0]]], stride=2),
            Conv1d([[[1.0, 2.0, 3.0]]], padding=1),
            Conv1d([[[1.0, 2.0, 3.0]]], dilation=2),
            Conv1d([[[1.0, 2.0, 3.0]]], groups=2, in_channels=1, out_channels=1),
        ]
        for module in unsupported_modules:
            with self.subTest(module=module):
                graph_module = FakeGraphModule([])
                graph_module.conv = module
                x = FakeNode("x", "placeholder", "x")
                x.meta["tensor_meta"] = FakeTensorMeta((1, 1, 5))
                conv = FakeNode("conv", "call_module", "conv", args=(x,))
                graph_module.graph.nodes = [x, conv]

                self.assertEqual(extract_torch_fx_operators(graph_module), {})

    def test_extracts_fake_multi_channel_conv1d_module(self):
        graph_module = FakeGraphModule([])
        graph_module.conv = Conv1d(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
            ],
            bias=[0.1, -0.2],
            in_channels=2,
            out_channels=2,
        )
        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 2, 5))
        conv = FakeNode("conv", "call_module", "conv", args=(x,))
        graph_module.graph.nodes = [x, conv]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("conv", captured)
        operator = captured["conv"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (6, 10))
        self.assertEqual(operator.bias, [0.1, 0.1, 0.1, -0.2, -0.2, -0.2])
        self.assertEqual(captured["conv"].event.notes["capture"], "conv1d_module")

    def test_extracts_fake_grouped_conv1d_module(self):
        graph_module = FakeGraphModule([])
        graph_module.conv = Conv1d(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
                [[0.5, -0.5, 1.5], [1.0, 0.0, -1.0]],
                [[-0.25, 0.5, 0.75], [2.0, -1.0, 0.25]],
            ],
            in_channels=4,
            out_channels=4,
            groups=2,
        )
        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 4, 5))
        conv = FakeNode("conv", "call_module", "conv", args=(x,))
        graph_module.graph.nodes = [x, conv]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("conv", captured)
        operator = captured["conv"].operator
        self.assertIsInstance(operator, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (12, 20))
        self.assertEqual(operator.groups, 2)
        self.assertEqual(operator.metadata.lowerings[0], "conv1d_grouped_direct")
        self.assertEqual(captured["conv"].event.notes["groups"], "2")

    def test_extracts_fake_functional_conv1d(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [
            [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
            [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
        ]
        graph_module.bias = [0.1, -0.2]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 2, 5))
        weight = FakeNode("weight", "get_attr", "weight")
        bias = FakeNode("bias", "get_attr", "bias")
        conv = FakeNode("conv1d", "call_function", "torch.nn.functional.conv1d", args=(x, weight, bias))
        graph_module.graph.nodes = [x, weight, bias, conv]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("conv1d", captured)
        operator = captured["conv1d"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (6, 10))
        self.assertEqual(operator.bias, [0.1, 0.1, 0.1, -0.2, -0.2, -0.2])
        self.assertEqual(captured["conv1d"].event.notes["capture"], "conv1d_function")

    def test_extracts_fake_functional_depthwise_conv1d(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [
            [[1.0, 0.0, -1.0]],
            [[0.5, -0.5, 1.5]],
            [[2.0, 1.0, 0.25]],
        ]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 3, 5))
        weight = FakeNode("weight", "get_attr", "weight")
        conv = FakeNode("conv1d", "call_function", "torch.nn.functional.conv1d", args=(x, weight), kwargs={"groups": 3})
        graph_module.graph.nodes = [x, weight, conv]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("conv1d", captured)
        operator = captured["conv1d"].operator
        self.assertIsInstance(operator, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (9, 15))
        self.assertEqual(operator.groups, 3)
        self.assertEqual(operator.metadata.lowerings[0], "conv1d_depthwise_direct")
        self.assertEqual(captured["conv1d"].event.notes["group_type"], "depthwise")

    def test_ignores_unsupported_fake_functional_conv1d_variants(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [[[1.0, 2.0, 3.0]]]
        graph_module.bias = [0.1]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 1, 5))
        dynamic_weight = FakeNode("dynamic_weight", "placeholder", "dynamic_weight")
        dynamic_bias = FakeNode("dynamic_bias", "placeholder", "dynamic_bias")
        weight = FakeNode("weight", "get_attr", "weight")
        bias = FakeNode("bias", "get_attr", "bias")
        dynamic_weight_conv = FakeNode("conv1d", "call_function", "torch.nn.functional.conv1d", args=(x, dynamic_weight, bias))
        dynamic_bias_conv = FakeNode("conv1d_1", "call_function", "torch.nn.functional.conv1d", args=(x, weight, dynamic_bias))
        strided_conv = FakeNode("conv1d_2", "call_function", "torch.nn.functional.conv1d", args=(x, weight, bias), kwargs={"stride": 2})
        invalid_grouped_conv = FakeNode(
            "conv1d_3",
            "call_function",
            "torch.nn.functional.conv1d",
            args=(x, weight, bias),
            kwargs={"groups": 2},
        )
        graph_module.graph.nodes = [
            x,
            dynamic_weight,
            dynamic_bias,
            weight,
            bias,
            dynamic_weight_conv,
            dynamic_bias_conv,
            strided_conv,
            invalid_grouped_conv,
        ]

        self.assertEqual(extract_torch_fx_operators(graph_module), {})

    def test_extracts_fixed_weight_matmul_pattern_as_dense(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((2, 3))
        weight = FakeNode("weight", "get_attr", "weight")
        weight_t = FakeNode("getattr_1", "call_function", getattr, args=(weight, "T"))
        matmul = FakeNode("matmul", "call_function", py_operator.matmul, args=(x, weight_t))
        graph_module.graph.nodes = [x, weight, weight_t, matmul]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("matmul", captured)
        operator = captured["matmul"].operator
        self.assertIsInstance(operator, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(operator.metadata.provenance.framework, "torch.fx")
        self.assertEqual(captured["matmul"].event.notes["capture"], "dense_matmul")
        self.assertMatrixAlmostEqual(operator.apply([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]), [[-2.0, -4.5], [7.5, 4.25]])

    def test_extracts_fixed_weight_addmm_pattern_as_affine_dense(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]
        graph_module.bias = [0.25, -0.75]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 3))
        bias = FakeNode("bias", "get_attr", "bias")
        weight = FakeNode("weight", "get_attr", "weight")
        weight_t = FakeNode("getattr_1", "call_function", getattr, args=(weight, "T"))
        addmm = FakeNode("addmm", "call_function", "torch.addmm", args=(bias, x, weight_t))
        graph_module.graph.nodes = [x, bias, weight, weight_t, addmm]

        captured = extract_torch_fx_operators(graph_module)

        self.assertIn("addmm", captured)
        operator = captured["addmm"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(operator.bias, [0.25, -0.75])
        self.assertMatrixAlmostEqual(operator.apply([[1.0, 0.0, -1.0]]), [[-1.75, -5.25]])

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=1, calls=32, sample_inputs=[[1.0, 0.0, -1.0]], codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "dense_gemm_bias")
        self.assertTrue(plan.selected.exact)

    def test_extracts_exported_addmm_from_signature_state_dict(self):
        graph_module = FakeGraphModule([])
        state_dict = {
            "weight": [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]],
            "bias": [0.25, -0.75],
        }
        exported = FakeExportedProgram(
            graph_module,
            state_dict,
            [
                FakeInputSpec("p_weight", "weight"),
                FakeInputSpec("p_bias", "bias"),
            ],
        )

        p_weight = FakeNode("p_weight", "placeholder", "p_weight")
        p_bias = FakeNode("p_bias", "placeholder", "p_bias")
        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((2, 3))
        weight_t = FakeNode("numpy_t", "call_function", "aten.numpy_T.default", args=(p_weight,))
        addmm = FakeNode("addmm", "call_function", "aten.addmm.default", args=(p_bias, x, weight_t))
        graph_module.graph.nodes = [p_weight, p_bias, x, weight_t, addmm]

        captured = extract_torch_fx_operators(exported)

        self.assertIn("addmm", captured)
        operator = captured["addmm"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(operator.bias, [0.25, -0.75])
        self.assertEqual(captured["addmm"].event.notes["capture"], "dense_addmm")
        self.assertEqual(captured["addmm"].event.notes["rhs_recovery"], "exported_graph_state")
        self.assertEqual(captured["addmm"].event.notes["bias_recovery"], "exported_graph_state")
        self.assertIn("exported_graph_constant_recovery", operator.metadata.provenance.transform_history)
        self.assertMatrixAlmostEqual(operator.apply([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]), [[-1.75, -5.25], [7.75, 3.5]])

    def test_extracts_exported_matmul_from_signature_state_dict(self):
        graph_module = FakeGraphModule([])
        state_dict = {
            "weight": [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]],
        }
        exported = FakeExportedProgram(
            graph_module,
            state_dict,
            [
                FakeInputSpec("p_weight", "weight"),
            ],
        )

        p_weight = FakeNode("p_weight", "placeholder", "p_weight")
        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((2, 3))
        weight_t = FakeNode("numpy_t", "call_function", "aten.numpy_T.default", args=(p_weight,))
        matmul = FakeNode("matmul", "call_function", "aten.matmul.default", args=(x, weight_t))
        graph_module.graph.nodes = [p_weight, x, weight_t, matmul]

        captured = extract_torch_fx_operators(exported)

        self.assertIn("matmul", captured)
        operator = captured["matmul"].operator
        self.assertIsInstance(operator, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(captured["matmul"].event.notes["capture"], "dense_matmul")
        self.assertEqual(captured["matmul"].event.notes["rhs_recovery"], "exported_graph_state")
        self.assertIn("exported_graph_constant_recovery", operator.metadata.provenance.transform_history)
        self.assertMatrixAlmostEqual(operator.apply([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]]), [[-2.0, -4.5], [7.5, 4.25]])

    def test_extracts_exported_nested_linear_from_signature_state_dict(self):
        graph_module = FakeGraphModule([])
        state_dict = {
            "right": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "left": [[0.5, 1.0], [-1.0, 2.0]],
            "bias": [0.0, 1.0],
        }
        exported = FakeExportedProgram(
            graph_module,
            state_dict,
            [
                FakeInputSpec("p_right", "right"),
                FakeInputSpec("p_left", "left"),
                FakeInputSpec("p_bias", "bias"),
            ],
        )

        x = FakeNode("x", "placeholder", "x")
        p_right = FakeNode("p_right", "placeholder", "p_right")
        p_left = FakeNode("p_left", "placeholder", "p_left")
        p_bias = FakeNode("p_bias", "placeholder", "p_bias")
        inner = FakeNode("linear", "call_function", "aten.linear.default", args=(x, p_right))
        outer = FakeNode("linear_1", "call_function", "aten.linear.default", args=(inner, p_left, p_bias))
        graph_module.graph.nodes = [p_right, p_left, p_bias, x, inner, outer]

        captured = extract_torch_fx_low_rank_operators(exported)

        self.assertIn("linear_1", captured)
        operator = captured["linear_1"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(operator.bias, [0.0, 1.0])
        self.assertEqual(captured["linear_1"].event.notes["weight_recovery"], "exported_graph_state")
        self.assertIn("exported_graph_constant_recovery", operator.metadata.provenance.transform_history)
        self.assertEqual(operator.apply([[1.0, 0.0, -1.0]]), [[-3.0, -1.0]])

    def test_ignores_unsupported_matmul_and_addmm_variants(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]
        graph_module.bias = [0.25, -0.75]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 3))
        bad_x = FakeNode("bad_x", "placeholder", "bad_x")
        bad_x.meta["tensor_meta"] = FakeTensorMeta((1, 4))
        y = FakeNode("y", "placeholder", "y")
        bias = FakeNode("bias", "get_attr", "bias")
        dynamic_bias = FakeNode("dynamic_bias", "placeholder", "dynamic_bias")
        weight = FakeNode("weight", "get_attr", "weight")
        weight_t = FakeNode("getattr_1", "call_function", getattr, args=(weight, "T"))
        dynamic_matmul = FakeNode("matmul", "call_function", py_operator.matmul, args=(x, y))
        incompatible_matmul = FakeNode("matmul_1", "call_function", py_operator.matmul, args=(bad_x, weight_t))
        scaled_addmm = FakeNode("addmm", "call_function", "torch.addmm", args=(bias, x, weight_t), kwargs={"alpha": 0.5})
        dynamic_bias_addmm = FakeNode("addmm_1", "call_function", "torch.addmm", args=(dynamic_bias, x, weight_t))

        graph_module.graph.nodes = [
            x,
            bad_x,
            y,
            dynamic_bias,
            bias,
            weight,
            weight_t,
            dynamic_matmul,
            incompatible_matmul,
            scaled_addmm,
            dynamic_bias_addmm,
        ]

        self.assertEqual(extract_torch_fx_operators(graph_module), {})

    def test_ignores_export_like_dynamic_or_ambiguous_dense_operands(self):
        graph_module = FakeGraphModule([])
        graph_module.weight = [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]

        x = FakeNode("x", "placeholder", "x")
        x.meta["tensor_meta"] = FakeTensorMeta((1, 3))
        dynamic_weight = FakeNode("dynamic_weight", "placeholder", "dynamic_weight")
        fixed_weight = FakeNode("fixed_weight", "get_attr", "weight")
        dynamic_weight_t = FakeNode("numpy_t", "call_function", "aten.numpy_T.default", args=(dynamic_weight,))
        dynamic_matmul = FakeNode("matmul", "call_function", "aten.matmul.default", args=(x, dynamic_weight_t))
        ambiguous_matmul = FakeNode("matmul_1", "call_function", "aten.matmul.default", args=(x, fixed_weight))
        graph_module.graph.nodes = [x, dynamic_weight, fixed_weight, dynamic_weight_t, dynamic_matmul, ambiguous_matmul]

        self.assertEqual(extract_torch_fx_operators(graph_module), {})

    def test_extracts_biased_functional_linear_pattern_as_affine(self):
        graph_module = FakeGraphModule([])
        graph_module.right = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        graph_module.left = [[0.5, 1.0], [-1.0, 2.0]]
        graph_module.bias = [0.0, 1.0]

        x = FakeNode("x", "placeholder", "x")
        right = FakeNode("right", "get_attr", "right")
        left = FakeNode("left", "get_attr", "left")
        bias = FakeNode("bias", "get_attr", "bias")
        inner = FakeNode("linear", "call_function", "torch.nn.functional.linear", args=(x, right))
        outer = FakeNode("linear_1", "call_function", "torch.nn.functional.linear", args=(inner, left, bias))
        graph_module.graph.nodes = [x, right, inner, left, bias, outer]

        captured = extract_torch_fx_low_rank_operators(graph_module)

        self.assertIn("linear_1", captured)
        operator = captured["linear_1"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertEqual(operator.bias, [0.0, 1.0])
        self.assertEqual(operator.apply([[1.0, 0.0, -1.0]]), [[-3.0, -1.0]])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_linear_modules(self):
        class LowRankModules(nn.Module):
            def __init__(self):
                super().__init__()
                self.down = nn.Linear(3, 2, bias=False)
                self.up = nn.Linear(2, 4, bias=False)
                with torch.no_grad():
                    self.down.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
                    self.up.weight.copy_(torch.tensor([[0.5, 1.0], [-1.0, 2.0], [3.0, -0.5], [0.25, 0.75]]))

            def forward(self, x):
                return self.up(self.down(x))

        module = LowRankModules().eval()

        captured = capture_torch_fx_linear_operators(module)

        self.assertIn("up", captured)
        operator = captured["up"].operator
        inputs = [[1.0, 0.0, -1.0]]
        self.assertEqual(operator.shape, (4, 3))
        self.assertEqual(operator.rank, 2)
        self.assertEqual(operator.apply(inputs), [[-3.0, -2.0, -5.0, -2.0]])
        self.assertMatrixAlmostEqual(module(torch.tensor(inputs)).detach().tolist(), operator.apply(inputs))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_biased_torch_linear_modules_as_affine(self):
        class BiasedLowRankModules(nn.Module):
            def __init__(self):
                super().__init__()
                self.down = nn.Linear(3, 2, bias=True)
                self.up = nn.Linear(2, 2, bias=True)
                with torch.no_grad():
                    self.down.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
                    self.down.bias.copy_(torch.tensor([0.25, -0.5]))
                    self.up.weight.copy_(torch.tensor([[0.5, 1.0], [-1.0, 2.0]]))
                    self.up.bias.copy_(torch.tensor([0.0, 1.0]))

            def forward(self, x):
                return self.up(self.down(x))

        module = BiasedLowRankModules().eval()
        inputs = [[1.0, 0.0, -1.0]]

        captured = capture_torch_fx_linear_operators(module)

        self.assertIn("up", captured)
        operator = captured["up"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertMatrixAlmostEqual(operator.apply(inputs), module(torch.tensor(inputs)).detach().tolist())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_matmul_operator_pattern(self):
        class MatmulProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]]))

            def forward(self, x):
                return x @ self.weight.T

        module = MatmulProjection().eval()
        torch_input = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
        input_rows = torch_input.tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("matmul", captured)
        operator = captured["matmul"].operator
        self.assertIsInstance(operator, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "dense_gemm")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_matmul_function_pattern(self):
        class MatmulProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]]))

            def forward(self, x):
                return torch.matmul(x, self.weight.T)

        module = MatmulProjection().eval()
        torch_input = torch.tensor([[2.0, -1.0, 0.5], [-0.25, 1.5, 2.0]])
        input_rows = torch_input.tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("matmul", captured)
        operator = captured["matmul"].operator
        self.assertIsInstance(operator, DenseOperator)
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().tolist())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_mm_function_pattern(self):
        class MmProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[1.25, -0.5, 0.75], [-1.0, 2.0, 0.5]]))

            def forward(self, x):
                return torch.mm(x, self.weight.T)

        module = MmProjection().eval()
        torch_input = torch.tensor([[0.25, 1.0, -2.0], [1.5, -0.5, 0.75]])
        input_rows = torch_input.tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("mm", captured)
        operator = captured["mm"].operator
        self.assertIsInstance(operator, DenseOperator)
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().tolist())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_addmm_pattern_as_affine(self):
        class AddmmProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [0.75, 1.25, -1.5]]))
                self.bias = nn.Parameter(torch.tensor([0.25, -0.5]))

            def forward(self, x):
                return torch.addmm(self.bias, x, self.weight.T)

        module = AddmmProjection().eval()
        torch_input = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
        input_rows = torch_input.tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("addmm", captured)
        operator = captured["addmm"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "dense_gemm_bias")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_torch_exported_addmm_pattern_as_affine(self):
        class AddmmProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[1.0, -2.0, 0.5], [0.75, 1.25, -1.5]]))
                self.bias = nn.Parameter(torch.tensor([0.25, -0.5]))

            def forward(self, x):
                return torch.addmm(self.bias, x, self.weight.T)

        module = AddmmProjection().eval()
        torch_input = torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
        exported = torch.export.export(module, (torch_input,))

        captured = capture_torch_fx_operators(exported)

        self.assertIn("addmm", captured)
        operator = captured["addmm"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, DenseOperator)
        self.assertEqual(operator.shape, (2, 3))
        self.assertEqual(captured["addmm"].event.notes["rhs_recovery"], "exported_graph_state")
        self.assertEqual(captured["addmm"].event.notes["bias_recovery"], "exported_graph_state")
        self.assertMatrixAlmostEqual(operator.apply(torch_input.tolist()), module(torch_input).detach().tolist())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_biasless_torch_conv1d_module(self):
        class TinyConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, kernel_size=3, bias=False)
                with torch.no_grad():
                    self.conv.weight.copy_(torch.tensor([[[1.0, -2.0, 0.5]]]))

            def forward(self, x):
                return self.conv(x)

        module = TinyConv().eval()
        torch_input = torch.tensor([
            [[1.0, 0.0, -1.0, 2.0, 0.5, -0.5]],
            [[0.5, -1.5, 1.0, 0.0, 2.0, -2.0]],
        ])
        input_rows = torch_input.squeeze(1).tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("conv", captured)
        operator = captured["conv"].operator
        self.assertIsInstance(operator, Convolution1DOperator)
        self.assertEqual(operator.shape, (4, 6))
        self.assertEqual(operator.metadata.provenance.framework, "torch.fx")
        self.assertMatrixAlmostEqual(operator.apply(input_rows), DenseOperator(operator.to_dense()).apply(input_rows))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().squeeze(1).tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "conv1d_direct")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_biased_torch_conv1d_module_as_affine(self):
        class TinyBiasedConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 1, kernel_size=3, bias=True)
                with torch.no_grad():
                    self.conv.weight.copy_(torch.tensor([[[0.25, 1.5, -0.75]]]))
                    self.conv.bias.copy_(torch.tensor([0.125]))

            def forward(self, x):
                return self.conv(x)

        module = TinyBiasedConv().eval()
        torch_input = torch.tensor([[[1.0, -1.0, 2.0, 0.0, -0.5, 1.5]]])
        input_rows = torch_input.squeeze(1).tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("conv", captured)
        operator = captured["conv"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, Convolution1DOperator)
        self.assertEqual(operator.shape, (4, 6))
        self.assertEqual(operator.bias, [0.125, 0.125, 0.125, 0.125])
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().squeeze(1).tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "conv1d_direct_bias")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_multi_channel_torch_conv1d_module(self):
        class MultiChannelConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 2, kernel_size=3, bias=False)
                with torch.no_grad():
                    self.conv.weight.copy_(
                        torch.tensor([
                            [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                            [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
                        ])
                    )

            def forward(self, x):
                return self.conv(x)

        module = MultiChannelConv().eval()
        torch_input = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.5, 2.0, -2.0, 1.5]],
            [[0.0, -1.0, 1.5, 2.5, -0.5], [2.0, -0.5, 1.0, 0.25, -1.5]],
        ])
        input_rows = torch_input.flatten(1).tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)

        self.assertIn("conv", captured)
        operator = captured["conv"].operator
        self.assertIsInstance(operator, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (6, 10))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), DenseOperator(operator.to_dense()).apply(input_rows))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().flatten(1).tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "conv1d_channel_direct")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_functional_torch_conv1d_without_bias(self):
        class FunctionalConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.tensor([[[1.0, -2.0, 0.5]]]))

            def forward(self, x):
                return F.conv1d(x, self.weight)

        module = FunctionalConv().eval()
        torch_input = torch.tensor([
            [[1.0, 0.0, -1.0, 2.0, 0.5]],
            [[0.5, -1.5, 1.0, 0.0, 2.0]],
        ])
        input_rows = torch_input.squeeze(1).tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)
        functional = next((item for item in captured.values() if item.event.notes.get("capture") == "conv1d_function"), None)

        self.assertIsNotNone(functional)
        operator = functional.operator
        self.assertIsInstance(operator, Convolution1DOperator)
        self.assertEqual(operator.shape, (3, 5))
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().squeeze(1).tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "conv1d_direct")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_real_functional_torch_conv1d_as_affine(self):
        class FunctionalConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight",
                    torch.tensor([
                        [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                        [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
                    ]),
                )
                self.register_buffer("bias", torch.tensor([0.1, -0.2]))

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias)

        module = FunctionalConv().eval()
        torch_input = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.5, 2.0, -2.0, 1.5]],
            [[0.0, -1.0, 1.5, 2.5, -0.5], [2.0, -0.5, 1.0, 0.25, -1.5]],
        ])
        input_rows = torch_input.flatten(1).tolist()

        captured = capture_torch_fx_operators(module, sample_inputs=torch_input)
        functional = next((item for item in captured.values() if item.event.notes.get("capture") == "conv1d_function"), None)

        self.assertIsNotNone(functional)
        operator = functional.operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertIsInstance(operator.linear, MultiChannelConvolution1DOperator)
        self.assertEqual(operator.shape, (6, 10))
        for actual, expected in zip(operator.bias, [0.1, 0.1, 0.1, -0.2, -0.2, -0.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertMatrixAlmostEqual(operator.apply(input_rows), module(torch_input).detach().flatten(1).tolist())

        plan = plan_fixed_weight(
            operator,
            PlanningRequest(batch_size=len(input_rows), calls=32, sample_inputs=input_rows, codebook_sizes=(2,)),
        )
        self.assertEqual(plan.selected.name, "conv1d_channel_direct_bias")
        self.assertTrue(plan.selected.exact)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_named_adapter_factors_with_merged_weight_hint(self):
        class MergedAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Linear(3, 2, bias=False)
                self.lora_B = nn.Linear(2, 4, bias=False)
                with torch.no_grad():
                    self.lora_A.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
                    self.lora_B.weight.copy_(torch.tensor([[0.5, 1.0], [-1.0, 2.0], [3.0, -0.5], [0.25, 0.75]]))
                self.register_buffer("merged_weight", self.lora_B.weight @ self.lora_A.weight)

            def forward(self, x):
                return F.linear(x, self.merged_weight)

        captured = capture_torch_named_adapter_operators(MergedAdapter().eval())

        self.assertIn("lora_B", captured)
        self.assertEqual(captured["lora_B"].event.notes["capture"], "named_adapter_pair")
        self.assertEqual(captured["lora_B"].event.notes["merged_weight_hint"], "true")
        self.assertEqual(captured["lora_B"].operator.shape, (4, 3))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_captures_embedding_projection_over_one_hot_inputs(self):
        class EmbeddingProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(5, 3)
                self.proj = nn.Linear(3, 2, bias=True)
                with torch.no_grad():
                    self.embedding.weight.copy_(
                        torch.tensor([
                            [1.0, 0.0, 0.5],
                            [0.0, 1.0, -0.5],
                            [0.25, 0.75, 1.0],
                            [-1.0, 0.5, 0.0],
                            [0.5, -0.25, 0.25],
                        ])
                    )
                    self.proj.weight.copy_(torch.tensor([[0.5, 1.0, -1.0], [1.5, -0.5, 0.25]]))
                    self.proj.bias.copy_(torch.tensor([0.25, -0.75]))
                self.embedding.weight.requires_grad_(False)

            def forward(self, ids):
                return self.proj(self.embedding(ids))

        module = EmbeddingProjection().eval()
        captured = capture_torch_fx_linear_operators(module)
        one_hot_token_two = [[0.0, 0.0, 1.0, 0.0, 0.0]]

        self.assertIn("proj", captured)
        operator = captured["proj"].operator
        self.assertIsInstance(operator, AffineOperator)
        self.assertEqual(operator.shape, (2, 5))
        self.assertEqual(captured["proj"].event.notes["input_basis"], "one_hot")
        self.assertMatrixAlmostEqual(operator.apply(one_hot_token_two), module(torch.tensor([2])).detach().tolist())


if __name__ == "__main__":
    unittest.main()
