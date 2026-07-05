import operator as py_operator
import unittest

from beyond_matmul.frontend import (
    capture_torch_fx_operators,
    capture_torch_fx_linear_operators,
    extract_torch_fx_operators,
    capture_torch_named_adapter_operators,
    extract_torch_fx_low_rank_operators,
)
from beyond_matmul.ir import AffineOperator, Convolution1DOperator, DenseOperator
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
            Conv1d([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], in_channels=2),
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
