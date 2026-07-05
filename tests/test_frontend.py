import unittest

from beyond_matmul.frontend import (
    capture_torch_fx_linear_operators,
    capture_torch_named_adapter_operators,
    extract_torch_fx_low_rank_operators,
)
from beyond_matmul.ir import AffineOperator

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
