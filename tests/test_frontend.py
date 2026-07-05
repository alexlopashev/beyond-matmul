import unittest

from beyond_matmul.frontend import extract_torch_fx_low_rank_operators


class FakeNode:
    def __init__(self, name, op, target, args=()):
        self.name = name
        self.op = op
        self.target = target
        self.args = args


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class FakeGraphModule:
    def __init__(self, nodes):
        self.graph = FakeGraph(nodes)


class FrontendTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
