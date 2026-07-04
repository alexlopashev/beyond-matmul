import unittest

from beyond_matmul import _linalg as la
from beyond_matmul.ir import (
    CodebookOperator,
    Convolution1DOperator,
    DenseOperator,
    DiagonalOperator,
    LowRankOperator,
    SparseCOOOperator,
)
from beyond_matmul.planner import PlanningRequest, plan_fixed_weight


class OperatorTests(unittest.TestCase):
    def test_structured_operators_match_dense_application(self):
        inputs = [[2.0, -1.0, 3.0]]

        diagonal = DiagonalOperator([2.0, 3.0, 4.0])
        self.assertEqual(diagonal.apply(inputs), DenseOperator(diagonal.to_dense()).apply(inputs))

        sparse = SparseCOOOperator([0, 1, 2], [0, 2, 1], [2.0, 5.0, -1.0], (3, 3))
        self.assertEqual(sparse.apply(inputs), DenseOperator(sparse.to_dense()).apply(inputs))

        low_rank = LowRankOperator([[1.0], [2.0], [3.0]], [[4.0, 5.0, 6.0]])
        self.assertEqual(low_rank.apply(inputs), DenseOperator(low_rank.to_dense()).apply(inputs))

        codebook = CodebookOperator([[0, 1, 0], [1, 0, 1], [0, 0, 1]], [0.0, 2.0])
        self.assertEqual(codebook.apply(inputs), DenseOperator(codebook.to_dense()).apply(inputs))

    def test_conv1d_to_dense(self):
        conv = Convolution1DOperator([1.0, -1.0, 2.0], input_length=5)
        inputs = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))


class PlannerTests(unittest.TestCase):
    def test_planner_selects_diagonal_kernel_for_diagonal_weight(self):
        weight = DiagonalOperator([1.0, 2.0, 3.0, 4.0])
        request = PlanningRequest(batch_size=8, calls=10)
        plan = plan_fixed_weight(weight, request)
        self.assertEqual(plan.selected.name, "diagonal_kernel")
        self.assertTrue(plan.selected.exact)

    def test_planner_uses_dense_fallback_when_error_contract_disallows_approximation(self):
        weight = la.random_matrix(6, 6, seed=42)
        request = PlanningRequest(batch_size=2, calls=1, allow_approximate=False)
        plan = plan_fixed_weight(weight, request)
        self.assertEqual(plan.selected.name, "dense_gemm")
        self.assertTrue(plan.selected.valid)

    def test_product_error_contract_can_accept_low_rank(self):
        left = [[1.0], [2.0], [3.0], [4.0]]
        right = [[2.0, -1.0, 0.5, 3.0]]
        weight = LowRankOperator(left, right)
        inputs = la.random_batch(4, 4, seed=7)
        request = PlanningRequest(
            batch_size=4,
            calls=16,
            allow_approximate=True,
            max_relative_error=1e-8,
            sample_inputs=inputs,
        )
        plan = plan_fixed_weight(weight, request)
        self.assertEqual(plan.selected.name, "low_rank_product")
        self.assertLessEqual(plan.selected.relative_error, 1e-8)


if __name__ == "__main__":
    unittest.main()
