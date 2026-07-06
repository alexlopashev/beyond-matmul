import unittest

from beyond_matmul import _linalg as la
from beyond_matmul.ir import (
    AffineOperator,
    CodebookOperator,
    Convolution1DOperator,
    DenseOperator,
    DiagonalOperator,
    LowRankOperator,
    MultiChannelConvolution1DOperator,
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

        affine = AffineOperator(low_rank, [0.5, -1.0, 2.0])
        dense_outputs = DenseOperator(affine.to_dense()).apply(inputs)
        expected = [[row[index] + affine.bias[index] for index in range(len(row))] for row in dense_outputs]
        self.assertEqual(affine.apply(inputs), expected)

        codebook = CodebookOperator([[0, 1, 0], [1, 0, 1], [0, 0, 1]], [0.0, 2.0])
        self.assertEqual(codebook.apply(inputs), DenseOperator(codebook.to_dense()).apply(inputs))

    def test_conv1d_to_dense(self):
        conv = Convolution1DOperator([1.0, -1.0, 2.0], input_length=5)
        inputs = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))

    def test_strided_padded_and_dilated_conv1d_to_dense(self):
        cases = [
            (
                Convolution1DOperator([1.0, -1.0], input_length=5, stride=2),
                [
                    [1.0, -1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0],
                ],
            ),
            (
                Convolution1DOperator([2.0, 3.0], input_length=3, padding=1),
                [
                    [3.0, 0.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [0.0, 2.0, 3.0],
                    [0.0, 0.0, 2.0],
                ],
            ),
            (
                Convolution1DOperator([1.0, -1.0, 2.0], input_length=7, dilation=2),
                [
                    [1.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 2.0],
                ],
            ),
            (
                Convolution1DOperator([1.0, -1.0, 2.0], input_length=7, stride=2, padding=1, dilation=2),
                [
                    [0.0, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                ],
            ),
        ]
        inputs = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]

        for conv, expected_dense in cases:
            with self.subTest(structure=conv.metadata.structure):
                expected_outputs = DenseOperator(expected_dense).apply([inputs[0][: conv.in_features]])
                self.assertEqual(conv.to_dense(), expected_dense)
                self.assertEqual(conv.apply([inputs[0][: conv.in_features]]), expected_outputs)
                self.assertEqual(conv.metadata.structure["stride"], conv.stride)
                self.assertEqual(conv.metadata.structure["padding"], conv.padding)
                self.assertEqual(conv.metadata.structure["dilation"], conv.dilation)

    def test_multi_channel_conv1d_to_dense(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
            ],
            input_length=5,
        )
        inputs = [[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 0.5, 2.0, -2.0, 1.5]]

        self.assertEqual(conv.shape, (6, 10))
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))

    def test_grouped_conv1d_to_dense(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
                [[0.5, -0.5, 1.5], [1.0, 0.0, -1.0]],
                [[-0.25, 0.5, 0.75], [2.0, -1.0, 0.25]],
            ],
            input_length=5,
            groups=2,
        )
        inputs = [[
            1.0, 2.0, 3.0, 4.0, 5.0,
            -1.0, 0.5, 2.0, -2.0, 1.5,
            0.0, -1.0, 1.5, 2.5, -0.5,
            2.0, -0.5, 1.0, 0.25, -1.5,
        ]]

        self.assertEqual(conv.shape, (12, 20))
        self.assertEqual(conv.metadata.lowerings[0], "conv1d_grouped_direct")
        self.assertEqual(conv.metadata.structure["groups"], 2)
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))
        dense = conv.to_dense()
        self.assertEqual(dense[6][:10], [0.0 for _ in range(10)])
        self.assertNotEqual(dense[6][10:], [0.0 for _ in range(10)])

    def test_depthwise_conv1d_to_dense(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.0, -1.0]],
                [[0.5, -0.5, 1.5]],
                [[2.0, 1.0, 0.25]],
            ],
            input_length=5,
            groups=3,
        )
        inputs = [[
            1.0, 2.0, 3.0, 4.0, 5.0,
            -1.0, 0.5, 2.0, -2.0, 1.5,
            0.0, -1.0, 1.5, 2.5, -0.5,
        ]]

        self.assertEqual(conv.shape, (9, 15))
        self.assertEqual(conv.metadata.lowerings[0], "conv1d_depthwise_direct")
        self.assertEqual(conv.metadata.structure["group_type"], "depthwise")
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))

    def test_strided_padded_dilated_grouped_conv1d_to_dense(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, -1.0], [0.5, 0.25]],
                [[0.5, 2.0], [-1.0, 1.25]],
                [[-0.25, 0.75], [1.0, -0.5]],
                [[1.5, -0.5], [0.25, 0.5]],
            ],
            input_length=5,
            groups=2,
            stride=2,
            padding=1,
            dilation=2,
        )
        inputs = [[
            1.0, 2.0, 3.0, 4.0, 5.0,
            -1.0, 0.5, 2.0, -2.0, 1.5,
            0.0, -1.0, 1.5, 2.5, -0.5,
            2.0, -0.5, 1.0, 0.25, -1.5,
        ]]

        self.assertEqual(conv.shape, (12, 20))
        self.assertEqual(conv.output_length, 3)
        self.assertEqual(conv.metadata.structure["stride"], 2)
        self.assertEqual(conv.metadata.structure["padding"], 1)
        self.assertEqual(conv.metadata.structure["dilation"], 2)
        self.assertEqual(conv.metadata.lowerings[0], "conv1d_grouped_direct")
        self.assertEqual(conv.apply(inputs), DenseOperator(conv.to_dense()).apply(inputs))
        dense = conv.to_dense()
        self.assertEqual(dense[6][:10], [0.0 for _ in range(10)])
        self.assertNotEqual(dense[6][10:], [0.0 for _ in range(10)])

    def test_grouped_conv1d_validation(self):
        with self.assertRaisesRegex(ValueError, "groups must be a positive integer"):
            MultiChannelConvolution1DOperator([[[1.0, 2.0, 3.0]]], input_length=5, groups=0)
        with self.assertRaisesRegex(ValueError, "output channels must be divisible by groups"):
            MultiChannelConvolution1DOperator(
                [
                    [[1.0, 2.0, 3.0]],
                    [[1.0, 2.0, 3.0]],
                ],
                input_length=5,
                groups=3,
            )

    def test_conv1d_parameter_validation(self):
        with self.assertRaisesRegex(ValueError, "conv1d stride must be a positive integer"):
            Convolution1DOperator([1.0, 2.0], input_length=4, stride=0)
        with self.assertRaisesRegex(ValueError, "conv1d padding must be a non-negative integer"):
            Convolution1DOperator([1.0, 2.0], input_length=4, padding=-1)
        with self.assertRaisesRegex(ValueError, "conv1d dilation must be a positive integer"):
            MultiChannelConvolution1DOperator([[[1.0, 2.0]]], input_length=4, dilation=0)
        with self.assertRaisesRegex(ValueError, "conv1d output length must be positive"):
            Convolution1DOperator([1.0, 2.0, 3.0], input_length=2, dilation=2)
        with self.assertRaisesRegex(ValueError, "conv1d stride must be a positive integer"):
            MultiChannelConvolution1DOperator([[[1.0, 2.0]]], input_length=4, stride=(1, 1))

    def test_conv1d_apply_rejects_mismatched_input_width_after_padding(self):
        conv = Convolution1DOperator([1.0, -1.0], input_length=4, padding=2)

        with self.assertRaisesRegex(ValueError, "input width does not match convolution operator"):
            conv.apply([[1.0, 2.0, 3.0]])


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

    def test_planner_preserves_affine_low_rank_bias(self):
        left = [[1.0], [2.0], [3.0]]
        right = [[4.0, 5.0, 6.0]]
        weight = AffineOperator(LowRankOperator(left, right), [0.5, -1.0, 2.0])
        inputs = [[2.0, -1.0, 3.0]]
        request = PlanningRequest(batch_size=1, calls=8, sample_inputs=inputs)

        plan = plan_fixed_weight(weight, request)

        self.assertEqual(plan.selected.name, "low_rank_product_bias")
        self.assertTrue(plan.selected.exact)
        self.assertEqual(plan.selected.operator.apply(inputs), weight.apply(inputs))

    def test_planner_preserves_affine_multi_channel_conv1d_bias(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
            ],
            input_length=5,
        )
        weight = AffineOperator(conv, [0.1, 0.1, 0.1, -0.2, -0.2, -0.2])
        inputs = [[1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 0.5, 2.0, -2.0, 1.5]]
        request = PlanningRequest(batch_size=1, calls=32, sample_inputs=inputs, codebook_sizes=(2,))

        plan = plan_fixed_weight(weight, request)

        self.assertEqual(plan.selected.name, "conv1d_channel_direct_bias")
        self.assertTrue(plan.selected.exact)
        self.assertEqual(plan.selected.operator.apply(inputs), weight.apply(inputs))

    def test_planner_distinguishes_grouped_and_depthwise_conv1d(self):
        grouped = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.5, -1.0], [0.25, 2.0, -0.5]],
                [[-1.5, 0.75, 0.25], [1.25, -0.25, 0.5]],
                [[0.5, -0.5, 1.5], [1.0, 0.0, -1.0]],
                [[-0.25, 0.5, 0.75], [2.0, -1.0, 0.25]],
            ],
            input_length=5,
            groups=2,
        )
        depthwise = MultiChannelConvolution1DOperator(
            [
                [[1.0, 0.0, -1.0]],
                [[0.5, -0.5, 1.5]],
                [[2.0, 1.0, 0.25]],
            ],
            input_length=5,
            groups=3,
        )

        grouped_plan = plan_fixed_weight(grouped, PlanningRequest(batch_size=2, calls=16, codebook_sizes=(2,)))
        depthwise_plan = plan_fixed_weight(depthwise, PlanningRequest(batch_size=2, calls=16, codebook_sizes=(2,)))

        self.assertEqual(grouped_plan.selected.name, "conv1d_grouped_direct")
        self.assertEqual(grouped_plan.selected.cost.apply_ops, 144)
        self.assertIn("dense_gemm", {option.name for option in grouped_plan.options})
        self.assertEqual(depthwise_plan.selected.name, "conv1d_depthwise_direct")
        self.assertEqual(depthwise_plan.selected.cost.apply_ops, 54)

    def test_planner_costs_strided_grouped_conv1d_from_explicit_metadata(self):
        conv = MultiChannelConvolution1DOperator(
            [
                [[1.0, -1.0], [0.5, 0.25]],
                [[0.5, 2.0], [-1.0, 1.25]],
                [[-0.25, 0.75], [1.0, -0.5]],
                [[1.5, -0.5], [0.25, 0.5]],
            ],
            input_length=5,
            groups=2,
            stride=2,
            padding=1,
            dilation=2,
        )

        plan = plan_fixed_weight(conv, PlanningRequest(batch_size=2, calls=16, codebook_sizes=(2,)))

        self.assertEqual(plan.selected.name, "conv1d_grouped_direct")
        self.assertEqual(plan.selected.cost.apply_ops, 96)
        self.assertIn("dense_gemm", {option.name for option in plan.options})

    def test_plan_option_exposes_cost_breakdown(self):
        weight = DiagonalOperator([1.0, 2.0, 3.0, 4.0])
        plan = plan_fixed_weight(weight, PlanningRequest(batch_size=8, calls=10))

        self.assertGreater(plan.selected.cost.apply_ops, 0.0)
        self.assertGreater(plan.selected.cost.memory_bytes_moved, 0)
        self.assertEqual(plan.selected.cost.cache_bytes, plan.selected.estimated_memory_bytes)
        self.assertEqual(plan.selected.amortized_cost, plan.selected.cost.score)


if __name__ == "__main__":
    unittest.main()
