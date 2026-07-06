import unittest

from beyond_matmul.analyzer import ReuseTracker, analyze_dense


class AnalyzerTests(unittest.TestCase):
    def test_detects_diagonal_and_sparse_candidates(self):
        matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
        candidates = analyze_dense(matrix)
        by_kind = {candidate.kind: candidate for candidate in candidates}
        self.assertTrue(by_kind["diagonal"].exact)
        self.assertGreater(by_kind["sparse"].confidence, 0.5)

    def test_detects_small_codebook(self):
        matrix = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
        candidates = analyze_dense(matrix, max_codebook_size=2)
        by_kind = {candidate.kind: candidate for candidate in candidates}
        self.assertTrue(by_kind["codebook"].exact)

    def test_reuse_tracker_confidence_increases(self):
        tracker = ReuseTracker()
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        first = tracker.observe(matrix)
        third = None
        tracker.observe(matrix)
        third = tracker.observe(matrix)
        self.assertLess(first.confidence, third.confidence)
        self.assertEqual(third.evidence["observed_calls"], 3)

    def test_ambiguous_dense_matrix_candidates_have_bounded_confidence(self):
        matrix = [
            [0.37, -1.12, 2.41, 0.58],
            [1.73, 0.24, -0.91, 3.17],
            [-2.08, 1.46, 0.63, -1.39],
            [0.81, -2.34, 1.19, 0.07],
        ]

        candidates = analyze_dense(matrix, max_codebook_size=2, ranks=(1,))
        by_kind = {candidate.kind: candidate for candidate in candidates}

        self.assertFalse(by_kind["diagonal"].exact)
        self.assertLessEqual(by_kind["diagonal"].confidence, 0.1)
        self.assertLessEqual(by_kind["sparse"].confidence, 0.1)
        self.assertFalse(by_kind["codebook"].exact)
        self.assertLessEqual(by_kind["codebook"].confidence, 0.25)
        self.assertFalse(by_kind["low_rank"].exact)
        self.assertLessEqual(by_kind["low_rank"].confidence, 0.1)

    def test_records_output_validation_evidence_when_sample_inputs_are_available(self):
        matrix = [
            [2.0, 4.0],
            [1.0, 2.0],
        ]
        sample_inputs = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        candidates = analyze_dense(matrix, ranks=(1,), sample_inputs=sample_inputs)
        low_rank = next(candidate for candidate in candidates if candidate.kind == "low_rank")

        validation = low_rank.evidence["validation"]
        self.assertEqual(validation["metric"], "output_relative_l2")
        self.assertEqual(validation["sample_count"], 2)
        self.assertLessEqual(validation["output_relative_error"], 1e-8)
        self.assertTrue(validation["exact_on_samples"])

    def test_output_validation_caps_confidence_for_bad_candidate_behavior(self):
        matrix = [
            [1.0, 0.5],
            [0.5, 1.0],
        ]

        candidates = analyze_dense(matrix, ranks=(), sample_inputs=[[1.0, -1.0]])
        diagonal = next(candidate for candidate in candidates if candidate.kind == "diagonal")

        validation = diagonal.evidence["validation"]
        self.assertGreater(validation["output_relative_error"], 0.05)
        self.assertEqual(validation["confidence_bound"], 0.0)
        self.assertEqual(diagonal.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
