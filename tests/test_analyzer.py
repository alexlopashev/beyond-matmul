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


if __name__ == "__main__":
    unittest.main()
