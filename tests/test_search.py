# tests/test_search.py
# Integration-style tests that call the real semantic_search() function.
# These tests DO load the actual trained model files.
# Mark tests in this file as 'integration' so they can be skipped when
# model files are unavailable (e.g., in CI).

import os
import sys
import unittest

# Ensure project root is on sys.path.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Skip all tests in this module if model files are absent.
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "word2vec_model.model")

@unittest.skipUnless(
    os.path.exists(MODEL_PATH),
    "Skipping real-model tests: word2vec_model.model not found.",
)
class TestRealSemanticSearch(unittest.TestCase):
    """
    These tests call the real semantic_search() function with actual
    model artifacts.  They verify end-to-end inference correctness.
    """

    @classmethod
    def setUpClass(cls):
        from src.inference import semantic_search, MODEL, DOCUMENT_VECTORS

        # Wrap in staticmethod so Python does not inject 'self' when called
        # on an instance (plain functions stored on a class become unbound methods).
        cls.search  = staticmethod(semantic_search)
        cls.model   = MODEL
        cls.vectors = DOCUMENT_VECTORS

    # ------------------------------------------------------------------
    # Model integrity
    # ------------------------------------------------------------------

    def test_model_vector_size_is_100(self):
        self.assertEqual(self.model.vector_size, 100)

    def test_document_vectors_shape(self):
        self.assertEqual(self.vectors.ndim,    2)
        self.assertEqual(self.vectors.shape[1], 100)

    def test_document_count_is_18227(self):
        self.assertEqual(len(self.vectors), 18227)

    # ------------------------------------------------------------------
    # Valid queries
    # ------------------------------------------------------------------

    def test_query_returns_list(self):
        results = self.search("space exploration", top_n=5)
        self.assertIsInstance(results, list)

    def test_query_returns_correct_top_n(self):
        for n in (1, 3, 5, 10):
            results = self.search("computer graphics", top_n=n)
            self.assertLessEqual(len(results), n,
                msg=f"Expected at most {n} results, got {len(results)}")

    def test_result_has_required_keys(self):
        results = self.search("network security", top_n=3)
        self.assertGreater(len(results), 0, "Expected at least one result.")
        for key in ("rank", "document_id", "category", "text", "similarity_score"):
            self.assertIn(key, results[0])

    def test_ranks_are_sequential(self):
        results = self.search("machine learning", top_n=5)
        for i, r in enumerate(results, start=1):
            self.assertEqual(r["rank"], i)

    def test_scores_are_descending(self):
        results = self.search("space exploration nasa", top_n=5)
        scores  = [r["similarity_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_in_valid_range(self):
        results = self.search("religion ethics", top_n=5)
        for r in results:
            self.assertGreaterEqual(r["similarity_score"], -1.0)
            self.assertLessEqual(   r["similarity_score"],  1.0)

    def test_category_is_non_empty_string(self):
        results = self.search("computer hardware", top_n=3)
        for r in results:
            self.assertIsInstance(r["category"], str)
            self.assertGreater(len(r["category"]), 0)

    def test_text_is_non_empty_string(self):
        results = self.search("motorcycle engine", top_n=3)
        for r in results:
            self.assertIsInstance(r["text"], str)
            self.assertGreater(len(r["text"]), 0)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.search("", top_n=5)

    def test_whitespace_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.search("   ", top_n=5)

    def test_invalid_top_n_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.search("space", top_n=0)

    def test_invalid_top_n_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.search("space", top_n=-1)

    def test_query_with_no_vocab_words_returns_empty_list(self):
        # Unlikely to match anything; not guaranteed to be empty but covers path.
        results = self.search("xyzzy1234quux9999", top_n=5)
        self.assertIsInstance(results, list)

    def test_top_n_1_returns_single_result(self):
        results = self.search("baseball sports", top_n=1)
        self.assertEqual(len(results), 1)

    def test_top_n_string_raises_value_error(self):
        with self.assertRaises((ValueError, TypeError)):
            self.search("space", top_n="five")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
