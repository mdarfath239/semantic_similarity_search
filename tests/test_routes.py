# tests/test_routes.py
# Unit tests for Flask routes.
# Models are mocked so the test suite runs without loading the large model files.

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight mock for src.inference so pytest does not need model files.
# ---------------------------------------------------------------------------

def _build_mock_inference():
    """Return a minimal fake for src.inference."""
    mock_model = MagicMock()
    mock_model.vector_size = 100
    mock_model.wv = MagicMock()

    fake_vectors = MagicMock()
    fake_vectors.__len__ = lambda self: 18227
    fake_vectors.shape   = (18227, 100)
    fake_vectors.ndim    = 2

    import numpy as np
    fake_vectors.__class__ = np.ndarray

    mock_meta = MagicMock()
    mock_meta.__len__ = lambda self: 18227

    fake_module = types.ModuleType("src.inference")
    fake_module.MODEL             = mock_model
    fake_module.DOCUMENT_VECTORS  = fake_vectors
    fake_module.DOCUMENT_METADATA = mock_meta

    sample_results = [
        {
            "rank": 1,
            "document_id": "doc_001",
            "category": "sci.space",
            "text": "NASA launched a new space probe.",
            "similarity_score": 0.87,
        },
        {
            "rank": 2,
            "document_id": "doc_002",
            "category": "sci.space",
            "text": "The Hubble telescope captured new images.",
            "similarity_score": 0.74,
        },
    ]

    def mock_semantic_search(query, top_n=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Please enter a valid search query.")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        return sample_results[:top_n]

    fake_module.semantic_search = mock_semantic_search
    return fake_module


# Inject fake module before importing our app packages.
_fake_inference = _build_mock_inference()
sys.modules["src"]           = types.ModuleType("src")
sys.modules["src.inference"] = _fake_inference
sys.modules["src.preprocessing"] = types.ModuleType("src.preprocessing")
sys.modules["src.preprocessing"].preprocess_text = lambda text: text.split()


class TestRoutes(unittest.TestCase):

    def setUp(self):
        """Create the Flask test client."""
        # Reload services with the mocked inference module in place.
        import importlib, app.services as svc
        # Force re-load of inference reference.
        svc._search_fn  = _fake_inference.semantic_search
        svc._model_meta = {
            "model_name": "Word2Vec",
            "embedding_dimensions": 100,
            "total_indexed_documents": 18227,
            "dataset": "20 Newsgroups",
            "similarity_metric": "Cosine Similarity",
        }
        svc._load_error = None

        from app import create_app
        self.app    = create_app()
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    # ------------------------------------------------------------------
    # Page routes
    # ------------------------------------------------------------------

    def test_home_route_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"SemanticSearch", res.data)

    def test_about_route_returns_200(self):
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"About", res.data)

    # ------------------------------------------------------------------
    # Health route
    # ------------------------------------------------------------------

    def test_health_returns_200_when_model_ready(self):
        res  = self.client.get("/api/health")
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["status"], "healthy")

    def test_health_returns_503_when_model_not_ready(self):
        import app.services as svc
        svc._search_fn  = None
        svc._load_error = "Model file missing."
        res  = self.client.get("/api/health")
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 503)
        self.assertFalse(data["model_loaded"])
        # Restore
        svc._search_fn  = _fake_inference.semantic_search
        svc._load_error = None

    # ------------------------------------------------------------------
    # Search API — input validation
    # ------------------------------------------------------------------

    def test_search_requires_json_content_type(self):
        res = self.client.post(
            "/api/search",
            data="query=test",
            content_type="application/x-www-form-urlencoded",
        )
        self.assertEqual(res.status_code, 415)

    def test_search_empty_query_returns_400(self):
        res  = self.client.post(
            "/api/search",
            json={"query": "", "top_n": 5},
        )
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    def test_search_whitespace_query_returns_400(self):
        res  = self.client.post("/api/search", json={"query": "   ", "top_n": 5})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    def test_search_missing_query_returns_400(self):
        res  = self.client.post("/api/search", json={"top_n": 5})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    def test_search_invalid_top_n_string_returns_400(self):
        res  = self.client.post("/api/search", json={"query": "space", "top_n": "abc"})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    def test_search_top_n_zero_returns_400(self):
        res  = self.client.post("/api/search", json={"query": "space", "top_n": 0})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    def test_search_top_n_exceeds_max_returns_400(self):
        res  = self.client.post("/api/search", json={"query": "space", "top_n": 999})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data["success"])

    # ------------------------------------------------------------------
    # Search API — valid requests
    # ------------------------------------------------------------------

    def test_valid_search_returns_200(self):
        res  = self.client.post("/api/search", json={"query": "space exploration", "top_n": 5})
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data["success"])

    def test_valid_search_response_shape(self):
        res  = self.client.post("/api/search", json={"query": "space exploration", "top_n": 5})
        data = json.loads(res.data)
        self.assertIn("query",             data)
        self.assertIn("total_results",     data)
        self.assertIn("execution_time_ms", data)
        self.assertIn("results",           data)

    def test_valid_search_result_fields(self):
        res     = self.client.post("/api/search", json={"query": "space", "top_n": 1})
        data    = json.loads(res.data)
        result  = data["results"][0]
        self.assertIn("rank",             result)
        self.assertIn("document_id",      result)
        self.assertIn("category",         result)
        self.assertIn("similarity_score", result)
        self.assertIn("text",             result)

    def test_search_top_n_1_returns_one_result(self):
        res  = self.client.post("/api/search", json={"query": "space", "top_n": 1})
        data = json.loads(res.data)
        self.assertEqual(len(data["results"]), 1)

    def test_search_query_echoed_in_response(self):
        res  = self.client.post("/api/search", json={"query": "space exploration", "top_n": 5})
        data = json.loads(res.data)
        self.assertEqual(data["query"], "space exploration")

    def test_similarity_score_between_0_and_1(self):
        res  = self.client.post("/api/search", json={"query": "space", "top_n": 2})
        data = json.loads(res.data)
        for r in data["results"]:
            self.assertGreaterEqual(r["similarity_score"], 0.0)
            self.assertLessEqual(r["similarity_score"],    1.0)

    # ------------------------------------------------------------------
    # Search API — empty results (vocabulary miss)
    # ------------------------------------------------------------------

    def test_search_no_vocab_match_returns_empty_results(self):
        import app.services as svc

        original_fn = svc._search_fn

        def empty_fn(query, top_n=5):
            return []

        svc._search_fn = empty_fn

        res  = self.client.post(
            "/api/search",
            json={"query": "xyzzy quux frobnicator", "top_n": 5},
        )
        data = json.loads(res.data)

        self.assertTrue(data["success"])
        self.assertEqual(data["total_results"], 0)
        self.assertEqual(data["results"], [])

        svc._search_fn = original_fn

    # ------------------------------------------------------------------
    # 404 handler
    # ------------------------------------------------------------------

    def test_unknown_route_returns_404(self):
        res  = self.client.get("/this/does/not/exist")
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 404)
        self.assertFalse(data["success"])


if __name__ == "__main__":
    unittest.main()

