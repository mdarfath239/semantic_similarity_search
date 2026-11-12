# app/services.py
# Business logic layer: wraps the existing semantic_search() function and
# adds request validation, timing, and error normalisation.

import time
from typing import Any

# MAX_TOP_N is imported from the Flask app config inside the request context,
# but we also define a module-level constant as a fallback for unit tests.
MAX_TOP_N = 20
MIN_TOP_N = 1


# ---------------------------------------------------------------------------
# Model loading (once at import time, shared across requests)
# ---------------------------------------------------------------------------

_search_fn = None
_model_meta: dict[str, Any] = {}
_load_error: str | None = None


def _load_inference() -> None:
    """
    Import and cache the existing semantic_search function.
    Called once when the Flask application starts.
    Sets module-level _search_fn and _model_meta, or records _load_error.
    """
    global _search_fn, _model_meta, _load_error

    try:
        # This import triggers model loading inside inference.py (module-level
        # globals).  It is intentionally deferred here so the Flask app can
        # start and report a health status even if models are missing.
        from src.inference import semantic_search, MODEL, DOCUMENT_VECTORS

        _search_fn = semantic_search

        _model_meta = {
            "model_name": "Word2Vec",
            "embedding_dimensions": int(MODEL.vector_size),
            "total_indexed_documents": int(len(DOCUMENT_VECTORS)),
            "dataset": "20 Newsgroups",
            "similarity_metric": "Cosine Similarity",
        }

        print("[services] Inference module loaded successfully.")

    except FileNotFoundError as exc:
        _load_error = f"Model file not found: {exc}"
        print(f"[services] ERROR – {_load_error}")

    except ValueError as exc:
        _load_error = f"Model integrity error: {exc}"
        print(f"[services] ERROR – {_load_error}")

    except Exception as exc:  # noqa: BLE001
        _load_error = f"Unexpected error loading model: {exc}"
        print(f"[services] ERROR – {_load_error}")


# Load immediately when the module is first imported.
_load_inference()


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


def get_model_info() -> dict[str, Any]:
    """Return static metadata about the loaded model."""
    if _load_error:
        return {"error": _load_error}
    return dict(_model_meta)


def is_model_ready() -> bool:
    """Return True when the inference function is available."""
    return _search_fn is not None


def get_load_error() -> str | None:
    """Return the model-loading error message, or None if healthy."""
    return _load_error


def run_search(query: str, top_n: int) -> dict[str, Any]:
    """
    Validate inputs, call semantic_search(), and return a structured
    response dict.

    Parameters
    ----------
    query : str
        Natural-language search query from the user.
    top_n : int
        Number of results requested.

    Returns
    -------
    dict with keys:
        success        bool
        query          str
        total_results  int
        execution_time_ms float
        results        list[dict]   (empty list on no match)
    or on error:
        success  False
        error    str (user-safe message, no internal paths)
    """

    # --- Guard: model not ready ---
    if not is_model_ready():
        return {
            "success": False,
            "error": "The search model is not available. Please check the server logs.",
        }

    # --- Validate query ---
    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "error": "Query must be a non-empty string.",
        }

    if len(query) > 2000:
        return {
            "success": False,
            "error": "Query is too long. Please keep it under 2 000 characters.",
        }

    # --- Validate top_n ---
    if not isinstance(top_n, int) or isinstance(top_n, bool):
        return {
            "success": False,
            "error": f"top_n must be an integer between {MIN_TOP_N} and {MAX_TOP_N}.",
        }

    if top_n < MIN_TOP_N or top_n > MAX_TOP_N:
        return {
            "success": False,
            "error": f"top_n must be between {MIN_TOP_N} and {MAX_TOP_N}.",
        }

    # --- Run inference ---
    t0 = time.perf_counter()

    try:
        results = _search_fn(query.strip(), top_n=top_n)

    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    except Exception:  # noqa: BLE001
        return {
            "success": False,
            "error": "An internal error occurred during the search. Please try again.",
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # --- Check for no-vocabulary match ---
    if not results:
        return {
            "success": True,
            "query": query.strip(),
            "total_results": 0,
            "execution_time_ms": elapsed_ms,
            "results": [],
            "message": (
                "No matching documents found. "
                "The query words may not be in the model vocabulary. "
                "Try rephrasing your query."
            ),
        }

    # Coerce any NumPy scalar types to native Python so Flask can serialise them.
    serialisable_results = []
    for r in results:
        serialisable_results.append({
            "rank":             int(r["rank"]),
            "document_id":      str(r["document_id"]),
            "category":         str(r["category"]),
            "text":             str(r["text"]),
            "similarity_score": float(r["similarity_score"]),
        })

    return {
        "success": True,
        "query": query.strip(),
        "total_results": len(serialisable_results),
        "execution_time_ms": elapsed_ms,
        "results": serialisable_results,
    }
