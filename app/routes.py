# app/routes.py
# Flask routes — thin controllers that delegate to services.py.

from flask import Blueprint, jsonify, render_template, request

from app import services

main_bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@main_bp.route("/")
def index():
    """Render the main search dashboard."""
    model_info = services.get_model_info()
    return render_template("index.html", model_info=model_info)


@main_bp.route("/about")
def about():
    """Render the About / Model Info page."""
    model_info = services.get_model_info()
    return render_template("about.html", model_info=model_info)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@main_bp.route("/api/health")
def health():
    """Return application and model health status."""
    ready = services.is_model_ready()
    error = services.get_load_error()
    model_info = services.get_model_info()

    payload = {
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "model_info": model_info,
    }

    if error:
        payload["error"] = error

    status_code = 200 if ready else 503
    return jsonify(payload), status_code


@main_bp.route("/api/search", methods=["POST"])
def search():
    """
    Accept a JSON search request and return ranked results.

    Request body:
        { "query": "...", "top_n": 5 }

    Response:
        { "success": true, "query": "...", "total_results": 5,
          "execution_time_ms": 12.5, "results": [...] }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be JSON (Content-Type: application/json).",
        }), 415

    data = request.get_json(silent=True) or {}

    query = data.get("query", "")
    top_n_raw = data.get("top_n", 5)

    # Coerce top_n to int if the client sends a string.
    try:
        top_n = int(top_n_raw)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "top_n must be a positive integer.",
        }), 400

    result = services.run_search(query, top_n)
    print(f"[SEARCH DEBUG] query={query!r}, top_n={top_n}, total_results={result.get('total_results')}", flush=True)
    if not result.get("success"):
        print(f"[SEARCH DEBUG ERROR] error={result.get('error')}", flush=True)
        return jsonify(result), 400

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@main_bp.app_errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@main_bp.app_errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@main_bp.app_errorhandler(413)
def request_too_large(_error):
    return jsonify({"success": False, "error": "Request body too large."}), 413


@main_bp.app_errorhandler(500)
def internal_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500

