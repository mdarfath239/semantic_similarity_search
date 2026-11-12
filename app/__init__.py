# app/__init__.py
# Flask application factory.

from flask import Flask
from flask_cors import CORS


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Allow cross-origin requests in development.
    CORS(app)

    # Apply configuration from environment variables with safe defaults.
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request cap
    app.config["MAX_RESULTS"] = 20  # hard upper limit for top_n
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching in development

    # Register blueprints.
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app

