# app.py
# Entry point for the Semantic Similarity Search Flask application.
# Run from the project root:  python app.py

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n{'=' * 60}")
    print(" Semantic Similarity Search — Flask Server")
    print(f"{'=' * 60}")
    print(f" URL   : http://{host}:{port}")
    print(f" Debug : {debug}")
    print(f"{'=' * 60}\n")

    app.run(host=host, port=port, debug=debug)

