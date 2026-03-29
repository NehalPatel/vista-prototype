from __future__ import annotations

import os

from flask import Flask

from pipeline.paths import ensure_directories, RESULTS_DIR


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Ensure runtime directories (including training_data) exist at startup
    ensure_directories()

    # Ensure known_faces dir exists so "Train faces" can write face_database.npy
    try:
        from face_pipeline.paths import KNOWN_FACES_DIR

        os.makedirs(str(KNOWN_FACES_DIR), exist_ok=True)
    except Exception:
        pass

    # Blueprints
    from web.main_routes import main_bp
    from web.api.processing import processing_bp
    from web.api.training import training_bp
    from web.api.system import system_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(processing_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(system_bp)

    # Provide results base for send_from_directory
    app.config["RESULTS_BASE"] = RESULTS_DIR

    return app

