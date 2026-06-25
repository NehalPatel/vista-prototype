from __future__ import annotations

from flask import Blueprint, current_app, render_template, send_from_directory


main_bp = Blueprint("main", __name__)


@main_bp.get("/")
def index():
    return render_template("index.html")


@main_bp.get("/training")
def training_page():
    """Training Data Manager page: face and monument dataset upload/train."""
    return render_template("training.html")


@main_bp.get("/results/<path:filename>")
def serve_results(filename: str):
    return send_from_directory(current_app.config["RESULTS_BASE"], filename)

