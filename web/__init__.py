"""VISTA Flask web application package.

Entry point remains `web/app.py` for backwards compatibility.
"""

from .factory import create_app

__all__ = ["create_app"]

