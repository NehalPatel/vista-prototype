from __future__ import annotations

import os
import sys


# Keep `python web/app.py` working regardless of CWD.
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from web import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)