#!/usr/bin/env python3
"""Test MongoDB connection using MONGODB_URI from .env or environment.

Run from repo root: python scripts/test_mongodb_connection.py
"""

import os
import sys

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.mongodb_store import get_db, ensure_indexes

def main():
    uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    if not uri:
        print("MONGODB_URI (or MONGO_URI) is not set. Add it to .env in the repo root or set the environment variable.")
        return 1
    # Mask password in output
    if "@" in uri and "://" in uri:
        display_uri = uri.split("@", 1)[-1]
    else:
        display_uri = uri
    print(f"Connecting to MongoDB ({display_uri}) ...")
    db = get_db()
    if db is None:
        print("Connection failed. Check that MongoDB is running and MONGODB_URI is correct.")
        return 1
    if not ensure_indexes():
        print("Connected but index creation failed.")
        return 1
    print("MongoDB connection OK. Indexes ensured.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
