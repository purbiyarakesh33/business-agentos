"""Read-only connectivity preflight for the local RAG regression matrix."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import rag

    with rag._connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
    if not row or row[0] != 1:
        raise RuntimeError("RAG database returned an unexpected health-check result.")
    print("RAG database connectivity passed; no rows were read or changed.")
except Exception as exc:
    print(f"RAG database connectivity failed ({type(exc).__name__}); retrieval cases skipped.")
    raise SystemExit(1)
