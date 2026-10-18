"""Index only bundled synthetic documents and reviews into the public RAG tenant.

Set RAG_DATABASE_URL to the hosted pgvector connection and keep the Featherless
key in the local .env before running this one-time bootstrap script.
"""
from __future__ import annotations

import os

os.environ["RAG_TENANT_ID"] = "public_demo"

from rag import ingest_company_sources  # noqa: E402


if __name__ == "__main__":
    result = ingest_company_sources(include_uploads=False)
    print(
        "Public synthetic RAG data indexed: "
        f"{result['document_chunks']} document chunks, "
        f"{result['review_records']} review records, "
        f"{result['new_sources']} new sources, "
        f"{result['updated_sources']} updated sources."
    )
