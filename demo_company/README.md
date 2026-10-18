# Cedar & Stone — synthetic AgentOS demo data

This is a fictional direct-to-consumer home-goods retailer created for the Business AgentOS prototype. Every record, person, company name, event, and review is synthetic. It is not evidence about a real company.

## Contents

- `data/`: linked CSV tables and `business_demo.sqlite3` for orders, customers, products, inventory, marketing, fulfillment, returns, and reviews.
- Structured demo business facts remain in SQLite and are queried with SQL. Semantic document/review search stores extracted passages, provenance, and 1024-dimensional Featherless embeddings in the separate PostgreSQL + pgvector store. The analyst invokes hybrid vector + full-text retrieval only when relevant; numeric findings still require SQL evidence. The synthetic evaluation truth file is excluded from retrieval.
- `docs/company_profile.md`: business context and operating assumptions.
- `docs/data_dictionary.md`: table grain, keys, fields, units, and caveats.
- `docs/kpi_definitions.md`: shared metric formulas and inclusion rules.
- `docs/scenario_catalog.md`: planted observations, expected analysis tasks, and unsupported causal claims to avoid.
- `docs/source_catalog.json`: machine-readable evidence source catalog.
- `docs/evidence_queries.sql`: reproducible starter queries; result rows can be cited by their IDs.
- `docs/demo_questions.md`: example questions for the four specialists and verifier.
- `guardrails/jev_cases.jsonl`: small, synthetic input-screening examples with expected labels.
- `generate_demo_data.py`: deterministic generator (Python standard library only).

Regenerate the CSVs and SQLite database with `python generate_demo_data.py`. The fixed seed makes the output repeatable. The data is intentionally compact enough for local development; production use should query the database rather than put full tables in an LLM prompt.


## Semantic document retrieval

The document index is separate from the SQLite business tables. rag.py extracts text from Markdown, TXT, JSON, SQL, DOCX, and text-based PDF files, splits it into overlapping passages, and preserves document names and PDF page references. The user can also upload documents in the Streamlit sidebar; uploaded originals are stored under uploads/ and ignored by Git. Scanned PDFs with no text layer need OCR before indexing.

Featherless Qwen/Qwen3-Embedding-8B generates 1024-dimensional embeddings. PostgreSQL with pgvector stores passage text, source metadata, and vectors. Retrieval combines vector similarity with PostgreSQL full-text search (hybrid ranking). Re-indexing skips unchanged files and reviews, and embedding requests are batched. The app does not send source text to Featherless at startup; indexing requires a deliberate sidebar action and confirmation. Retrieval queries are embedded only when a specialist explicitly invokes the retrieval tool.

### Local setup

1. Install Docker Desktop and start the vector database from D:\thon:
   docker compose up -d
2. Install Python dependencies in the project environment:
   py -m pip install -r requirements.txt
3. The project .env contains RAG_DATABASE_URL and FEATHERLESS_EMBEDDING_MODEL; the Featherless API key remains in that same local file.
4. Run the app:
   py -m streamlit run app.py --server.port 8502
5. Open Semantic document knowledge base in the sidebar, confirm Featherless embedding use, add documents if needed, and click Index documents and demo reviews.

The bundled business metrics still come from the synthetic SQLite database. This local prototype uses a single RAG_TENANT_ID (default demo); production document sources need company authentication, per-document ACL filtering, and approved source connectors before multiple companies can share the service. The local Compose password is for development only.
## Important

The September 2025 events are planted co-occurring signals for analysis practice. Low/zero on-hand inventory snapshots, lower paid-search spend, and weaker sales appear in the same period. The dataset does not establish which event caused the sales change. Agents must report supported observations and qualify causal hypotheses.

The Streamlit UI connects to `business_demo.sqlite3` through `D:\thon\business_data.py`. Planner context includes live row counts and monthly sales aggregates. After the user approves a plan, specialists receive bounded query results from a local read-only SQLite connector. The CSV and database content is synthetic and should be described as demo evidence, not real-company findings.


