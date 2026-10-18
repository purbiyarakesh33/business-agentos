# Business AgentOS

A local, synthetic-data demo of a guarded multi-agent business analysis workflow.

## Workflow

User → Jev input screening → planner → human plan approval → specialist agents/tools → evidence verification → output privacy guard → report

The app includes a Data Analyst, Customer Analyst, Product & Operations Analyst, Forecasting Agent, and Verification Agent. Structured Cedar & Stone demo records are queried from SQLite. Company documents and review text can be indexed for semantic retrieval in PostgreSQL with pgvector. Featherless provides the Jev, chat, planning, specialist, and embedding model APIs.

## Run with Docker

1. Copy .env.example to .env.
2. Put your Featherless API key in .env as FEATHERLESS_API_KEY=... Keep .env private; it is excluded from Git and Docker build context.
3. From the project directory, run:

        docker compose up --build -d
        docker compose ps

4. Open http://localhost:8501.
5. In the sidebar, open Semantic document knowledge base, check the authorization box, and select Index documents and demo reviews to populate a new vector database. Indexing sends the selected demo/source text to Featherless for embeddings.
6. Ask a supported business question. Review and approve the planner’s proposed work before specialists run.

Follow container output with docker compose logs -f app. Stop the services with docker compose down; named database and uploaded-document volumes are retained. The database and UI ports bind to localhost.

The default PostgreSQL password is only for local demo use. Before using Docker beyond localhost, set POSTGRES_PASSWORD in .env to a strong URL-safe value. Compose passes the Featherless key at runtime; it is not copied into the image.

## Run on Windows without Docker for the app

Start the vector database:

        docker compose up -d ragdb

Install dependencies and launch Streamlit from the project directory:

        py -m pip install -r requirements.txt
        py -m streamlit run app.py --server.port 8501

For this mode, RAG_DATABASE_URL in .env should point to localhost:5433. The supplied .env.example has that local-demo URL.

## Local checks

From PowerShell in the project directory:

        py -B evals/validate_golden_dataset.py
        py -B analyst_eval.py
        py -B planner_eval.py
        py -B evals/public_safety_eval.py

These checks are offline. The expected current results are golden-set validation passed, 33/33 analyst checks, 7/7 planner checks, and 5/5 public-safety contract checks. See evals/EVALUATION_STATUS.md for component metrics, DeepEval report metrics, current live smoke results, and caveats.

The eight live Streamlit smoke scenarios make Featherless requests. They are documented in evals/README.md and should be run when changing the workflow.

## Deploy on Streamlit Community Cloud

The app entrypoint, dependency file, and Streamlit configuration are already at the repository root (app.py, requirements.txt, and .streamlit/config.toml). Community Cloud deploys the app from a GitHub repository. Configure credentials in the app's Streamlit Secrets settings; never commit .env or secrets.toml. The app reads these root-level secrets: FEATHERLESS_API_KEY and, for document RAG, RAG_DATABASE_URL pointing to a remotely reachable PostgreSQL database with the pgvector extension. Optional model overrides use the names in .env.example.

The SQLite business demo data is included with the source. The local Docker Compose database is not reachable from Community Cloud, so persistent document retrieval needs a hosted PostgreSQL database with pgvector. Supabase is the setup guide used here: enable the `vector` extension in Database → Extensions, then copy the Session pooler connection string from Connect. The app requires SSL and disables prepared statements for transaction-pooler compatibility; it also handles Supabase's `extensions` schema. Supabase's free database currently includes 500 MB and may pause after a week of low activity, so treat it as a demo service and check current plan details before relying on it. [pgvector setup](https://supabase.com/docs/guides/database/extensions/pgvector) · [connection modes](https://supabase.com/docs/guides/database/connecting-to-postgres) · [free-plan pausing](https://supabase.com/docs/guides/platform/free-project-pausing)

For a public demo, add these Streamlit Secrets (never commit them): `FEATHERLESS_API_KEY`, `RAG_DATABASE_URL`, `PUBLIC_DEMO_MODE = "true"`, `PUBLIC_DAILY_REQUEST_LIMIT = "20"`, and `PUBLIC_HOURLY_REQUEST_LIMIT = "6"`. Keep the first two values private. Public mode uses one shared synthetic-data tenant, enforces atomic shared hourly/daily caps on each Featherless API request through PostgreSQL, and fails closed if the usage database is unavailable. PDF uploads are kept in a random per-browser-session tenant in Postgres and expire after 24 hours; visitor uploads are not written to the app's shared local folder. A consent checkbox explains that prompts and document text go to Featherless. This is still an unauthenticated demo, not a production security boundary.

Before deploying, seed the public vector database with only the bundled synthetic documents and reviews. Set `RAG_DATABASE_URL` in your local environment to the hosted connection string, keep `FEATHERLESS_API_KEY` in the local `.env`, and run `py seed_public_demo_rag.py`. The seed command explicitly excludes the local uploads folder, so previous test PDFs are not included. Visitors should not enter confidential information. Streamlit Community Cloud stores app secrets in the app settings and deploys from GitHub: [secrets setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) · [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Demo limits

All business records are synthetic. This is a local portfolio/demo application, not a production-ready business deployment. It does not yet provide user authentication, tenant isolation, managed secret storage, TLS/reverse proxy, production monitoring, backup/restore policy, or a deployment/SLA setup. LLM judge scores are small-sample regression signals; the holdout set is intentionally still unused. See the evaluation status for known SQL-generation and RAG ranking limitations.

The bundled scenario_truth.json is evaluator-only and is excluded from the app image and RAG indexing.
