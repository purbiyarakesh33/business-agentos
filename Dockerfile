FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml
COPY --chown=app:app app.py analysis_methods.py analyst.py business_agents.py customer_analyst.py demo_sql.py forecasting_agent.py jev.py output_guardrails.py planner.py product_operations_analyst.py rag.py ./
COPY --chown=app:app demo_company/data/ ./demo_company/data/
COPY --chown=app:app demo_company/docs/ ./demo_company/docs/
RUN mkdir -p /app/demo_company/uploads \
    && chown -R app:app /app/demo_company/uploads

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
