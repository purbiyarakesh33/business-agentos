"""Fail-closed global usage limits for an unauthenticated public demo."""
from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from db_diagnostics import log_database_error


_LOGGER = logging.getLogger(__name__)


class PublicDemoError(RuntimeError):
    """The public demo cannot safely accept another model-backed request."""


class PublicDemoLimitReached(PublicDemoError):
    """A shared hourly or daily public request cap has been reached."""


def public_demo_enabled() -> bool:
    return os.getenv("PUBLIC_DEMO_MODE", "").strip().casefold() in {"1", "true", "yes", "on"}


def configure_public_demo() -> None:
    """Pin every anonymous visitor to the public synthetic-data RAG tenant."""
    if public_demo_enabled():
        os.environ["RAG_TENANT_ID"] = "public_demo"


def _positive_limit(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise PublicDemoError("Public demo limits are misconfigured; requests are paused.") from exc
    if value < 1 or value > maximum:
        raise PublicDemoError("Public demo limits are misconfigured; requests are paused.")
    return value


def reserve_public_request() -> dict[str, int] | None:
    """Atomically reserve one public request; no DB means fail closed in public mode."""
    if not public_demo_enabled():
        return None

    database_url = os.getenv("RAG_DATABASE_URL", "").strip()
    if not database_url:
        raise PublicDemoError("The public demo is temporarily unavailable while its usage database is configured.")

    daily_limit = _positive_limit("PUBLIC_DAILY_REQUEST_LIMIT", 20, 1000)
    hourly_limit = _positive_limit("PUBLIC_HOURLY_REQUEST_LIMIT", 6, 100)
    host = (urlsplit(database_url).hostname or "").casefold()
    sslmode = "prefer" if host in {"localhost", "127.0.0.1", "::1"} else "require"

    try:
        import psycopg

        with psycopg.connect(
            database_url,
            connect_timeout=5,
            prepare_threshold=None,
            sslmode=sslmode,
        ) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS public.agentos_public_usage (
                    window_name TEXT NOT NULL CHECK (window_name IN ('hour', 'day')),
                    window_start TIMESTAMPTZ NOT NULL,
                    requests_used INTEGER NOT NULL CHECK (requests_used >= 0),
                    PRIMARY KEY (window_name, window_start)
                )"""
            )
            connection.execute(
                "DELETE FROM public.agentos_public_usage WHERE window_start < now() - interval '8 days'"
            )
            used: dict[str, int] = {}
            for name, precision, limit in (
                ("day", "day", daily_limit),
                ("hour", "hour", hourly_limit),
            ):
                row = connection.execute(
                    """INSERT INTO public.agentos_public_usage (window_name, window_start, requests_used)
                       VALUES (%s, date_trunc(%s, now()), 1)
                       ON CONFLICT (window_name, window_start) DO UPDATE
                       SET requests_used = public.agentos_public_usage.requests_used + 1
                       WHERE agentos_public_usage.requests_used < %s
                       RETURNING requests_used""",
                    (name, precision, limit),
                ).fetchone()
                if row is None:
                    raise PublicDemoLimitReached(
                        "The public demo has reached its shared request limit for this time window. Please try again later."
                    )
                used[name] = int(row[0])
            return used
    except PublicDemoLimitReached:
        raise
    except PublicDemoError:
        raise
    except Exception as exc:
        # Log only safe classification fields; raw driver text may expose connection details.
        log_database_error(_LOGGER, "reserve_public_request", exc)
        raise PublicDemoError(
            "The public demo is temporarily unavailable because its usage database could not be reached."
        ) from exc
