"""Offline contract checks for public-demo request caps and upload isolation."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from public_safety import PublicDemoError, PublicDemoLimitReached, reserve_public_request
import rag
from output_guardrails import screen_business_output


def check_local_mode_needs_no_usage_database() -> None:
    with patch.dict(os.environ, {"PUBLIC_DEMO_MODE": "false"}, clear=False):
        os.environ.pop("RAG_DATABASE_URL", None)
        assert reserve_public_request() is None
    print("PASS local mode does not require hosted usage storage")


def check_public_mode_fails_closed_without_database() -> None:
    with patch.dict(os.environ, {"PUBLIC_DEMO_MODE": "true"}, clear=False):
        os.environ.pop("RAG_DATABASE_URL", None)
        try:
            reserve_public_request()
        except PublicDemoError:
            pass
        else:
            raise AssertionError("public request was allowed without the quota database")
    print("PASS public mode holds model calls when usage storage is unavailable")


def check_quota_is_atomic() -> None:
    counts = {"day": 0, "hour": 0}

    class Cursor:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __enter__(self):
            self.before = counts.copy()
            return self

        def __exit__(self, kind, value, traceback):
            if kind is not None:
                counts.update(self.before)

        def execute(self, query, params=None):
            if query.lstrip().startswith("INSERT INTO public.agentos_public_usage"):
                name, _precision, limit = params
                if counts[name] >= limit:
                    return Cursor()
                counts[name] += 1
                return Cursor((counts[name],))
            return Cursor()

    fake_driver = types.ModuleType("psycopg")
    fake_driver.connect = lambda *args, **kwargs: Connection()
    settings = {
        "PUBLIC_DEMO_MODE": "true",
        "RAG_DATABASE_URL": "postgresql://db.example/test",
        "PUBLIC_DAILY_REQUEST_LIMIT": "5",
        "PUBLIC_HOURLY_REQUEST_LIMIT": "1",
    }
    with patch.dict(os.environ, settings, clear=False), patch.dict(sys.modules, {"psycopg": fake_driver}):
        assert reserve_public_request() == {"day": 1, "hour": 1}
        try:
            reserve_public_request()
        except PublicDemoLimitReached:
            pass
        else:
            raise AssertionError("hourly quota was not enforced")
    assert counts == {"day": 1, "hour": 1}, counts
    print("PASS hourly cap is atomic and rolls back the daily increment")


def check_public_uploads_are_parsed_in_memory() -> None:
    uploaded = type(
        "Upload",
        (),
        {"name": "private-note.txt", "getvalue": lambda self: b"Only this session should retrieve this note."},
    )()
    sources, count = rag._uploaded_document_sources([uploaded])
    assert count == 1
    source_type, source_name = next(iter(sources))
    assert source_type == "company_document"
    assert source_name == "session_uploads/private-note.txt"
    assert sources[(source_type, source_name)]["chunks"][0]["content"] == "Only this session should retrieve this note."
    print("PASS uploaded text is parsed in memory into a session source")


def check_output_guard_blocks_hidden_prompt_disclosure() -> None:
    leaked = "You are a friendly general assistant in a business application."
    assert screen_business_output(leaked).action == "BLOCK"
    assert screen_business_output("The report cites the uploaded document.").action == "ALLOW"
    print("PASS output screen blocks system-prompt markers")


def main() -> int:
    check_local_mode_needs_no_usage_database()
    check_public_mode_fails_closed_without_database()
    check_quota_is_atomic()
    check_public_uploads_are_parsed_in_memory()
    check_output_guard_blocks_hidden_prompt_disclosure()
    print("Public safety contracts: 5/5 passed. No network/API was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
