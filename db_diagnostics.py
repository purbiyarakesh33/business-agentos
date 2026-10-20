"""Safe database error diagnostics: never log raw driver messages or credentials."""
from __future__ import annotations

import logging
from typing import Any


def log_database_error(logger: logging.Logger, operation: str, exc: BaseException) -> None:
    """Log only a coarse category and SQLSTATE, never a raw exception/DSN."""
    message = str(exc).casefold()
    raw_state: Any = getattr(exc, "sqlstate", None)
    sqlstate = raw_state if isinstance(raw_state, str) and len(raw_state) == 5 and raw_state.isalnum() else "unknown"

    if "password authentication failed" in message or "authentication failed" in message:
        category = "authentication_failed"
    elif "no pg_hba.conf entry" in message:
        category = "server_access_rejected"
    elif "could not translate host name" in message or "name or service not known" in message:
        category = "dns_resolution_failed"
    elif "connection timed out" in message or "timeout expired" in message or "timed out" in message:
        category = "connection_timed_out"
    elif "connection refused" in message:
        category = "connection_refused"
    elif "network is unreachable" in message or "no route to host" in message:
        category = "network_unreachable"
    elif "role" in message and "does not exist" in message:
        category = "database_role_missing"
    elif "database" in message and "does not exist" in message:
        category = "database_missing"
    elif "permission denied" in message or "must be owner" in message or "insufficient privilege" in message:
        category = "permission_denied"
    elif "extension" in message and "does not exist" in message:
        category = "extension_missing"
    elif sqlstate.startswith("28"):
        category = "authentication_or_authorization"
    elif sqlstate == "42501":
        category = "permission_denied"
    elif sqlstate == "3D000":
        category = "database_missing"
    elif sqlstate.startswith("08"):
        category = "connection_error"
    else:
        category = "unclassified_database_error"

    logger.error(
        "Database operation failed: operation=%s exception_type=%s sqlstate=%s category=%s",
        operation,
        type(exc).__name__,
        sqlstate,
        category,
    )