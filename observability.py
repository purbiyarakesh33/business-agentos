"""Optional LangSmith tracing with strict content redaction."""
from __future__ import annotations

from functools import wraps
import os
from typing import Any, Callable, TypeVar

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # Keep offline/local component evaluation usable without the optional SDK.
    _langsmith_traceable = None

F = TypeVar("F", bound=Callable[..., Any])
_TRUE_VALUES = {"1", "true", "yes", "on"}


class _SafeTraceFailure(RuntimeError):
    """A generic trace failure that cannot carry business content."""


def configure_langsmith() -> bool:
    """Enable LangSmith only with an explicit key and opt-in setting.

    Inputs are hidden globally. Each trace wrapper also replaces inputs and outputs
    with safe metadata, so prompts, SQL, reports, and uploaded document text are
    never sent as trace payloads.
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    requested = os.getenv("LANGSMITH_TRACING", "true" if api_key else "false").strip().lower()
    if not api_key or requested not in _TRUE_VALUES or _langsmith_traceable is None:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "BusinessAgentOS")
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    return True


def _safe_trace_output(output: Any) -> dict[str, str]:
    """Keep only a run status and exception class in LangSmith output records."""
    if not isinstance(output, dict):
        return {"status": "completed"}
    safe = {"status": str(output.get("_trace_status", "completed"))[:24]}
    error_type = output.get("_trace_error_type")
    if isinstance(error_type, str):
        safe["error_type"] = error_type[:80]
    return safe


def trace_component(name: str, *, run_type: str = "chain") -> Callable[[F], F]:
    """Add an optional span while keeping data and exception messages private."""
    def decorate(function: F) -> F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if (
                _langsmith_traceable is None
                or os.getenv("LANGSMITH_TRACING", "").strip().lower() not in _TRUE_VALUES
                or not os.getenv("LANGSMITH_API_KEY", "").strip()
            ):
                return function(*args, **kwargs)

            caught: list[Exception] = []
            completed: list[Any] = []

            @_langsmith_traceable(
                name=name,
                run_type=run_type,
                project_name=os.getenv("LANGSMITH_PROJECT", "BusinessAgentOS"),
                tags=["business-agentos", run_type],
                metadata={"privacy": "redacted"},
                process_inputs=lambda _inputs: {},
                process_outputs=_safe_trace_output,
            )
            def traced_call() -> dict[str, Any]:
                try:
                    value = function(*args, **kwargs)
                    completed.append(value)
                    return {"_trace_status": "completed"}
                except Exception as exc:
                    caught.append(exc)
                    # Keep the trace's failure signal useful without retaining
                    # the original message, which may contain user or source text.
                    raise _SafeTraceFailure(
                        f"Component failed ({type(exc).__name__}); details redacted."
                    ) from None

            try:
                traced_call()
            except _SafeTraceFailure:
                if caught:
                    raise caught[0]
                if completed:
                    return completed[0]
                return function(*args, **kwargs)
            except Exception:
                # Observability must never prevent the business workflow from running.
                if caught:
                    raise caught[0]
                if completed:
                    return completed[0]
                return function(*args, **kwargs)
            if caught:
                raise caught[0]
            return completed[0] if completed else None

        return wrapped  # type: ignore[return-value]

    return decorate
