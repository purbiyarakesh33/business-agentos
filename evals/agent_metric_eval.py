"""Capture workflow traces and score the seven Business AgentOS eval metrics.

This module uses only local data and standard-library scoring. Live traces are
collected by run_workflow_smoke.py; scoring a saved trace never calls a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASET_PATH = EVAL_DIR / "agent_metric_dataset.json"
GOLD_PATH = EVAL_DIR / "golden_dataset.json"
METRICS = (
    "task_completion", "step_efficiency", "argument_correctness",
    "tool_correctness", "plan_adherence", "plan_quality", "answer_correctness",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _task_fields(task: Any) -> dict[str, Any]:
    if task is None:
        return {}
    if isinstance(task, dict):
        return {
            "task_id": task.get("task_id"), "agent": task.get("agent"),
            "data_sources": list(task.get("data_sources", [])),
            "depends_on": list(task.get("depends_on", [])),
        }
    return {
        "task_id": getattr(task, "task_id", None),
        "agent": getattr(task, "agent", None),
        "data_sources": list(getattr(task, "data_sources", []) or []),
        "depends_on": list(getattr(task, "depends_on", []) or []),
    }


def create_trace_recorder() -> Any:
    """Create a recorder that captures safe, ordered events for one workflow run."""
    class Recorder:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []
            self._sequence = 0
            self._workflow_phase: list[str] = []

        def wrap(
            self,
            module: Any,
            attribute: str,
            kind: str,
            name: str,
            arguments: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
            result: Callable[[Any], dict[str, Any]] | None = None,
        ) -> None:
            original = getattr(module, attribute)
            if getattr(original, "_agent_metric_trace_wrapper", False):
                raise RuntimeError(f"{module.__name__}.{attribute} is already instrumented")

            @wraps(original)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                self._sequence += 1
                event: dict[str, Any] = {
                    "sequence": self._sequence,
                    "kind": kind,
                    "name": name,
                    "status": "running",
                }
                event["workflow_phase"] = (
                    self._workflow_phase[-1] if self._workflow_phase else "request_setup"
                )
                if arguments is not None:
                    try:
                        event["arguments"] = arguments(args, kwargs)
                    except Exception:
                        event["arguments"] = {"summary_error": True}
                started = time.perf_counter()
                self.events.append(event)
                if kind == "workflow_step":
                    self._workflow_phase.append(name)
                try:
                    value = original(*args, **kwargs)
                except Exception as exc:
                    event["status"] = "failed"
                    event["error_type"] = type(exc).__name__
                    raise
                else:
                    event["status"] = "completed"
                    if result is not None:
                        try:
                            event["result"] = result(value)
                        except Exception:
                            event["result"] = {"summary_error": True}
                    return value
                finally:
                    if kind == "workflow_step":
                        self._workflow_phase.pop()
                    event["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)

            setattr(wrapped, "_agent_metric_trace_wrapper", True)
            setattr(module, attribute, wrapped)

    return Recorder()


def install_runtime_trace(recorder: Any) -> None:
    """Instrument the existing app boundaries without sending payloads to tracing SaaS."""
    import business_agents
    import forecasting_agent
    import jev
    import output_guardrails
    import planner

    def one_text(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        value = args[0] if args else ""
        return {"input_chars": len(value) if isinstance(value, str) else 0}

    def decision_result(value: Any) -> dict[str, Any]:
        return {
            "action": getattr(value, "action", None),
            "security": getattr(value, "security", None),
            "intent": getattr(value, "intent", None),
        }

    def plan_request_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_chars": len(args[0]) if args and isinstance(args[0], str) else 0,
            "repair": kwargs.get("repair_error") is not None,
        }

    def model_result(value: Any) -> dict[str, Any]:
        return {"model": value[1]} if isinstance(value, tuple) and len(value) > 1 else {}

    def plan_arg(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        plan = args[0] if args else None
        return {
            "plan_status": getattr(plan, "status", None),
            "approval_status": getattr(plan, "approval_status", None),
        }

    def plan_result(value: Any) -> dict[str, Any]:
        return {
            "plan_status": getattr(value, "status", None),
            "approval_status": getattr(value, "approval_status", None),
        }

    def coverage_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        question = args[0] if args and isinstance(args[0], str) else ""
        return {"question_chars": len(question)}

    def coverage_result(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"status": "within_coverage"}
        evidence = value.get("evidence", [])
        tables = sorted({
            str(table)
            for item in evidence if isinstance(item, dict)
            for table in item.get("tables_read", [])
        }) if isinstance(evidence, list) else []
        return {
            "status": value.get("status", "data_unavailable"),
            "evidence_count": len(value.get("evidence", [])) if isinstance(value.get("evidence"), list) else 0,
            "tables_read": tables,
        }

    def specialist_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        task = _task_fields(args[1] if len(args) > 1 else None)
        state = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
        task["iteration"] = state.get("iteration")
        return task

    def specialist_model_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "purpose": args[1] if len(args) > 1 else None,
            "max_tokens": args[2] if len(args) > 2 else None,
            "structured_output": len(args) > 3 and args[3] is not None,
        }

    def text_result(value: Any) -> dict[str, Any]:
        return {"response_chars": len(value)} if isinstance(value, str) else {}

    def forecast_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        observations = args[0] if args and isinstance(args[0], list) else []
        horizon = args[1] if len(args) > 1 else None
        metric = args[2] if len(args) > 2 else None
        return {"history_months": len(observations), "horizon": horizon, "metric": metric}

    def forecast_result(value: Any) -> dict[str, Any]:
        return {
            "model": value.get("model"), "backtest_folds": value.get("backtest_folds"),
            "forecast_period_count": len(value.get("periods", [])),
        }

    def action_result(value: Any) -> dict[str, Any]:
        return {"action": value.get("action")} if isinstance(value, dict) else {}

    def sql_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        task = _task_fields(args[1] if len(args) > 1 else None)
        sql = args[0] if args and isinstance(args[0], str) else ""
        return {**task, "sql": sql}

    def sql_result(value: Any) -> dict[str, Any]:
        validation = value.get("sql_verification", {}) if isinstance(value, dict) else {}
        return {
            "tables_read": value.get("tables_read", []) if isinstance(value, dict) else [],
            "row_count_returned": value.get("row_count_returned") if isinstance(value, dict) else None,
            "sql_verification": validation.get("status") if isinstance(validation, dict) else None,
        }

    def rag_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        query = args[0] if args and isinstance(args[0], str) else ""
        top_k = args[1] if len(args) > 1 else kwargs.get("top_k", 5)
        return {"query": query, "top_k": top_k}

    def retrieval_result(value: Any) -> dict[str, Any]:
        sources = sorted({
            str(item.get("source_name") or item.get("source_ref") or "")
            for item in value if isinstance(item, dict)
        }) if isinstance(value, list) else []
        return {"result_count": len(value) if isinstance(value, list) else None, "sources": sources}

    def specialist_execution_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        return _task_fields(args[1] if len(args) > 1 else None)

    def specialist_execution_result(value: Any) -> dict[str, Any]:
        items = value if isinstance(value, list) else [value]
        items = [item for item in items if isinstance(item, dict)]
        return {
            "evidence_count": len(items),
            "evidence_types": sorted({str(item.get("evidence_type", "sql" if item.get("specialist_sql") else "unknown")) for item in items}),
            "forecast_templates": sorted({str(item["forecast_template"]) for item in items if item.get("forecast_template")}),
        }

    def evidence_count(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        evidence = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        return {"evidence_count": len(evidence)}

    def verification_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        findings = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        evidence = args[2] if len(args) > 2 and isinstance(args[2], list) else []
        return {"finding_count": len(findings), "evidence_count": len(evidence)}

    def report_result(value: Any) -> dict[str, Any]:
        findings = value.get("findings", []) if isinstance(value, dict) else []
        return {"finding_count": len(findings) if isinstance(findings, list) else None}

    def verify_result(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {"issue_count": len(value.get("issues", [])) if isinstance(value.get("issues"), list) else None}
        if isinstance(value, list):
            return {"issue_count": len(value)}
        return {}

    def guard_args(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
        report = args[0] if args and isinstance(args[0], str) else ""
        evidence = args[1] if len(args) > 1 and isinstance(args[1], list) else []
        return {"report_chars": len(report), "evidence_count": len(evidence)}

    def guard_result(value: Any) -> dict[str, Any]:
        return {"action": getattr(value, "action", None)}

    def workflow_result(value: Any) -> dict[str, Any]:
        return {
            "status": value.get("status") if isinstance(value, dict) else None,
            "evidence_count": len(value.get("evidence", [])) if isinstance(value, dict) and isinstance(value.get("evidence"), list) else None,
        }

    recorder.wrap(jev, "classify_request", "model_call", "jev_classification", one_text, decision_result)
    recorder.wrap(planner, "_request_plan", "model_call", "planner_request", plan_request_args, model_result)
    recorder.wrap(planner, "approve_plan", "workflow_step", "human_approval", plan_arg, plan_result)
    recorder.wrap(business_agents, "relative_period_unavailable", "tool_call", "date_coverage_check", coverage_args, coverage_result)
    recorder.wrap(business_agents, "_featherless_text", "model_call", "specialist_model", specialist_model_args, text_result)
    recorder.wrap(business_agents, "_make_analysis_action", "workflow_step", "specialist_action", specialist_args, action_result)
    recorder.wrap(business_agents, "_execute_read_only", "tool_call", "read_only_sql", sql_args, sql_result)
    recorder.wrap(business_agents, "_retrieve_documents", "tool_call", "document_search", rag_args, retrieval_result)
    recorder.wrap(forecasting_agent, "forecast_one_series", "tool_call", "deterministic_forecast", forecast_args, forecast_result)
    recorder.wrap(business_agents, "_run_specialist", "workflow_step", "specialist_execution", specialist_execution_args, specialist_execution_result)
    recorder.wrap(business_agents, "_draft_findings", "workflow_step", "report_generation", evidence_count, report_result)
    recorder.wrap(business_agents, "_verify_findings", "workflow_step", "evidence_verification", verification_args, verify_result)
    recorder.wrap(business_agents, "execute_approved_plan", "workflow_step", "approved_workflow", lambda a, _k: plan_arg((getattr(a[1], "plan", a[1]) if len(a) > 1 else None,), {}), workflow_result)
    recorder.wrap(output_guardrails, "screen_business_output", "workflow_step", "output_guard", guard_args, guard_result)

    # app.py imports urlopen after this recorder is installed. Other workflow
    # modules above already hold their original imports, so this observes the
    # chat boundary without double-counting Jev, planner, or specialist calls.
    import urllib.request
    original_urlopen = urllib.request.urlopen

    @wraps(original_urlopen)
    def traced_urlopen(request: Any, *args: Any, **kwargs: Any) -> Any:
        title = request.get_header("X-title") if hasattr(request, "get_header") else None
        if title != "Business AgentOS Chat":
            return original_urlopen(request, *args, **kwargs)
        recorder._sequence += 1
        event = {
            "sequence": recorder._sequence, "kind": "model_call",
            "name": "general_chat_model", "status": "running",
            "arguments": {"request_title": title},
        }
        recorder.events.append(event)
        started = time.perf_counter()
        try:
            response = original_urlopen(request, *args, **kwargs)
        except Exception as exc:
            event["status"] = "failed"
            event["error_type"] = type(exc).__name__
            raise
        else:
            event["status"] = "completed"
            return response
        finally:
            event["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)

    urllib.request.urlopen = traced_urlopen


def _normalized_set_scores(expected: list[str], actual: list[str]) -> dict[str, Any]:
    gold, pred = set(expected), set(actual)
    overlap = len(gold & pred)
    precision = overlap / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = overlap / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "exact_match": float(gold == pred),
        "missing": sorted(gold - pred), "extra": sorted(pred - gold),
    }


def _fraction(passed: int, total: int) -> float:
    return passed / total if total else 1.0


def _checks_metric(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(item["passed"] is True for item in checks)
    return {
        "status": "scored", "score": round(_fraction(passed, len(checks)), 4),
        "passed": passed, "total": len(checks),
        "failures": [item["name"] for item in checks if item["passed"] is not True],
        "checks": checks,
    }


def _plan_tasks(plan: Any) -> list[dict[str, Any]]:
    tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
    return [item for item in tasks if isinstance(item, dict)] if isinstance(tasks, list) else []


def _specialist_tasks(plan: Any) -> list[dict[str, Any]]:
    return [task for task in _plan_tasks(plan) if task.get("agent") != "verification_agent"]


def _evidence_types(prediction: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in prediction.get("evidence", []):
        if not isinstance(item, dict):
            continue
        evidence_type = item.get("evidence_type")
        if evidence_type == "document_context":
            result.add("document_context")
        if evidence_type == "forecast" or item.get("forecast_template"):
            result.add("forecast")
        if isinstance(item.get("specialist_sql") or item.get("sql"), str):
            result.add("sql")
        if item.get("date_coverage"):
            result.add("date_coverage")
        if str(item.get("evidence_id", "")).startswith("dataset_coverage:"):
            result.add("date_coverage")
    for event in prediction.get("events", []):
        if not isinstance(event, dict):
            continue
        summary = event.get("result", {})
        if isinstance(summary, dict) and "forecast" in summary.get("evidence_types", []):
            result.add("forecast")
    return result


def _citation_check(answer: str, evidence: list[dict[str, Any]]) -> bool:
    ids = {str(item.get("evidence_id")) for item in evidence if isinstance(item, dict) and item.get("evidence_id")}
    cited = {
        value.strip()
        for match in re.findall(r"\[Evidence:\s*([^\]]+)\]", answer)
        for value in match.split(",")
        if value.strip()
    }
    return bool(cited) and cited.issubset(ids)


def _term_coverage(
    text: str,
    terms: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    lowered = text.casefold()
    aliases = aliases or {}
    matched: list[str] = []
    matched_as: dict[str, str] = {}
    missing: list[str] = []
    for term in terms:
        candidates = [term, *aliases.get(term, [])]
        found = next((candidate for candidate in candidates if candidate.casefold() in lowered), None)
        if found is None:
            missing.append(term)
        else:
            matched.append(term)
            if found != term:
                matched_as[term] = found
    return {
        "score": round(_fraction(len(matched), len(terms)), 4),
        "matched": matched,
        "matched_as": matched_as,
        "missing": missing,
        "total": len(terms),
    }


def _numeric_presence(text: str, values: list[Any], tolerance: float = 0.011) -> dict[str, Any]:
    tokens = re.findall(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?%?", text)
    parsed: list[Decimal] = []
    for token in tokens:
        try:
            parsed.append(Decimal(token.replace(",", "").rstrip("%")))
        except InvalidOperation:
            continue
    tol = Decimal(str(tolerance))
    missing = [value for value in values if not any(abs(Decimal(str(value)) - number) <= tol for number in parsed)]
    return {
        "expected_count": len(values), "matched_count": len(values) - len(missing),
        "recall": round(_fraction(len(values) - len(missing), len(values)), 4),
        "missing_values": missing,
    }


def _actual_tools(prediction: dict[str, Any]) -> list[str]:
    tools = {
        str(event.get("name"))
        for event in prediction.get("events", [])
        if isinstance(event, dict) and event.get("kind") == "tool_call"
    }
    for item in prediction.get("evidence", []):
        if not isinstance(item, dict):
            continue
        if item.get("action") == "clean_dataset":
            tools.add("dataset_cleaning")
        if item.get("statistical_tests"):
            tools.add("welch_test")
    return sorted(tools)


def _actual_tables(prediction: dict[str, Any]) -> list[str]:
    tables: set[str] = set()
    for event in prediction.get("events", []):
        if event.get("kind") == "tool_call":
            result = event.get("result", {})
            if isinstance(result, dict):
                tables.update(str(value) for value in result.get("tables_read", []))
    return sorted(tables)


def _workflow_events(prediction: dict[str, Any], kind: str | None = None) -> list[dict[str, Any]]:
    events = [item for item in prediction.get("events", []) if isinstance(item, dict)]
    if kind:
        events = [item for item in events if item.get("kind") == kind]
    return sorted(events, key=lambda item: item.get("sequence", 0))


def _valid_plan_contract(plan: Any) -> tuple[bool, list[str]]:
    if not isinstance(plan, dict):
        return False, ["plan_missing"]
    issues: list[str] = []
    tasks = _plan_tasks(plan)
    status = plan.get("status")
    if status == "ready":
        if plan.get("requires_approval") is not True:
            issues.append("approval_not_required")
        if plan.get("approval_status") != "pending":
            issues.append("plan_not_pending_approval")
        if not tasks:
            issues.append("ready_plan_has_no_tasks")
        if not any(task.get("agent") == "verification_agent" for task in tasks):
            issues.append("verification_task_missing")
        task_ids = {task.get("task_id") for task in tasks}
        for task in tasks:
            if not isinstance(task.get("objective"), str) or not task.get("objective", "").strip():
                issues.append(f"empty_objective:{task.get('task_id')}")
            if not isinstance(task.get("data_sources"), list):
                issues.append(f"invalid_sources:{task.get('task_id')}")
            if any(dep not in task_ids for dep in task.get("depends_on", [])):
                issues.append(f"unknown_dependency:{task.get('task_id')}")
    elif status in {"needs_clarification", "data_unavailable"}:
        if _specialist_tasks(plan):
            issues.append("non_ready_plan_has_executable_specialist")
        if plan.get("requires_approval") is not False:
            issues.append("non_ready_plan_requires_approval")
        if plan.get("approval_status") != "not_required":
            issues.append("non_ready_approval_status_invalid")
        if status == "needs_clarification" and not plan.get("missing_information"):
            issues.append("clarification_reason_missing")
        if status == "data_unavailable" and not plan.get("unavailable_data"):
            issues.append("unavailable_data_reason_missing")
    else:
        issues.append("unsupported_plan_status")
    return not issues, issues


def _check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    row = {"name": name, "passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    return row


def score_case(
    case: dict[str, Any],
    prediction: dict[str, Any],
    golden_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score one recorded workflow against its case contract and answer gold."""
    jev_decision = prediction.get("jev_decision") or {}
    jev_failed = any(
        event.get("name") == "jev_classification" and event.get("status") == "failed"
        for event in prediction.get("events", []) if isinstance(event, dict)
    )
    if jev_failed or not jev_decision.get("security") or not jev_decision.get("intent"):
        reason = (
            "Jev did not produce both a security and intent classification; "
            "the workflow did not reach the agent, so this case cannot score agent metrics."
        )
        return {
            "case_id": case["case_id"],
            "case_status": "blocked",
            "blocked_reason": reason,
            **{
                metric: {"status": "blocked", "score": None, "reason": reason}
                for metric in METRICS
            },
        }
    answer = str(prediction.get("answer", ""))
    plan_before = prediction.get("plan_before_approval")
    final_plan = prediction.get("plan")
    completion = case.get("completion", {})
    plan_spec = case.get("plan", {})
    tool_spec = case.get("tools", {})
    argument_spec = case.get("arguments", {})
    expected_tools = sorted(tool_spec.get("expected", []))
    actual_tools = _actual_tools(prediction)
    tool_events = [event for event in _workflow_events(prediction) if event.get("kind") == "tool_call"]

    # 1. Task Completion
    checks: list[dict[str, Any]] = []
    expected_status = completion.get("workflow_status")
    if expected_status is not None:
        checks.append(_check("expected_workflow_status", prediction.get("workflow_status") == expected_status,
                             {"expected": expected_status, "actual": prediction.get("workflow_status")}))
    expected_actions = completion.get("jev_actions", [])
    if expected_actions:
        actual_action = (prediction.get("jev_decision") or {}).get("action")
        checks.append(_check("expected_jev_action", actual_action in expected_actions,
                             {"expected_any_of": expected_actions, "actual": actual_action}))
    expected_plan_status = completion.get("plan_status")
    if expected_plan_status:
        actual_status = (plan_before or {}).get("status")
        checks.append(_check("ready_plan_produced", actual_status == expected_plan_status,
                             {"expected": expected_plan_status, "actual": actual_status}))
    if completion.get("requires_answer"):
        checks.append(_check("nonempty_answer", bool(answer.strip())))
    expected_guard = completion.get("output_guard")
    if expected_guard:
        checks.append(_check("output_guard_result", prediction.get("output_guard_action") == expected_guard,
                             {"expected": expected_guard, "actual": prediction.get("output_guard_action")}))
    evidence_types = _evidence_types(prediction)
    for evidence_type in completion.get("evidence_types", []):
        checks.append(_check("evidence:" + evidence_type, evidence_type in evidence_types,
                             {"actual": sorted(evidence_types)}))
    if completion.get("require_citations"):
        checks.append(_check("citations_resolve", _citation_check(answer, prediction.get("evidence", []))))
    answer_term_check = _term_coverage(answer, completion.get("answer_terms", []))
    if completion.get("answer_terms"):
        checks.append(_check("requested_content_markers", answer_term_check["score"] == 1.0,
                             answer_term_check))
    if completion.get("must_not_create_plan"):
        checks.append(_check("no_plan_created", not isinstance(final_plan, dict)))
    if completion.get("must_not_use_tools"):
        checks.append(_check("no_tools_used", not tool_events))
    task_completion = _checks_metric(checks)

    # 2. Step Efficiency: count steps and penalize redundant/failed/over-budget work.
    events = _workflow_events(prediction)
    countable = [event for event in events if event.get("kind") in {"model_call", "tool_call", "workflow_step"}]
    model_calls = [event for event in countable if event.get("kind") == "model_call"]
    calls = [event for event in countable if event.get("kind") == "tool_call"]
    duplicate_keys: Counter[str] = Counter()
    duplicate_count = 0
    for event in calls:
        # Identical calls in different workflow phases can be intentional:
        # coverage is rechecked after approval and deterministic results are
        # independently recomputed during evidence verification. Only flag a
        # repeat inside the same phase as redundant work.
        key = json.dumps(
            [event.get("workflow_phase"), event.get("name"), event.get("arguments")],
            sort_keys=True,
            ensure_ascii=False,
        )
        duplicate_keys[key] += 1
        if duplicate_keys[key] > 1:
            duplicate_count += 1
    failed_calls = sum(event.get("status") != "completed" for event in [*model_calls, *calls])
    planner_repairs = sum(
        event.get("name") == "planner_request"
        and isinstance(event.get("arguments"), dict)
        and event["arguments"].get("repair") is True
        for event in model_calls
    )
    budgets = case.get("efficiency", {})
    over_model = max(0, len(model_calls) - int(budgets.get("max_model_calls", len(model_calls))))
    over_tools = max(0, len(calls) - int(budgets.get("max_tool_calls", len(calls))))
    over_steps = max(0, len(countable) - int(budgets.get("max_total_steps", len(countable))))
    over_repairs = max(0, planner_repairs - int(budgets.get("max_planner_repairs", planner_repairs)))
    unnecessary = duplicate_count + failed_calls + over_model + over_tools + over_steps + over_repairs
    step_efficiency = {
        "status": "scored",
        "score": round(max(0.0, 1.0 - unnecessary / max(1, len(countable))), 4),
        "passed": unnecessary == 0,
        "total_steps": len(countable),
        "model_calls": len(model_calls),
        "tool_calls": len(calls),
        "failed_calls": failed_calls,
        "duplicate_tool_calls": duplicate_count,
        "planner_repairs": planner_repairs,
        "over_budget": {"model_calls": over_model, "tool_calls": over_tools,
                        "total_steps": over_steps, "planner_repairs": over_repairs},
        "unnecessary_step_penalties": unnecessary,
        "wall_time_ms": prediction.get("wall_time_ms"),
    }

    # 3. Argument Correctness
    argument_checks: list[dict[str, Any]] = []
    sql_events = [event for event in tool_events if event.get("name") == "read_only_sql"]
    rag_events = [event for event in tool_events if event.get("name") == "document_search"]
    all_sql = "\n".join(
        str((event.get("arguments") or {}).get("sql", "")) for event in sql_events
    )
    expected_tables = set(tool_spec.get("required_tables", []))
    actual_tables = set(_actual_tables(prediction))
    if expected_tables:
        argument_checks.append(_check("required_tables_read", expected_tables.issubset(actual_tables),
                                      {"required": sorted(expected_tables), "actual": sorted(actual_tables)}))
    if "read_only_sql" in expected_tools:
        argument_checks.append(_check("sql_tool_was_called", bool(sql_events)))
        for index, event in enumerate(sql_events, 1):
            args = event.get("arguments", {})
            result = event.get("result", {})
            argument_checks.append(_check(f"sql_call_{index}_has_arguments", bool(args.get("sql"))))
            if argument_spec.get("require_sql_verification"):
                argument_checks.append(_check(f"sql_call_{index}_verified", result.get("sql_verification") == "passed",
                                              result.get("sql_verification")))
            task_sources = {
                str(source)
                for task in _plan_tasks(plan_before)
                if task.get("task_id") == args.get("task_id")
                for source in task.get("data_sources", [])
            }
            observed_tables = set(result.get("tables_read", []))
            if task_sources:
                argument_checks.append(_check(f"sql_call_{index}_within_approved_sources",
                                              observed_tables.issubset(task_sources),
                                              {"approved": sorted(task_sources), "read": sorted(observed_tables)}))
    for pattern in argument_spec.get("required_sql_patterns", []):
        argument_checks.append(_check("sql_argument_pattern:" + pattern,
                                      bool(re.search(pattern, all_sql, re.IGNORECASE)), pattern))
    for pattern in argument_spec.get("forbidden_sql_patterns", []):
        argument_checks.append(_check("forbidden_sql_pattern_absent:" + pattern,
                                      not bool(re.search(pattern, all_sql, re.IGNORECASE)), pattern))
    rag_query = "\n".join(str((event.get("arguments") or {}).get("query", "")) for event in rag_events)
    for term in argument_spec.get("required_rag_terms", []):
        argument_checks.append(_check("retrieval_argument:" + term, term.casefold() in rag_query.casefold(), term))
    if rag_events and argument_spec.get("max_rag_top_k") is not None:
        max_k = int(argument_spec["max_rag_top_k"])
        argument_checks.append(_check("retrieval_top_k_within_limit",
                                      all((event.get("arguments") or {}).get("top_k", max_k + 1) <= max_k for event in rag_events),
                                      {"maximum": max_k,
                                       "actual": [(event.get("arguments") or {}).get("top_k") for event in rag_events]}))
    if argument_spec.get("require_forecast_template"):
        forecast_templates = [item.get("forecast_template") for item in prediction.get("evidence", [])
                              if isinstance(item, dict) and item.get("forecast_template")]
        argument_checks.append(_check("forecast_template_used", bool(forecast_templates), forecast_templates))
    if not expected_tools and not argument_checks:
        argument_checks.append(_check("no_tool_arguments_expected", not tool_events))
    argument_correctness = _checks_metric(argument_checks)

    # 4. Tool Correctness
    actual_agents = sorted({
        str((event.get("arguments") or {}).get("agent"))
        for event in events
        if event.get("name") == "specialist_execution"
        and isinstance(event.get("arguments"), dict)
        and event["arguments"].get("agent")
    })
    expected_agents = sorted(plan_spec.get("agents", [])) if not plan_spec.get("not_applicable") else []
    actual_source_tables = sorted(actual_tables)
    expected_source_tables = sorted(tool_spec.get("required_tables", []))
    tool_scores = _normalized_set_scores(expected_tools, actual_tools)
    agent_scores = _normalized_set_scores(expected_agents, actual_agents)
    source_scores = _normalized_set_scores(expected_source_tables, actual_source_tables)
    tool_components = [tool_scores["exact_match"], agent_scores["exact_match"]]
    if expected_source_tables or actual_source_tables:
        tool_components.append(source_scores["f1"])
    tool_correctness = {
        "status": "scored",
        "score": round(sum(tool_components) / len(tool_components), 4),
        "passed": all(value == 1.0 for value in tool_components),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "tool_set_scores": tool_scores,
        "expected_agents": expected_agents,
        "executed_agents": actual_agents,
        "agent_set_scores": agent_scores,
        "expected_tables": expected_source_tables,
        "actual_tables": actual_source_tables,
        "table_set_scores": source_scores,
    }

    # 5. Plan Adherence
    adherence_checks: list[dict[str, Any]] = []
    if completion.get("must_not_create_plan"):
        adherence_checks.append(_check("did_not_plan_disallowed_request", not isinstance(final_plan, dict)))
    if completion.get("must_not_use_tools"):
        adherence_checks.append(_check("did_not_use_tools_for_disallowed_request", not tool_events))
    if completion.get("plan_status") == "ready":
        approval_events = [event for event in events if event.get("name") == "human_approval"]
        first_execution = next((event for event in events if event.get("name") in {"specialist_execution", "read_only_sql", "document_search"}), None)
        approval_seq = min((event.get("sequence", 10**9) for event in approval_events), default=10**9)
        adherence_checks.extend([
            _check("plan_was_pending_before_approval",
                   isinstance(plan_before, dict) and plan_before.get("status") == "ready"
                   and plan_before.get("approval_status") == "pending"),
            _check("user_approval_was_recorded", bool(prediction.get("approval_clicked")) and bool(approval_events)),
            _check("no_evidence_before_approval", prediction.get("preapproval_evidence_count") == 0),
            _check("approval_preceded_execution", first_execution is None or approval_seq < first_execution.get("sequence", 0)),
            _check("executed_plan_is_approved", isinstance(final_plan, dict) and final_plan.get("approval_status") == "approved"),
        ])
        planned = _specialist_tasks(final_plan)
        proposed = _specialist_tasks(plan_before)
        def task_signature(task: dict[str, Any]) -> tuple[Any, ...]:
            return (
                task.get("task_id"), task.get("agent"), task.get("objective"),
                tuple(sorted(task.get("data_sources", []))),
                tuple(task.get("depends_on", [])),
                tuple(task.get("evidence_required", [])),
            )
        adherence_checks.append(_check(
            "approved_plan_matches_proposed_specialist_tasks",
            [task_signature(task) for task in proposed] == [task_signature(task) for task in planned],
        ))
        planned_ids = [str(task.get("task_id")) for task in planned]
        execution_events = [event for event in events if event.get("name") == "specialist_execution"]
        executed_ids = [str((event.get("arguments") or {}).get("task_id")) for event in execution_events]
        adherence_checks.append(_check("planned_specialists_executed_once_in_order",
                                       executed_ids == planned_ids,
                                       {"planned": planned_ids, "executed": executed_ids}))
        seen: set[str] = set()
        dep_ok = True
        for task in planned:
            task_id = str(task.get("task_id"))
            if any(str(dep) not in seen for dep in task.get("depends_on", [])):
                dep_ok = False
            seen.add(task_id)
        adherence_checks.append(_check("task_dependencies_respected", dep_ok))
    if not adherence_checks:
        adherence = {"status": "not_applicable", "score": None, "reason": "No execution-plan contract applies to this case."}
    else:
        adherence = _checks_metric(adherence_checks)

    # 6. Plan Quality
    if plan_spec.get("not_applicable"):
        plan_quality = {"status": "not_applicable", "score": None, "reason": "Case has no expected executable plan."}
    else:
        plan_checks: list[dict[str, Any]] = []
        valid, contract_issues = _valid_plan_contract(plan_before)
        plan_checks.append(_check("plan_contract_valid", valid, contract_issues))
        plan_agents = sorted({str(task.get("agent")) for task in _specialist_tasks(plan_before) if task.get("agent")})
        plan_sources = sorted({str(source) for task in _specialist_tasks(plan_before) for source in task.get("data_sources", [])})
        agent_set = _normalized_set_scores(sorted(plan_spec.get("agents", [])), plan_agents)
        required_sources = sorted(plan_spec.get("required_sources", []))
        source_set = _normalized_set_scores(required_sources, plan_sources)
        objectives = " ".join(
            str(value)
            for task in _specialist_tasks(plan_before)
            for value in [task.get("objective", ""), *(task.get("evidence_required", []) or [])]
        )
        objective_coverage = _term_coverage(
            objectives,
            plan_spec.get("objective_terms", []),
            plan_spec.get("objective_term_aliases", {}),
        )
        specialists = _specialist_tasks(plan_before)
        max_tasks = int(plan_spec.get("max_specialist_tasks", len(specialists) or 1))
        minimality = 1.0 if len(specialists) <= max_tasks else max_tasks / max(1, len(specialists))
        plan_checks.extend([
            _check("agent_route_exact", agent_set["exact_match"] == 1.0, agent_set),
            _check("required_sources_selected_without_extras", source_set["exact_match"] == 1.0, source_set),
            _check("plan_objectives_cover_request", objective_coverage["score"] == 1.0, objective_coverage),
            _check("plan_is_minimal", minimality == 1.0, {"specialist_tasks": len(specialists), "maximum": max_tasks}),
        ])
        scored_checks = _checks_metric(plan_checks)
        plan_quality = {
            **scored_checks,
            "planned_agents": plan_agents,
            "planned_sources": plan_sources,
            "agent_set_scores": agent_set,
            "source_set_scores": source_set,
            "objective_coverage": objective_coverage,
            "minimality": round(minimality, 4),
        }

    # 7. Answer Correctness
    answer_checks: list[dict[str, Any]] = []
    answer_spec = case.get("answer", {})
    if answer_spec.get("golden_case_id"):
        gold_case = golden_by_id.get(answer_spec["golden_case_id"])
        if not gold_case:
            answer_checks.append(_check("linked_answer_gold_exists", False, answer_spec["golden_case_id"]))
            expected_facts: list[Any] = []
        else:
            expected_facts = list(gold_case.get("expected", {}).get("numeric_facts", []))
            if expected_facts:
                numeric = _numeric_presence(answer, expected_facts)
                answer_checks.append(_check("expected_numeric_facts_present", numeric["recall"] == 1.0, numeric))
    else:
        expected_facts = []
    answer_terms = list(dict.fromkeys([*answer_spec.get("required_terms", []), *completion.get("answer_terms", [])]))
    if answer_terms:
        coverage = _term_coverage(answer, answer_terms)
        answer_checks.append(_check("required_answer_markers_present", coverage["score"] == 1.0, coverage))
    forbidden_matches = [pattern for pattern in answer_spec.get("forbidden_patterns", []) if re.search(pattern, answer)]
    answer_checks.append(_check("forbidden_claims_absent", not forbidden_matches, forbidden_matches))
    answer_checks.append(_check("answer_nonempty", bool(answer.strip())))
    if completion.get("require_citations"):
        answer_checks.append(_check("answer_citations_resolve", _citation_check(answer, prediction.get("evidence", []))))
    answer_correctness = _checks_metric(answer_checks)

    return {
        "case_id": case["case_id"],
        "task_completion": task_completion,
        "step_efficiency": step_efficiency,
        "argument_correctness": argument_correctness,
        "tool_correctness": tool_correctness,
        "plan_adherence": adherence,
        "plan_quality": plan_quality,
        "answer_correctness": answer_correctness,
    }


def validate_dataset(dataset: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        return ["agent metric dataset cases must be a list"]
    if dataset.get("metrics") != list(METRICS):
        errors.append("dataset metrics must list the seven supported metrics in order")
    gold_by_id = {case.get("case_id"): case for case in golden.get("cases", []) if isinstance(case, dict)}
    seen_ids: set[str] = set()
    seen_scenarios: set[str] = set()
    for index, case in enumerate(cases):
        where = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{where} must be an object")
            continue
        case_id = case.get("case_id")
        scenario = case.get("scenario")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            errors.append(f"{where} has a missing or duplicate case_id")
        else:
            seen_ids.add(case_id)
        if not isinstance(scenario, str) or not scenario or scenario in seen_scenarios:
            errors.append(f"{case_id}: missing or duplicate scenario")
        else:
            seen_scenarios.add(scenario)
        if case.get("split") not in {"development", "holdout"}:
            errors.append(f"{case_id}: split must be development or holdout")
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            errors.append(f"{case_id}: query is required")
        for key in ("completion", "plan", "tools", "arguments", "efficiency", "answer"):
            if not isinstance(case.get(key), dict):
                errors.append(f"{case_id}: {key} must be an object")
        if isinstance(case.get("plan"), dict):
            aliases = case["plan"].get("objective_term_aliases", {})
            if not isinstance(aliases, dict) or any(
                not isinstance(term, str)
                or not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                for term, values in aliases.items()
            ):
                errors.append(f"{case_id}: objective_term_aliases must map terms to nonempty string lists")
        golden_id = case.get("answer", {}).get("golden_case_id") if isinstance(case.get("answer"), dict) else None
        if golden_id:
            gold_case = gold_by_id.get(golden_id)
            if (not gold_case or gold_case.get("component") != "answer_quality"
                    or gold_case.get("split") != case.get("split")):
                errors.append(f"{case_id}: answer gold must reference an answer_quality case from the same split")
    return errors


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prediction JSON at line {line_number}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("case_id"), str):
            raise ValueError(f"Prediction line {line_number} needs a string case_id")
        if value["case_id"] in predictions:
            raise ValueError(f"Duplicate prediction case_id: {value['case_id']}")
        predictions[value["case_id"]] = value
    return predictions


def score_predictions(
    dataset: dict[str, Any], golden: dict[str, Any], predictions: dict[str, dict[str, Any]],
    split: str = "development", *, use_judge: bool = False,
) -> dict[str, Any]:
    gold_by_id = {
        case["case_id"]: case
        for case in golden.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    cases = [case for case in dataset["cases"] if case.get("split") == split]
    scored: list[dict[str, Any]] = []
    for case in cases:
        prediction = predictions.get(case["case_id"])
        if prediction is None:
            continue
        row = score_case(case, prediction, gold_by_id)
        if use_judge and row.get("case_status") != "blocked" and case.get("answer", {}).get("golden_case_id"):
            gold = gold_by_id.get(case["answer"]["golden_case_id"])
            try:
                from evals.score_components import judge
                judged = judge(gold, {
                    "answer": prediction.get("answer", ""),
                    "evidence": json.dumps(prediction.get("evidence", []), ensure_ascii=False),
                })
                row["answer_quality_judge"] = {"status": "scored", **judged}
            except Exception as exc:
                row["answer_quality_judge"] = {
                    "status": "judge_error", "error_type": type(exc).__name__,
                }
        scored.append(row)
    summary: dict[str, Any] = {}
    for metric in METRICS:
        rows = [row[metric] for row in scored if row[metric].get("status") == "scored"]
        values = [row["score"] for row in rows if isinstance(row.get("score"), (int, float))]
        summary[metric] = {
            "applicable_cases": len(rows),
            "mean_score": round(sum(values) / len(values), 4) if values else None,
            "perfect_case_rate": round(sum(value == 1.0 for value in values) / len(values), 4) if values else None,
        }
    if use_judge:
        judge_rows = [row.get("answer_quality_judge", {}) for row in scored]
        valid_judges = [row for row in judge_rows if row.get("status") == "scored"]
        summary["answer_quality_judge"] = {
            "scored_cases": len(valid_judges),
            "judge_errors": sum(row.get("status") == "judge_error" for row in judge_rows),
            "pass_rate": round(_fraction(sum(row.get("overall_pass") is True for row in valid_judges), len(valid_judges)), 4)
            if valid_judges else None,
            "interpretation": "Supplementary model-judge signal; review disagreements and sample agreements.",
        }
    expected_ids = {case["case_id"] for case in cases}
    return {
        "dataset_version": dataset.get("dataset_version"),
        "split": split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unique_scenario_count": len(cases),
        "prediction_count": len(predictions),
        "case_result_count": len(scored),
        "scored_case_count": sum(
            any(row[metric].get("status") == "scored" for metric in METRICS)
            for row in scored
        ),
        "blocked_case_count": sum(row.get("case_status") == "blocked" for row in scored),
        "missing_case_ids": sorted(expected_ids - set(predictions)),
        "unexpected_case_ids": sorted(set(predictions) - expected_ids),
        "total_metric_case_evaluations": sum(
            row["applicable_cases"] for row in summary.values()
            if isinstance(row, dict) and isinstance(row.get("applicable_cases"), int)
        ),
        "metric_summary": summary,
        "cases": scored,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate the workflow dataset and its development gold references.")
    parser.add_argument("--predictions", type=Path, help="JSONL traces recorded by run_workflow_smoke.py")
    parser.add_argument("--output", type=Path, help="Optional JSON scorecard path")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--judge", action="store_true", help="Opt in to billable Featherless answer-quality judging for linked gold cases.")
    args = parser.parse_args()
    dataset = read_json(DATASET_PATH)
    golden = read_json(GOLD_PATH)
    errors = validate_dataset(dataset, golden)
    if errors:
        print("DATASET INVALID:")
        for error in errors:
            print(" -", error)
        return 1
    if args.validate or not args.predictions:
        print(f"Agent metric dataset {dataset.get('dataset_version')}: {len(dataset['cases'])} cases")
        print("Metrics:", ", ".join(dataset.get("metrics", [])))
        print("DATASET VALID")
        if not args.predictions:
            return 0
    if not any(case.get("split") == args.split for case in dataset.get("cases", [])):
        raise ValueError(f"The agent metric dataset has no {args.split} cases; no scorecard was generated.")
    predictions = load_predictions(args.predictions)
    result = score_predictions(dataset, golden, predictions, args.split, use_judge=args.judge)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved scorecard: {args.output}")
    print(
        f"Scored {result['scored_case_count']} of {len(result['missing_case_ids']) + result['case_result_count']} "
        f"{args.split} cases; blocked={result['blocked_case_count']}"
    )
    for metric, row in result["metric_summary"].items():
        if metric == "answer_quality_judge":
            print(f"{metric}: pass_rate={row['pass_rate']} n={row['scored_cases']}")
        else:
            print(f"{metric}: mean={row['mean_score']} perfect={row['perfect_case_rate']} cases={row['applicable_cases']}")
    print(
        "Total metric-case evaluations:", result["total_metric_case_evaluations"],
        "across", result["unique_scenario_count"], "unique scenario(s)",
    )
    if result["missing_case_ids"]:
        print("Missing cases:", ", ".join(result["missing_case_ids"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("Agent eval error: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
