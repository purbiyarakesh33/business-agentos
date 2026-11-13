"""Exercise the Streamlit request workflow against live configured services.

Each scenario sends one request to Jev. Business planning and ordinary chat can
also call Featherless. Business scenarios simulate the explicit human approval
click in Streamlit's test harness and use synthetic Cedar & Stone demo data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.agent_metric_eval import create_trace_recorder, install_runtime_trace, read_json, score_case

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DATASET = read_json(EVAL_DIR / "agent_metric_dataset.json")
GOLD_DATASET = read_json(EVAL_DIR / "golden_dataset.json")
AGENT_CASES = {case["scenario"]: case for case in AGENT_DATASET["cases"]}
GOLD_BY_ID = {case["case_id"]: case for case in GOLD_DATASET["cases"] if isinstance(case, dict)}

SCENARIOS: dict[str, dict[str, Any]] = {}
for _scenario, _case in AGENT_CASES.items():
    _expected_agents = _case.get("plan", {}).get("agents", [])
    _expected_sources = _case.get("plan", {}).get("required_sources", [])
    _expected_tools = _case.get("tools", {}).get("expected", [])
    SCENARIOS[_scenario] = {
        "query": _case["query"],
        "agent": _expected_agents[0] if _expected_agents else None,
        "sources": set(_expected_sources),
        "requires_sql": "read_only_sql" in _expected_tools,
        "requires_rag": "document_search" in _expected_tools,
    }

_CURRENT_PREDICTION: dict[str, Any] | None = None
_CURRENT_CASE: dict[str, Any] | None = None
_CURRENT_RECORDER: Any = None
_RECORD_PATH: Path | None = None
_APPEND_RECORD = False
_FINALIZED = False
_RUN_STARTED = 0.0


def install_sql_trace() -> None:
    """Print SQL and SQLite errors from this smoke run only; never enabled by default."""
    import business_agents

    original = business_agents._execute_read_only

    def traced(sql: str, task: Any) -> dict[str, Any]:
        try:
            result = original(sql, task)
        except Exception as exc:
            print(f"SQL_ERROR task={getattr(task, 'task_id', '?')}: {' '.join(sql.split())[:3000]}")
            print(f"SQL_ERROR detail: {type(exc).__name__}: {' '.join(str(exc).split())[:300]}")
            raise
        print(f"SQL_OK task={getattr(task, 'task_id', '?')} rows={result.get('row_count_returned')}")
        return result

    business_agents._execute_read_only = traced


def install_findings_trace() -> None:
    """Print rejected report claims and compact cited evidence metadata for diagnosis."""
    import business_agents

    original = business_agents._validate_findings

    def traced(raw: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            return original(raw, evidence)
        except business_agents.BusinessAgentError as exc:
            findings = raw.get("findings", []) if isinstance(raw, dict) else []
            for finding in findings if isinstance(findings, list) else []:
                if not isinstance(finding, dict):
                    continue
                refs = finding.get("evidence_ids", [])
                refs = refs if isinstance(refs, list) else []
                cited = [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "evidence_type": item.get("evidence_type"),
                        "has_sql": bool(item.get("specialist_sql") or item.get("sql")),
                        "excerpt": str(item.get("excerpt", ""))[:240],
                    }
                    for item in evidence
                    if item.get("evidence_id") in refs
                ]
                print("REJECTED_FINDING:", json.dumps({
                    "claim": finding.get("claim"),
                    "evidence_ids": refs,
                    "cited_evidence": cited,
                    "validator_error": str(exc),
                }, ensure_ascii=False))
            raise

    business_agents._validate_findings = traced


def _plan_dict(value: Any) -> dict[str, Any] | None:
    if is_dataclass(value):
        return asdict(value)
    return value if isinstance(value, dict) else None


def _safe_evidence(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    safe: list[dict[str, Any]] = []
    allowed = {
        "evidence_id", "task_id", "agent", "objective", "evidence_type", "source_ref",
        "source_name", "excerpt", "sql", "specialist_sql", "tables_read", "columns",
        "row_count_returned", "truncated", "query_template", "forecast_template",
        "date_coverage", "metric_label", "dimension_label", "forecast_status",
        "forecast_model", "analysis_method", "action", "statistical_tests",
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        safe.append({key: item[key] for key in allowed if key in item})
    return safe


def _finalize_prediction() -> None:
    global _FINALIZED
    if _FINALIZED or _CURRENT_PREDICTION is None or _CURRENT_CASE is None:
        return
    _FINALIZED = True
    _CURRENT_PREDICTION["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _CURRENT_PREDICTION["wall_time_ms"] = round((time.perf_counter() - _RUN_STARTED) * 1000, 2)
    if _CURRENT_RECORDER is not None:
        _CURRENT_PREDICTION["events"] = sorted(
            _CURRENT_RECORDER.events, key=lambda event: event.get("sequence", 0)
        )
    _CURRENT_PREDICTION["metrics"] = score_case(_CURRENT_CASE, _CURRENT_PREDICTION, GOLD_BY_ID)
    print("Agent metric results:")
    for metric in (
        "task_completion", "step_efficiency", "argument_correctness", "tool_correctness",
        "plan_adherence", "plan_quality", "answer_correctness",
    ):
        row = _CURRENT_PREDICTION["metrics"][metric]
        score = row.get("score")
        print(f"- {metric}: {score if score is not None else row.get('status')}")
    if _RECORD_PATH is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if _APPEND_RECORD else "x"
        with _RECORD_PATH.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_CURRENT_PREDICTION, ensure_ascii=False) + "\n")
        print(f"Saved trace: {_RECORD_PATH}")


def fail(message: str) -> int:
    print("FAIL: " + message, file=sys.stderr)
    if _CURRENT_PREDICTION is not None:
        _CURRENT_PREDICTION["failure"] = message
    _finalize_prediction()
    return 1


def visible_messages(at: AppTest) -> list[dict[str, Any]]:
    return at.session_state["visible_messages"]


def main() -> int:
    global _CURRENT_PREDICTION, _CURRENT_CASE, _CURRENT_RECORDER
    global _RECORD_PATH, _APPEND_RECORD, _FINALIZED, _RUN_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--trace-sql", action="store_true", help="Print generated SQL and SQLite errors for this diagnostic run.")
    parser.add_argument("--trace-findings", action="store_true", help="Print rejected report claims and cited evidence metadata for diagnosis.")
    parser.add_argument("--record", type=Path, help="Append this run's structured trace to a JSONL file.")
    parser.add_argument("--append", action="store_true", help="Append to an existing --record file.")
    args = parser.parse_args()
    if args.append and not args.record:
        parser.error("--append requires --record")
    if args.record and args.record.exists() and not args.append:
        parser.error("--record file already exists; pass --append to add another case")
    if args.append and args.record and not args.record.is_file():
        parser.error("--append requires an existing --record file")
    scenario = SCENARIOS[args.scenario]
    case = AGENT_CASES[args.scenario]
    _CURRENT_CASE = case
    _CURRENT_RECORDER = create_trace_recorder()
    _RECORD_PATH = args.record
    _APPEND_RECORD = args.append
    _FINALIZED = False
    _RUN_STARTED = time.perf_counter()
    _CURRENT_PREDICTION = {
        "case_id": case["case_id"], "scenario": args.scenario, "split": case["split"],
        "query": case["query"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_status": None, "jev_decision": None, "plan_before_approval": None,
        "plan": None, "preapproval_evidence_count": None, "approval_clicked": False,
        "output_guard_action": None, "answer": "", "evidence": [], "events": [],
    }
    install_runtime_trace(_CURRENT_RECORDER)
    if args.trace_sql:
        install_sql_trace()
    if args.trace_findings:
        install_findings_trace()
    query = scenario["query"]

    try:
        # Streamlit 1.53's AppTest chat-input widget does not populate the
        # file-upload protobuf field expected by chat_input when accept_file is
        # enabled. These text-only benchmark scenarios don't attach PDFs.
        os.environ["BUSINESS_AGENTOS_EVAL_MODE"] = "1"
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=args.timeout).run()
        if at.exception:
            return fail("app startup raised a Streamlit exception")
        at.chat_input[0].set_value(query).run()
        if at.exception:
            return fail("request submission raised a Streamlit exception")
        messages = visible_messages(at)
        if len(messages) < 2:
            return fail("request did not produce a user/assistant message pair")
        user, assistant = messages[-2], messages[-1]
        decision = user.get("jev")
        if not isinstance(decision, dict):
            return fail("Jev decision was not persisted on the user message")
        _CURRENT_PREDICTION["jev_decision"] = decision
        if not decision.get("security") or not decision.get("intent"):
            return fail("Jev did not produce security and intent labels; workflow scoring is blocked")

        if args.scenario == "injection":
            _CURRENT_PREDICTION["workflow_status"] = assistant.get("_workflow_status")
            _CURRENT_PREDICTION["plan"] = _plan_dict(assistant.get("_plan"))
            _CURRENT_PREDICTION["answer"] = str(assistant.get("content", ""))
            _CURRENT_PREDICTION["evidence"] = _safe_evidence(assistant.get("_evidence", []))
            if decision.get("action") not in {"BLOCK", "REVIEW"}:
                return fail("hostile request was not blocked or held")
            if assistant.get("_plan") or assistant.get("_workflow_status") in {"pending", "verified"}:
                return fail("hostile request reached business planning or execution")
            print(f"PASS input safety: Jev {decision.get('action')}; no business plan was created")
            _finalize_prediction()
            return 0

        if decision.get("action") != "ALLOW":
            return fail(f"expected Jev ALLOW, received {decision.get('action')}")

        if args.scenario == "unsupported_period":
            _CURRENT_PREDICTION["workflow_status"] = assistant.get("_workflow_status")
            _CURRENT_PREDICTION["plan"] = _plan_dict(assistant.get("_plan"))
            _CURRENT_PREDICTION["answer"] = str(assistant.get("content", ""))
            _CURRENT_PREDICTION["evidence"] = _safe_evidence(assistant.get("_evidence", []))
            if assistant.get("_workflow_status") != "data_unavailable":
                return fail("out-of-range dates did not return the data-coverage response")
            if not assistant.get("_evidence"):
                return fail("out-of-range response did not include date-coverage evidence")
            print("PASS unsupported period: safely reported date coverage with query evidence")
            _finalize_prediction()
            return 0

        if args.scenario == "general_chat":
            content = str(assistant.get("content", ""))
            _CURRENT_PREDICTION["workflow_status"] = assistant.get("_workflow_status")
            _CURRENT_PREDICTION["plan"] = _plan_dict(assistant.get("_plan"))
            _CURRENT_PREDICTION["answer"] = content
            _CURRENT_PREDICTION["evidence"] = _safe_evidence(assistant.get("_evidence", []))
            if "HTTP 403" in content or "could not be reached" in content.lower():
                return fail("chat gateway did not return a usable answer")
            if assistant.get("_plan"):
                return fail("general chat was incorrectly routed to business planning")
            print("PASS general chat: Jev allowed the request and the chat model replied")
            _finalize_prediction()
            return 0

        plan = assistant.get("_plan")
        if plan is not None:
            _CURRENT_PREDICTION["plan_before_approval"] = _plan_dict(plan)
            _CURRENT_PREDICTION["preapproval_evidence_count"] = len(assistant.get("_evidence", []))
        if assistant.get("_workflow_status") != "pending" or plan is None:
            return fail("business request did not reach the pending human-approval state")
        if plan.status != "ready" or plan.approval_status != "pending":
            return fail("planner did not return a ready plan pending approval")
        specialists = [task for task in plan.tasks if task.agent != "verification_agent"]
        if not specialists or any(task.agent != scenario["agent"] for task in specialists):
            actual = [
                {"task_id": task.task_id, "agent": task.agent, "objective": task.objective,
                 "data_sources": task.data_sources}
                for task in specialists
            ]
            return fail(f"planner selected {actual}; expected tasks assigned to {scenario['agent']}")
        selected_sources = {source for task in specialists for source in task.data_sources}
        missing_sources = scenario.get("sources", set()) - selected_sources
        if missing_sources:
            return fail("planner omitted required data source(s): " + ", ".join(sorted(missing_sources)))
        if assistant.get("_evidence"):
            return fail("business evidence appeared before the approval control was used")

        approvals = [item for item in at.checkbox if item.label == "I reviewed and approve this plan"]
        if len(approvals) != 1:
            return fail("could not find the explicit plan-approval checkbox")
        approvals[0].check().run()
        buttons = [item for item in at.button if item.label == "Approve plan and run specialists"]
        if len(buttons) != 1:
            return fail("could not find the plan-approval submit control")
        _CURRENT_PREDICTION["approval_clicked"] = True
        buttons[0].click().run()
        if at.exception:
            return fail("approved workflow raised a Streamlit exception")
        final = visible_messages(at)[-1]
        _CURRENT_PREDICTION["workflow_status"] = final.get("_workflow_status")
        _CURRENT_PREDICTION["plan"] = _plan_dict(final.get("_plan"))
        _CURRENT_PREDICTION["output_guard_action"] = final.get("_output_guard", {}).get("action")
        _CURRENT_PREDICTION["answer"] = str(final.get("content", ""))
        _CURRENT_PREDICTION["evidence"] = _safe_evidence(final.get("_evidence", []))
        if final.get("_workflow_status") != "verified":
            print("WORKFLOW STATUS:", final.get("_workflow_status"), file=sys.stderr)
            print("WORKFLOW MESSAGE:", final.get("content", ""), file=sys.stderr)
            return fail("approved analysis did not produce a verified report")
        if final.get("_output_guard", {}).get("action") != "ALLOW":
            return fail("output privacy guard did not allow the synthetic business report")
        evidence = final.get("_evidence", [])
        report = str(final.get("content", ""))
        evidence_ids = {str(item.get("evidence_id")) for item in evidence}
        cited_ids = {
            item.strip() for match in re.findall(r"\[Evidence:\s*([^\]]+)\]", report)
            for item in match.split(",")
        }
        if not cited_ids or not cited_ids.issubset(evidence_ids):
            return fail("report citations do not resolve to returned evidence IDs")
        sql_items = [item for item in evidence if isinstance(item.get("specialist_sql") or item.get("sql"), str)]
        rag_items = [item for item in evidence if item.get("evidence_type") == "document_context"]
        if scenario.get("requires_sql", False) and not sql_items:
            return fail("verified report has no SQL evidence")
        if scenario.get("requires_rag", False) and not rag_items:
            return fail("verified report has no retrieved document/review passages")
        print(f"PASS live workflow: Jev -> planner -> approval -> {scenario['agent']} -> verification -> output guard")
        print("Status: verified | SQL evidence:", len(sql_items), "| RAG passages:", len(rag_items), "| Evidence items:", len(evidence))
        summary = " ".join(report.split())[:2400]
        summary = summary.encode("ascii", "backslashreplace").decode("ascii")
        print("Report summary:", summary)
        _finalize_prediction()
        return 0
    except Exception as exc:
        # Avoid logging request objects or configuration that might contain credentials.
        return fail(f"{type(exc).__name__}: workflow smoke run did not complete")


if __name__ == "__main__":
    raise SystemExit(main())



