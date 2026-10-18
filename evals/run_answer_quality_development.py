"""Execute development answer-quality cases using saved, approved-plan fixtures.

Planner predictions are reused to isolate specialist execution, verification,
reporting, and output screening. Approval is simulated only for this synthetic
benchmark; this runner does not change the application's human-approval gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD = Path(__file__).with_name("golden_dataset.json")
DEFAULT_PLANS = Path(__file__).with_name("predictions_planner_development_v050.jsonl")
PLAN_CASE_ALIASES = {
    "report_category_sales": "planner_category_sales",
    "report_customer_lifecycle_3m_dev": "planner_v050_09",
    "report_customer_cohort_retention_dev": "planner_v050_09",
    "report_ops_return_rate_category_dev": "planner_v050_11",
    "report_ops_inventory_availability_dev": "planner_v050_05",
}
PLAN_TASK_OVERRIDES = {
    "report_customer_lifecycle_3m_dev": {
        "task_id": "customer_lifecycle_jul_sep_2025",
        "agent": "customer_analyst",
        "objective": "Compare monthly customer lifecycle metrics by first-recorded month status for July, August, and September 2025.",
        "data_sources": ["customers", "orders", "order_items"],
    },
    "report_customer_cohort_retention_dev": {
        "task_id": "customer_cohort_retention_jul_sep_2025",
        "agent": "customer_analyst",
        "objective": "Analyze customer cohort retention from month 0 through month 3 for the July, August, and September 2025 first-recorded cohorts.",
        "data_sources": ["customers", "orders"],
    },
    "report_ops_return_rate_category_dev": {
        "task_id": "return_rate_category_aug_sep_2025",
        "agent": "product_operations_analyst",
        "objective": "Compare returns by product category for August and September 2025 using returned-unit rates.",
        "data_sources": ["orders", "order_items", "products", "returns"],
    },
    "report_ops_inventory_availability_dev": {
        "task_id": "inventory_snapshot_availability_aug_sep_2025",
        "agent": "product_operations_analyst",
        "objective": "Compare inventory snapshot availability by product category for August and September 2025.",
        "data_sources": ["inventory_snapshots", "products"],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name} line {line_no}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("case_id"), str):
            raise ValueError(f"Invalid prediction object in {path.name} line {line_no}.")
        rows.append(value)
    return rows


def safe_evidence(items: list[dict[str, Any]]) -> str:
    """Keep traceable SQL and aggregate rows; strip identifiers and personal fields."""
    safe_items = []
    allow = {
        "evidence_id", "agent", "sql", "specialist_sql", "tables_read", "columns", "rows",
        "row_count_returned", "truncated", "query_template", "forecast_template",
        "forecast_status", "forecast_model", "metric_label", "dimension_label",
        "model_comparison", "backtest", "reconciliation", "analysis_method",
        "statistical_tests", "result_validation", "date_coverage", "source_ref",
        "source_name", "section", "excerpt", "evidence_type",
        "causal_review",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): clean(item) for key, item in value.items()
                if str(key).lower() != "order_history_evidence" and (str(key) in allow or (
                    not re.search(r"(?:^|_)(?:customer|order|item|product|review|campaign|shipment)?_?id$", str(key), re.IGNORECASE)
                    and not re.search(r"email|phone|address|password|token|secret", str(key), re.IGNORECASE)
                ))
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    for item in items:
        safe_items.append(clean({key: value for key, value in item.items() if key in allow}))
    return json.dumps(safe_items, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()

    cases = json.loads(GOLD.read_text(encoding="utf-8"))["cases"]
    cases = [case for case in cases
             if case.get("component") == "answer_quality" and case.get("split") == "development"]
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in requested]
        found = {case["case_id"] for case in cases}
        if requested != found:
            raise ValueError("Unknown/non-development answer-quality case ID(s): " + ", ".join(sorted(requested - found)))

    by_question = {}
    by_case_id = {}
    for row in load_jsonl(args.plans):
        raw_plan = row.get("plan")
        if not isinstance(raw_plan, str):
            continue
        try:
            plan_data = json.loads(raw_plan)
        except json.JSONDecodeError:
            continue
        by_case_id[row.get("case_id")] = (row.get("case_id"), plan_data)
        question = plan_data.get("question") if isinstance(plan_data, dict) else None
        if isinstance(question, str):
            by_question[question] = (row.get("case_id"), plan_data)

    def has_plan(case: dict[str, Any]) -> bool:
        return case["query"] in by_question or PLAN_CASE_ALIASES.get(case["case_id"]) in by_case_id

    selected = [case for case in cases if has_plan(case)]
    skipped = [case["case_id"] for case in cases if not has_plan(case)]
    if not selected:
        raise ValueError("No selected answer-quality case has an exact matching saved planner question.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).with_name(f"predictions_answer_quality_development_{stamp}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions file: {output}")

    from business_agents import BusinessAgentError, execute_approved_plan
    from output_guardrails import screen_business_output
    from planner import PlannerPlan, PlannerTask, approve_plan

    successes = 0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(selected, 1):
            if case["query"] in by_question:
                planner_case_id, raw_plan = by_question[case["query"]]
            else:
                planner_case_id = PLAN_CASE_ALIASES[case["case_id"]]
                _, raw_plan = by_case_id[planner_case_id]
                raw_plan = {**raw_plan, "question": case["query"]}
            override = PLAN_TASK_OVERRIDES.get(case["case_id"])
            if override:
                raw_plan = {**raw_plan, "tasks": [dict(task) for task in raw_plan.get("tasks", [])]}
                specialist_tasks = [task for task in raw_plan["tasks"] if task.get("agent") != "verification_agent"]
                verify_tasks = [task for task in raw_plan["tasks"] if task.get("agent") == "verification_agent"]
                if len(specialist_tasks) != 1 or len(verify_tasks) != 1:
                    raise ValueError(f"Plan fixture {planner_case_id} must have one specialist and one verifier.")
                specialist_tasks[0].update({
                    "task_id": override["task_id"],
                    "agent": override["agent"],
                    "objective": override["objective"],
                    "data_sources": list(override["data_sources"]),
                    "depends_on": [],
                })
                verify_tasks[0]["data_sources"] = list(override["data_sources"])
                verify_tasks[0]["depends_on"] = [override["task_id"]]
                raw_plan.update({
                    "question": case["query"],
                    "summary": override["objective"],
                    "assumptions": ["This is a read-only evaluation against synthetic Cedar & Stone data."],
                    "missing_information": [],
                    "unavailable_data": [],
                    "approval_status": "pending",
                })
            row: dict[str, Any] = {"case_id": case["case_id"], "plan_fixture": planner_case_id}
            try:
                tasks = [PlannerTask(**task) for task in raw_plan.get("tasks", [])]
                plan = PlannerPlan(**{**raw_plan, "tasks": tasks})
                if plan.question != case["query"] or plan.status != "ready":
                    raise BusinessAgentError("The matching saved plan is not ready for this exact question.")
                approved = approve_plan(plan)
                result = execute_approved_plan(case["query"], approved)
                answer = str(result.get("content", ""))
                evidence = result.get("evidence", [])
                guard = screen_business_output(answer, evidence)
                expected_execution = case.get("expected_execution")
                execution_path = None
                path_passed = True
                if isinstance(expected_execution, dict):
                    specialist = expected_execution.get("specialist")
                    specialist_evidence = [
                        item for item in evidence
                        if isinstance(item, dict)
                        and item.get("agent") == specialist
                        and (isinstance(item.get("specialist_sql"), str)
                             or (item.get("evidence_type") == "forecast" and item.get("forecast_template")))
                    ]
                    sql_evidence = [item for item in specialist_evidence if isinstance(item.get("specialist_sql"), str)]
                    forecast_evidence = [item for item in specialist_evidence if item.get("evidence_type") == "forecast"]
                    actual_templates = sorted({
                        item["query_template"] for item in sql_evidence
                        if isinstance(item.get("query_template"), str)
                    })
                    actual_forecast_templates = sorted({
                        item["forecast_template"] for item in forecast_evidence
                        if isinstance(item.get("forecast_template"), str)
                    })
                    actual_sources = sorted({
                        source for item in sql_evidence
                        for source in item.get("tables_read", []) if isinstance(source, str)
                    })
                    expected_sources = sorted(expected_execution.get("sources", []))
                    expected_query_template = expected_execution.get("query_template")
                    expected_templates = [expected_query_template] if isinstance(expected_query_template, str) else []
                    expected_forecast_template = expected_execution.get("forecast_template")
                    expected_forecast_templates = [expected_forecast_template] if isinstance(expected_forecast_template, str) else []
                    actual_tools = []
                    if sql_evidence:
                        actual_tools.append("read_only_sql")
                    if forecast_evidence:
                        actual_tools.append("deterministic_forecast")
                    expected_tools = expected_execution.get("tools")
                    if not isinstance(expected_tools, list):
                        expected_tool = expected_execution.get("tool")
                        expected_tools = [expected_tool] if isinstance(expected_tool, str) else []
                    expected_tools = sorted(expected_tools)
                    actual_tools = sorted(actual_tools)
                    path_passed = (
                        bool(specialist_evidence)
                        and actual_tools == expected_tools
                        and actual_templates == expected_templates
                        and actual_forecast_templates == expected_forecast_templates
                        and actual_sources == expected_sources
                    )
                    execution_path = {
                        "passed": path_passed,
                        "expected_specialist": specialist,
                        "actual_specialists": sorted({item.get("agent") for item in specialist_evidence}),
                        "expected_tools": expected_tools,
                        "actual_tools": actual_tools,
                        "expected_query_template": expected_query_template,
                        "actual_query_templates": actual_templates,
                        "expected_forecast_template": expected_forecast_template,
                        "actual_forecast_templates": actual_forecast_templates,
                        "expected_sources": expected_sources,
                        "actual_sources": actual_sources,
                    }
                row["execution_path"] = execution_path
                row["workflow_status"] = result.get("status")
                row["output_guard"] = guard.action
                row["execution_path_passed"] = path_passed
                if result.get("status") == "verified" and guard.action == "ALLOW" and path_passed:
                    row["answer"] = answer
                    row["evidence"] = safe_evidence(evidence)
                    row["verification_issues"] = result.get("verification_issues", [])
                    successes += 1
                    print(f"{index}/{len(selected)} {case['case_id']}: verified, allowed, execution path matched")
                else:
                    row["answer"] = ""
                    row["evidence"] = ""
                    row["error"] = "Report, output screen, or expected specialist/tool/source path did not pass."
                    print(f"{index}/{len(selected)} {case['case_id']}: held safely")
            except Exception as exc:
                row["answer"] = ""
                row["evidence"] = ""
                row["error"] = f"{type(exc).__name__}: report generation failed; details omitted"
                print(f"{index}/{len(selected)} {case['case_id']}: failed ({type(exc).__name__})")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

    if skipped:
        print("Skipped answer-quality cases without a matching saved plan: " + ", ".join(skipped))
    print(f"Saved {successes}/{len(selected)} verified reports to {output}")
    return 0 if successes == len(selected) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("Answer-quality runner error: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
