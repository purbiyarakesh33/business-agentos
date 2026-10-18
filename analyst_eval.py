"""Offline regression suite for the synthetic Data Analyst fixture.

Run with ``py analyst_eval.py`` from any directory. Uses the local SQLite demo
DB and scripted analyst actions only: no API key is loaded and no LLM is called.
This checks tool execution, KPI SQL, expected fixture values, and deterministic
result checks; it does not measure a hosted model's query-selection quality.
"""
from __future__ import annotations

import math
import re
import sqlite3
from typing import Any
from unittest.mock import patch

from analyst import AnalystError, DataAnalystAgent, _repair_misplaced_compound_order_by
from business_agents import (
    BusinessAgentError,
    _deterministic_category_period_findings,
    _deterministic_demo_cohort_findings,
    _deterministic_demo_customer_findings,
    _deterministic_demo_customer_segment_findings,
    _deterministic_demo_product_operations_findings,
    _deterministic_demo_sales_findings,
    _deterministic_demo_gross_profit_findings,
    _deterministic_review_text_findings,
    _draft_findings,
    _execute_read_only,
    _make_analysis_action,
    _asserts_causal_attribution,
    _period_comparison_sql,
    _schema_context,
    _validate_findings,
    _verify_findings,
)
from customer_analyst import CustomerAnalystAgent
from demo_sql import build_demo_query
from forecasting_agent import (
    ForecastingAgent,
    ForecastingError,
    deterministic_forecast_findings,
    parse_forecast_request,
    validate_forecast_evidence,
)
from planner import PlannerTask


CASES: list[dict[str, Any]] = [
    {
        "name": "delivered net sales, three calendar months",
        "sources": ["orders", "order_items"],
        "sql": """SELECT
          ROUND(SUM(CASE WHEN o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01' THEN oi.line_net_sales ELSE 0 END),2) AS sep_2025,
          ROUND(SUM(CASE WHEN o.order_date >= '2025-08-01' AND o.order_date < '2025-09-01' THEN oi.line_net_sales ELSE 0 END),2) AS aug_2025,
          ROUND(SUM(CASE WHEN o.order_date >= '2024-09-01' AND o.order_date < '2024-10-01' THEN oi.line_net_sales ELSE 0 END),2) AS sep_2024
        FROM orders o JOIN order_items oi ON oi.order_id=o.order_id
        WHERE o.status='delivered'""",
        "expected": [{"sep_2025": 11008.18, "aug_2025": 18318.63, "sep_2024": 12591.85}],
    },
    {
        "name": "delivered net sales and gross margin by category",
        "sources": ["orders", "order_items", "products"],
        "sql": """SELECT p.category,
          ROUND(SUM(oi.line_net_sales),2) AS net_sales,
          ROUND(SUM(oi.line_gross_profit),2) AS gross_profit,
          ROUND(100.0*SUM(oi.line_gross_profit)/NULLIF(SUM(oi.line_net_sales),0),2) AS gross_margin_pct
        FROM orders o JOIN order_items oi ON oi.order_id=o.order_id
        JOIN products p ON p.product_id=oi.product_id
        WHERE o.status='delivered' AND o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01'
        GROUP BY p.category ORDER BY p.category""",
        "expected": [
            {"category": "Bedding", "net_sales": 2833.43, "gross_profit": 1543.78, "gross_margin_pct": 54.48},
            {"category": "Decor", "net_sales": 1558.61, "gross_profit": 722.20, "gross_margin_pct": 46.34},
            {"category": "Kitchen", "net_sales": 1815.53, "gross_profit": 1006.68, "gross_margin_pct": 55.45},
            {"category": "Lighting", "net_sales": 2009.60, "gross_profit": 962.42, "gross_margin_pct": 47.89},
            {"category": "Storage", "net_sales": 2791.01, "gross_profit": 1321.59, "gross_margin_pct": 47.35},
        ],
    },
    {
        "name": "on-time delivery by carrier",
        "sources": ["fulfillment", "orders"],
        "sql": """SELECT f.carrier, COUNT(*) AS fulfilled_orders,
          SUM(CASE WHEN f.delivered_date <= f.promised_delivery_date THEN 1 ELSE 0 END) AS on_time_orders,
          ROUND(100.0*SUM(CASE WHEN f.delivered_date <= f.promised_delivery_date THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS on_time_pct
        FROM fulfillment f JOIN orders o ON o.order_id=f.order_id
        WHERE o.status='delivered' AND f.delivered_date IS NOT NULL AND f.promised_delivery_date IS NOT NULL
        GROUP BY f.carrier ORDER BY f.carrier""",
        "expected": [
            {"carrier": "GroundLine", "fulfilled_orders": 844, "on_time_orders": 736, "on_time_pct": 87.20},
            {"carrier": "ParcelNorth", "fulfilled_orders": 817, "on_time_orders": 719, "on_time_pct": 88.00},
            {"carrier": "SwiftPost", "fulfilled_orders": 843, "on_time_orders": 746, "on_time_pct": 88.49},
        ],
    },
    {
        "name": "refund totals by recorded reason",
        "sources": ["returns", "order_items", "orders"],
        "sql": """SELECT r.reason, COUNT(*) AS returned_items,
          ROUND(SUM(r.refund_amount),2) AS refund_total
        FROM returns r JOIN order_items oi ON oi.order_item_id=r.order_item_id
        JOIN orders o ON o.order_id=oi.order_id
        GROUP BY r.reason ORDER BY r.reason""",
        "expected": [
            {"reason": "changed_mind", "returned_items": 56, "refund_total": 4080.25},
            {"reason": "damaged", "returned_items": 44, "refund_total": 3341.61},
            {"reason": "not_as_expected", "returned_items": 55, "refund_total": 4471.73},
            {"reason": "wrong_item", "returned_items": 78, "refund_total": 6579.59},
        ],
    },
    {
        "name": "monthly repeat-customer definition",
        "sources": ["orders"],
        "sql": """WITH per_customer AS (
          SELECT o.customer_id, COUNT(DISTINCT o.order_id) AS delivered_orders
          FROM orders o WHERE o.status='delivered'
            AND o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01'
          GROUP BY o.customer_id
        )
        SELECT COUNT(*) AS ordering_customers,
          SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers,
          ROUND(100.0*SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),2) AS repeat_customer_pct
        FROM per_customer""",
        "expected": [{"ordering_customers": 71, "repeat_customers": 1, "repeat_customer_pct": 1.41}],
    },
]


def rows_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, f"row count {len(actual)} != expected {len(expected)}"
    for index, (got, want) in enumerate(zip(actual, expected), 1):
        if set(got) != set(want):
            return False, f"row {index} columns differ: {sorted(got)} vs {sorted(want)}"
        for key, expected_value in want.items():
            actual_value = got[key]
            if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                if not isinstance(actual_value, (int, float)) or not math.isclose(
                    float(actual_value), float(expected_value), rel_tol=0, abs_tol=0.011
                ):
                    return False, f"row {index} {key}: {actual_value!r} != {expected_value!r}"
            elif actual_value != expected_value:
                return False, f"row {index} {key}: {actual_value!r} != {expected_value!r}"
    return True, "fixture values and row shapes match"


def run() -> int:
    passed = 0
    total = len(CASES) + 28
    for index, case in enumerate(CASES, 1):
        task = PlannerTask(
            task_id=f"eval_{index}", agent="data_analyst", objective=case["name"],
            data_sources=case["sources"], depends_on=[], evidence_required=["executed SQL evidence"],
        )
        result = _execute_read_only(case["sql"], task)
        ok, detail = rows_match(result["rows"], case["expected"])
        checks = DataAnalystAgent.validate_result(result)
        ok = ok and checks["all_checks_passed"]
        if not checks["all_checks_passed"]:
            detail += "; deterministic checks failed: " + "; ".join(checks["failed_checks"])
        print(f"{'PASS' if ok else 'FAIL'} {case['name']}: {detail}")
        passed += int(ok)

    causal_guard_findings = [{
        "claim": "Competitor advertising caused the September sales decline.",
        "evidence_ids": ["category_sales_compare:query1"],
        "calculation": "",
        "limitations": [],
    }]
    causal_guard_evidence = [{
        "evidence_id": "category_sales_compare:query1",
        "query_template": "deterministic_category_period_comparison_v1",
    }]
    causal_guard_issues = _verify_findings(
        "Why did delivered sales decline?",
        causal_guard_findings,
        causal_guard_evidence,
        PlannerTask("eval_causal_guard", "verification_agent", "Check causal claims", [], [], []),
        {"causal_claim_allowed": False, "status": "observational_association_only"},
    )
    unsupported_cause = any("causal_review says" in issue for issue in causal_guard_issues)
    print(f"{'PASS' if unsupported_cause else 'FAIL'} verification holds an unsupported causal assertion: {causal_guard_issues}")
    passed += int(unsupported_cause)
    causal_caveat = not _asserts_causal_attribution(
        "The data does not prove that competitor advertising caused the decline; the comparison shows association only."
    )
    print(f"{'PASS' if causal_caveat else 'FAIL'} accepts an explicit association-only caveat")
    passed += int(causal_caveat)

    question = (
        "For delivered orders, compare net product sales in September 2025 with "
        "August 2025 and September 2024. Break down by region and product category."
    )
    task = PlannerTask(
        task_id="eval_period_rollup", agent="data_analyst",
        objective="Compare delivered net product sales by region and category",
        data_sources=["orders", "order_items", "products"], depends_on=[],
        evidence_required=["total, region and category sales changes"],
    )
    from business_agents import _period_comparison_sql, _schema_context
    allowed, schemas = _schema_context(task.data_sources)
    comparison_sql = _period_comparison_sql(question, task, allowed, schemas)
    if not comparison_sql:
        print("FAIL analyst loop and three-way rollup: deterministic query was not selected")
        return 1

    def scripted_action(_question: str, _task: PlannerTask, state: dict[str, Any]) -> dict[str, Any]:
        return {"action": "done"} if state["query_history"] else {"action": "query", "sql": comparison_sql}

    agent = DataAnalystAgent(scripted_action, _execute_read_only)
    try:
        evidence = agent.run_task(question, task)
        total_row = next(
            row for row in evidence[0]["rows"] if row.get("dimension_type") == "total"
        )
        independent_sql = """SELECT
          ROUND(SUM(CASE WHEN o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01' THEN oi.line_net_sales ELSE 0 END), 2) AS period_1_sales,
          ROUND(SUM(CASE WHEN o.order_date >= '2025-08-01' AND o.order_date < '2025-09-01' THEN oi.line_net_sales ELSE 0 END), 2) AS period_2_sales,
          ROUND(SUM(CASE WHEN o.order_date >= '2024-09-01' AND o.order_date < '2024-10-01' THEN oi.line_net_sales ELSE 0 END), 2) AS period_3_sales
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id
        WHERE o.status = 'delivered'
          AND o.order_date >= '2024-09-01' AND o.order_date < '2025-10-01'"""
        independent_total = _execute_read_only(independent_sql, task)["rows"][0]
        ok = (
            len(evidence) == 1
            and evidence[0]["analysis_complete"]
            and evidence[0]["result_validation"]["all_checks_passed"]
            and evidence[0]["result_validation"]["reconciliation_status"].startswith("checked")
            and "independent_total AS" in comparison_sql
            and all(
                math.isclose(total_row[key], independent_total[key], abs_tol=0.011)
                for key in ("period_1_sales", "period_2_sales", "period_3_sales")
            )
            and math.isclose(total_row["change_vs_period_2_usd"], -7310.45, abs_tol=0.011)
            and math.isclose(total_row["change_vs_period_2_pct"], -39.91, abs_tol=0.011)
        )
        print(f"{'PASS' if ok else 'FAIL'} analyst loop and three-way rollup: "
              f"{len(evidence)} query evidence item(s); "
              f"{evidence[0]['result_validation']['reconciliation_status']}; "
              "independent period totals match")
        passed += int(ok)
    except (AnalystError, StopIteration, KeyError) as exc:
        print(f"FAIL analyst loop and three-way rollup: {exc}")

    category_question = (
        "Compare delivered net product sales by product category in September 2025 "
        "versus August 2025. Show dollar and percentage change, rank declines, and reconcile totals."
    )
    category_task = PlannerTask(
        task_id="eval_category_sales_two_period_independent_total", agent="data_analyst",
        objective="Compare delivered net product sales by category for September and August 2025",
        data_sources=["orders", "order_items", "products"], depends_on=[],
        evidence_required=["complete category rows, independent overall total, and evidence-backed report"],
    )
    allowed, schemas = _schema_context(category_task.data_sources)
    category_sql = _period_comparison_sql(category_question, category_task, allowed, schemas)
    category_ok = False
    category_detail = "comparison query was not selected"
    if category_sql:
        try:
            category_result = _execute_read_only(category_sql, category_task)
            category_result.update({
                "evidence_id": "eval_category_sales:query1",
                "query_template": "deterministic_category_period_comparison_v1",
                "objective": category_task.objective,
            })
            category_result["result_validation"] = DataAnalystAgent.validate_result(category_result)
            domain_result = _execute_read_only(
                "SELECT DISTINCT category AS dimension_value FROM products WHERE category IS NOT NULL ORDER BY category",
                category_task,
            )
            domain_result.update({
                "evidence_id": "eval_category_sales:domain",
                "evidence_type": "dimension_domain",
                "dimension": "product_category",
            })
            report = _deterministic_category_period_findings(
                category_question, [category_result, domain_result]
            )
            total_row = next(
                row for row in category_result["rows"] if row.get("dimension_type") == "total"
            )
            category_rows = [
                row for row in category_result["rows"]
                if row.get("dimension_type") == "product_category"
            ]
            independent = _execute_read_only("""SELECT
              ROUND(SUM(CASE WHEN o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01' THEN oi.line_net_sales ELSE 0 END), 2) AS september_sales,
              ROUND(SUM(CASE WHEN o.order_date >= '2025-08-01' AND o.order_date < '2025-09-01' THEN oi.line_net_sales ELSE 0 END), 2) AS august_sales
            FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
            WHERE o.status = 'delivered'""", category_task)["rows"][0]
            report_text = " ".join(
                [finding.get("claim", "") for finding in report["findings"]]
                + [limitation for finding in report["findings"] for limitation in finding.get("limitations", [])]
            )
            category_ok = (
                category_result["result_validation"]["all_checks_passed"]
                and "independent_total AS" in category_sql
                and len(category_rows) == len(domain_result["rows"]) == 5
                and math.isclose(total_row["period_1_sales"], independent["september_sales"], abs_tol=0.011)
                and math.isclose(total_row["period_2_sales"], independent["august_sales"], abs_tol=0.011)
                and "order_date" in report_text
                and "USD -7,310.45" in report_text
                and all(row.get("decline_rank_vs_period_2") is not None for row in category_rows)
            )
            category_detail = (
                "category rows reconcile to a separately queried orders/order_items total; "
                "report states order-date basis; rank, percentage, and limitation checks pass"
                if category_ok else
                f"validation={category_result['result_validation']}; rows={len(category_rows)}; report={report_text[:400]}"
            )
        except (AnalystError, BusinessAgentError, KeyError, StopIteration, TypeError, ValueError) as exc:
            category_detail = str(exc)
    print(f"{'PASS' if category_ok else 'FAIL'} two-period category report and independent total: {category_detail}")
    passed += int(category_ok)

    customer_question = (
        "Compare customer lifecycle in September 2025, August 2025, and September 2024; "
        "show active customers, repeat rate, and net sales per active customer."
    )
    customer_task = PlannerTask(
        task_id="eval_customer_lifecycle", agent="customer_analyst",
        objective="Compare customer lifecycle metrics across three months",
        data_sources=["customers", "orders", "order_items"], depends_on=[],
        evidence_required=["customer lifecycle SQL and aggregate identity preflight evidence"],
    )
    allowed, schemas = _schema_context(customer_task.data_sources)
    customer_query = build_demo_query(customer_question, customer_task, allowed, schemas)
    customer_ok = customer_query is not None
    customer_detail = "fixed lifecycle template selected" if customer_ok else "no fixed lifecycle template selected"
    if customer_query is not None:
        def scripted_customer_action(_question: str, _task: PlannerTask, state: dict[str, Any]) -> dict[str, Any]:
            if state["query_history"]:
                return {"action": "done"}
            return {
                "action": "query", "sql": customer_query.sql,
                "template_id": customer_query.template_id,
                "template_limitations": list(customer_query.limitations),
            }

        try:
            customer_evidence = CustomerAnalystAgent(
                scripted_customer_action, _execute_read_only
            ).run_task(customer_question, customer_task)
            quality = [
                item for item in customer_evidence
                if item.get("evidence_type") == "customer_data_quality"
            ]
            lifecycle = [
                item for item in customer_evidence
                if item.get("query_template") == "demo_customer_lifecycle_period_comparison_v1"
            ]
            proposed = _deterministic_demo_customer_findings(customer_question, customer_evidence)
            verified = _verify_findings(
                customer_question, _validate_findings(proposed, customer_evidence), customer_evidence,
                PlannerTask("verify_customer", "verification_agent", "Verify lifecycle report", [], [], []),
            )
            claims = [finding["claim"] for finding in proposed.get("findings", [])]
            lifecycle_rows = {
                (int(row["period_number"]), str(row["customer_type"])): row
                for row in lifecycle[0].get("rows", [])
            }

            def shown_pct_change(value: float, reference: float) -> str:
                return "not computable (zero baseline)" if reference == 0 else f"{(value - reference) * 100.0 / reference:+.2f}%"

            def shown_rate(value: Any) -> str:
                return "not computable" if value is None else f"{float(value):.2f}%"

            direction_checks = []
            for group_index, kind in enumerate(("all", "new", "returning", "unknown")):
                claim = claims[group_index] if group_index < len(claims) else ""
                reference = lifecycle_rows[(1, kind)]
                for period in (2, 3):
                    comparison = lifecycle_rows[(period, kind)]
                    active_change = int(comparison["active_customers"]) - int(reference["active_customers"])
                    expected_active = (
                        f"{comparison['period_label']} vs {reference['period_label']}: active customers "
                        f"{int(comparison['active_customers']):,} vs {int(reference['active_customers']):,} "
                        f"(change {active_change:+,}, {shown_pct_change(float(comparison['active_customers']), float(reference['active_customers']))})"
                    )
                    repeat_change = (
                        "not computable" if comparison["repeat_customer_pct"] is None or reference["repeat_customer_pct"] is None
                        else f"{float(comparison['repeat_customer_pct']) - float(reference['repeat_customer_pct']):+.2f} percentage points"
                    )
                    expected_repeat = (
                        f"repeat rate {shown_rate(comparison['repeat_customer_pct'])} vs "
                        f"{shown_rate(reference['repeat_customer_pct'])} ({repeat_change})"
                    )
                    direction_checks.append(expected_active in claim and expected_repeat in claim)
            customer_ok = (
                len(quality) == 1 and len(lifecycle) == 1
                and quality[0].get("blocking_checks_passed") is True
                and quality[0].get("result_validation", {}).get("all_checks_passed") is True
                and not any("customer_id" in column.lower() for column in quality[0].get("columns", []))
                and len(claims) == 4
                and not verified
                and all(quality[0]["evidence_id"] in finding["evidence_ids"]
                        for finding in proposed["findings"])
                # Verify every displayed change uses comparison period minus the explicitly named reference.
                and len(direction_checks) == 8 and all(direction_checks)
            )
            customer_detail = (
                f"preflight rows={len(quality)}; lifecycle rows={len(lifecycle)}; "
                "aggregate-only quality evidence is cited in the deterministic report"
            )
        except (AnalystError, KeyError, TypeError, AttributeError) as exc:
            customer_ok = False
            customer_detail = str(exc)
    print(f"{'PASS' if customer_ok else 'FAIL'} customer identity preflight and lifecycle evidence: {customer_detail}")
    passed += int(customer_ok)

    cohort_question = (
        "Compare cohort retention for September 2025, August 2025, and September 2024 "
        "from M+0 through M+3."
    )
    cohort_task = PlannerTask(
        task_id="eval_customer_cohort", agent="customer_analyst",
        objective="Compare customer cohort retention through four monthly offsets",
        data_sources=["customers", "orders"], depends_on=[],
        evidence_required=["retention numerators, cohort denominators, and observable dates"],
    )
    allowed, schemas = _schema_context(cohort_task.data_sources)
    cohort_query = build_demo_query(cohort_question, cohort_task, allowed, schemas)
    cohort_ok = cohort_query is not None
    cohort_detail = "fixed cohort template selected" if cohort_ok else "no fixed cohort template selected"
    if cohort_query is not None:
        def scripted_cohort_action(_question: str, _task: PlannerTask, state: dict[str, Any]) -> dict[str, Any]:
            if state["query_history"]:
                return {"action": "done"}
            return {
                "action": "query", "sql": cohort_query.sql,
                "template_id": cohort_query.template_id,
                "template_limitations": list(cohort_query.limitations),
            }

        try:
            cohort_evidence = CustomerAnalystAgent(
                scripted_cohort_action, _execute_read_only
            ).run_task(cohort_question, cohort_task)
            quality = [item for item in cohort_evidence if item.get("evidence_type") == "customer_data_quality"]
            results = [item for item in cohort_evidence if item.get("query_template") == cohort_query.template_id]
            report = _deterministic_demo_cohort_findings(cohort_question, cohort_evidence)
            verified = _verify_findings(
                cohort_question, _validate_findings(report, cohort_evidence), cohort_evidence,
                PlannerTask("verify_cohort", "verification_agent", "Verify cohort report", [], [], []),
            )
            cohort_rows = results[0]["rows"] if len(results) == 1 else []
            cohort_ok = (
                len(quality) == 1 and len(results) == 1 and len(cohort_rows) == 12
                and results[0].get("result_validation", {}).get("all_checks_passed") is True
                and {int(row["month_number"]) for row in cohort_rows} == {0, 1, 2, 3}
                and all(quality[0]["evidence_id"] in finding["evidence_ids"]
                        for finding in report.get("findings", []))
                and len(report.get("findings", [])) == 3
                and not verified
            )
            cohort_detail = f"{len(cohort_rows)} validated cohort-offset rows; {len(report.get('findings', []))} evidence-linked findings"
        except (AnalystError, BusinessAgentError, KeyError, TypeError, AttributeError) as exc:
            cohort_ok = False
            cohort_detail = str(exc)
    print(f"{'PASS' if cohort_ok else 'FAIL'} customer cohort retention and evidence validation: {cohort_detail}")
    passed += int(cohort_ok)

    segment_question = (
        "Compare customer behavior in September 2025, August 2025, and September 2024 "
        "by customer segment, customer region, and acquisition channel."
    )
    segment_task = PlannerTask(
        task_id="eval_customer_segments", agent="customer_analyst",
        objective="Compare customer segment, customer region, and acquisition channel metrics",
        data_sources=["customers", "orders", "order_items"], depends_on=[],
        evidence_required=["customer-grain activity, sales, and segment rollups"],
    )
    allowed, schemas = _schema_context(segment_task.data_sources)
    segment_query = build_demo_query(segment_question, segment_task, allowed, schemas)
    segment_ok = segment_query is not None
    segment_detail = "fixed segment template selected" if segment_ok else "no fixed segment template selected"
    if segment_query is not None:
        def scripted_segment_action(_question: str, _task: PlannerTask, state: dict[str, Any]) -> dict[str, Any]:
            if state["query_history"]:
                return {"action": "done"}
            return {
                "action": "query", "sql": segment_query.sql,
                "template_id": segment_query.template_id,
                "template_limitations": list(segment_query.limitations),
            }

        try:
            segment_evidence = CustomerAnalystAgent(
                scripted_segment_action, _execute_read_only
            ).run_task(segment_question, segment_task)
            quality = [item for item in segment_evidence if item.get("evidence_type") == "customer_data_quality"]
            results = [item for item in segment_evidence if item.get("query_template") == segment_query.template_id]
            report = _deterministic_demo_customer_segment_findings(segment_question, segment_evidence)
            verified = _verify_findings(
                segment_question, _validate_findings(report, segment_evidence), segment_evidence,
                PlannerTask("verify_segments", "verification_agent", "Verify segment report", [], [], []),
            )
            segment_rows = results[0]["rows"] if len(results) == 1 else []
            segment_ok = (
                len(quality) == 1 and len(results) == 1 and len(segment_rows) >= 15
                and results[0].get("result_validation", {}).get("all_checks_passed") is True
                and len(report.get("findings", [])) == 4
                and not verified
                and all(quality[0]["evidence_id"] in finding["evidence_ids"]
                        for finding in report.get("findings", []))
            )
            segment_detail = f"{len(segment_rows)} rows; segment rollups reconciled; {len(report.get('findings', []))} deterministic findings"
        except (AnalystError, BusinessAgentError, KeyError, TypeError, AttributeError) as exc:
            segment_ok = False
            segment_detail = str(exc)
    print(f"{'PASS' if segment_ok else 'FAIL'} customer segment analysis and reconciliation: {segment_detail}")
    passed += int(segment_ok)

    repeat_question = (
        "For July through December 2025, which customer segments had the lowest "
        "repeat-purchase rate? Define repeat purchase using delivered orders and cite the order-history evidence."
    )
    repeat_task = PlannerTask(
        task_id="eval_repeat_rate_by_segment", agent="customer_analyst",
        objective="Compare within-window repeat-purchase rates by customer segment",
        data_sources=["customers", "orders"], depends_on=[],
        evidence_required=["all segment counts, rates, and SQL evidence"],
    )
    allowed_repeat, schemas_repeat = _schema_context(repeat_task.data_sources)
    repeat_query = build_demo_query(repeat_question, repeat_task, allowed_repeat, schemas_repeat)
    if repeat_query is None or repeat_query.template_id != "demo_customer_repeat_window_v1":
        raise BusinessAgentError("Six-month repeat-rate analysis did not select its deterministic SQL template.")

    def scripted_repeat_action(_question: str, _task: PlannerTask, state: dict[str, Any]) -> dict[str, Any]:
        return {"action": "done"} if state["query_history"] else {
            "action": "query", "sql": repeat_query.sql,
            "template_id": repeat_query.template_id,
            "template_limitations": list(repeat_query.limitations),
        }

    try:
        repeat_evidence = CustomerAnalystAgent(
            scripted_repeat_action, _execute_read_only
        ).run_task(repeat_question, repeat_task)
        repeat_draft = _draft_findings(repeat_question, repeat_evidence)
        repeat_findings = _validate_findings(repeat_draft or {}, repeat_evidence)
        repeat_issues = _verify_findings(
            repeat_question, repeat_findings, repeat_evidence,
            PlannerTask("verify_repeat", "verification_agent", "Verify repeat-rate report", [], [], []),
        )
        repeat_claim = repeat_findings[0]["claim"] if repeat_findings else ""
        repeat_calculation = repeat_findings[0]["calculation"] if repeat_findings else ""
        repeat_sql_evidence = [item for item in repeat_evidence if item.get("query_template") == repeat_query.template_id]
        incomplete = [dict(repeat_findings[0])] if repeat_findings else []
        if incomplete:
            incomplete[0]["claim"] = incomplete[0]["claim"].replace(
                "Core: 99 of 261 active customers (37.93%)", ""
            )
        omission_issues = _verify_findings(
            repeat_question, incomplete, repeat_evidence,
            PlannerTask("verify_repeat_omission", "verification_agent", "Reject omitted segment", [], [], []),
        ) if incomplete else ["missing test finding"]
        repeat_ok = (
            not repeat_issues and bool(omission_issues) and len(repeat_sql_evidence) == 1
            and repeat_sql_evidence[0].get("result_validation", {}).get("all_checks_passed") is True
            and all(value in repeat_claim for value in (
                "Premium: 24 of 68 active customers (35.29%)",
                "Value: 54 of 153 active customers (35.29%)",
                "Core: 99 of 261 active customers (37.93%)",
                "tied for the lowest rate",
            ))
            and "During July through December 2025" in repeat_claim
            and "within-window measure" in repeat_calculation
            and "not lifetime returning-customer status" in repeat_calculation
            and "current customers-table values, not historical snapshots" in " ".join(repeat_findings[0]["limitations"])
        )
        repeat_detail = (
            "all three segments and counts/rates reported; within-window/current-segment limits stated; "
            "verifier rejects an omitted segment"
            if repeat_ok else
            f"issues={repeat_issues}; omitted-row issues={omission_issues}; claim={repeat_claim}"
        )
    except (AnalystError, BusinessAgentError, KeyError, TypeError, AttributeError) as exc:
        repeat_ok = False
        repeat_detail = str(exc)
    print(f"{'PASS' if repeat_ok else 'FAIL'} customer repeat-rate report completeness and verification: {repeat_detail}")
    passed += int(repeat_ok)

    privacy_task = PlannerTask(
        task_id="eval_customer_privacy", agent="customer_analyst",
        objective="Check that customer-level identifiers are not returned",
        data_sources=["customers"], depends_on=[], evidence_required=["aggregate output only"],
    )
    try:
        _execute_read_only("SELECT customer_id AS customer_reference FROM customers LIMIT 1", privacy_task)
        privacy_ok = False
        privacy_detail = "individual customer ID was returned"
    except BusinessAgentError as exc:
        privacy_ok = "individual customer identifiers" in str(exc)
        privacy_detail = "individual customer ID query was refused" if privacy_ok else str(exc)
    print(f"{'PASS' if privacy_ok else 'FAIL'} customer analyst privacy output guard: {privacy_detail}")
    passed += int(privacy_ok)

    out_of_range_question = (
        "Compare customer lifecycle in September 2026, August 2026, and September 2024."
    )
    out_of_range_task = PlannerTask(
        task_id="eval_customer_date_range", agent="customer_analyst",
        objective="Compare customer lifecycle in named months",
        data_sources=["customers", "orders", "order_items"], depends_on=[],
        evidence_required=["observed source coverage"],
    )
    def should_not_plan(_question: str, _task: PlannerTask, _state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("The analyst action generator should not run for out-of-range months.")
    try:
        CustomerAnalystAgent(should_not_plan, _execute_read_only).run_task(
            out_of_range_question, out_of_range_task
        )
        range_ok = False
        range_detail = "out-of-range periods were not blocked"
    except AnalystError as exc:
        range_ok = "outside the observed order-date range" in str(exc)
        range_detail = "out-of-range periods were stopped before specialist analysis" if range_ok else str(exc)
    print(f"{'PASS' if range_ok else 'FAIL'} customer source-date coverage gate: {range_detail}")
    passed += int(range_ok)

    category_forecast_question = (
        "Forecast delivered net product sales by product category for 3 months "
        "after the latest available month."
    )
    try:
        category_request = parse_forecast_request(category_forecast_question)
        category_ok = category_request.dimension == "category"
        try:
            parse_forecast_request(
                "Forecast delivered net product sales by product for 3 months "
                "after the latest available month."
            )
            category_ok = False
        except ForecastingError:
            pass
        category_task = PlannerTask(
            task_id="eval_forecast_category", agent="forecasting_agent",
            objective="Forecast delivered net product sales by product category",
            data_sources=["orders", "order_items", "products"], depends_on=[],
            evidence_required=["category monthly history, verified forecast, aggregate reconciliation"],
        )
        allowed, schemas = _schema_context(category_task.data_sources)
        category_evidence = ForecastingAgent(_execute_read_only).run_task(
            category_forecast_question, category_task, allowed, schemas
        )
        category_forecast = next(
            item for item in category_evidence if item.get("forecast_template")
        )
        category_reconciliation = category_forecast.get("reconciliation")
        category_ok = (
            category_ok
            and category_forecast.get("forecast_status") != "unavailable"
            and category_forecast.get("dimension") == "category"
            and len(category_forecast.get("rows", [])) == 15
            and isinstance(category_reconciliation, dict)
            and len(category_reconciliation.get("rows", [])) == 3
            and not validate_forecast_evidence(category_evidence)
        )
        if isinstance(category_reconciliation, dict):
            for aggregate in category_reconciliation.get("rows", []):
                grouped = [
                    row for row in category_forecast["rows"]
                    if row.get("period_start") == aggregate.get("period_start")
                ]
                category_ok = category_ok and math.isclose(
                    sum(float(row["forecast"]) for row in grouped),
                    float(aggregate["forecast"]), rel_tol=0, abs_tol=0.001,
                )
        category_report = deterministic_forecast_findings(
            category_forecast_question, category_evidence
        )
        category_ok = category_ok and bool(category_report) and any(
            "Across all product category groups" in finding.get("claim", "")
            for finding in category_report.get("findings", [])
        )
        category_detail = (
            f"{len(category_forecast.get('rows', []))} group-forecast rows; "
            "category wording accepted, product-level wording rejected, totals verified"
        )
    except (ForecastingError, BusinessAgentError, StopIteration, KeyError, TypeError, ValueError) as exc:
        category_ok = False
        category_detail = str(exc)
    print(f"{'PASS' if category_ok else 'FAIL'} forecasting by product category and total reconciliation: {category_detail}")
    passed += int(category_ok)

    channel_forecast_question = (
        "What if delivered net product sales increase by 10% by channel for 3 months "
        "after the latest available month?"
    )
    try:
        channel_task = PlannerTask(
            task_id="eval_forecast_channel_scenario", agent="forecasting_agent",
            objective="Forecast channel sales with a mechanical 10 percent sensitivity",
            data_sources=["orders", "order_items", "products"], depends_on=[],
            evidence_required=["channel monthly history, verified forecast, scenario reconciliation"],
        )
        allowed, schemas = _schema_context(channel_task.data_sources)
        channel_evidence = ForecastingAgent(_execute_read_only).run_task(
            channel_forecast_question, channel_task, allowed, schemas
        )
        channel_forecast = next(
            item for item in channel_evidence if item.get("forecast_template")
        )
        channel_reconciliation = channel_forecast.get("reconciliation")
        channel_ok = (
            channel_forecast.get("forecast_status") != "unavailable"
            and channel_forecast.get("dimension") == "channel"
            and channel_forecast.get("scenario_change_pct") == 10.0
            and len(channel_forecast.get("rows", [])) == 9
            and isinstance(channel_reconciliation, dict)
            and len(channel_reconciliation.get("rows", [])) == 3
            and not validate_forecast_evidence(channel_evidence)
        )
        if isinstance(channel_reconciliation, dict):
            for aggregate in channel_reconciliation.get("rows", []):
                grouped = [
                    row for row in channel_forecast["rows"]
                    if row.get("period_start") == aggregate.get("period_start")
                ]
                channel_ok = channel_ok and math.isclose(
                    sum(float(row["forecast"]) for row in grouped),
                    float(aggregate["forecast"]), rel_tol=0, abs_tol=0.001,
                ) and math.isclose(
                    sum(float(row["scenario_forecast"]) for row in grouped),
                    float(aggregate["scenario_forecast"]), rel_tol=0, abs_tol=0.001,
                )
        channel_report = deterministic_forecast_findings(
            channel_forecast_question, channel_evidence
        )
        channel_ok = channel_ok and bool(channel_report) and any(
            "Across all sales channel groups" in finding.get("claim", "")
            for finding in channel_report.get("findings", [])
        )
        channel_detail = (
            f"{len(channel_forecast.get('rows', []))} channel forecast rows; "
            "base and +10% scenario totals independently verified"
        )
    except (ForecastingError, BusinessAgentError, StopIteration, KeyError, TypeError, ValueError) as exc:
        channel_ok = False
        channel_detail = str(exc)
    print(f"{'PASS' if channel_ok else 'FAIL'} channel forecast and scenario reconciliation: {channel_detail}")
    passed += int(channel_ok)

    malformed_union_sql = "SELECT 2 AS n ORDER BY n UNION ALL SELECT 1 AS n"
    repaired_union_sql = _repair_misplaced_compound_order_by(
        malformed_union_sql, "ORDER BY clause should come after UNION ALL not before"
    )
    try:
        with sqlite3.connect(":memory:") as connection:
            repaired_rows = connection.execute(repaired_union_sql or malformed_union_sql).fetchall()
        union_repair_ok = repaired_rows == [(1,), (2,)] and repaired_union_sql is not None
        union_detail = "relocated the rejected sort and SQLite executed the repaired compound query"
    except sqlite3.Error as exc:
        union_repair_ok = False
        union_detail = str(exc)
    print(f"{'PASS' if union_repair_ok else 'FAIL'} repairs misplaced ORDER BY before UNION ALL: {union_detail}")
    passed += int(union_repair_ok)

    historical_forecast_question = (
        "Run a historical forecast exercise, not a current forecast: treat 2025-12-31 as the forecast origin, "
        "use monthly delivered net product sales through December 2025, and forecast January–March 2026. "
        "Compare models with rolling-origin backtesting and report 80% prediction intervals."
    )
    try:
        historical_request = parse_forecast_request(historical_forecast_question)
        historical_task = PlannerTask(
            task_id="eval_historical_forecast", agent="forecasting_agent",
            objective="Retrospective monthly sales forecast from the approved 2025 year-end origin",
            data_sources=["orders", "order_items"], depends_on=[],
            evidence_required=["origin-bounded history, selected model, backtest, and forecast intervals"],
        )
        allowed, schemas = _schema_context(historical_task.data_sources)
        historical_evidence = ForecastingAgent(_execute_read_only).run_task(
            historical_forecast_question, historical_task, allowed, schemas
        )
        historical_result = next(item for item in historical_evidence if item.get("forecast_template"))
        historical_rows = historical_result.get("rows", [])
        historical_ok = (
            historical_request.forecast_origin.isoformat() == "2025-12-31"
            and historical_request.horizon == 3
            and historical_result.get("forecast_status") == "retrospective"
            and historical_result.get("source_cutoff_exclusive") == "2026-01-01"
            and len(historical_rows) == 3
            and [row.get("period_start") for row in historical_rows]
                == ["2026-01-01", "2026-02-01", "2026-03-01"]
            and not validate_forecast_evidence(historical_evidence)
        )
        historical_detail = (
            "fixed 2025-12-31 origin; December included, later rows excluded; Jan–Mar 2026 forecast independently validated"
            if historical_ok else str(historical_result.get("reason") or validate_forecast_evidence(historical_evidence))
        )
    except (BusinessAgentError, ForecastingError, StopIteration, KeyError, TypeError, ValueError, AttributeError) as exc:
        historical_ok = False
        historical_detail = str(exc)
    print(f"{'PASS' if historical_ok else 'FAIL'} retrospective forecast respects explicit historical origin: {historical_detail}")
    passed += int(historical_ok)

    lifecycle_sales_question = (
        "How did delivered net product sales and delivered order count change in September 2025 "
        "versus August 2025 and September 2024?"
    )
    lifecycle_sales_task = PlannerTask(
        task_id="eval_sales_counts", agent="data_analyst",
        objective="Compare overall delivered net sales and order counts",
        data_sources=["orders", "order_items"], depends_on=[],
        evidence_required=["sales changes and delivered order-count changes"],
    )
    allowed, schemas = _schema_context(lifecycle_sales_task.data_sources)
    sales_query = build_demo_query(lifecycle_sales_question, lifecycle_sales_task, allowed, schemas)
    if sales_query is None:
        sales_counts_ok = False
        sales_counts_detail = "overall comparison SQL was not selected"
    else:
        sales_result = _execute_read_only(sales_query.sql, lifecycle_sales_task)
        sales_result.update({
            "evidence_id": "eval_sales_counts:query1",
            "task_id": lifecycle_sales_task.task_id,
            "agent": lifecycle_sales_task.agent,
            "objective": lifecycle_sales_task.objective,
            "query_template": sales_query.template_id,
            "template_limitations": list(sales_query.limitations),
        })
        sales_result["result_validation"] = DataAnalystAgent.validate_result(sales_result)
        count_findings = _deterministic_demo_sales_findings(lifecycle_sales_question, [sales_result])
        count_claims = [finding["claim"] for finding in (count_findings or {}).get("findings", [])]
        chart = DataAnalystAgent.chart_hint(
            [
                {"month": "Sep", "period_1_sales": 1000.0, "period_2_sales": 800.0,
                 "period_1_order_count": 20, "period_2_order_count": 16},
                {"month": "Aug", "period_1_sales": 900.0, "period_2_sales": 700.0,
                 "period_1_order_count": 18, "period_2_order_count": 14},
            ],
            ["month", "period_1_sales", "period_2_sales", "period_1_order_count", "period_2_order_count"],
        )
        sales_counts_ok = (
            len(count_claims) == 2
            and all("Delivered orders changed from" in claim for claim in count_claims)
            and chart is not None
            and all("order_count" not in name for name in chart.get("y", []))
        )
        sales_counts_detail = (
            "report states both exact period order counts; chart selects sales only, not mixed dollar/count units"
            if sales_counts_ok else
            f"findings={len(count_claims)} claim={(count_claims[0][:240] if count_claims else 'missing')} chart={chart}"
        )
    print(f"{'PASS' if sales_counts_ok else 'FAIL'} sales report completeness and unit-consistent chart: {sales_counts_detail}")
    passed += int(sales_counts_ok)

    ranking_question = (
        "Compare delivered net product sales by product category in September 2025 "
        "versus August 2025 and September 2024. Rank declining categories and reconcile to the total."
    )
    ranking_task = PlannerTask(
        task_id="eval_rollup_rank_exclusion", agent="data_analyst",
        objective="Compare net product sales by category and rank category declines",
        data_sources=["orders", "order_items", "products", "customers"],
        depends_on=[], evidence_required=["category decline ranks and total reconciliation"],
    )
    allowed, schemas = _schema_context(ranking_task.data_sources)
    explicit_rank_sql = _period_comparison_sql(
        ranking_question, ranking_task, allowed, schemas
    )
    demo_rank_query = build_demo_query(
        ranking_question, ranking_task, allowed, schemas
    )
    rank_query_cases = [
        ("explicit category comparison", explicit_rank_sql),
        ("demo segmented comparison", demo_rank_query.sql if demo_rank_query else None),
    ]
    rank_case_results = []
    try:
        for label, sql in rank_query_cases:
            if not sql:
                rank_case_results.append((label, False, "query template unavailable"))
                continue
            result = _execute_read_only(sql, ranking_task)
            result["specialist_sql"] = sql
            checks = DataAnalystAgent.validate_result(result)
            rows = result.get("rows", [])
            totals = [
                row for row in rows
                if str(row.get("dimension_type", "")).casefold() == "total"
            ]
            categories = [
                row for row in rows
                if row.get("dimension_type") == "product_category"
            ]
            ranks = sorted(row.get("decline_rank_vs_period_2") for row in categories)
            rank_columns = [
                column for column in result.get("columns", [])
                if "rank" in column.casefold()
            ]
            ok = (
                len(totals) == 1 and bool(rank_columns)
                and all(totals[0].get(column) is None for column in rank_columns)
                and len(categories) == 5 and ranks == [1, 2, 3, 4, 5]
                and checks.get("all_checks_passed") is True
            )
            rank_case_results.append((
                label, ok,
                f"total ranks={ [totals[0].get(column) for column in rank_columns] if totals else 'missing' }; category ranks={ranks}",
            ))
        generated_rank_sql_ok = bool(rank_case_results) and all(
            ok for _label, ok, _detail in rank_case_results
        )
        generated_rank_sql_detail = "; ".join(
            f"{label}: {detail}" for label, _ok, detail in rank_case_results
        )
    except (BusinessAgentError, AnalystError, KeyError, TypeError, ValueError, AttributeError) as exc:
        generated_rank_sql_ok = False
        generated_rank_sql_detail = str(exc)
    print(f"{'PASS' if generated_rank_sql_ok else 'FAIL'} SQL templates leave total rows unranked: {generated_rank_sql_detail}")
    passed += int(generated_rank_sql_ok)

    misranked_rollup = {
        "columns": [
            "dimension_type", "dimension_value",
            "change_vs_period_2_usd", "decline_rank_vs_period_2",
        ],
        "rows": [
            {"dimension_type": "total", "dimension_value": "All delivered orders",
             "change_vs_period_2_usd": -5.0, "decline_rank_vs_period_2": 1},
            {"dimension_type": "product_category", "dimension_value": "Bedding",
             "change_vs_period_2_usd": -5.0, "decline_rank_vs_period_2": 1},
        ],
        "specialist_sql": "SELECT total_and_category_rank",
    }
    bad_rank_checks = DataAnalystAgent.validate_result(misranked_rollup)
    bad_rank_rejected = any(
        failure.startswith("ranking:rollup:decline_rank_vs_period_2")
        for failure in bad_rank_checks["failed_checks"]
    )
    misranked_rollup["rows"][0]["decline_rank_vs_period_2"] = None
    good_rank_checks = DataAnalystAgent.validate_result(misranked_rollup)
    good_rank_accepted = not any(
        failure.startswith("ranking:rollup:")
        for failure in good_rank_checks["failed_checks"]
    )
    rank_guard_ok = bad_rank_rejected and good_rank_accepted
    print(f"{'PASS' if rank_guard_ok else 'FAIL'} verifier rejects ranked totals and accepts NULL total rank")
    passed += int(rank_guard_ok)

    operations_question = (
        "Compare delivered-order on-time delivery rates and returned-unit rates in September 2025 versus August 2025. "
        "Show on-time and delivered counts by carrier, returned units by recorded reason, cite SQL evidence, "
        "and state that co-movement is an association rather than proven causation."
    )
    operation_tasks = [
        PlannerTask(
            task_id="eval_on_time_delivery", agent="data_analyst",
            objective="Compare on-time delivery counts and rates by carrier",
            data_sources=["orders", "fulfillment"], depends_on=[], evidence_required=["on-time/delivered counts by carrier"],
        ),
        PlannerTask(
            task_id="eval_return_rate", agent="data_analyst",
            objective="Compare returned-unit rate and breakdown by recorded return reason",
            data_sources=["orders", "order_items", "returns"], depends_on=[], evidence_required=["return-unit rate and reason breakdown"],
        ),
    ]
    operation_evidence = []
    operation_detail = ""
    try:
        for operation_task in operation_tasks:
            allowed, schemas = _schema_context(operation_task.data_sources)
            operation_query = build_demo_query(operations_question, operation_task, allowed, schemas)
            if operation_query is None:
                raise BusinessAgentError(f"No deterministic comparison query routed for {operation_task.task_id}.")
            result = _execute_read_only(operation_query.sql, operation_task)
            result.update({
                "evidence_id": operation_task.task_id + ":query1",
                "task_id": operation_task.task_id,
                "agent": operation_task.agent,
                "objective": operation_task.objective,
                "query_template": operation_query.template_id,
                "template_limitations": list(operation_query.limitations),
            })
            result["result_validation"] = DataAnalystAgent.validate_result(result)
            operation_evidence.append(result)
        operation_report = _draft_findings(operations_question, operation_evidence)
        operation_findings = _validate_findings(operation_report, operation_evidence)
        operation_issues = _verify_findings(
            operations_question, operation_findings, operation_evidence,
            PlannerTask("eval_verify_operations", "verification_agent", "Verify rates and SQL evidence", [], [], []),
        )
        all_operation_rows = [row for item in operation_evidence for row in item.get("rows", [])]
        has_carrier_rows = any(row.get("dimension_type") == "carrier" for row in all_operation_rows)
        has_reason_rows = any(row.get("dimension_type") == "return_reason" for row in all_operation_rows)
        operation_ok = not operation_issues and has_carrier_rows and has_reason_rows and all(
            item.get("result_validation", {}).get("all_checks_passed") is True for item in operation_evidence
        )
        operation_detail = (
            "two-month tasks routed to read-only SQL; carrier/reason rows, rates, evidence IDs, and deterministic report all validate"
            if operation_ok else "; ".join(operation_issues) or "expected operation groups or SQL checks were missing"
        )
    except (BusinessAgentError, AnalystError, KeyError, TypeError, ValueError, AttributeError) as exc:
        operation_ok = False
        operation_detail = str(exc)
    print(f"{'PASS' if operation_ok else 'FAIL'} two-period delivery/returns report and deterministic verification: {operation_detail}")
    passed += int(operation_ok)

    carrier_ship_month_question = "Compare delivered on-time rates by carrier in September 2025 and August 2025."
    carrier_ship_month_task = PlannerTask(
        task_id="eval_carrier_ship_month", agent="product_operations_analyst",
        objective="Compare delivered on-time delivery rates by carrier for September 2025 versus August 2025.",
        data_sources=["orders", "fulfillment"], depends_on=[],
        evidence_required=["on-time and delivered shipment counts grouped by fulfillment ship month"],
    )
    allowed, schemas = _schema_context(carrier_ship_month_task.data_sources)
    carrier_query = build_demo_query(carrier_ship_month_question, carrier_ship_month_task, allowed, schemas)
    carrier_ship_month_ok = carrier_query is not None and "f.ship_date >= p.start_date" in carrier_query.sql
    carrier_detail = "period filtering uses fulfillment.ship_date" if carrier_ship_month_ok else "query did not filter by fulfillment ship month"
    if carrier_query is not None:
        try:
            carrier_evidence = _execute_read_only(carrier_query.sql, carrier_ship_month_task)
            carrier_evidence.update({
                "evidence_id": carrier_ship_month_task.task_id + ":query1",
                "task_id": carrier_ship_month_task.task_id,
                "agent": carrier_ship_month_task.agent,
                "objective": carrier_ship_month_task.objective,
                "query_template": carrier_query.template_id,
                "template_limitations": list(carrier_query.limitations),
            })
            carrier_evidence["result_validation"] = DataAnalystAgent.validate_result(carrier_evidence)
            carrier_rows = carrier_evidence["rows"]
            expected_carrier_rates = {
                ("GroundLine", "August 2025"): (27, 32, 84.38),
                ("GroundLine", "September 2025"): (23, 29, 79.31),
                ("ParcelNorth", "August 2025"): (37, 42, 88.10),
                ("ParcelNorth", "September 2025"): (12, 15, 80.00),
                ("SwiftPost", "August 2025"): (40, 42, 95.24),
                ("SwiftPost", "September 2025"): (28, 32, 87.50),
            }
            actual_rates = {
                (row["dimension_value"], row["period_label"]):
                    (row["numerator"], row["denominator"], row["rate_pct"])
                for row in carrier_rows
            }
            carrier_report = _draft_findings(carrier_ship_month_question, [carrier_evidence])
            carrier_findings = _validate_findings(carrier_report, [carrier_evidence])
            carrier_findings_text = " ".join(
                [finding.get("claim", "") for finding in carrier_findings]
                + [finding.get("calculation", "") for finding in carrier_findings]
                + [note for finding in carrier_findings for note in finding.get("limitations", [])]
            )
            carrier_report_issues = _verify_findings(
                carrier_ship_month_question, carrier_findings, [carrier_evidence],
                PlannerTask("verify_carrier_report", "verification_agent", "Verify carrier report", [], [], []),
            )
            focused_report = (
                len(carrier_findings) == 1
                and not carrier_report_issues
                and "on-time delivery rate" in carrier_findings_text.lower()
                and "fulfillment ship month" in carrier_findings_text.lower()
                and "both promised and actual delivery dates" in carrier_findings_text.lower()
                and "ParcelNorth had the largest observed on-time rate decline" in carrier_findings_text
                and not re.search(r"\binventory\b|\breturns?\b", carrier_findings_text, re.IGNORECASE)
            )
            carrier_ship_month_ok = (
                carrier_ship_month_ok
                and actual_rates == expected_carrier_rates
                and all(row.get("metric_type") == "on_time_delivery_rate" for row in carrier_rows)
                and carrier_evidence["result_validation"]["all_checks_passed"] is True
                and focused_report
            )
            if not carrier_ship_month_ok:
                carrier_detail = (
                    "ship-month counts/rates, report focus, evidence verification, or delivery caveats failed; "
                    + carrier_findings_text[:320]
                )
        except (BusinessAgentError, KeyError, TypeError):
            carrier_ship_month_ok = False
            carrier_detail = "carrier ship-month SQL failed or returned an invalid shape"
    print(f"{'PASS' if carrier_ship_month_ok else 'FAIL'} carrier comparison period follows fulfillment ship_date: {carrier_detail}")
    passed += int(carrier_ship_month_ok)

    bad_arithmetic = {
        "columns": ["current_value", "comparison_value", "change_usd", "change_pct"],
        "rows": [{"current_value": 110, "comparison_value": 100, "change_usd": 50, "change_pct": 50}],
        "specialist_sql": "SELECT 110, 100, 50, 50",
    }
    arithmetic_checks = DataAnalystAgent.validate_result(bad_arithmetic)
    arithmetic_ok = not arithmetic_checks["all_checks_passed"] and any(
        "arithmetic:change_usd" in failure for failure in arithmetic_checks["failed_checks"]
    )
    print(f"{'PASS' if arithmetic_ok else 'FAIL'} detects incorrect reported change")
    passed += int(arithmetic_ok)

    bad_rollup = {
        "columns": ["dimension_type", "dimension_value", "period_1_sales", "period_2_sales",
                    "period_3_sales", "change_vs_period_2_usd", "change_vs_period_2_pct",
                    "change_vs_period_3_usd", "change_vs_period_3_pct"],
        "rows": [
            {"dimension_type": "total", "dimension_value": "all", "period_1_sales": 100,
             "period_2_sales": 90, "period_3_sales": 80, "change_vs_period_2_usd": 10,
             "change_vs_period_2_pct": 11.11, "change_vs_period_3_usd": 20, "change_vs_period_3_pct": 25},
            {"dimension_type": "region", "dimension_value": "A", "period_1_sales": 40,
             "period_2_sales": 40, "period_3_sales": 30, "change_vs_period_2_usd": 0,
             "change_vs_period_2_pct": 0, "change_vs_period_3_usd": 10, "change_vs_period_3_pct": 33.33},
            {"dimension_type": "region", "dimension_value": "B", "period_1_sales": 50,
             "period_2_sales": 50, "period_3_sales": 40, "change_vs_period_2_usd": 0,
             "change_vs_period_2_pct": 0, "change_vs_period_3_usd": 10, "change_vs_period_3_pct": 25},
        ],
        "specialist_sql": "SELECT total_and_region_rollup",
    }
    rollup_checks = DataAnalystAgent.validate_result(bad_rollup)
    rollup_ok = any("reconciliation:region:period_1_sales" in failure
                    for failure in rollup_checks["failed_checks"])
    print(f"{'PASS' if rollup_ok else 'FAIL'} detects an inconsistent total/detail rollup")
    passed += int(rollup_ok)

    bad_zero_baseline = {
        "columns": ["current_value", "comparison_value", "change_usd", "change_pct"],
        "rows": [{"current_value": 5, "comparison_value": 0, "change_usd": 5, "change_pct": 0}],
        "specialist_sql": "SELECT zero_baseline_example",
    }
    zero_checks = DataAnalystAgent.validate_result(bad_zero_baseline)
    zero_ok = any("arithmetic:change_pct" in failure and "NULL" in failure
                  for failure in zero_checks["failed_checks"])
    print(f"{'PASS' if zero_ok else 'FAIL'} rejects a percentage with a zero comparison baseline")
    passed += int(zero_ok)

    gross_profit_question = "Rank product categories by gross profit for the second half of 2025."
    gross_profit_task = PlannerTask(
        task_id="eval_gross_profit_category_h2", agent="data_analyst",
        objective="Rank delivered product-category gross profit for the second half of 2025",
        data_sources=["orders", "order_items", "products"], depends_on=[],
        evidence_required=["category gross-profit rows, total reconciliation, and exact SQL"],
    )
    gross_profit_ok = False
    gross_profit_detail = "query template unavailable"
    try:
        allowed, schemas = _schema_context(gross_profit_task.data_sources)
        gross_profit_query = build_demo_query(gross_profit_question, gross_profit_task, allowed, schemas)
        if gross_profit_query is None:
            raise BusinessAgentError("No deterministic gross-profit query was selected.")
        gross_profit_evidence = _execute_read_only(gross_profit_query.sql, gross_profit_task)
        gross_profit_evidence.update({
            "evidence_id": gross_profit_task.task_id + ":query1",
            "agent": gross_profit_task.agent,
            "query_template": gross_profit_query.template_id,
            "template_limitations": list(gross_profit_query.limitations),
        })
        gross_profit_rows = gross_profit_evidence["rows"]
        category_rows = [row for row in gross_profit_rows if row["dimension_type"] == "product_category"]
        total_rows = [row for row in gross_profit_rows if row["dimension_type"] == "total"]
        expected_profit = [
            ("Storage", 17104.52), ("Decor", 10782.16), ("Bedding", 9666.75),
            ("Lighting", 9025.16), ("Kitchen", 8696.95),
        ]
        actual_profit = [(row["dimension_value"], row["gross_profit_usd"]) for row in category_rows]
        gross_profit_draft = _draft_findings(gross_profit_question, [gross_profit_evidence])
        gross_profit_findings = _validate_findings(gross_profit_draft, [gross_profit_evidence])
        gross_claim = gross_profit_findings[0]["claim"] if gross_profit_findings else ""
        gross_profit_ok = (
            gross_profit_query.template_id == "demo_gross_profit_category_window_v1"
            and actual_profit == expected_profit and len(total_rows) == 1
            and total_rows[0]["gross_profit_usd"] == 55275.54
            and round(sum(value for _, value in actual_profit), 2) == 55275.54
            and all(name in gross_claim for name, _ in expected_profit)
            and all(f"{value:,.2f}" in gross_claim for _, value in expected_profit)
        )
        gross_profit_detail = (
            f"{len(category_rows)} categories; rank and USD 55,275.54 total independently match; "
            "fixed SQL evidence and deterministic verified report"
            if gross_profit_ok else
            f"rows={actual_profit}; total={total_rows}; claim={gross_claim}"
        )
    except (BusinessAgentError, AnalystError, KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        gross_profit_detail = str(exc)
    print(f"{'PASS' if gross_profit_ok else 'FAIL'} gross-profit category query, reconciliation, and report: {gross_profit_detail}")
    passed += int(gross_profit_ok)

    channel_question = "Compare delivered net sales by sales channel for September and August 2025."
    channel_task = PlannerTask(
        task_id="eval_channel_sales_two_months", agent="data_analyst",
        objective="Compare delivered net product sales for each sales channel across September and August 2025",
        data_sources=["orders", "order_items"], depends_on=[],
        evidence_required=["channel rows, order counts, and an overall reconciliation"],
    )
    channel_sales_ok = False
    channel_sales_detail = "query template unavailable"
    try:
        allowed, schemas = _schema_context(channel_task.data_sources)
        channel_query = build_demo_query(channel_question, channel_task, allowed, schemas)
        if channel_query is None:
            raise BusinessAgentError("No deterministic two-period channel query was selected.")
        channel_evidence = _execute_read_only(channel_query.sql, channel_task)
        channel_evidence.update({
            "evidence_id": channel_task.task_id + ":query1",
            "agent": channel_task.agent,
            "query_template": channel_query.template_id,
            "template_limitations": list(channel_query.limitations),
        })
        channel_evidence["result_validation"] = DataAnalystAgent.validate_result(channel_evidence)
        channel_rows = channel_evidence["rows"]
        total_row = next(row for row in channel_rows if row["dimension_type"] == "total")
        by_channel = [row for row in channel_rows if row["dimension_type"] == "sales_channel"]
        expected_channel_changes = {
            "Marketplace": (6485.98, 4357.30, -2128.68, -32.82),
            "Retail Partner": (598.36, 508.70, -89.66, -14.98),
            "Web": (11234.29, 6142.18, -5092.11, -45.33),
        }
        actual_channel_changes = {
            row["dimension_value"]: (
                round(row["period_2_sales"], 2), round(row["period_1_sales"], 2),
                round(row["change_vs_period_2_usd"], 2), round(row["change_vs_period_2_pct"], 2),
            ) for row in by_channel
        }
        channel_draft = _draft_findings(channel_question, [channel_evidence])
        channel_findings = _validate_findings(channel_draft, [channel_evidence])
        channel_claims = " ".join(item["claim"] for item in channel_findings)
        channel_sales_ok = (
            channel_query.template_id == "demo_sales_channel_two_period_v1"
            and channel_evidence["result_validation"].get("all_checks_passed") is True
            and actual_channel_changes == expected_channel_changes
            and round(total_row["period_2_sales"], 2) == 18318.63
            and round(total_row["period_1_sales"], 2) == 11008.18
            and round(total_row["change_vs_period_2_usd"], 2) == -7310.45
            and len(channel_findings) == 3
            and "USD -7,310.45" in channel_claims
            and "Web: USD -5,092.11" in channel_claims
            and "channel sales and delivered order counts also reconcile" in channel_claims
        )
        channel_sales_detail = (
            "September/August channel sales and percentages match the independent SQL fixture; "
            "all channels, order counts, total change, and evidence-linked report reconcile"
            if channel_sales_ok else
            f"rows={actual_channel_changes}; total={total_row}; claims={channel_claims}"
        )
    except (BusinessAgentError, AnalystError, KeyError, TypeError, ValueError, AttributeError, StopIteration) as exc:
        channel_sales_detail = str(exc)
    print(f"{'PASS' if channel_sales_ok else 'FAIL'} two-period channel sales analysis and reconciliation: {channel_sales_detail}")
    passed += int(channel_sales_ok)

    rag_text_evidence = [{
        "evidence_id": "review_task:rag1:1",
        "evidence_type": "document_context",
        "source_name": "business_demo.reviews",
        "source_ref": "business_demo.reviews#fixture:1",
        "excerpt": "Review text: Setup was simple and the finish looked good.",
    }]
    bounded_review_finding = {
        "claim": "The retrieved excerpt praises the simple setup and product finish (evidence_id: review_task:rag1:1).",
        "evidence_ids": ["review_task:rag1:1"],
        "calculation": "Qualitative summary of the cited passage.",
        "limitations": ["The excerpt is illustrative, not a population estimate."],
    }
    rag_bounded_ok = False
    rag_bounded_detail = "sample wording was not validated"
    try:
        bounded = _validate_findings({"findings": [bounded_review_finding]}, rag_text_evidence)
        numeric_guess = {**bounded_review_finding, "claim": "Three themes appear in the reviews."}
        prevalence_guess = {**bounded_review_finding, "claim": "Reviews often praise the finish."}
        numeric_blocked = prevalence_blocked = False
        try:
            _validate_findings({"findings": [numeric_guess]}, rag_text_evidence)
        except BusinessAgentError as exc:
            numeric_blocked = "must cite SQL result evidence" in str(exc)
        try:
            _validate_findings({"findings": [prevalence_guess]}, rag_text_evidence)
        except BusinessAgentError as exc:
            prevalence_blocked = "cannot support review-wide frequency" in str(exc)
        rating_evidence = [{
            **rag_text_evidence[0],
            "excerpt": "Product category: Storage. Rating: 4/5. Review text: Nice product and easy to use.",
        }]
        rating_finding = {
            **bounded_review_finding,
            "claim": "A retrieved Storage review was rated 4/5 and praised its easy use.",
        }
        supported_rating = unsupported_rating_blocked = False
        try:
            _validate_findings({"findings": [rating_finding]}, rating_evidence)
            supported_rating = True
        except BusinessAgentError:
            pass
        wrong_rating = {**rating_finding, "claim": "A retrieved Storage review was rated 5/5."}
        try:
            _validate_findings({"findings": [wrong_rating]}, rating_evidence)
        except BusinessAgentError as exc:
            unsupported_rating_blocked = "must match a cited review passage" in str(exc)
        alternate_rating_wording = {
            **rating_finding,
            "claim": "A 4/5 review praised the product's ease of use.",
        }
        alternate_rating_supported = False
        try:
            _validate_findings({"findings": [alternate_rating_wording]}, rating_evidence)
            alternate_rating_supported = True
        except BusinessAgentError:
            pass
        rag_bounded_ok = (
            len(bounded) == 1 and numeric_blocked and prevalence_blocked
            and supported_rating and unsupported_rating_blocked and alternate_rating_supported
        )
        rag_bounded_detail = (
            "bounded text accepted; counts/prevalence rejected; cited ratings allowed in two phrasings and unsupported rating rejected"
            if rag_bounded_ok else
            f"bounded={len(bounded)}; numeric_blocked={numeric_blocked}; prevalence_blocked={prevalence_blocked}; "
            f"supported_rating={supported_rating}; unsupported_rating_blocked={unsupported_rating_blocked}; "
            f"alternate_rating_supported={alternate_rating_supported}"
        )
    except BusinessAgentError as exc:
        rag_bounded_detail = str(exc)
    print(f"{'PASS' if rag_bounded_ok else 'FAIL'} RAG faithfulness boundary: {rag_bounded_detail}")
    passed += int(rag_bounded_ok)

    rag_task = PlannerTask(
        task_id="eval_rag_search_limit", agent="data_analyst",
        objective="Summarize representative review themes from retrieved text only",
        data_sources=["reviews"], depends_on=[],
        evidence_required=["retrieved review passages and an explicit coverage limitation"],
    )
    rag_calls = {"count": 0}

    def keep_searching(_question, _task, _state):
        return {"action": "retrieve", "query": "Storage customer review themes"}

    def one_review_passage(_query, _top_k):
        rag_calls["count"] += 1
        return [{
            "source_name": "business_demo.reviews",
            "source_ref": f"business_demo.reviews#fixture:{rag_calls['count']}",
            "content": "Product category: Storage. Review text: Useful and sturdy.",
        }]

    rag_agent = DataAnalystAgent(
        keep_searching, lambda *_args: {}, retrieval_tool=one_review_passage,
    )
    rag_limit_ok = False
    rag_limit_detail = "search loop did not complete"
    try:
        rag_evidence = rag_agent.run_task("Summarize Storage reviews", rag_task)
        passages = [item for item in rag_evidence if item.get("evidence_type") == "document_context"]
        limit_notes = [item for item in rag_evidence if item.get("status") == "limit_reached"]
        rag_limit_ok = (
            rag_calls["count"] == 3 and len(passages) == 3 and len(limit_notes) == 1
            and rag_evidence[-1].get("analysis_complete") is True
            and "not be presented as a full-corpus count" in limit_notes[0].get("message", "")
        )
        rag_limit_detail = (
            "three searches retained as cited text evidence; over-limit search stopped with an explicit bound"
            if rag_limit_ok else
            f"calls={rag_calls['count']}; passage_count={len(passages)}; limit_notes={len(limit_notes)}"
        )
    except AnalystError as exc:
        rag_limit_detail = str(exc)
    print(f"{'PASS' if rag_limit_ok else 'FAIL'} bounded RAG search exhaustion: {rag_limit_detail}")
    passed += int(rag_limit_ok)

    review_route_task = PlannerTask(
        task_id="eval_review_theme_route", agent="customer_analyst",
        objective="Summarize positive and negative Storage review themes from passages",
        data_sources=["reviews"], depends_on=[],
        evidence_required=["RAG excerpts supporting bounded qualitative themes"],
    )
    review_route_state = {
        "query_history": [], "retrieval_history": [], "rag_calls_remaining": 3,
        "rag_unavailable": False,
    }
    review_route_ok = False
    review_route_detail = "review task did not deterministically route to RAG"
    try:
        first_action = _make_analysis_action(
            "Using indexed customer-review text, summarize positive and negative themes for Storage products.",
            review_route_task, review_route_state,
        )
        after_retrieval = {
            **review_route_state,
            "retrieval_history": [{"query": first_action.get("query", ""), "result_count": 5}],
            "rag_calls_remaining": 2,
        }
        finish_action = _make_analysis_action(
            "Using indexed customer-review text, summarize positive and negative themes for Storage products.",
            review_route_task, after_retrieval,
        )
        numeric_task = PlannerTask(
            task_id="eval_review_numeric_route", agent="customer_analyst",
            objective="Compute the average rating of Storage product reviews",
            data_sources=["reviews"], depends_on=[],
            evidence_required=["SQL evidence for average review rating"],
        )
        with patch("business_agents._make_sql", return_value="SELECT AVG(r.rating) AS avg_rating FROM reviews AS r"):
            numeric_action = _make_analysis_action(
                "What was the average rating for Storage product reviews?",
                numeric_task, review_route_state,
            )
        review_route_ok = (
            first_action.get("action") == "retrieve"
            and "Storage" in first_action.get("query", "")
            and finish_action.get("action") == "done"
            and numeric_action.get("action") == "query"
            and "AVG" in numeric_action.get("sql", "")
        )
        review_route_detail = (
            "qualitative review themes retrieve passages then finish; explicit average-rating metric routes to SQL"
            if review_route_ok else
            f"first={first_action}; after_retrieval={finish_action}; numeric={numeric_action}"
        )
    except (BusinessAgentError, KeyError, TypeError, ValueError) as exc:
        review_route_detail = str(exc)
    print(f"{'PASS' if review_route_ok else 'FAIL'} qualitative review routing versus numeric review SQL: {review_route_detail}")
    passed += int(review_route_ok)

    review_fixture_evidence = [
        {
            "evidence_id": "eval_reviews:rag1:1", "evidence_type": "document_context",
            "source_name": "business_demo.reviews", "source_ref": "business_demo.reviews#fixture:1",
            "excerpt": "Product category: Storage. Rating: 5/5. Review text: Beautiful finish and fast delivery.",
        },
        {
            "evidence_id": "eval_reviews:rag1:2", "evidence_type": "document_context",
            "source_name": "business_demo.reviews", "source_ref": "business_demo.reviews#fixture:2",
            "excerpt": "Product category: Storage. Rating: 4/5. Review text: Nice product and easy to use.",
        },
        {
            "evidence_id": "eval_reviews:rag1:3", "evidence_type": "document_context",
            "source_name": "business_demo.reviews", "source_ref": "business_demo.reviews#fixture:3",
            "excerpt": "Product category: Storage. Rating: 1/5. Review text: Arrived damaged and did not match the listing.",
        },
    ]
    deterministic_review_ok = False
    deterministic_review_detail = "deterministic review report was not validated"
    try:
        review_report = _deterministic_review_text_findings(
            "Summarize review themes and sentiment for Storage products.", review_fixture_evidence,
        )
        review_findings = _validate_findings(review_report or {}, review_fixture_evidence)
        review_claims = " ".join(item["claim"] for item in review_findings)
        deterministic_review_ok = (
            len(review_findings) == 2
            and "Beautiful finish and fast delivery" in review_claims
            and "Nice product and easy to use" in review_claims
            and "Arrived damaged and did not match the listing" in review_claims
            and set(review_findings[0]["evidence_ids"]) == {"eval_reviews:rag1:1", "eval_reviews:rag1:2"}
            and review_findings[1]["evidence_ids"] == ["eval_reviews:rag1:3"]
            and all("full review population" in " ".join(item["limitations"]) for item in review_findings)
        )
        deterministic_review_detail = (
            "review quotes and citations are deterministic; grouping and sample limitations are explicit"
            if deterministic_review_ok else f"findings={review_findings}"
        )
    except (BusinessAgentError, KeyError, TypeError, ValueError) as exc:
        deterministic_review_detail = str(exc)
    print(f"{'PASS' if deterministic_review_ok else 'FAIL'} deterministic, passage-cited review report: {deterministic_review_detail}")
    passed += int(deterministic_review_ok)

    print(f"Offline benchmark: {passed}/{total} passed. No LLM/API was called.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(run())




