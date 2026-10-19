"""Customer-specialist role over the shared read-only SQL and on-demand RAG tools."""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
import re
from typing import Any, Callable

from analyst import AnalystError, DataAnalystAgent
from observability import trace_component


CUSTOMER_ANALYST_GUIDANCE = """CUSTOMER ANALYST SPECIALTY
Analyze customer behavior using any relevant approved data columns and
schema-grounded read-only SQL. Do not require fixed templates or exactly three
months. Match the requested metric, period, comparison, and customer grain.
Keep customer-level joins accurate; aggregate before joining order items when
needed to avoid duplicate customer/order counts.

Return aggregate results only; never expose customer IDs or personal fields.
Use customers.first_order_date only when defining first-recorded new versus
returning status, and explain that it is not necessarily first-ever history.
State the definitions and denominators used for active customers, repeat
purchases, retention, or churn. Ask for clarification when a required definition
is missing and cannot be safely assumed. Do not relabel repeat rate as churn or
historical revenue as predictive lifetime value.

Use SQL for exact counts, rates, revenue, and explicitly requested numeric review
metrics such as rating averages or distributions. Qualitative review themes,
sentiment, quotations, representative examples, and review wording MUST be
retrieved through RAG; never concatenate review text in SQL or infer themes from
rating counts. Retrieved examples do not establish population-wide sentiment or
business totals. Report data quality, observation-window, and
maturation limits that affect the result. Do not infer demographics or claim a
cause from observational segments. Customer-key integrity checks must remain
aggregate and must not disclose individual identifiers.""".strip()


CUSTOMER_QUALITY_SQL = """SELECT
  (SELECT COUNT(*) FROM customers
   WHERE customer_id IS NULL OR TRIM(CAST(customer_id AS TEXT)) = '') AS blank_customer_keys,
  (SELECT COUNT(*) FROM (
     SELECT customer_id FROM customers
     WHERE customer_id IS NOT NULL AND TRIM(CAST(customer_id AS TEXT)) <> ''
     GROUP BY customer_id HAVING COUNT(*) > 1
   )) AS duplicate_customer_keys,
  (SELECT COUNT(*) FROM orders AS o
   LEFT JOIN customers AS c ON c.customer_id = o.customer_id
   WHERE o.status = 'delivered'
     AND (o.customer_id IS NULL OR c.customer_id IS NULL)) AS unlinked_delivered_orders,
  (SELECT COUNT(*) FROM customers
   WHERE first_order_date IS NOT NULL
     AND TRIM(CAST(first_order_date AS TEXT)) <> ''
     AND (date(first_order_date) IS NULL
          OR date(first_order_date) <> substr(TRIM(CAST(first_order_date AS TEXT)), 1, 10)))
    AS invalid_first_order_dates,
  (SELECT MIN(order_date) FROM orders) AS orders_min_date,
  (SELECT MAX(order_date) FROM orders) AS orders_max_date"""


def _named_month_starts(question: str) -> list[date]:
    month_numbers = {calendar.month_name[number].lower(): number for number in range(1, 13)}
    pattern = r"\b(" + "|".join(month_numbers) + r")\s+(20\d{2})\b"
    starts = []
    for match in re.finditer(pattern, question, flags=re.IGNORECASE):
        starts.append(date(int(match.group(2)), month_numbers[match.group(1).lower()], 1))
    return list(dict.fromkeys(starts))


class CustomerAnalystAgent(DataAnalystAgent):
    """Customer-focused analyst using the shared bounded query/RAG execution loop."""

    def __init__(
        self,
        action_generator: Callable[[str, Any, dict[str, Any]], dict[str, Any]],
        query_executor: Callable[[str, Any], dict[str, Any]],
        retrieval_tool: Callable[[str, int], list[dict[str, Any]]] | None = None,
        db_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            action_generator,
            query_executor,
            db_path=db_path,
            catalog_path=catalog_path,
            retrieval_tool=retrieval_tool,
            agent_name="customer_analyst",
            allow_cleaning=False,
        )

    @trace_component("Customer Analyst", run_type="chain")
    def run_task(self, question: str, task: Any) -> list[dict[str, Any]]:
        """Run aggregate identity checks before any customer-grain analysis."""
        required = {"customers", "orders"}
        if not required.issubset(set(getattr(task, "data_sources", []))):
            return super().run_task(question, task)

        audit = self.query_executor(CUSTOMER_QUALITY_SQL, task)
        rows = audit.get("rows") if isinstance(audit, dict) else None
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise AnalystError("Customer identity preflight returned no usable aggregate evidence; analysis was stopped.")
        row = rows[0]
        required_counts = (
            "blank_customer_keys", "duplicate_customer_keys", "unlinked_delivered_orders",
            "invalid_first_order_dates",
        )
        if any(
            isinstance(row.get(name), bool) or not isinstance(row.get(name), int) or row[name] < 0
            for name in required_counts
        ):
            raise AnalystError("Customer identity preflight returned invalid counts; analysis was stopped.")
        min_date, max_date = row.get("orders_min_date"), row.get("orders_max_date")
        if not isinstance(min_date, str) or not isinstance(max_date, str):
            raise AnalystError("Customer order-date coverage could not be established; analysis was stopped.")
        try:
            min_month = date.fromisoformat(min_date[:10]).replace(day=1)
            max_month = date.fromisoformat(max_date[:10]).replace(day=1)
        except ValueError:
            raise AnalystError("Customer order-date coverage is invalid; analysis was stopped.") from None
        named_months = _named_month_starts(question)
        if named_months and any(month < min_month or month > max_month for month in named_months):
            raise AnalystError(
                f"Customer analysis was stopped because the request includes a month outside the observed "
                f"order-date range ({min_date[:10]} through {max_date[:10]})."
            )
        audit["result_validation"] = DataAnalystAgent.validate_result(audit)
        if not audit["result_validation"].get("all_checks_passed"):
            raise AnalystError("Customer identity preflight failed result validation; analysis was stopped.")
        if row["duplicate_customer_keys"]:
            raise AnalystError(
                "Customer analysis was stopped because duplicate customer IDs can multiply customer-level joins. "
                "Resolve the duplicate identity records before retrying."
            )
        if row["unlinked_delivered_orders"] or row["blank_customer_keys"]:
            raise AnalystError(
                "Customer analysis was stopped because delivered orders or customer records have missing identity keys. "
                "Resolve the customer-to-order links before retrying."
            )

        flags = []
        for name, label in (
            ("blank_customer_keys", "customer rows have blank IDs"),
            ("unlinked_delivered_orders", "delivered orders have no matching customer record"),
            ("invalid_first_order_dates", "customer first-order dates are invalid"),
        ):
            if row[name]:
                flags.append(f"{row[name]:,} {label}")

        audit_evidence = {
            **audit,
            "evidence_id": f"{task.task_id}:customer_quality",
            "task_id": task.task_id,
            "agent": self.agent_name,
            "objective": "Preflight customer identity and join-key quality using aggregate counts only.",
            "evidence_type": "customer_data_quality",
            "quality_flags": flags,
            "blocking_checks_passed": True,
            "date_coverage": {"orders_min_date": min_date[:10], "orders_max_date": max_date[:10]},
        }
        results = super().run_task(question, task)
        for item in results:
            if item.get("specialist_sql"):
                item["customer_quality_evidence_id"] = audit_evidence["evidence_id"]
                item["customer_quality_flags"] = flags
                item["customer_quality_sql"] = audit.get("specialist_sql", CUSTOMER_QUALITY_SQL)
                item["customer_data_coverage"] = audit_evidence["date_coverage"]
                item.setdefault("template_limitations", []).append(
                    f"Observed order dates run from {min_date[:10]} through {max_date[:10]}; the latest observed date is not a source-completeness watermark."
                )
                if flags:
                    item.setdefault("template_limitations", []).append(
                        "Customer data-quality preflight found: " + "; ".join(flags) + "."
                    )
        return [audit_evidence, *results]
