"""Product and operations specialist over approved read-only business data."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from analyst import AnalystError, DataAnalystAgent
from planner import PlannerTask


PRODUCT_OPERATIONS_GUIDANCE = """PRODUCT AND OPERATIONS ANALYST
Analyze product performance and operational questions using any relevant
approved data columns. Select dimensions, filters, date ranges, and calculations
from the user's request and the supplied schema; do not assume a fixed query
template or fixed number of periods. Generate one focused read-only SQL query at
a time, preserve the correct join grain, and use SQL results as evidence.

Use only approved sources and exact schema fields. Product sales should use
order-item measures and the requested order-status filter. Distinguish product
sales from shipping and later refunds when relevant. Treat inventory snapshots
as point-in-time observations; state the eligible population for delivery and
return rates. Do not claim inventory, supplier, marketing, competitor, or
fulfillment causes unless the approved data measures that relationship. Separate
measured facts from associations and causes. Never mutate source data, expose
customer identifiers, or claim a dimension absent from the approved schema.""".strip()


class ProductOperationsAgent(DataAnalystAgent):
    """Product/operations role using the shared bounded read-only analysis loop."""

    def __init__(
        self,
        action_generator: Callable[[str, Any, dict[str, Any]], dict[str, Any]],
        query_executor: Callable[[str, Any], dict[str, Any]],
        retrieval_tool: Callable[[str, int], list[dict[str, Any]]] | None = None,
        db_path: str | Path | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        def guided_action(question: str, task: Any, state: dict[str, Any]) -> dict[str, Any]:
            enriched_state = dict(state)
            enriched_state["specialist_guidance"] = PRODUCT_OPERATIONS_GUIDANCE
            return action_generator(question, task, enriched_state)

        super().__init__(
            guided_action,
            query_executor,
            db_path=db_path,
            catalog_path=catalog_path,
            retrieval_tool=retrieval_tool,
            agent_name="product_operations_analyst",
            allow_cleaning=False,
        )

    def run_task(self, question: str, task: PlannerTask) -> list[dict[str, Any]]:
        if task.agent != self.agent_name:
            raise AnalystError(f"{self.agent_name} cannot accept a {task.agent} task.")
        if not task.data_sources:
            raise AnalystError("Product and operations analysis needs at least one approved data source.")
        return super().run_task(question, task)
