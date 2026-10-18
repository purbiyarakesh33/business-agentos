"""Offline planner contract checks; no API key or model call is used."""
from __future__ import annotations

import json
from unittest.mock import patch

import planner
from jev import JevDecision
from planner import PlannerError


CATALOG = [{"id": "orders"}, {"id": "order_items"}]


def ready_plan(objective: str | None = "Compare delivered sales by month.") -> dict:
    task = {
        "task_id": "sales_compare",
        "agent": "data_analyst",
        "data_sources": ["orders", "order_items"],
        "depends_on": [],
        "evidence_required": ["Query result and filters"],
    }
    if objective is not None:
        task["objective"] = objective
    return {
        "status": "ready",
        "summary": "Compare the requested periods using delivered orders.",
        "assumptions": [],
        "missing_information": [],
        "unavailable_data": [],
        "tasks": [task],
    }


def expect_objective_error(plan: dict, expected: str) -> None:
    try:
        planner._validate(plan, "Compare sales by month.", "fixture-model", CATALOG)
    except PlannerError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected objective error containing {expected!r}")


def main() -> None:
    valid = planner._validate(
        ready_plan(), "Compare sales by month.", "fixture-model", CATALOG
    )
    assert valid.status == "ready" and valid.requires_approval
    print("PASS valid planner plan remains pending human approval")

    expect_objective_error(ready_plan(None), "objective is missing")
    print("PASS missing objective gets a specific diagnostic")

    wrong_type = ready_plan()
    wrong_type["tasks"][0]["objective"] = ["not", "text"]
    expect_objective_error(wrong_type, "must be a string (received list)")
    print("PASS non-string objective gets a specific diagnostic")

    blank = ready_plan("   ")
    expect_objective_error(blank, "objective is empty")
    print("PASS blank objective gets a specific diagnostic")

    too_long = ready_plan("x" * (planner.MAX_TASK_OBJECTIVE_CHARS + 1))
    expect_objective_error(too_long, "is 701 characters; the maximum is 700")
    print("PASS overlong objective reports actual and maximum lengths")

    bad = ready_plan("x" * (planner.MAX_TASK_OBJECTIVE_CHARS + 1))
    fixed = ready_plan("Compare delivered sales across the requested months.")
    decision = JevDecision(
        action="ALLOW",
        security="normal",
        intent="business_performance",
        security_confidence=1.0,
        intent_confidence=1.0,
        model="fixture-jev",
    )
    with (
        patch("planner._read_context", return_value=(CATALOG, "fixture dictionary")),
        patch(
            "planner._request_plan",
            side_effect=[
                (json.dumps(bad), "fixture-model"),
                (json.dumps(fixed), "fixture-model"),
            ],
        ) as request,
    ):
        repaired = planner.plan_request("Compare sales by month.", decision)
    assert repaired.status == "ready"
    assert request.call_count == 2
    repair_kwargs = request.call_args.kwargs
    assert "maximum is 700" in repair_kwargs["repair_error"]
    assert len(repair_kwargs["previous_response"]) > planner.MAX_TASK_OBJECTIVE_CHARS
    print("PASS malformed objective gets one targeted repair before approval")

    with (
        patch("planner._read_context", return_value=(CATALOG, "fixture dictionary")),
        patch(
            "planner._request_plan",
            side_effect=[
                (json.dumps(bad), "fixture-model"),
                (json.dumps(ready_plan(None)), "fixture-model"),
            ],
        ),
    ):
        try:
            planner.plan_request("Compare sales by month.", decision)
        except PlannerError as exc:
            assert "still invalid after one correction attempt" in str(exc)
            assert "objective is missing" in str(exc)
        else:
            raise AssertionError("A second invalid response must be rejected")
    print("PASS failed repair is rejected with the precise remaining cause")
    print("Offline planner contract checks: 7/7 passed. No LLM/API was called.")


if __name__ == "__main__":
    main()

