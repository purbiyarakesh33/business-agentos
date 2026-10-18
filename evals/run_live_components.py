"""Collect Jev, planner, or local output-guard predictions for the golden benchmark.

This makes model/API calls. Use development first; keep holdout untouched until
prompts and code are frozen.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD = Path(__file__).with_name("golden_dataset.json")


def planner_intent(query: str) -> str:
    text = query.lower()
    if any(term in text for term in ("forecast", "predict", "next three", "what if")):
        return "forecast_or_what_if"
    if any(term in text for term in ("customer", "repeat", "retention", "review", "sentiment")):
        return "customer_analysis"
    if any(term in text for term in ("carrier", "delivery", "fulfillment", "inventory", "return")):
        return "product_or_operations_analysis"
    return "business_performance"


def run_jev(query: str) -> dict:
    from jev import classify_request
    decision = classify_request(query)
    return {
        "action": decision.action, "security": decision.security, "intent": decision.intent,
        "security_confidence": decision.security_confidence,
        "intent_confidence": decision.intent_confidence, "model": decision.model,
    }


def run_planner(query: str) -> dict:
    from jev import JevDecision
    from planner import plan_request
    # A fixed ALLOW decision isolates planner behavior from Jev in this component test.
    decision = JevDecision("ALLOW", "normal", planner_intent(query), 1.0, 1.0, "eval-fixture")
    plan = plan_request(query, decision).to_dict()
    tasks = plan.get("tasks", [])
    return {
        "plan_status": plan.get("status"),
        "approval_status": plan.get("approval_status"),
        "agents": sorted({t.get("agent") for t in tasks if t.get("agent")}),
        "sources": sorted({s for t in tasks for s in t.get("data_sources", [])}),
        "plan": json.dumps(plan, ensure_ascii=False),
        "model": plan.get("model"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=("jev", "planner", "output_guard"))
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="development")
    parser.add_argument("--case-id", action="append", dest="case_ids",
                        help="Run only the named case(s); may be supplied more than once.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    component_name = {"jev": "jev", "planner": "planner_selection", "output_guard": "output_guard"}[args.component]
    cases = json.loads(GOLD.read_text(encoding="utf-8"))["cases"]
    cases = [c for c in cases if c["component"] == component_name
             and (args.split == "all" or c["split"] == args.split)]
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in requested]
        found = {case["case_id"] for case in cases}
        if found != requested:
            raise ValueError("Unknown/non-selected case ID(s): " + ", ".join(sorted(requested - found)))
    if not cases:
        raise ValueError("No cases found for selected component and split.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).with_name(
        f"predictions_{args.component}_{args.split}_{stamp}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions file: {output}")

    if args.component in {"jev", "planner"}:
        from jev import load_local_environment
        load_local_environment()
    successes = 0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(cases, 1):
            row = {"case_id": case["case_id"]}
            try:
                if args.component == "jev":
                    row.update(run_jev(case["query"]))
                elif args.component == "planner":
                    row.update(run_planner(case["query"]))
                else:
                    from output_guardrails import screen_business_output
                    report = case.get("report", case["query"])
                    evidence = case.get("evidence", [])
                    decision = screen_business_output(report, evidence)
                    row.update({"action": decision.action, "reasons": list(decision.reasons)})
                successes += 1
                print(f"{index}/{len(cases)} {case['case_id']}: completed")
            except Exception as exc:
                # Do not serialize tracebacks or request objects that could contain credentials.
                row["error"] = f"{type(exc).__name__}: request failed; details withheld"
                if args.component == "jev":
                    row.update({"action": "REVIEW", "security": "ambiguous"})
                else:
                    row.update({"plan_status": "error", "agents": [], "sources": [], "plan": row["error"]})
                print(f"{index}/{len(cases)} {case['case_id']}: failed ({type(exc).__name__})")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved {successes}/{len(cases)} predictions to {output}")
    return 0 if successes == len(cases) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("Evaluation runner error: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
