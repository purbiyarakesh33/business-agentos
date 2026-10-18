"""Check whether the configured DeepEval judge penalizes known bad answer mutations."""
from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from run_deepeval_specialists import (
    EVALS,
    GOLD_PATH,
    PREDICTIONS_PATH,
    evidence_context,
    expected_reference,
    make_judge,
    read_jsonl,
)
from deepeval.test_case import LLMTestCase

OUTPUT = EVALS / "results_deepeval_sensitivity_controls_2026-10-15_faithfulness_v3.json"
CASE_ID = "report_category_sales"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()
    dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = next(c for c in dataset["cases"] if c["case_id"] == CASE_ID)
    prediction = next(p for p in read_jsonl(args.predictions) if p["case_id"] == CASE_ID)
    evidence = evidence_context(prediction["evidence"])
    specialist_evidence = evidence_context(prediction["evidence"], specialist_only=True)
    judge = make_judge()

    # Focus this rerun on the known Faithfulness miss. Other bad-answer controls
    # remain recorded in the first sensitivity artifact.
    controls = [
        {
            "control": "verified_answer_baseline",
            "actual_output": prediction["answer"],
            "metrics": ["Faithfulness"],
            "context": evidence,
        },
        {
            "control": "unsupported_causal_claim",
            "actual_output": (
                "Delivered net product sales declined by USD 7,310.45 (-39.91%) from August to September 2025. "
                "Storage declined by USD 2,800.75, Decor by USD 1,930.93, Bedding by USD 1,156.62, "
                "Kitchen by USD 991.80, and Lighting by USD 430.35. Competitor advertising caused the decline."
            ),
            "metrics": ["Faithfulness"],
            "context": evidence,
        },
    ]

    metric_factory = __import__("run_deepeval_specialists").create_metrics
    metric_map = {metric.__name__: metric for metric in metric_factory(judge)}
    results = []
    for control in controls:
        test_case = LLMTestCase(
            input=gold["query"],
            actual_output=control["actual_output"],
            expected_output=(
                expected_reference(gold["expected"])
            ),
            retrieval_context=specialist_evidence,
            name=control["control"],
        )
        row = {"control": control["control"], "metrics": {}}
        for name in control["metrics"]:
            metric = metric_map[name]
            try:
                metric.measure(test_case)
                metric_score = float(metric.score) if metric.score is not None else None
                metric_threshold = getattr(metric, "threshold", None)
                row["metrics"][name] = {
                    "score": metric_score,
                    "reason": metric.reason,
                    "threshold": metric_threshold,
                    "judge_verdict": (
                        "pass" if metric_threshold is not None and metric_score is not None and metric_score >= metric_threshold
                        else "fail" if metric_threshold is not None and metric_score is not None
                        else "score_only"
                    ),
                }
                print(f"{control['control']} | {name}: {metric.score}", flush=True)
            except Exception as exc:
                row["metrics"][name] = {"score": None, "reason": f"ERROR {type(exc).__name__}: {str(exc)[:300]}"}
                print(f"{control['control']} | {name}: ERROR {type(exc).__name__}", flush=True)
        results.append(row)

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deepeval_version": "4.2.8",
        "judge_model": judge.model_id,
        "basis": "One synthetic category-sales report plus a single unsupported-cause mutation. Other failure mutations are recorded in the prior sensitivity artifact.",
        "predictions_path": str(args.predictions),
        "production_predictions_modified": False,
        "controls": results,
        "interpretation": "These probes check whether the DeepEval faithfulness judge penalizes unsupported causal claims when the sanitized evidence includes causal_review metadata.",
    }
    by_control = {row["control"]: row["metrics"] for row in results}

    def score(control: str, metric: str) -> float | None:
        value = by_control.get(control, {}).get(metric, {}).get("score")
        return float(value) if isinstance(value, (int, float)) else None

    checks = [
        {
            "check": "known_good_report_scores_high",
            "score": score("verified_answer_baseline", "Faithfulness"),
            "expected": ">= 0.8",
        },
        {
            "check": "unsupported_cause_scores_low",
            "score": score("unsupported_causal_claim", "Faithfulness"),
            "expected": "<= 0.9",
        },
    ]
    for item in checks:
        actual = item["score"]
        if actual is None:
            item["passed"] = False
        elif item["expected"] == ">= 0.8":
            item["passed"] = actual >= 0.8
        else:
            item["passed"] = actual <= 0.9
    artifact["sensitivity_checks"] = checks
    artifact["sensitivity_checks_passed"] = all(item["passed"] for item in checks)
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['check']}: {item['score']} (expected {item['expected']})")
    return 0 if artifact["sensitivity_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
