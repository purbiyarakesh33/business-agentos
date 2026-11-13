"""Score synthetic toxicity controls and verified app reports with DeepEval."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
CASES_PATH = HERE / "safety_toxicity_cases.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name} line {line_number}.") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["expected_toxic"] and row["predicted_toxic"] for row in rows)
    fp = sum(not row["expected_toxic"] and row["predicted_toxic"] for row in rows)
    fn = sum(row["expected_toxic"] and not row["predicted_toxic"] for row in rows)
    tn = sum(not row["expected_toxic"] and not row["predicted_toxic"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "case_count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def evaluate_outputs(
    outputs: list[tuple[str, str, bool]], judge: Any, threshold: float,
) -> list[dict[str, Any]]:
    from deepeval.metrics import ToxicityMetric
    from deepeval.test_case import LLMTestCase

    rows = []
    for index, (case_id, output, expected_toxic) in enumerate(outputs, 1):
        metric = ToxicityMetric(
            threshold=threshold,
            model=judge,
            async_mode=False,
            include_reason=False,
            verbose_mode=False,
        )
        try:
            metric.measure(LLMTestCase(input=case_id, actual_output=output, name=case_id))
            score = float(metric.score)
            if not math.isfinite(score):
                raise ValueError("judge returned a non-finite score")
            rows.append({
                "case_id": case_id,
                "expected_toxic": expected_toxic,
                "non_toxic_score": round(score, 4),
                "predicted_toxic": score < threshold,
                "passed": (score < threshold) == expected_toxic,
            })
            print(f"toxicity {index}/{len(outputs)} {case_id}: scored", flush=True)
        except Exception as exc:
            rows.append({
                "case_id": case_id,
                "expected_toxic": expected_toxic,
                "error_type": type(exc).__name__,
                "passed": False,
            })
            print(f"toxicity {index}/{len(outputs)} {case_id}: failed ({type(exc).__name__})", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--agent-predictions", type=Path,
                        help="Fresh verified answer-quality predictions JSONL to screen as non-toxic outputs.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    dataset = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    controls = dataset.get("cases", [])
    if not isinstance(controls, list) or not controls:
        raise ValueError("Toxicity matrix has no cases.")
    from run_deepeval_specialists import make_judge
    judge = make_judge()

    control_outputs = [
        (item["case_id"], item["output"], bool(item["expected_toxic"]))
        for item in controls
    ]
    control_rows = evaluate_outputs(control_outputs, judge, args.threshold)
    successful_controls = [row for row in control_rows if "predicted_toxic" in row]

    app_outputs: list[tuple[str, str, bool]] = []
    if args.agent_predictions:
        for row in read_jsonl(args.agent_predictions):
            answer = row.get("answer")
            guard_action = row.get("output_guard", row.get("output_guard_action"))
            if (row.get("workflow_status") == "verified"
                    and guard_action == "ALLOW"
                    and isinstance(answer, str) and answer.strip()):
                app_outputs.append((str(row.get("case_id", "agent_report")), answer, False))
    app_rows = evaluate_outputs(app_outputs, judge, args.threshold) if app_outputs else []
    scored_app_rows = [row for row in app_rows if "non_toxic_score" in row]

    artifact = {
        "dataset_version": dataset.get("dataset_version"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "DeepEval ToxicityMetric",
        "judge_model": getattr(judge, "model_id", judge.get_model_name()),
        "threshold": args.threshold,
        "score_semantics": "higher non_toxic_score means safer; predicted_toxic when score is below threshold",
        "controls": {
            "metrics": confusion(successful_controls),
            "scored_count": len(successful_controls),
            "error_count": len(control_rows) - len(successful_controls),
            "cases": control_rows,
        },
        "verified_app_reports": {
            "input_count": len(app_outputs),
            "scored_count": len(scored_app_rows),
            "error_count": len(app_rows) - len(scored_app_rows),
            "safe_rate": (
                sum(row["non_toxic_score"] >= args.threshold for row in scored_app_rows) / len(scored_app_rows)
                if scored_app_rows else None
            ),
            "mean_non_toxic_score": (
                round(sum(row["non_toxic_score"] for row in scored_app_rows) / len(scored_app_rows), 4)
                if scored_app_rows else None
            ),
            "cases": app_rows,
        },
        "limitations": [
            "The toxicity judge is an LLM evaluator and may vary; compare case-level scores across runs and human-review disagreements.",
            "Synthetic control cases check judge sensitivity, not the prevalence of toxic responses in real user traffic.",
            "A safe report score does not replace the application's scope, privacy, or evidence checks.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved toxicity evaluation: {args.output}")
    if len(successful_controls) != len(control_outputs) or len(scored_app_rows) != len(app_outputs):
        return 1
    return 0 if artifact["controls"]["metrics"]["accuracy"] == 1.0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Toxicity evaluation stopped ({type(exc).__name__}).", file=sys.stderr)
        raise SystemExit(2)
