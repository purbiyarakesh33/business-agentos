"""Score user-visible whole-application answers for presentation style."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {number} in {path.name}.") from exc
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError(f"Trace line {number} needs a case_id object.")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Whole-application JSONL trace from run_workflow_smoke.py or run_agent_eval.py.")
    parser.add_argument("--output", required=True, type=Path, help="New JSON scorecard path.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite style scorecard: {args.output}")

    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    from run_deepeval_specialists import make_judge

    judge = make_judge()
    rows = read_jsonl(args.predictions)
    candidates = [row for row in rows if isinstance(row.get("answer"), str) and row["answer"].strip()]
    scored: list[dict[str, Any]] = []
    for number, row in enumerate(candidates, 1):
        case_id = row["case_id"]
        status = row.get("workflow_status", row.get("status", "unknown"))
        metric = GEval(
            name="Application Style",
            criteria=(
                "Score the presentation of this user-facing business-agent answer: clarity, concise organization, "
                "professional and appropriate tone, readable wording, and clear communication of uncertainty. "
                "Penalize confusing or repetitive phrasing, unnecessary jargon, and overconfident wording. "
                "Do not score factual correctness, evidence support, or completeness; those are measured separately."
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=judge,
            async_mode=False,
            threshold=None,
        )
        try:
            metric.measure(LLMTestCase(
                input=str(row.get("query", row.get("question", ""))),
                actual_output=row["answer"],
                name=case_id,
            ))
            score = float(metric.score)
            result = {
                "case_id": case_id,
                "workflow_status": status,
                "score": round(score, 4),
                "reason": getattr(metric, "reason", None),
                "threshold": None,
                "verdict": "score only; no calibrated binary threshold",
            }
        except Exception as exc:
            result = {
                "case_id": case_id,
                "workflow_status": status,
                "score": None,
                "reason": f"Judge error: {type(exc).__name__}: {str(exc)[:240]}",
                "verdict": "error",
            }
        scored.append(result)
        print(f"{number}/{len(candidates)} {case_id}: {result['score']}", flush=True)

    values = [row["score"] for row in scored if isinstance(row.get("score"), (int, float))]
    artifact = {
        "metric": "Application Style [GEval]",
        "judge_model": getattr(judge, "model_id", judge.get_model_name()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_trace": str(args.predictions.resolve()),
        "input_case_count": len(rows),
        "answer_case_count": len(candidates),
        "scored_cases": len(values),
        "judge_errors": len(scored) - len(values),
        "mean_score": round(sum(values) / len(values), 4) if values else None,
        "cases": scored,
        "limitations": [
            "Style is a model-judge signal, not a fact or correctness metric.",
            "No pass threshold has been calibrated against human ratings; inspect case-level reasons.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Application style mean: {artifact['mean_score']} ({artifact['scored_cases']}/{artifact['answer_case_count']} scored)")
    print(f"Saved style scorecard: {args.output}")
    return 0 if artifact["answer_case_count"] > 0 and artifact["judge_errors"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Application style evaluation failed: {type(exc).__name__}: {str(exc)[:240]}", file=sys.stderr)
        raise SystemExit(2)
