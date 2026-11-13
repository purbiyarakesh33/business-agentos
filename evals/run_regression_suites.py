"""Run and save the development regression suites for agent quality and safety.

The default run uses only development matrices. It makes Featherless/DeepEval
calls for current planner, Jev, specialist-report, RAG, and toxicity predictions;
it never runs the holdout split or charges the public demo's shared request cap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
RUNS = EVALS / "regression_runs"
GOLD = EVALS / "golden_dataset.json"
PLANNER_FIXTURES = EVALS / "predictions_planner_development_v050.jsonl"


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def get_numeric(data: dict[str, Any], *path: str) -> float | None:
    value: Any = data
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_check(log_path: Path, exit_code: int) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    match = re.search(r"\b(\d+)\s*/\s*(\d+)\s+(?:checks?\s+)?passed\b", text, re.IGNORECASE)
    if match:
        passed, total = int(match.group(1)), int(match.group(2))
    else:
        passed = sum(bool(re.match(r"\s*PASS\b", line)) for line in text.splitlines())
        failed = sum(bool(re.match(r"\s*FAIL\b", line)) for line in text.splitlines())
        total = passed + failed if exit_code == 0 else None
    return {
        "status": "passed" if exit_code == 0 else "failed",
        "passed": passed if passed else None,
        "total": total,
        "pass_rate": passed / total if total else (1.0 if exit_code == 0 else None),
    }


class RegressionRun:
    def __init__(self, run_dir: Path, env: dict[str, str]):
        self.run_dir = run_dir
        self.env = env
        self.steps: list[dict[str, Any]] = []

    def run(self, name: str, args: list[str], *, stream: bool = True) -> dict[str, Any]:
        log_path = self.run_dir / "logs" / f"{slug(name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", *args]
        print(f"[{name}] started", flush=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if stream and line.strip() and re.search(
                    r"\b(PASS|FAIL|Saved|completed|verified|retrieved|scored|failed|checks?)\b|^\s*\d+/\d+",
                    line,
                    re.IGNORECASE,
                ):
                    display_line = line.strip()[:180].encode("ascii", errors="replace").decode("ascii")
                    print("  " + display_line, flush=True)
            exit_code = process.wait()
        row = {
            "name": name,
            "status": "passed" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "log": str(log_path.relative_to(self.run_dir)),
        }
        self.steps.append(row)
        print(f"[{name}] {'passed' if exit_code == 0 else 'failed'}", flush=True)
        return row

    def score(self, name: str, predictions: Path, component: str) -> tuple[dict[str, Any], dict[str, Any]]:
        output = self.run_dir / "artifacts" / f"{slug(name)}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        args = [
            sys.executable, "-B", str(EVALS / "score_components.py"),
            "--predictions", str(predictions), "--split", "development",
            "--component", component,
        ]
        print(f"[{name}] scoring current predictions", flush=True)
        process = subprocess.run(args, cwd=ROOT, env=self.env, capture_output=True, text=True, encoding="utf-8")
        row: dict[str, Any] = {
            "name": name,
            "status": "passed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "artifact": str(output.relative_to(self.run_dir)),
        }
        try:
            report = json.loads(process.stdout)
            json_write(output, report)
        except (json.JSONDecodeError, OSError):
            report = {}
            row["status"] = "failed"
            row["error_type"] = "InvalidScorecardOutput"
        if process.returncode != 0:
            row["error_type"] = "ScorerFailed"
        self.steps.append(row)
        print(f"[{name}] {row['status']}", flush=True)
        return report, row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
    return result


def add_set_metrics(
    metrics: dict[str, float], prefix: str, data: dict[str, Any],
    component: str, selection: str,
) -> None:
    for source, target in (("precision", "precision"), ("recall", "recall"),
                           ("f1", "f1"), ("exact_match", "exact_match")):
        value = get_numeric(data, "components", component, "macro_scores", selection, target)
        if value is not None:
            metrics[f"{prefix}_{source}"] = value


def run_task1(run: RegressionRun, env: dict[str, str]) -> dict[str, Any]:
    task_start = len(run.steps)
    task_dir = run.run_dir / "artifacts" / "task1"
    task_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}

    for name, script in (
        ("golden_dataset_validation", "evals/validate_golden_dataset.py"),
        ("analyst_deterministic_regression", "analyst_eval.py"),
        ("planner_contract_regression", "planner_eval.py"),
    ):
        step = run.run(name, [script])
        if name != "golden_dataset_validation":
            check = summarize_check(run.run_dir / step["log"], step["exit_code"])
            if check["total"]:
                metrics[f"{name}_pass_rate"] = float(check["pass_rate"])
                metrics[f"{name}_passed"] = float(check["passed"])

    planner_predictions = task_dir / "planner_predictions.jsonl"
    planner_step = run.run("planner_development_matrix", [
        "evals/run_live_components.py", "--component", "planner", "--split", "development",
        "--output", str(planner_predictions),
    ])
    if planner_step["exit_code"] == 0:
        planner_score, _ = run.score("planner_selection_scorecard", planner_predictions, "planner_selection")
        add_set_metrics(metrics, "planner_agents", planner_score, "planner_selection", "agents")
        add_set_metrics(metrics, "planner_sources", planner_score, "planner_selection", "sources")
        for target, path in (
            ("planner_status_accuracy", ("components", "planner_selection", "plan_status_accuracy", "accuracy")),
            ("planner_workflow_contract_pass_rate", ("components", "planner_selection", "workflow_contracts", "pass_rate")),
        ):
            value = get_numeric(planner_score, *path)
            if value is not None:
                metrics[target] = value

    rag_predictions = task_dir / "rag_predictions.jsonl"
    rag_preflight = run.run("rag_database_readiness", ["evals/check_rag_connectivity.py"])
    if rag_preflight["exit_code"] == 0:
        rag_step = run.run("rag_development_matrix", [
            "evals/run_rag_retrieval_eval.py", "--top-k", "5", "--output", str(rag_predictions),
        ])
    else:
        rag_step = {
            "name": "rag_development_matrix", "status": "skipped",
            "exit_code": 0, "skip_reason": "RAG database preflight did not pass.",
        }
        run.steps.append(rag_step)
    if rag_preflight["exit_code"] == 0 and rag_step["exit_code"] == 0:
        rag_rows = read_jsonl(rag_predictions) if rag_predictions.is_file() else []
        if not rag_rows or any("error" in row for row in rag_rows):
            rag_step["status"] = "failed"
            rag_step["exit_code"] = 1
            rag_step["failed_cases"] = sum("error" in row for row in rag_rows)
        else:
            rag_score, _ = run.score("rag_retrieval_scorecard", rag_predictions, "rag_retrieval")
            for key, fields in (
                ("sources", ("precision", "recall", "f1", "exact_match")),
                ("passages", ("precision_at_k", "recall_at_k", "hit_at_k", "mrr_at_k", "ndcg_at_k")),
            ):
                for field in fields:
                    value = get_numeric(rag_score, "components", "rag_retrieval", "macro_scores", key, field)
                    if value is not None:
                        metrics[f"rag_{key}_{field}"] = value

    answer_predictions = task_dir / "answer_predictions.jsonl"
    answer_step = run.run("specialist_report_generation", [
        "evals/run_answer_quality_development.py", "--plans", str(PLANNER_FIXTURES),
        "--output", str(answer_predictions),
    ])
    answer_score_path = task_dir / "specialist_report_deepeval.json"
    answer_score: dict[str, Any] = {}
    if answer_predictions.is_file():
        run.run("specialist_report_deepeval", [
            "evals/run_deepeval_specialists.py", "--predictions", str(answer_predictions),
            "--output", str(answer_score_path),
        ])
        if answer_score_path.is_file():
            answer_score = json_read(answer_score_path)
            for name, item in answer_score.get("overall_metrics", {}).items():
                value = item.get("mean") if isinstance(item, dict) else None
                if isinstance(value, (int, float)):
                    metrics[f"report_{slug(name)}"] = float(value)
            case_count = answer_score.get("case_count", 0)
            verified = answer_score.get("scored_case_count", 0)
            if isinstance(case_count, int) and case_count:
                metrics["report_verified_and_scored_rate"] = verified / case_count
        numeric_score, _ = run.score("report_numeric_coverage", answer_predictions, "answer_quality")
        numeric_rows = numeric_score.get("components", {}).get("answer_quality", {}).get("cases", [])
        recalls = [
            row.get("numeric_presence", {}).get("recall")
            for row in numeric_rows
            if isinstance(row.get("numeric_presence", {}).get("recall"), (int, float))
        ]
        if recalls:
            metrics["report_gold_numeric_fact_recall"] = sum(recalls) / len(recalls)
        predictions = read_jsonl(answer_predictions)
        if predictions:
            metrics["report_verified_path_pass_rate"] = sum(
                row.get("workflow_status") == "verified"
                and row.get("output_guard") == "ALLOW"
                and row.get("execution_path_passed") is True
                for row in predictions
            ) / len(predictions)

    return {
        "task": "task1_agent_quality",
        "status": "passed" if all(step["exit_code"] == 0 for step in run.steps[task_start:]) else "partial",
        "metrics": metrics,
        "steps": list(run.steps[task_start:]),
        "artifacts": {
            "planner_predictions": str(planner_predictions.relative_to(run.run_dir)) if planner_predictions.is_file() else None,
            "rag_predictions": str(rag_predictions.relative_to(run.run_dir)) if rag_predictions.is_file() else None,
            "answer_predictions": str(answer_predictions.relative_to(run.run_dir)) if answer_predictions.is_file() else None,
            "deepeval_scorecard": str(answer_score_path.relative_to(run.run_dir)) if answer_score_path.is_file() else None,
        },
    }


def run_task2(run: RegressionRun, env: dict[str, str], answer_predictions: Path | None) -> dict[str, Any]:
    task_start = len(run.steps)
    task_dir = run.run_dir / "artifacts" / "task2"
    task_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}

    jev_predictions = task_dir / "jev_predictions.jsonl"
    jev_step = run.run("jev_security_and_scope_matrix", [
        "evals/run_live_components.py", "--component", "jev", "--split", "development",
        "--output", str(jev_predictions),
    ])
    if jev_step["exit_code"] == 0:
        jev_score, _ = run.score("jev_safety_scorecard", jev_predictions, "jev")
        jev = jev_score.get("components", {}).get("jev", {})
        for name, path in (
            ("jev_security_accuracy", ("security_label_accuracy",)),
            ("jev_intent_accuracy", ("intent_accuracy",)),
            ("jev_injection_precision", ("injection_detection", "precision")),
            ("jev_injection_recall", ("injection_detection", "recall")),
            ("jev_injection_false_positive_rate", ("injection_detection", "false_positive_rate")),
            ("scope_accuracy", ("scope_gate", "accuracy")),
            ("scope_precision", ("scope_gate", "precision")),
            ("scope_recall", ("scope_gate", "recall")),
            ("scope_f1", ("scope_gate", "f1")),
            ("scope_false_positive_rate", ("scope_gate", "false_positive_rate")),
        ):
            value = get_numeric(jev, *path)
            if value is not None:
                metrics[name] = value

    guard_predictions = task_dir / "output_guard_predictions.jsonl"
    guard_step = run.run("leakage_output_guard_matrix", [
        "evals/run_live_components.py", "--component", "output_guard", "--split", "development",
        "--output", str(guard_predictions),
    ])
    if guard_step["exit_code"] == 0:
        guard_score, _ = run.score("leakage_scorecard", guard_predictions, "output_guard")
        guard = guard_score.get("components", {}).get("output_guard", {})
        for name in ("accuracy", "block_precision", "block_recall", "false_positive_rate"):
            value = guard.get(name)
            if isinstance(value, (int, float)):
                metrics[f"leakage_{name}"] = float(value)

    public_step = run.run("public_safety_contracts", ["evals/public_safety_eval.py"])
    public_check = summarize_check(run.run_dir / public_step["log"], public_step["exit_code"])
    if public_check["total"]:
        metrics["public_safety_contract_pass_rate"] = float(public_check["pass_rate"])

    if answer_predictions is None or not answer_predictions.is_file():
        answer_predictions = task_dir / "answer_predictions_for_toxicity.jsonl"
        run.run("current_reports_for_toxicity", [
            "evals/run_answer_quality_development.py", "--plans", str(PLANNER_FIXTURES),
            "--output", str(answer_predictions),
        ])
    toxicity_path = task_dir / "toxicity_scorecard.json"
    toxicity_args = ["evals/run_toxicity_eval.py", "--output", str(toxicity_path)]
    if answer_predictions is not None and answer_predictions.is_file():
        toxicity_args.extend(["--agent-predictions", str(answer_predictions)])
    tox_step = run.run("toxicity_control_and_report_eval", toxicity_args)
    if toxicity_path.is_file():
        tox = json_read(toxicity_path)
        for name, path in (
            ("toxicity_control_accuracy", ("controls", "metrics", "accuracy")),
            ("toxicity_control_precision", ("controls", "metrics", "precision")),
            ("toxicity_control_recall", ("controls", "metrics", "recall")),
            ("toxicity_control_f1", ("controls", "metrics", "f1")),
            ("toxicity_control_false_positive_rate", ("controls", "metrics", "false_positive_rate")),
            ("agent_report_non_toxic_rate", ("verified_app_reports", "safe_rate")),
            ("agent_report_mean_non_toxic_score", ("verified_app_reports", "mean_non_toxic_score")),
        ):
            value = get_numeric(tox, *path)
            if value is not None:
                metrics[name] = value

    return {
        "task": "task2_safety",
        "status": "passed" if all(step["exit_code"] == 0 for step in run.steps[task_start:]) else "partial",
        "metrics": metrics,
        "steps": list(run.steps[task_start:]),
        "artifacts": {
            "jev_predictions": str(jev_predictions.relative_to(run.run_dir)) if jev_predictions.is_file() else None,
            "output_guard_predictions": str(guard_predictions.relative_to(run.run_dir)) if guard_predictions.is_file() else None,
            "toxicity_scorecard": str(toxicity_path.relative_to(run.run_dir)) if toxicity_path.is_file() else None,
            "agent_predictions": str(answer_predictions.relative_to(run.run_dir)) if answer_predictions.is_file() else None,
        },
    }


LOWER_IS_BETTER = ("false_positive_rate", "error_rate")
JUDGE_METRIC_MARKERS = ("report_correctness", "report_faithfulness", "answer_relevancy", "contextual_")


def compare_to_previous(
    current: dict[str, float], previous: dict[str, Any] | None,
    previous_status: str | None = None,
) -> dict[str, Any]:
    if not previous:
        return {"status": "baseline_initialized", "compared_metrics": 0, "changes": []}
    changes = []
    for name, value in current.items():
        old = previous.get(name)
        if not isinstance(old, (int, float)) or isinstance(old, bool):
            continue
        delta = value - float(old)
        lower = any(marker in name for marker in LOWER_IS_BETTER)
        tolerance = 0.03 if any(marker in name for marker in JUDGE_METRIC_MARKERS) else 0.0
        improved = delta < -tolerance if lower else delta > tolerance
        regressed = delta > tolerance if lower else delta < -tolerance
        changes.append({
            "metric": name,
            "previous": float(old),
            "current": value,
            "delta": round(delta, 6),
            "status": "improved" if improved else "regressed" if regressed else "unchanged",
        })
    regressions = sum(item["status"] == "regressed" for item in changes)
    return {
        "status": "regressed" if regressions else "no_regressions_detected",
        "previous_task_status": previous_status,
        "compared_metrics": len(changes),
        "regression_count": regressions,
        "changes": changes,
    }


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "task1", "task2"), default="all")
    parser.add_argument("--run-id", help="Optional unique directory name for this regression run.")
    args = parser.parse_args()

    dataset = json_read(GOLD)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or timestamp
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    previous_pointer = json_read(RUNS / "latest.json") if (RUNS / "latest.json").is_file() else {}
    previous_task1_id = previous_pointer.get("task1_agent_quality") if isinstance(previous_pointer, dict) else None
    previous_task2_id = previous_pointer.get("task2_safety") if isinstance(previous_pointer, dict) else None
    previous_task1_dir = RUNS / previous_task1_id if isinstance(previous_task1_id, str) else None
    previous_task2_dir = RUNS / previous_task2_id if isinstance(previous_task2_id, str) else None

    env = os.environ.copy()
    env["PUBLIC_DEMO_MODE"] = "false"  # Evaluation requests must not consume the live demo's shared quota.
    env["PYTHONUNBUFFERED"] = "1"
    run = RegressionRun(run_dir, env)
    base = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "dataset_version": dataset.get("dataset_version"),
        "development_only": True,
        "holdout_used": False,
        "api_notes": "Model and judge calls use only synthetic development cases. Public demo quota reservation is disabled for this local evaluation process.",
    }
    latest = dict(previous_pointer) if isinstance(previous_pointer, dict) else {}
    latest["run_id"] = run_id
    latest["generated_at_utc"] = base["generated_at_utc"]

    if args.suite in {"all", "task1"}:
        task1 = run_task1(run, env)
        previous_file = previous_task1_dir / "task1_agent_quality.json" if previous_task1_dir else None
        previous_result = json_read(previous_file) if previous_file and previous_file.is_file() else None
        previous_metrics = previous_result.get("metrics") if isinstance(previous_result, dict) else None
        task1["regression"] = compare_to_previous(
            task1["metrics"], previous_metrics if isinstance(previous_metrics, dict) else None,
            previous_result.get("status") if isinstance(previous_result, dict) else None,
        )
        task1 = {**base, **task1}
        json_write(run_dir / "task1_agent_quality.json", task1)
        latest["task1_agent_quality"] = run_id
        answer_rel = task1.get("artifacts", {}).get("answer_predictions")
        answer_predictions = run_dir / answer_rel if answer_rel else None
    else:
        answer_predictions = None

    if args.suite in {"all", "task2"}:
        task2 = run_task2(run, env, answer_predictions)
        previous_file = previous_task2_dir / "task2_safety.json" if previous_task2_dir else None
        previous_result = json_read(previous_file) if previous_file and previous_file.is_file() else None
        previous_metrics = previous_result.get("metrics") if isinstance(previous_result, dict) else None
        task2["regression"] = compare_to_previous(
            task2["metrics"], previous_metrics if isinstance(previous_metrics, dict) else None,
            previous_result.get("status") if isinstance(previous_result, dict) else None,
        )
        task2 = {**base, **task2}
        json_write(run_dir / "task2_safety.json", task2)
        latest["task2_safety"] = run_id

    json_write(RUNS / "latest.json", latest)
    print(f"Regression results saved under {run_dir}", flush=True)
    for task_name in ("task1_agent_quality", "task2_safety"):
        result_path = run_dir / f"{task_name}.json"
        if result_path.is_file():
            result = json_read(result_path)
            regression = result.get("regression", {})
            print(
                f"{task_name}: {result['status']}; "
                f"regression={regression.get('status')}; "
                f"metrics={len(result.get('metrics', {}))}",
                flush=True,
            )
    failed_steps = [step for step in run.steps if step.get("exit_code") != 0]
    regression_found = any(
        result.get("regression", {}).get("status") == "regressed"
        for task_name in ("task1_agent_quality", "task2_safety")
        if (run_dir / f"{task_name}.json").is_file()
        for result in [json_read(run_dir / f"{task_name}.json")]
    )
    return 1 if failed_steps or regression_found else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Regression runner stopped ({type(exc).__name__}).", file=sys.stderr)
        raise SystemExit(2)
