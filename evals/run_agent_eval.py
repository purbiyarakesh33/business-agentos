"""Run the complete development workflow benchmark and print its seven scores.

Each benchmark scenario runs in its own Python process through the existing
Streamlit AppTest smoke runner. The collected local traces are then scored
offline by agent_metric_eval.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evals.agent_metric_eval import DATASET_PATH, read_json


def _development_scenarios() -> list[str]:
    dataset = read_json(DATASET_PATH)
    return [
        str(case["scenario"])
        for case in dataset.get("cases", [])
        if isinstance(case, dict) and case.get("split") == "development"
    ]


def _run_streaming(command: list[str], cwd: Path) -> int:
    """Print a child command's output as it arrives; don't hide case failures."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def main() -> int:
    # The Windows console may use cp1252 while AppTest output contains Unicode.
    # Keep streaming alive by escaping characters that the active code page lacks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", action="append",
        help="Run only this scenario (repeat the flag to select more than one). Default: all development scenarios.",
    )
    parser.add_argument("--trace", type=Path, help="Path for the combined JSONL workflow trace.")
    parser.add_argument("--output", type=Path, help="Path for the final seven-metric JSON scorecard.")
    parser.add_argument("--timeout", type=int, default=240, help="AppTest timeout for each workflow scenario.")
    parser.add_argument(
        "--judge", action="store_true",
        help="Opt in to additional billable Featherless answer-quality judging.",
    )
    args = parser.parse_args()

    all_scenarios = _development_scenarios()
    unknown = sorted(set(args.scenario or []) - set(all_scenarios))
    if unknown:
        parser.error("unknown or non-development scenario(s): " + ", ".join(unknown))
    selected = args.scenario or all_scenarios
    if len(set(selected)) != len(selected):
        parser.error("each scenario may be selected only once")
    if not selected:
        parser.error("the development benchmark has no scenarios")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace = args.trace or (EVAL_DIR / f"predictions_agent_metrics_{stamp}.jsonl")
    scorecard = args.output or (EVAL_DIR / f"results_agent_metrics_{stamp}.json")
    trace = trace if trace.is_absolute() else ROOT / trace
    scorecard = scorecard if scorecard.is_absolute() else ROOT / scorecard
    if trace.exists():
        parser.error(f"trace file already exists: {trace}")
    if scorecard.exists():
        parser.error(f"scorecard file already exists: {scorecard}")

    print(f"Business AgentOS agent eval: {len(selected)} development scenario(s)", flush=True)
    print("Per-scenario workflow runs make Featherless requests.", flush=True)
    print(f"Trace: {trace}", flush=True)

    failed_scenarios: list[str] = []
    for index, scenario in enumerate(selected):
        print(f"\n[{index + 1}/{len(selected)}] Running {scenario}", flush=True)
        command = [
            sys.executable, "-u", "-B", str(EVAL_DIR / "run_workflow_smoke.py"),
            "--scenario", scenario,
            "--record", str(trace),
            "--timeout", str(args.timeout),
        ]
        if trace.exists():
            command.append("--append")
        code = _run_streaming(command, ROOT)
        if code != 0:
            failed_scenarios.append(scenario)
            print(f"Scenario failed: {scenario} (exit {code}); continuing to collect the remaining cases.", flush=True)

    print("\nScoring the recorded traces offline...", flush=True)
    score_command = [
        sys.executable, "-u", "-B", str(EVAL_DIR / "agent_metric_eval.py"),
        "--predictions", str(trace),
        "--output", str(scorecard),
        "--split", "development",
    ]
    if args.judge:
        score_command.append("--judge")
    score_code = _run_streaming(score_command, ROOT)
    print(f"Scorecard: {scorecard}", flush=True)
    if failed_scenarios:
        print("Workflow failures:", ", ".join(failed_scenarios), flush=True)
    if score_code != 0:
        return score_code
    if failed_scenarios:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
