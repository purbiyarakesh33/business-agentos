# Business AgentOS seven-metric agent evaluation

- **Live run:** 20261112T175204Z (UTC)
- **Benchmark:** dataset v1.0.2, development split, 8 scenarios
- **Coverage:** 52 applicable metric-case evaluations; 0 blocked cases
- **Workflow result:** 8/8 scenarios passed; review analysis retrieved five indexed passages.
- **Command:** `python -B evals\run_agent_eval.py` from `D:\thon`.
- **Scope:** eight development regression scenarios. A perfect score here is a pass on this small, hand-authored set, not evidence of broad or production-level reliability.

## Metric scores

| Metric | Mean | Perfect-case rate | Applicable cases |
|---|---:|---:|---:|
| task_completion | 1.0000 | 100.0% | 8 |
| step_efficiency | 1.0000 | 100.0% | 8 |
| argument_correctness | 1.0000 | 100.0% | 8 |
| tool_correctness | 1.0000 | 100.0% | 8 |
| plan_adherence | 1.0000 | 100.0% | 7 |
| plan_quality | 1.0000 | 100.0% | 5 |
| answer_correctness | 1.0000 | 100.0% | 8 |

## Per-case scores

| Scenario | Workflow | Task completion | Step efficiency | Argument correctness | Tool correctness | Plan adherence | Plan quality | Answer correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| analytics | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| customer | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| operations | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| forecast | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| review_analysis | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unsupported_period | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | N/A | 1.0000 |
| injection | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.0000 |
| general_chat | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.0000 |

## What changed

- Forecast reports now spell out January, February, and March and label the empirical 80% ranges as uncertainty intervals.
- Review reports identify high/low rating groups as positive/negative rating proxies, include representative sample passages, and state that ratings do not establish sentiment.
- The benchmark expects the workflow date-coverage check, accepts reviewed ISO/month-name equivalents in plan checks, and recognizes `products` as a valid source for category-scoped review retrieval.
- Step-efficiency scoring now only treats a repeated identical tool call as redundant when it repeats in the same workflow phase; pre/post-approval checks and independent verification are distinct phases.
- **The rubric changed from v1.0.0 to v1.0.2, so compare this run only with future runs using v1.0.2.** The earlier lower values included penalties for valid calls and equivalent wording.

## Infrastructure and artifacts

- Docker Compose reports `ragdb` and `app` healthy. Review retrieval returned five passages and produced a verified, cited answer.
- `.env` was not modified.
- JSON scorecard: `D:\thon\evals\results_agent_metrics_20261112T175204Z.json`
- Structured trace: `D:\thon\evals\predictions_agent_metrics_20261112T175204Z.jsonl`
- Project report: `evals\agent_metric_report_20261112T175204Z.md`
