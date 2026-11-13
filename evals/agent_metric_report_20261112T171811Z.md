# Business AgentOS seven-metric agent evaluation

- **Run:** 2026-11-12T17:24:21.602215+00:00 (UTC)
- **Dataset:** v1.0.0, development split, 8 scenarios
- **Coverage:** 52 applicable metric-case evaluations; 0 Jev-blocked cases
- **Command:** `python -B evals\run_agent_eval.py`
- **Scope:** Development regression benchmark only. These eight scenarios are small and should not be treated as a statistically representative estimate or release threshold.

## Metric scores

| Metric | Mean score | Perfect cases | Applicable cases |
|---|---:|---:|---:|
| task_completion | 0.9236 | 75.0% | 8 |
| step_efficiency | 0.9299 | 37.5% | 8 |
| argument_correctness | 0.9896 | 87.5% | 8 |
| tool_correctness | 0.7709 | 37.5% | 8 |
| plan_adherence | 1.0000 | 100.0% | 7 |
| plan_quality | 0.9200 | 60.0% | 5 |
| answer_correctness | 0.8812 | 62.5% | 8 |

Scores are normalized from 0 to 1. `N/A` cases are excluded from that metric mean. The benchmark does not define one universal pass threshold for each metric; the table reports score values and exact failed checks below.

## Per-case results

| Scenario | Workflow | Task completion | Step efficiency | Argument correctness | Tool correctness | Plan adherence | Plan quality | Answer correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| analytics | PASS | 1.0000 | 0.9231 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 |
| customer | PASS | 1.0000 | 0.9286 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 0.8000 |
| operations | PASS | 1.0000 | 0.9231 | 1.0000 | 0.6667 | 1.0000 | 1.0000 | 1.0000 |
| forecast | PASS | 0.8889 | 0.8462 | 0.9167 | 0.6667 | 1.0000 | 0.8000 | 0.7500 |
| review_analysis | FAIL | 0.5000 | 0.8182 | 1.0000 | 0.5000 | 1.0000 | 0.8000 | 0.5000 |
| unsupported_period | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | N/A | 1.0000 |
| injection | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.0000 |
| general_chat | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 1.0000 |

## Failures and score gaps

- **review_analysis — workflow failed:** approved analysis did not produce a verified report. Final status: `held`.
- **customer — metric checks below contract:** answer_correctness: required_answer_markers_present (missing six-month).
- **forecast — metric checks below contract:** task_completion: requested_content_markers (missing January, February, March, uncertainty); argument_correctness: sql_argument_pattern:2025 ("2025"); plan_quality: plan_objectives_cover_request (missing 2025-12-31); answer_correctness: required_answer_markers_present (missing January, February, March, uncertainty).
- **review_analysis — metric checks below contract:** task_completion: expected_workflow_status, evidence:document_context, citations_resolve, requested_content_markers ({"expected": "verified", "actual": "held"}; {"actual": []}; missing Storage, positive, negative, passage, sample); plan_quality: required_sources_selected_without_extras ({"precision": 0.5, "recall": 1.0, "f1": 0.6667, "exact_match": 0.0, "missing": [], "extra": ["products"]}); answer_correctness: required_answer_markers_present, answer_citations_resolve (missing Storage, positive, negative, representative, sample, passage).

### Interpretation

- `review_analysis` did not complete because its workflow could not retrieve indexed review evidence. The configured local RAG database at `127.0.0.1:5433` timed out during this run. Its scores are included as observed failure results, not treated as a blocked case.
- SQL cases still ran and produced verified reports despite the RAG connection timeouts during startup.
- `tool_correctness` was the lowest aggregate mean (0.7709); inspect the per-case checks before changing tool-routing behavior. `plan_adherence` scored 1.0000 on its seven applicable cases.
- Forecast checks flagged missing requested date/uncertainty markers and an expected `2025` SQL argument pattern. The forecast workflow itself completed successfully.
- The customer answer omitted the exact `six-month` marker; the report otherwise included the measured segment rates.
- Do not score or tune against held-out data while changing code, prompts, or benchmark labels.

## Artifacts

- JSON scorecard: `D:\thon\evals\results_agent_metrics_20261112T171811Z.json`
- Structured trace: `D:\thon\evals\predictions_agent_metrics_20261112T171811Z.jsonl`
- Benchmark command: `python -B evals\run_agent_eval.py` from `D:\thon`.
