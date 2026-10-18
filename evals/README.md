# Business AgentOS evaluation

Component-wise golden-set evaluation for Jev input screening, planner routing, RAG, output privacy screening, and final answer quality. The current dataset is v0.6.1: **108 development cases and 8 separate holdout cases**. The holdout remains untouched. Iterate on development first; freeze code and prompts before using holdout. Current metrics, artifacts, and limitations are in [EVALUATION_STATUS.md](EVALUATION_STATUS.md).

## Components and metrics

- **Jev:** security/intent accuracy and injection precision, recall, and false-positive rate.
- **Planner:** precision, recall, F1, and exact match for selected agents and data sources; separate plan-status and workflow-contract checks.
- **RAG:** source and passage retrieval precision/recall/ranking metrics, plus topical labels for selected review searches. Retrieved text snippets are not a substitute for complete SQL counts or rates.
- **Output guard:** allow/block confusion matrix, precision, recall, false-positive rate, and accuracy over supplied report and evidence fields.
- **Answer quality:** deterministic expected-number presence plus optional hosted judge scores for correctness, completeness, relevance, evidence support, and caveat handling. The judge is a review signal, not ground truth.
- **DeepEval specialist reports:** correctness, completeness, answer relevancy, faithfulness, contextual precision/recall/relevancy, plus deterministic evidence-source precision/recall. Reports are scored only after specialist execution, evidence verification, and output screening. Contextual metrics treat SQL evidence results as context nodes; they are not the pgvector RAG benchmark.
- **Execution-path audit:** verifies the selected specialist, read-only SQL/forecast tool, fixed template where applicable, and expected source tables for each answer-quality report fixture.
- **Analyst/specialist logic:** deterministic fixture calculations, report/evidence contracts, privacy, forecasting, and verifier behavior.

## Reproduce local checks

From `D:\thon` in PowerShell:

```powershell
& .eval-venv\Scripts\python.exe -B evals\validate_golden_dataset.py
& .eval-venv\Scripts\python.exe -B analyst_eval.py
& .eval-venv\Scripts\python.exe -B planner_eval.py
```

These checks use local synthetic data and do not call an LLM API. Latest local run: golden-set validation passed, analyst checks 33/33, and planner contract checks 7/7.

## Run development component evaluations

```powershell
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component jev --split development
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component planner --split development
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component output_guard --split development
& .eval-venv\Scripts\python.exe -B evals\run_rag_retrieval_eval.py --top-k 5
```

Jev and planner runs make model API calls. RAG calls Featherless for query embeddings and uses the local pgvector index. Output-guard evaluation is local.

## Regenerate and score answer-quality reports

Saved planner fixtures are used to isolate report execution, verification, and screening. Synthetic benchmark approval is simulated; the live UI's human-approval gate is unchanged.

```powershell
$tag = Get-Date -Format yyyyMMdd_HHmmss
$predictions = "evals\predictions_answer_quality_development_$tag.jsonl"
$scorecard = "evals\results_deepeval_specialists_development_$tag.json"
& .eval-venv\Scripts\python.exe -B evals\run_answer_quality_development.py --plans evals\predictions_planner_development_v050.jsonl --output $predictions
& .eval-venv\Scripts\python.exe -B evals\run_deepeval_specialists.py --predictions $predictions --output $scorecard
```

The report runner checks expected specialist, tool(s), query/forecast template where applicable, and source tables. The DeepEval step uses Featherless and makes billable requests. Keep API credentials out of prediction and result files. The current 2026-10-16 development refresh regenerated all ten report fixtures, verified their paths, and scored all seven metrics across all ten with no holdout data. See EVALUATION_STATUS.md for the per-specialist results and small-sample limitations.

## Evaluation rules

1. Do not score holdout while changing prompts or code. Freeze first, then run the eight holdout cases once and preserve the results separately.
2. Keep `scenario_truth.json` out of application context and RAG; it is evaluator-only.
3. Independently verify SQL filters, join grain, denominators, calculations, and reconciliations. LLM judging does not prove arithmetic correctness.
4. Human-review benchmark labels, judge disagreements, and a sample of agreements.
5. Track missed attacks separately from false blocks of benign requests. Keep safety, privacy, evidence, and analysis metrics distinct.
6. Component scores and saved-plan report fixtures do not certify the live UI workflow. Test Jev → planner → human approval → agents/tools → evidence verification → output screen → final report separately.
7. Treat small synthetic benchmark scores as regression signals only; do not present them as production guarantees.

## Live Streamlit workflow smoke checks

These checks exercise the app through Streamlit `AppTest` and use the configured live Jev/Featherless services. They make API requests; the analytics scenario also executes against the local synthetic SQLite database and simulates the human approval click in the test harness.

```powershell
python -B evals\run_workflow_smoke.py --scenario analytics
python -B evals\run_workflow_smoke.py --scenario injection
python -B evals\run_workflow_smoke.py --scenario unsupported_period
python -B evals\run_workflow_smoke.py --scenario customer
python -B evals\run_workflow_smoke.py --scenario operations
python -B evals\run_workflow_smoke.py --scenario forecast
python -B evals\run_workflow_smoke.py --scenario review_analysis
python -B evals\run_workflow_smoke.py --scenario general_chat
```

Each command tests one scenario only. The analytics smoke asserts the plan is pending approval before executing, then requires verified status, SQL evidence, resolvable report citations, and output-guard `ALLOW`. The injection smoke passes only when Jev blocks or holds the request before planning. These are workflow regressions, not a statistical security evaluation or production certification.

Note: live smoke tests require the Python environment with Streamlit and app dependencies installed; .eval-venv does not include Streamlit. As of 2026-10-16, all eight named scenarios passed in the final sequential run. Customer repeat-rate uses its deterministic six-month delivered-order query. Review analysis passed with the actual planner, five RAG passages, verification, and output screening; its report quotes each cited passage directly and labels rating-based grouping and sample limitations. Unrestricted generated-SQL first-attempt reliability still needs broader evaluation. Add `--trace-sql` to print generated SQL and SQLite errors; `--trace-findings` prints rejected claims and cited-evidence metadata.
