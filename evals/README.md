# Business AgentOS evaluation

Component-wise golden-set evaluation for Jev input screening, planner routing, RAG, output privacy screening, and final answer quality. The current dataset is v0.6.3: **111 development cases and 8 separate holdout cases**. The holdout remains untouched. Iterate on development first; freeze code and prompts before using holdout. Current metrics, artifacts, and limitations are in [EVALUATION_STATUS.md](EVALUATION_STATUS.md). The combined seven-metric and RAG review is in [AGENT_EVAL_AND_RAG_REPORT_20261113.md](AGENT_EVAL_AND_RAG_REPORT_20261113.md).

## Components and metrics

- **Jev:** security/intent accuracy and injection precision, recall, and false-positive rate.
- **Planner:** precision, recall, F1, and exact match for selected agents and data sources; separate plan-status and workflow-contract checks.
- **RAG:** source and passage retrieval precision/recall/ranking metrics, plus topical labels for selected review searches. Retrieved text snippets are not a substitute for complete SQL counts or rates.
- **Output guard:** allow/block confusion matrix, precision, recall, false-positive rate, and accuracy over supplied report and evidence fields.
- **Answer quality:** deterministic expected-number presence plus optional hosted judge scores for correctness, completeness, relevance, evidence support, and caveat handling. The judge is a review signal, not ground truth.
- **DeepEval specialist reports:** correctness, completeness, style, answer relevancy, faithfulness, contextual precision/recall/relevancy, plus deterministic evidence-source precision/recall. Reports are scored only after specialist execution, evidence verification, and output screening. Contextual metrics treat SQL evidence results as context nodes; they are not the pgvector RAG benchmark.
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

## Seven-metric end-to-end agent evaluation

`agent_metric_dataset.json` defines eight initial development workflows, reusing the analytics, customer, operations, forecast, review, unsupported-period, injection, and general-chat smoke cases. Four business workflows link to independently labeled answer-quality cases in `golden_dataset.json`. The current set is a regression benchmark, not a statistically representative sample. It contains development cases only; add and review a separate holdout before using these scores for release decisions.

Run the complete development benchmark and print the aggregate scores for all seven metrics with one command from the project root:

```powershell
python -B evals\run_agent_eval.py
```

It runs every workflow scenario, prints per-case results as they finish, records a combined trace, then prints the seven-metric scorecard and its saved JSON path. It exits nonzero if any workflow scenario fails. Workflow runs make Featherless requests; review analysis also needs the local RAG database indexed. The scorer itself is offline.

To smoke one task first, run `python -B evals\run_agent_eval.py --scenario analytics`. Repeat `--scenario NAME` to select a subset; `--timeout 300` changes the AppTest timeout per scenario. `--judge` opts into extra billable Featherless answer-quality judgments and is off by default.

The runner records a local, structured trace from the existing workflow boundaries. It keeps prompts out of LangSmith and stores only the benchmark query, tool arguments, plan fields, evidence metadata, outcome, and timings in the requested JSONL file.

The seven scores are:

- **Task completion:** expected terminal state, requested-content markers, required evidence, and citations.
- **Step efficiency:** model/tool/workflow-step counts, failed and duplicate calls, planner repairs, configured call budgets, and wall time. Duplicate calls are counted within the same workflow phase, so approval rechecks and independent verification are not treated as redundant.
- **Argument correctness:** SQL source and query-pattern checks, read-only SQL verification status, approved-source use, and retrieval arguments.
- **Tool correctness:** expected versus actual specialists, tool names, and read tables.
- **Plan adherence:** pending-before-approval state, approval ordering, no pre-approval evidence, executed-task order, and dependencies.
- **Plan quality:** plan contract, expected specialist/source selection, objective coverage, and task minimality.
- **Answer correctness:** expected numeric facts, required answer markers, forbidden-claim patterns, and resolvable citations.

Run the local dataset and label checks without an API call:

```powershell
python -B evals/agent_metric_eval.py --validate
```

Each business case uses the synthetic SQLite database and simulates the human approval click through Streamlit `AppTest`.

The scorer is offline. It reports each metric independently and lists missing cases. Do not interpret the initial eight-case mean as a general performance estimate. The expected SQL patterns and step budgets are explicit starter contracts; review them alongside failures and expand the human-reviewed set. Never score held-out cases while changing code, prompts, or labels.

To add a supplementary answer-quality judge score for the four cases linked to answer gold, pass `--judge` to `agent_metric_eval.py`. This makes separate Featherless calls after trace collection; it does not change the seven deterministic metric scores. Treat judge scores as review signals and inspect disagreements plus a sample of agreements.

## Run development component evaluations

```powershell
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component jev --split development
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component planner --split development
& .eval-venv\Scripts\python.exe -B evals\run_live_components.py --component output_guard --split development
& .eval-venv\Scripts\python.exe -B evals\run_rag_retrieval_eval.py --top-k 5
```

Jev and planner runs make model API calls. RAG calls Featherless for query embeddings and uses the local pgvector index. Output-guard evaluation is local.

## Layered RAG evaluation: generator → retriever → pipeline → application

The expected RAG evaluation design and the precise metric/data boundary for each layer are documented in [RAG_EVALUATION_STRATEGY.md](RAG_EVALUATION_STRATEGY.md). In short:

- **Generator in isolation:** `run_rag_generator_eval.py --stage generator` supplies reviewed oracle passages and enforces zero retriever calls. Faithfulness, Answer Relevancy, and Citation Correctness are reference-free judge scores; resolved citation IDs are also audited deterministically.
- **Retriever in isolation:** `run_rag_retrieval_eval.py` retrieves from pgvector and `score_components.py --component rag_retrieval` scores ranked sources/passages against the human-reviewed labels in `golden_dataset.json`.
- **Wired RAG pipeline:** `run_rag_generator_eval.py --stage pipeline` uses actual retrieved passages and scores Contextual Relevancy, Faithfulness, Answer Relevancy, Citation Correctness, plus exact-match passage coverage diagnostics.
- **Whole application:** `run_agent_eval.py` scores workflow completion/mechanics; verified report evaluation includes G-Eval Correctness, Completeness, and Style; separate safety evals cover Jev, output guard, and toxicity. Existing traces include latency/call counts, while token and cost accounting remain uninstrumented.

The generator/pipeline runner uses DeepEval 4.2.8 and psycopg from `.eval-venv`, with Featherless as the judge. Install once with `& .\.eval-venv\Scripts\python.exe -m pip install -r evals\requirements-deepeval.txt`; start the Docker RAG database and keep the project `.env` configured. Run from `D:\thon`; these runs call embedding, generator, and judge services and can incur API usage. The latest cycle's case-level results are in [RAG_EVALUATION_REPORT_20261113.md](RAG_EVALUATION_REPORT_20261113.md).

```powershell
$tag = Get-Date -Format yyyyMMdd_HHmmss
& .\.eval-venv\Scripts\python.exe -B evals\run_rag_generator_eval.py --stage generator --predictions "evals\predictions_rag_generator_$tag.jsonl" --scorecard "evals\results_rag_generator_$tag.json"
& .\.eval-venv\Scripts\python.exe -B evals\run_rag_retrieval_eval.py --top-k 5 --output "evals\predictions_rag_retriever_$tag.jsonl"
& .\.eval-venv\Scripts\python.exe -B evals\score_components.py --component rag_retrieval --split development --predictions "evals\predictions_rag_retriever_$tag.jsonl" | Set-Content "evals\results_rag_retriever_$tag.json"
& .\.eval-venv\Scripts\python.exe -B evals\run_rag_generator_eval.py --stage pipeline --top-k 5 --predictions "evals\predictions_rag_pipeline_$tag.jsonl" --scorecard "evals\results_rag_pipeline_$tag.json"
$appTrace = "evals\predictions_agent_metrics_app_$tag.jsonl"
python -B evals\run_agent_eval.py --trace $appTrace --output "evals\results_agent_metrics_app_$tag.json" --judge
& .\.eval-venv\Scripts\python.exe -B evals\run_app_style_eval.py --predictions $appTrace --output "evals\results_app_style_$tag.json"
& .\.eval-venv\Scripts\python.exe -B evals\run_live_components.py --component output_guard --split development --output "evals\predictions_output_guard_$tag.jsonl"
& .\.eval-venv\Scripts\python.exe -B evals\score_components.py --component output_guard --split development --predictions "evals\predictions_output_guard_$tag.jsonl" | Tee-Object -FilePath "evals\results_output_guard_$tag.json"
& .\.eval-venv\Scripts\python.exe -B evals\run_toxicity_eval.py --output "evals\results_toxicity_$tag.json" --agent-predictions $appTrace
```

Judge means are exploratory until calibrated against human ratings. Inspect case-level reasons and report the number of cases scored beside each mean. The current four document queries and eight retrieval cases are small development sets; they are regression signals, not release guarantees.

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
