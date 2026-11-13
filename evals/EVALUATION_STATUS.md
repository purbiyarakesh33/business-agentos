# Component Evaluation Status

**Updated:** 2026-11-13 (layered RAG and whole-application development eval; see the detailed [RAG evaluation report](RAG_EVALUATION_REPORT_20261113.md))

## Dataset and split

`evals/golden_dataset.json` is v0.6.3: **119 cases** total, with **111 development cases** and **8 held-out cases**. The v0.6.3 update adds reviewed passage refs to five development RAG queries; holdout cases were not used or edited. For current and historical RAG metrics, see [RAG_EVALUATION_REPORT_20261113.md](RAG_EVALUATION_REPORT_20261113.md). For the seven-metric Agent Eval history, see [AGENT_EVAL_AND_RAG_REPORT_20261113.md](AGENT_EVAL_AND_RAG_REPORT_20261113.md).

| Component | Development | Holdout |
|---|---:|---:|
| Jev screening | 57 | 3 |
| Planner selection | 16 | 2 |
| Output guard | 20 | 1 |
| RAG retrieval | 8 | 1 |
| Verified answer-quality reports | 10 | 1 |

## Current development results

| Component | Cases / checks | Latest result | Interpretation and limits |
|---|---:|---|---|
| Jev input screening | 54 | Accuracy 100%; injection TP=26, FP=0, FN=0, TN=28; precision 100%, recall 100%, FPR 0% | Hand-authored synthetic cases. This is not an estimate of real-world attack detection. The Jev examples were unchanged in v0.6.0. |
| Planner selection | 16 | Workflow contracts 16/16; plan status 16/16; agent precision/recall/F1/exact match 100%; data-source precision 95.83%, recall 100%, F1 96.88%, exact match 93.75% | One query selected extra `orders`/`order_items` sources. Objective quality and real-world route coverage need more human-reviewed cases. |
| Output guard | 20 | TP=12, FP=0, FN=0, TN=8; accuracy, block precision/recall 100%; FPR 0% | Synthetic examples only; not a broad privacy or leakage certification. |
| RAG source retrieval | 8 | Latest macro precision 83.33%, recall 93.75%, F1 87.50%, exact match 62.50% (baseline F1 76.31%, exact match 12.5%) | Better source precision and exact-set matches; one delivery-eligibility case still misses the data-dictionary source. |
| RAG passage ranking | 7 passage-labeled queries, top 5 | Latest P@5 34.29%, returned-passage precision 80.95%, recall@5 78.57%, hit@5 100%, MRR 0.929, nDCG 0.801 | Passage labels increased from 2 to 7, so baseline P@5 20% is not a like-for-like comparison. One review-topic case has precision 1.0/recall 0.6 and is not a sentiment-quality evaluation. |
| RAG pipeline judges | 4 questions | Latest Contextual Relevancy 0.650 (4/4; 0.867 on 3 answerable cases), Faithfulness 1.0, Answer Relevancy 1.0, Citation Correctness 1.0 | Baseline Contextual Relevancy 0.135 (3/4; one judge error). The intentionally unanswerable gross-margin case scores 0 contextual relevance and passes its abstention check. |
| Verified specialist reports | 10 | 10/10 verified; 10/10 output guard `ALLOW`; **10/10 expected execution paths matched** | Reports use synthetic data and saved planner fixtures. This does not test live planner selection or the UI path. SQL templates cover specific task families; it does not establish support for arbitrary business questions. |
| Analyst deterministic suite | 33 | 33/33 passed | Offline checks cover fixture calculations, report/evidence contracts, privacy, forecasting, SQL repair, review RAG routing, exact citations, and verifier behavior. Synthetic data only. |
| Planner contract suite | 7 | 7/7 passed | Offline approval-state and malformed-objective checks; no API calls. |
| Live Streamlit workflow smoke | 8 scenarios | **8/8 passed in the final sequential run** after the analyst fixes | Live services; AppTest simulates the approval click. These are individual regressions on synthetic data, not a production-readiness estimate. |

On 2026-10-16, saved Jev (54), planner (16), output-guard (20), RAG (8), and answer-quality (10) development predictions were rescored locally against golden set v0.6.1. These were re-scores of saved predictions, not fresh model runs. Version 0.6.1 changed no component labels. The RAG improvement run on 2026-11-13 freshly reindexed focused passages and evaluated seven passage-labeled development queries; artifacts are linked in the report.

## Local Docker run

On 2026-10-16, docker compose config validation and the app image build passed. The Streamlit app and pgvector containers both started healthy; the UI returned HTTP 200 and its health endpoint returned ok. The app had runtime Featherless/RAG configuration, connected to pgvector, and reported 32 indexed company-document chunks plus 926 review records. The image contains the SQLite demo database and excludes evaluator-only scenario_truth.json. The existing named RAG volume survived the database container recreation.

This validates a localhost demo container, not an internet-facing deployment.

## DeepEval specialist report scorecard

Latest fresh scorecard: [results_deepeval_specialists_development_20261016_current_all10.json](results_deepeval_specialists_development_20261016_current_all10.json). Current report/path-audit inputs: [predictions_answer_quality_development_20261016_current_all10.jsonl](predictions_answer_quality_development_20261016_current_all10.jsonl).

DeepEval 4.2.8 used Featherless zai-org/GLM-5.3-Flash. All ten synthetic specialist reports were freshly regenerated from saved approved-plan fixtures against current code and scored on 2026-10-16. All seven judge metrics scored 1.0000 for all ten reports. No holdout cases were used.

| Metric | Mean | Scored reports |
|---|---:|---:|
| Correctness (GEval) | 1.0000 | 10/10 |
| Completeness (GEval) | 1.0000 | 10/10 |
| Answer relevancy | 1.0000 | 10/10 |
| Faithfulness | 1.0000 | 10/10 |
| Contextual precision | 1.0000 | 10/10 |
| Contextual recall | 1.0000 | 10/10 |
| Contextual relevancy | 1.0000 | 10/10 |
| Evidence-source precision / recall | 1.0000 / 1.0000 | 10/10 |

Faithfulness has a provisional 0.90 threshold; other judge metrics are scores without a pass threshold. The current sensitivity-control run scored a known-good answer 1.0000 and an unsupported causal mutation 0.8571, below the provisional threshold. This is one synthetic contrast, so it does not establish broad faithfulness reliability. Contextual metrics treat SQL result evidence as context nodes; they are separate from the pgvector RAG benchmark. Source precision/recall compares tables_read to manually recorded expected sources and does not validate SQL logic by itself.

| Specialist | Reports | Correctness | Completeness | Relevancy | Faithfulness | Evidence-source P / R |
|---|---:|---:|---:|---:|---:|---:|
| Data Analyst | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| Customer Analyst | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| Product/Operations Analyst | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| Forecasting Agent | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |

These per-specialist means are fresh on 2026-10-16 but exploratory because the sample sizes are small: Forecasting has one report and the other specialists have three each. All ten regenerated report paths matched expected specialists, tools, templates where applicable, and source tables. Forecasting checks both read-only SQL history access and the deterministic forecast tool. The customer repeat-segment case asserts the demo_customer_repeat_window_v1 template along with its specialist, read-only SQL tool, and source tables.

### Faithfulness defect fixed

The prior Customer Analyst lifecycle report paired an August/September customer count with a July-minus-comparison delta. For example, it displayed August's 109 customers beside `-3`, even though August had **3 more** than July's 106. The September comparison similarly displayed 71 beside `+35`, though September had **35 fewer** than July. The values and arithmetic were present, but the report wording made the comparison direction wrong for readers; the judge correctly penalized it.

The report now states each pair and direction explicitly: **August 109 vs July 106 (+3, +2.83%)**; **September 71 vs July 106 (-35, -33.02%)**. Repeat rate is shown as **10.09% vs 6.60% (+3.49 percentage points)** for August and **1.41% vs 6.60% (-5.19 points)** for September. The Customer Analyst Faithfulness score moved from **0.7143 to 1.0000**. Regression checks cover all customer groups and periods; the latest offline analyst suite passes 33/33.

The path audit also exposed that the deterministic operations formatter rejected two SQL tasks from the same Data Analyst as though they came from different specialists. It now combines multiple fixed operations results from one specialist and still declines to silently merge evidence from multiple specialists. The delivery/returns report regression passes.

## What this establishes—and what it does not

- All ten current report fixtures produced verified reports, passed the output guard, and matched the expected specialist/tool/source paths; the matching full DeepEval run scored all ten.
- The report judge metrics are positive regression signals on synthetic examples, not independent proof of general correctness. Arithmetic safety still depends on SQL/result checks and human-reviewed expected values.
- The saved-plan report benchmark uses synthetic fixtures and simulates approval. A separate live Streamlit smoke runner passed eight scenarios spanning five business report flows, injection, unsupported-period handling, and general chat. Its approval click is simulated in AppTest; this is regression coverage, not broad live-query coverage.
- A perfect ten-report score does not show arbitrary natural-language-to-SQL coverage. Several reports rely on known demo templates; broader query families and unsupported-input behavior still need evaluation.
- RAG remains a clear weakness: source precision is 63.33% and passage precision@5 is 20% in the current small benchmark.
- The benchmark is small, synthetic, and hand-authored. Do not describe these results as production readiness or a guarantee for a real company's data.

## Remaining evaluation work

1. Expand and independently human-review development examples for report types, SQL joins/filters, planner objectives and source selection, ambiguous requests, document retrieval, injection attacks, and output privacy.
2. Expand tests for unrestricted generated-SQL queries and monitor first-attempt failure rates; the common six-month customer repeat-rate task now uses a deterministic read-only template.
3. Improve RAG ranking and add a larger human-judged passage relevance set.
4. Add more independently calculated numeric assertions and human-review judge agreements as well as disagreements.
5. The final live suite passed all eight scenarios. Broaden UI coverage across more normal questions, unsupported-data cases, injection attempts, and safe failures; preserve the current run as a regression baseline.
6. The current development reports have all been freshly scored. Freeze code and prompts before using the eight held-out cases once; keep those results separate from development.

## Relevant artifacts

- Dataset and validator: `golden_dataset.json`, `validate_golden_dataset.py`.
- Current ten-report predictions and path audit: [predictions_answer_quality_development_20261016_current_all10.jsonl](predictions_answer_quality_development_20261016_current_all10.jsonl).
- Current fresh DeepEval scorecard: [results_deepeval_specialists_development_20261016_current_all10.json](results_deepeval_specialists_development_20261016_current_all10.json).
- Current faithfulness sensitivity controls: results_deepeval_sensitivity_controls_2026-10-15_faithfulness_v3.json (known-good 1.0000; unsupported cause 0.8571; both checks passed).
- Offline checks: `analyst_eval.py`, `planner_eval.py`.
- Saved component predictions: predictions_jev_development_v050_final.jsonl, predictions_planner_development_v050.jsonl, predictions_output_guard_development_v050.jsonl, and predictions_rag_development_20261015T180526Z.jsonl.
- Local golden-set v0.6.1 rescore artifacts: `results_jev_development_20261016_v061.json`, `results_planner_selection_development_20261016_v061.json`, `results_output_guard_development_20261016_v061.json`, `results_rag_retrieval_development_20261016_v061.json`, and `results_answer_quality_development_20261016_v061.json`.

- Latest RAG improvement artifacts: `results_rag_retriever_final4_20261113.json`, `results_rag_generator_focused_20261113.json`, and `results_rag_pipeline_final_20261113.json`; see the detailed report for the baseline comparison and per-case results.
- Historical component rescores: Jev 54/54 labels matched (26 injection TP, 0 FP, 0 FN); planner 16/16 contracts with agent P/R/F1 1.00 and source precision 0.9583/recall 1.00; output guard 20/20 (12 blocked TP, 0 FP, 0 FN).

**Status:** component evaluation catches report-direction and specialist-path defects, and the local Docker demo is running. Evaluation still lacks real-world coverage and broad live-query coverage; the app is not production-ready. All company records are synthetic Cedar & Stone demo data.

## Earlier live Streamlit smoke run (2026-10-15)

`run_workflow_smoke.py` exercised the real Streamlit script with the configured Jev and Featherless services:

- **Analytics:** passed Jev, produced a ready `data_analyst` plan using `orders`, `order_items`, and `products`, waited for the approval control, executed SQL, passed evidence verification and the output privacy screen, and generated a report whose citations resolve to evidence IDs. September vs August totals were USD 11,008.18 vs USD 18,318.63; the report reconciled a USD -7,310.45 change across five product categories.
- **Injection:** Jev returned `BLOCK`; no planner plan was created.
- **Unsupported period:** returned a date-coverage answer with SQL evidence rather than a forecast or guessed business conclusion.
- **General chat:** Jev allowed the greeting and the Featherless chat model returned a reply.

These are four individual smoke cases, not a broad accuracy estimate. The analytics scenario's HITL action was automated in AppTest, and the database is synthetic Cedar & Stone demo data.


## Current live workflow follow-up (2026-10-16)

run_workflow_smoke.py now covers eight named scenarios. It uses configured Jev/Featherless services and synthetic Cedar & Stone data. Business scenarios simulate the explicit approval click in Streamlit AppTest.

The final sequential run passed analytics, customer, operations, forecast, review_analysis, injection, unsupported_period, and general_chat. Business reports were verified with resolvable evidence citations and output-guard ALLOW; injection was held before planning; the out-of-range period returned coverage evidence.

The review-analysis route has now been verified with the actual planner:

- customer: the original generated SQL path failed inconsistently and once described non-delivered statuses. A deterministic six-month repeat-rate SQL template now filters only delivered orders; it returns Premium 24/68 (35.29%), Value 54/153 (35.29%), and Core 99/261 (37.93%). The deterministic report and final live run passed verification and output screening with two SQL evidence items.
- review_analysis: the actual planner initially created duplicate review work for two specialists; a planner instruction now keeps qualitative review themes in one Customer Analyst task. The specialist deterministically retrieves passages and produces directly quoted findings with every passage cited, grouped by recorded rating without claiming population prevalence. Exact ratings from cited excerpts remain allowed; mismatched ratings and unsupported numbers remain blocked. The live actual-planner route passed through verification and output screening with five RAG passages and no SQL.
- One diagnostic smoke attempt exposed a missing `json` import in the test-only rejection logger. That harness defect is fixed; it did not affect the app.

These eight named-scenario results are workflow regression checks, not an accuracy estimate or production certification. Unrestricted model-generated SQL still needs broader first-attempt reliability tests. The RAG benchmark remains weak: passage precision@5 is 20% on eight development cases. The eight held-out cases remain untouched.

## Seven-metric end-to-end agent evaluation — 2026-11-12

- Ran all 8 development scenarios with the seven-metric runner. All 8 produced score rows (52 applicable metric-case evaluations); the live workflow passed 7/8.
- Mean scores: task completion **0.9236**, step efficiency **0.9299**, argument correctness **0.9896**, tool correctness **0.7709**, plan adherence **1.0000** (7 applicable cases), plan quality **0.9200** (5 applicable cases), answer correctness **0.8812**.
- `review_analysis` failed safely in `held` status because the configured local RAG database at `127.0.0.1:5433` timed out and returned no review passages. This is a dependency failure, not a Jev-blocked score.
- Forecast checks missed date and uncertainty markers and one SQL argument pattern. The customer answer omitted the exact `six-month` label marker. See the per-case report for full details.
- Full report: `evals/agent_metric_report_20261112T171811Z.md`; scorecard: `evals/results_agent_metrics_20261112T171811Z.json`; trace: `evals/predictions_agent_metrics_20261112T171811Z.jsonl`.
- Re-run from the project root with `python -B evals\run_agent_eval.py`. This is a development regression benchmark, not a representative release-quality estimate.

## Seven-metric benchmark rerun — dataset v1.0.2 — 2026-11-12

- Full live rerun passed all 8 development workflows, including review analysis with five retrieved RAG passages; all 52 applicable metric-case evaluations scored 1.0. Applicable cases: plan adherence 7, plan quality 5, and the other five metrics 8 each.
- Review completed after Docker Compose `ragdb` and `app` reported healthy.
- Dataset/evaluator v1.0.2 accepts legitimate date forms and workflow-phase distinctions; forecast output wording and review proxy labeling were clarified. Do not compare aggregate scores directly with v1.0.0.
- Report: `evals/agent_metric_report_20261112T175204Z.md`; scorecard: `evals/results_agent_metrics_20261112T175204Z.json`; trace: `evals/predictions_agent_metrics_20261112T175204Z.jsonl`.
- The eight development scenarios are diagnostic regression checks, not a broad performance estimate.

## Layered RAG and whole-application evaluation — initial baseline — 2026-11-13

- Initial isolated generator: 4/4 cases produced answers from oracle passages, with no retriever calls. Mean Faithfulness 1.000, Answer Relevancy 1.000, and Citation Correctness 0.975.
- Initial isolated retriever: 8/8 development queries completed. Source precision/recall were 0.633/1.000; passage precision@5/recall@5 were 0.200/0.750 across the two passage-labeled cases.
- Initial wired pipeline: 4/4 answers generated from the live retriever. Contextual Relevancy averaged 0.135 on 3/4 scored cases (one structured-judge error); Faithfulness 0.938, Answer Relevancy 1.000, and Citation Correctness 0.975.
- Whole app: all 8 development workflows passed their deterministic checks. The optional gold-linked judge passed 4/4 linked cases. Style G-Eval averaged 0.75 across all 8 responses (0.86 across five business reports); output guard matched 20/20 labels; toxicity matched 8/8 controls and all 5/5 verified reports were judged non-toxic.
- Mean workflow latency was 31.8 seconds, with 14 model calls and 20 tool calls total. Token and cost usage remain uninstrumented.
- This is the initial baseline; it was superseded by the improvement cycle below. The whole-application workflow, style, and safety scores above were not rerun as part of the retrieval-only change.

## RAG retrieval improvement cycle — 2026-11-13

- Re-chunked Markdown definitions/table rows and JSON records, routed explicit review vs document queries by source type, and selected compact passages for multi-part questions. Reindexed 8 company/upload sources; all 926 review vectors remained unchanged.
- Updated 5 development retrieval query passage labels (2 → 7); holdout remained untouched. Retrieval succeeded on 8/8 cases.
- Source precision/recall/F1/exact match: **0.833 / 0.938 / 0.875 / 0.625**, versus initial **0.633 / 1.000 / 0.763 / 0.125** on the same queries and source labels.
- Passage P@5 0.343, returned-passage precision 0.810, recall@5 0.786, MRR 0.929, nDCG 0.801 (7 passage-labeled cases; not directly comparable to initial two-case labels).
- Pipeline: Contextual Relevancy **0.650 (4/4)**, Faithfulness 1.000, Answer Relevancy 1.000, Citation Correctness 1.000. Three answerable queries averaged 0.867 contextual relevance; the unsupported gross-margin control scores 0.000 for contextual relevance but passes deterministic abstention.
- Targeted whole-app `review_analysis` smoke after the retriever change passed all seven Agent Eval metrics on one scenario; it returned five cited passages and completed verification/output guard. The full eight-workflow app suite was not rerun.
- Remaining: delivery-eligibility query missed the expected data-dictionary source; reporting-context relevance is 0.600. See [RAG_EVALUATION_REPORT_20261113.md](RAG_EVALUATION_REPORT_20261113.md) for per-case reasons and artifacts. These are small synthetic dev results, not a production estimate.
