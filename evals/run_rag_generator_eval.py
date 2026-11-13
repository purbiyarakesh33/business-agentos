"""Score the isolated RAG generator or the live retriever→generator pipeline.

Both modes use reference-free DeepEval judges for answer quality. ``generator``
feeds reviewed oracle passages directly to the writer and forbids retrieval;
``pipeline`` calls the configured retriever, then passes its actual passages to
the same writer. Output artifacts include each judge's score/reason and a
deterministic citation-resolution / oracle-passage coverage audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "rag_generator_dataset.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))


def load_oracle_context(source_ref: str) -> dict[str, Any]:
    import rag

    source_name, marker, _chunk = source_ref.partition("#chunk-")
    if not marker or not source_name.startswith("company_docs/"):
        raise ValueError(f"Unsupported oracle reference: {source_ref}")
    path = rag.DEFAULT_DOCS_PATH / Path(source_name).name
    _content_hash, chunks = rag._read_file_chunks(path, source_name)
    for chunk in chunks:
        if chunk.get("source_ref") == source_ref:
            return chunk
    raise ValueError(f"Oracle passage not found in the current source files: {source_ref}")


def as_evidence(hit: dict[str, Any], index: int, *, oracle: bool = False) -> dict[str, Any]:
    source_ref = str(hit.get("source_ref", ""))
    return {
        "evidence_id": ("oracle:" + source_ref) if oracle else f"retrieved:{index}:{source_ref}",
        "evidence_type": "document_context" if hit.get("source_type") != "customer_review" else "customer_review",
        "source_name": str(hit.get("source_name", "")),
        "source_ref": source_ref,
        "section": str(hit.get("section", "")),
        "excerpt": str(hit.get("content", "")),
        "retrieval_query": "reviewed oracle passage; retriever bypassed" if oracle else "live RAG retrieval",
        "retrieval_method": hit.get("retrieval_method"),
        "rank": index,
    }


def render_findings(findings: list[dict[str, Any]]) -> str:
    sections = ["## Verified findings"]
    for index, finding in enumerate(findings, 1):
        claim = str(finding.get("claim", "")).strip()
        references = finding.get("evidence_ids", [])
        citation = ", ".join(str(value) for value in references)
        sections.append(f"{index}. {claim} [Evidence: {citation}]")
        if finding.get("calculation"):
            sections.append("Calculation: " + str(finding["calculation"]))
        limitations = finding.get("limitations", [])
        if limitations:
            sections.append("Limitations: " + " ".join(str(item) for item in limitations))
    return "\n\n".join(sections)


def validate_generator_findings(raw: Any, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the generator contract without mixing in SQL-specific app policy.

    The full application validator also enforces business/SQL rules. At the RAG
    generator boundary, document dates and other numeric reference facts must
    still be allowed when directly supported by cited passages.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("findings"), list):
        raise ValueError("Generator output does not contain a findings list.")
    findings = raw["findings"]
    if not findings or len(findings) > 12:
        raise ValueError("Generator output has an empty or oversized findings list.")
    known_ids = {str(item["evidence_id"]) for item in evidence}
    validated = []
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("Generator output contains a malformed finding.")
        claim = finding.get("claim")
        evidence_ids = finding.get("evidence_ids")
        calculation = finding.get("calculation", "")
        limitations = finding.get("limitations", [])
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Generator finding has no claim text.")
        if not isinstance(evidence_ids, list) or not evidence_ids or any(ref not in known_ids for ref in evidence_ids):
            raise ValueError("Generator finding has a missing or unresolved evidence citation.")
        if not isinstance(calculation, str) or not isinstance(limitations, list):
            raise ValueError("Generator finding fields do not match the expected schema.")
        validated.append({"claim": claim.strip(), "evidence_ids": evidence_ids,
                          "calculation": calculation,
                          "limitations": [str(item) for item in limitations]})
    return validated


def citation_audit(findings: list[dict[str, Any]], evidence: list[dict[str, Any]], expected_refs: list[str]) -> dict[str, Any]:
    id_to_ref = {str(item["evidence_id"]): str(item["source_ref"]) for item in evidence}
    cited_ids = {
        str(reference)
        for finding in findings
        for reference in finding.get("evidence_ids", [])
    }
    cited_refs = {id_to_ref[reference] for reference in cited_ids if reference in id_to_ref}
    resolved = cited_ids & set(id_to_ref)
    return {
        "citation_id_precision": round(len(resolved) / len(cited_ids), 4) if cited_ids else 0.0,
        "cited_expected_passage_recall": round(len(cited_refs & set(expected_refs)) / len(expected_refs), 4) if expected_refs else None,
        "cited_evidence_ids": sorted(cited_ids),
        "cited_source_refs": sorted(cited_refs),
        "unresolved_citations": sorted(cited_ids - set(id_to_ref)),
    }


def retrieval_label_audit(evidence: list[dict[str, Any]], expected_refs: list[str], top_k: int) -> dict[str, Any]:
    ranked_refs = [str(item.get("source_ref", "")) for item in evidence]
    expected = set(expected_refs)
    hits = [rank for rank, ref in enumerate(ranked_refs, 1) if ref in expected]
    denominator = max(1, top_k)
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(expected), top_k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "expected_passage_refs": sorted(expected),
        "retrieved_passage_refs": ranked_refs,
        "precision_at_k": round(len(hits) / denominator, 4),
        "recall_at_k": round(len(hits) / len(expected), 4) if expected else None,
        "hit_at_k": bool(hits),
        "mrr_at_k": round(1 / hits[0], 4) if hits else 0.0,
        "ndcg_at_k": round(dcg / idcg, 4) if idcg else None,
        "label_scope_note": "Exact source_ref match against the hand-reviewed oracle refs; labels may not enumerate every useful passage.",
    }


def _judge_metrics(stage: str, query: str, answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams
    from run_deepeval_specialists import make_judge

    judge = make_judge()
    context = [
        f"Evidence ID: {item['evidence_id']}\nSource ref: {item['source_ref']}\nSection: {item['section']}\n{item['excerpt']}"
        for item in evidence
    ]
    test_case = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)
    shared = {"model": judge, "async_mode": False, "include_reason": True, "threshold": None}
    metrics: list[Any] = []
    if stage == "pipeline":
        metrics.append(ContextualRelevancyMetric(**shared))
    metrics.extend((FaithfulnessMetric(**shared), AnswerRelevancyMetric(**shared)))
    metrics.append(GEval(
        name="Citation Correctness",
        criteria=(
            "Evaluate whether each material factual claim in the answer is supported by the passage(s) "
            "identified by its attached evidence citation. Citations must resolve to a supplied Evidence ID, "
            "support the specific nearby claim, and not be used to imply more than the cited text states. "
            "Penalize uncited material claims, mismatched citations, unsupported numbers, and claims presented "
            "as facts when the context only offers an example. Judge only the supplied input, answer, and context."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        model=judge,
        async_mode=False,
        threshold=None,
    ))

    results: dict[str, Any] = {}
    for metric in metrics:
        name = metric.__name__
        attempts = 0
        while True:
            attempts += 1
            try:
                metric.measure(test_case)
                break
            except Exception as exc:
                # Some hosted judges occasionally return a truncated/malformed
                # structured verdict. Retry that parse failure once; never turn
                # a second failure into a score or a pass.
                if attempts == 1 and (
                    type(exc).__name__ == "ValidationError" or "ValidationError" in str(exc)
                ):
                    continue
                results[name] = {
                    "score": None,
                    "reason": f"Judge error after {attempts} attempt(s): {type(exc).__name__}: {str(exc)[:240]}",
                    "threshold": None,
                    "binary_verdict": "error",
                    "attempts": attempts,
                }
                break
        if results.get(name, {}).get("score") is None and results.get(name, {}).get("binary_verdict") == "error":
            continue
        try:
            score = float(metric.score)
            results[name] = {
                "score": round(score, 4),
                "reason": getattr(metric, "reason", None),
                "threshold": None,
                "binary_verdict": "not calibrated; score only",
                "attempts": attempts,
            }
        except Exception as exc:
            results[name] = {
                "score": None,
                "reason": f"Judge error: {type(exc).__name__}: {str(exc)[:240]}",
                "threshold": None,
                "binary_verdict": "error",
            }
    return results


def safe_failure(exc: Exception) -> str:
    message = re.sub(r"(?i)(api[_ -]?key|password|secret|token)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", str(exc))
    return f"{type(exc).__name__}: {message[:300]}"


def evaluate_case(case: dict[str, Any], stage: str, top_k: int) -> dict[str, Any]:
    import business_agents
    from output_guardrails import screen_business_output

    expected_refs = list(case.get("oracle_context_refs", []))
    if stage == "generator":
        evidence = [as_evidence(load_oracle_context(ref), index, oracle=True) for index, ref in enumerate(expected_refs, 1)]
    else:
        evidence = []

    calls = {"report_generator": 0, "retriever": 0}
    original_text = business_agents._featherless_text
    original_retriever = business_agents._retrieve_documents

    def tracked_text(*args: Any, **kwargs: Any) -> Any:
        calls["report_generator"] += 1
        return original_text(*args, **kwargs)

    def live_retriever(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        calls["retriever"] += 1
        return original_retriever(query, top_k=top_k)

    def forbidden_retrieval(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls["retriever"] += 1
        raise RuntimeError("The isolated generator stage must bypass retrieval.")

    business_agents._featherless_text = tracked_text
    business_agents._retrieve_documents = forbidden_retrieval if stage == "generator" else live_retriever
    try:
        if stage == "pipeline":
            hits = business_agents._retrieve_documents(case["query"], top_k=top_k)
            evidence = [as_evidence(hit, index) for index, hit in enumerate(hits, 1)]
        raw = business_agents._draft_findings(case["query"], evidence)
        findings = validate_generator_findings(raw, evidence)
        answer = render_findings(findings)
        guard = screen_business_output(answer, evidence)
        guard_action = getattr(getattr(guard, "action", None), "value", getattr(guard, "action", None))
        audit = citation_audit(findings, evidence, expected_refs)
        case_result: dict[str, Any] = {
            "case_id": case["case_id"],
            "status": "generated",
            "stage": stage,
            "generator_calls": calls["report_generator"],
            "retriever_calls": calls["retriever"],
            "output_guard": str(guard_action),
            "citation_audit": audit,
            "supplementary_reference_diagnostics": {
                "expected_terms": case.get("expected_terms", []),
                "expected_terms_found": [term for term in case.get("expected_terms", []) if term.casefold() in answer.casefold()],
                "forbidden_patterns_found": [pattern for pattern in case.get("forbidden_patterns", []) if re.search(pattern, answer, re.IGNORECASE)],
                "must_abstain": bool(case.get("must_abstain", False)),
                "abstention_check": (
                    bool(re.search(r"\b(cannot|can't|unable to|not enough|not provided|does not provide|cannot be determined)\b", answer, re.IGNORECASE))
                    and not bool(re.search(r"\bUSD\s*[\d,]+|\b\d+(?:\.\d+)?\s*%", answer, re.IGNORECASE))
                ) if case.get("must_abstain") else None,
            },
            "retrieval_label_audit": retrieval_label_audit(evidence, expected_refs, top_k) if stage == "pipeline" else None,
            "answer": answer,
            "evidence": evidence,
        }
        case_result["metrics"] = _judge_metrics(stage, case["query"], answer, evidence)
        return case_result
    finally:
        business_agents._featherless_text = original_text
        business_agents._retrieve_documents = original_retriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generator", "pipeline"), default="generator",
                        help="generator = fixed oracle context/no retrieval; pipeline = live retrieval then generation")
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Run only named case(s); repeat the flag to select more.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth for pipeline stage.")
    parser.add_argument("--predictions", type=Path, help="JSONL output path; defaults to a timestamped evals/ file.")
    parser.add_argument("--scorecard", type=Path, help="JSON scorecard path; defaults to a timestamped evals/ file.")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10")

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = list(dataset.get("cases", []))
    unknown = sorted(set(args.case_ids or []) - {case["case_id"] for case in cases})
    if unknown:
        parser.error("unknown case id(s): " + ", ".join(unknown))
    selected_ids = set(args.case_ids or [])
    selected = [case for case in cases if not selected_ids or case["case_id"] in selected_ids]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictions_path = args.predictions or EVAL_DIR / f"predictions_rag_{args.stage}_{stamp}.jsonl"
    scorecard_path = args.scorecard or EVAL_DIR / f"results_rag_{args.stage}_{stamp}.json"
    for path in (predictions_path, scorecard_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        for number, case in enumerate(selected, 1):
            print(f"[{number}/{len(selected)}] {args.stage}: {case['case_id']}", flush=True)
            try:
                row = evaluate_case(case, args.stage, args.top_k)
            except Exception as exc:
                row = {"case_id": case["case_id"], "stage": args.stage, "status": "failed", "failure": safe_failure(exc), "metrics": {}}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if row.get("status") == "generated":
                metric_summary = ", ".join(
                    f"{name}={value.get('score')}" for name, value in row["metrics"].items()
                )
                print(f"  {metric_summary}", flush=True)
            else:
                print(f"  FAILED: {row.get('failure')}", flush=True)

    metric_names = sorted({name for row in rows for name in row.get("metrics", {})})
    summaries: dict[str, Any] = {}
    for name in metric_names:
        values = [row["metrics"][name].get("score") for row in rows if isinstance(row.get("metrics", {}).get(name, {}).get("score"), (int, float))]
        errors = sum(row.get("metrics", {}).get(name, {}).get("score") is None for row in rows if name in row.get("metrics", {}))
        summaries[name] = {
            "mean_score": round(sum(values) / len(values), 4) if values else None,
            "scored_cases": len(values),
            "total_selected_cases": len(rows),
            "judge_errors": errors,
        }
    scorecard = {
        "dataset_version": dataset.get("dataset_version"),
        "split": dataset.get("split"),
        "stage": args.stage,
        "judge_type": "reference-free LLM-as-a-judge; expected_output is not supplied",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "generated_case_count": sum(row.get("status") == "generated" for row in rows),
        "failed_case_count": sum(row.get("status") != "generated" for row in rows),
        "metric_summaries": summaries,
        "cases": rows,
        "limitations": [
            "The four document questions are a small development set, not a production-quality estimate.",
            "Judge scores are review signals; inspect case-level reasons and calibrate thresholds against human ratings before calling cases passed/failed.",
            "Exact passage-label precision/recall are supplementary and may undercount useful passages because oracle refs are not exhaustive.",
        ],
    }
    scorecard_path.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved scorecard: {scorecard_path}")
    print(f"Generated: {scorecard['generated_case_count']}/{scorecard['case_count']}; failed: {scorecard['failed_case_count']}/{scorecard['case_count']}")
    for name, values in summaries.items():
        print(f"{name}: {values['mean_score']} ({values['scored_cases']}/{values['total_selected_cases']} scored)")
    return 0 if scorecard["failed_case_count"] == 0 and all(value["judge_errors"] == 0 for value in summaries.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print("RAG evaluation error: " + safe_failure(exc), file=sys.stderr)
        raise SystemExit(2)
