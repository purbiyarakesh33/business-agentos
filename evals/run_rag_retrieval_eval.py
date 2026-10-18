"""Run the development-only RAG retrieval benchmark and save passage IDs/evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD = Path(__file__).with_name("golden_dataset.json")


def review_topics(text: str) -> set[str]:
    """Apply a small, explicit lexical taxonomy to retrieved review snippets.

    These tags are a deterministic retrieval-coverage proxy, not an LLM sentiment
    evaluation and not a substitute for counting the full review population.
    """
    value = text.casefold()
    topics = set()
    if any(term in value for term in ("excellent quality", "good quality", "beautiful finish")):
        topics.add("product_quality_positive")
    if any(term in value for term in ("disappointed with the quality", "damaged")):
        topics.add("product_quality_negative")
    if "fast delivery" in value or "arrived on time" in value:
        topics.add("delivery_positive")
    if "delivery took too long" in value:
        topics.add("delivery_negative")
    if "did not match the listing" in value:
        topics.add("listing_mismatch")
    if "cheaper than expected" in value:
        topics.add("value_negative")
    if "easy to use" in value:
        topics.add("usability_positive")
    if "nothing special" in value or "average experience" in value:
        topics.add("neutral_or_mixed")
    if "would buy it again" in value:
        topics.add("repurchase_intent_positive")
    if "works as described" in value:
        topics.add("functionality_as_described")
    return topics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 10:
        raise ValueError("--top-k must be between 1 and 10")
    cases = json.loads(GOLD.read_text(encoding="utf-8"))["cases"]
    cases = [case for case in cases
             if case["component"] == "rag_retrieval" and case["split"] == "development"]
    if not cases:
        raise ValueError("No development RAG cases found.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(__file__).with_name(f"predictions_rag_development_{stamp}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite predictions file: {output}")

    import rag
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for number, case in enumerate(cases, 1):
            row = {"case_id": case["case_id"]}
            try:
                hits = rag.retrieve_relevant(case["query"], top_k=args.top_k)
                row["relevant_sources"] = sorted({str(hit["source_name"]) for hit in hits})
                row["relevant_passages"] = [str(hit["source_ref"]) for hit in hits]
                row["retrieved_passages"] = [
                    {key: hit.get(key) for key in
                     ("source_type", "source_name", "source_ref", "title", "section", "content", "score", "retrieval_method")}
                    for hit in hits
                ]
                topics = set()
                for hit in hits:
                    if hit.get("source_type") == "customer_review":
                        match = re.search(r"Review text:\s*(.*)$", str(hit.get("content", "")))
                        if match:
                            topics.update(review_topics(match.group(1).strip()))
                row["retrieved_topics"] = sorted(topics)
                print(f"{number}/{len(cases)} {case['case_id']}: retrieved {len(hits)} passage(s)")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: retrieval failed; details withheld"
                row["relevant_sources"] = []
                row["relevant_passages"] = []
                print(f"{number}/{len(cases)} {case['case_id']}: failed ({type(exc).__name__})")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved predictions to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("RAG evaluation runner error: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
