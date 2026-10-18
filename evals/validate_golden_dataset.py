"""Validate golden-set structure, component labels, and split leakage guards."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

GOLD = Path(__file__).with_name("golden_dataset.json")
REQUIRED_EXPECTED = {
    "jev": {"security", "intent"},
    "planner_selection": {"agents", "sources"},
    "rag_retrieval": {"relevant_sources"},
    "answer_quality": {"required_claims", "relevant_sources", "numeric_facts"},
    "output_guard": {"action"},
}
COMPONENTS = set(REQUIRED_EXPECTED)
SPLITS = {"development", "holdout"}


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def validate(data: dict) -> list[str]:
    errors = []
    if not isinstance(data.get("cases"), list):
        return ["cases must be a list"]
    seen_ids: set[str] = set()
    seen_queries: dict[tuple[str, str], tuple[str, str]] = {}
    family_splits: dict[str, set[str]] = {}
    for index, item in enumerate(data["cases"]):
        where = f"case[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            continue
        case_id = item.get("case_id")
        component, split = item.get("component"), item.get("split")
        query, expected = item.get("query"), item.get("expected")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{where}: missing case_id")
        elif case_id in seen_ids:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            seen_ids.add(case_id)
        if component not in COMPONENTS:
            errors.append(f"{case_id}: unsupported component {component!r}")
            continue
        if split not in SPLITS:
            errors.append(f"{case_id}: unsupported split {split!r}")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{case_id}: query must be non-empty text")
        if not isinstance(expected, dict):
            errors.append(f"{case_id}: expected must be an object")
        elif missing := REQUIRED_EXPECTED[component] - set(expected):
            errors.append(f"{case_id}: missing expected fields {sorted(missing)}")
        if isinstance(query, str) and split in SPLITS:
            key = (component, normalize(query))
            previous = seen_queries.get(key)
            if previous and previous[1] != split:
                errors.append(f"normalized query crosses splits: {previous[0]} and {case_id}")
            else:
                seen_queries[key] = (case_id, split)
        family = item.get("family_id")
        if family:
            family_splits.setdefault(str(family), set()).add(str(split))
        if component == "output_guard":
            if item.get("report") is not None and not isinstance(item["report"], str):
                errors.append(f"{case_id}: report must be text")
            if item.get("evidence") is not None and not isinstance(item["evidence"], list):
                errors.append(f"{case_id}: evidence must be a list")
        if component == "rag_retrieval" and isinstance(expected, dict):
            if "relevant_passages" in expected and not isinstance(expected["relevant_passages"], list):
                errors.append(f"{case_id}: relevant_passages must be a list")
            if ("relevant_passages" in expected and expected.get("evaluate_passage_ranking", True)
                    and not expected["relevant_passages"]):
                errors.append(f"{case_id}: passage ranking is enabled but relevance list is empty")
            if "relevant_topics" in expected and not isinstance(expected["relevant_topics"], list):
                errors.append(f"{case_id}: relevant_topics must be a list")
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(f"family_id crosses development and holdout: {family}")
    counts = Counter((item.get("component"), item.get("split")) for item in data["cases"] if isinstance(item, dict))
    if data.get("dataset_version") in {"0.5.0", "0.6.0"}:
        minimums = {("jev", "development"): 50, ("planner_selection", "development"): 12,
                    ("rag_retrieval", "development"): 8, ("output_guard", "development"): 20}
        if data.get("dataset_version") == "0.6.0":
            minimums[("answer_quality", "development")] = 10
        for key, minimum in minimums.items():
            if counts[key] < minimum:
                errors.append(f"{key[0]} development has {counts[key]} cases; minimum is {minimum}")
    return errors


if __name__ == "__main__":
    dataset = json.loads(GOLD.read_text(encoding="utf-8"))
    problems = validate(dataset)
    counts = Counter((item["component"], item["split"]) for item in dataset["cases"])
    print(f"Dataset {dataset.get('dataset_version')}: {len(dataset.get('cases', []))} cases")
    for (component, split), count in sorted(counts.items()):
        print(f"  {split:11} {component:18} {count:3}")
    if problems:
        print("VALIDATION FAILED:")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print("VALIDATION PASSED")
