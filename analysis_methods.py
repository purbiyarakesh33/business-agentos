"""Small, deterministic analysis methods used by the Data Analyst Agent."""
from __future__ import annotations

import itertools
import math
import re
from typing import Any


class AnalysisMethodError(ValueError):
    """A requested statistical method cannot be applied to the supplied evidence."""


_CAUSAL_WORDS = re.compile(
    r"\b(why|cause|causal|causation|caused|driver|drivers|driven|drive|drives|driving|impact|effect|influence|explain|explains|explained|because|reason|reasons|led)\b",
    flags=re.IGNORECASE,
)
_DESIGN_FIELD_WORDS = ("treatment", "control_group", "experiment", "randomized", "randomised", "assignment")


def causal_evidence_review(question: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Conservatively assess whether supplied evidence includes a causal design.

    A column name alone never authorizes a causal claim. This demo has no validated
    randomization protocol or other identification strategy, so effect estimates
    remain withheld even if a candidate assignment field is discovered.
    """
    if not _CAUSAL_WORDS.search(question):
        return None
    column_names: set[str] = set()
    for item in evidence:
        for source in item.get("source_discovery", []) if isinstance(item.get("source_discovery"), list) else []:
            for column in source.get("columns", []) if isinstance(source, dict) else []:
                if isinstance(column, dict) and isinstance(column.get("name"), str):
                    column_names.add(column["name"].lower())
    candidates = sorted(
        name for name in column_names
        if any(token in name for token in _DESIGN_FIELD_WORDS)
    )
    if candidates:
        status = "candidate_design_fields_unverified"
        explanation = (
            "Candidate experiment/assignment columns exist, but their meaning, assignment process, "
            "compliance, timing, and confounder handling are not validated. No causal effect estimate was made."
        )
    else:
        status = "observational_association_only"
        explanation = (
            "The approved data contains descriptive business records but no verified randomized assignment "
            "or causal identification strategy. The comparisons show association only; no cause can be attributed."
        )
    return {
        "status": status,
        "causal_claim_allowed": False,
        "candidate_design_fields": candidates,
        "explanation": explanation,
        "requirements_for_causal_estimate": [
            "A clearly defined treatment/exposure and outcome",
            "A documented assignment or identification design (for example, a randomized experiment or justified quasi-experiment)",
            "A valid comparison group, measurement window, and checks for bias/confounding",
        ],
    }


def _beta_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iterations = 250
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise AnalysisMethodError("The Student t distribution calculation did not converge.")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if not (a > 0 and b > 0 and 0 <= x <= 1):
        raise AnalysisMethodError("Invalid parameters for the Student t distribution.")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_freedom: float) -> float:
    if degrees_freedom <= 0 or not math.isfinite(degrees_freedom):
        raise AnalysisMethodError("Welch test produced invalid degrees of freedom.")
    if value == 0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    tail = 0.5 * _regularized_beta(x, degrees_freedom / 2.0, 0.5)
    return 1.0 - tail if value > 0 else tail


def _student_t_critical(confidence: float, degrees_freedom: float) -> float:
    target = (1.0 + confidence) / 2.0
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_freedom) < target:
        high *= 2.0
        if high > 1e8:
            raise AnalysisMethodError("Could not calculate the requested confidence interval.")
    for _ in range(100):
        middle = (low + high) / 2.0
        if _student_t_cdf(middle, degrees_freedom) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _welch_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    n1, n2 = int(left["n"]), int(right["n"])
    sum1, sum2 = float(left["sum_value"]), float(right["sum_value"])
    sq1, sq2 = float(left["sum_squares"]), float(right["sum_squares"])
    mean1, mean2 = sum1 / n1, sum2 / n2
    variance1 = max(0.0, (sq1 - sum1 * sum1 / n1) / (n1 - 1))
    variance2 = max(0.0, (sq2 - sum2 * sum2 / n2) / (n2 - 1))
    component1, component2 = variance1 / n1, variance2 / n2
    standard_error = math.sqrt(component1 + component2)
    if standard_error == 0:
        raise AnalysisMethodError(
            "Welch test is undefined because both groups have zero observed variance."
        )
    degrees = (component1 + component2) ** 2 / (
        component1 * component1 / (n1 - 1) + component2 * component2 / (n2 - 1)
    )
    difference = mean1 - mean2
    t_statistic = difference / standard_error
    p_value = min(1.0, max(0.0, 2.0 * (1.0 - _student_t_cdf(abs(t_statistic), degrees))))
    critical = _student_t_critical(0.95, degrees)
    pooled_df = n1 + n2 - 2
    pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / pooled_df
    if pooled_variance > 0:
        correction = 1.0 - 3.0 / (4.0 * pooled_df - 1.0)
        hedges_g = correction * difference / math.sqrt(pooled_variance)
    else:
        hedges_g = None
    return {
        "group_1": left["group_label"],
        "group_2": right["group_label"],
        "n_1": n1,
        "n_2": n2,
        "mean_1": mean1,
        "mean_2": mean2,
        "mean_difference_1_minus_2": difference,
        "welch_t": t_statistic,
        "degrees_of_freedom": degrees,
        "p_value_two_sided": p_value,
        "confidence_level": 0.95,
        "confidence_interval_adjustment": "Unadjusted per-pair 95% interval; Holm correction is applied to p-values only.",
        "confidence_interval_difference": [
            difference - critical * standard_error,
            difference + critical * standard_error,
        ],
        "hedges_g": hedges_g,
        "effect_size_note": "Hedges' g uses the pooled within-group standard deviation; undefined when pooled variance is zero.",
        "significant_at_0_05_before_adjustment": p_value < 0.05,
    }


def welch_tests_from_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run pairwise Welch tests from SQL group n/sum/sum-of-squares summaries.

    Expected row fields are ``group_label``, ``n``, ``sum_value``, and
    ``sum_squares``. Holm correction is applied across all returned pairs.
    """
    if not isinstance(rows, list) or not 2 <= len(rows) <= 8:
        raise AnalysisMethodError("Statistical comparison requires between two and eight groups.")
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise AnalysisMethodError("Statistical summary rows have an invalid shape.")
        label = row.get("group_label")
        n = row.get("n")
        total = row.get("sum_value")
        squares = row.get("sum_squares")
        if not isinstance(label, str) or not label.strip() or label in seen:
            raise AnalysisMethodError("Statistical groups must have distinct, non-empty labels.")
        if (
            isinstance(n, bool) or not isinstance(n, (int, float))
            or not math.isfinite(float(n)) or int(n) != n or n < 2
        ):
            raise AnalysisMethodError(f"Group {label} needs at least two non-null independent observations.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
               for value in (total, squares)):
            raise AnalysisMethodError(f"Group {label} has invalid numeric sufficient statistics.")
        n = int(n)
        total, squares = float(total), float(squares)
        # Catch a bad/rounded aggregate that would imply a materially negative variance.
        raw_numerator = squares - total * total / n
        if raw_numerator < -1e-7 * max(1.0, abs(squares)):
            raise AnalysisMethodError(f"Group {label} has inconsistent sum and sum-of-squares values.")
        seen.add(label)
        groups.append({"group_label": label, "n": n, "sum_value": total, "sum_squares": squares})

    comparisons = [_welch_pair(a, b) for a, b in itertools.combinations(groups, 2)]
    ordered = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_value_two_sided"])
    adjusted = [0.0] * len(comparisons)
    running_max = 0.0
    count = len(comparisons)
    for rank, index in enumerate(ordered):
        running_max = max(running_max, (count - rank) * comparisons[index]["p_value_two_sided"])
        adjusted[index] = min(1.0, running_max)
    for item, adjusted_p in zip(comparisons, adjusted):
        item["p_value_holm_adjusted"] = adjusted_p
        item["significant_at_0_05_after_holm"] = adjusted_p < 0.05
    return {
        "test": "pairwise Welch independent-samples t-test",
        "null_hypothesis": "The two groups have equal population means.",
        "alternative": "Two-sided difference in means.",
        "multiple_comparison_correction": "Holm step-down correction across all group pairs.",
        "assumptions": [
            "The input is a numeric outcome summarized at the independent observation grain.",
            "Observations are independent within and between groups.",
            "The sampling process supports inference to the population of interest.",
            "Within-group outcomes are sufficiently well-behaved for a t approximation, especially with small samples.",
            "Welch's test does not assume equal group variances; repeated observations, severe skew, or clustering can still invalidate the result.",
        ],
        "interpretation_limit": "A p-value is not the probability that the null is true and does not establish causality. Statistical significance does not imply business importance. These are synthetic demo records, not a probability sample; population generalization is not justified without a valid sampling design.",
        "groups": [
            {"group_label": g["group_label"], "n": g["n"], "mean": g["sum_value"] / g["n"]}
            for g in groups
        ],
        "comparisons": comparisons,
    }
