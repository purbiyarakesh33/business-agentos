from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import calendar
import math
import re
from typing import Any, Callable
from observability import trace_component


class ForecastingError(RuntimeError):
    """Raised when a forecast request cannot be safely interpreted."""


@dataclass(frozen=True)
class ForecastRequest:
    metric: str
    metric_label: str
    horizon: int
    dimension: str | None
    dimension_label: str
    retrospective_from_latest: bool
    forecast_origin: date | None
    scenario_change_pct: float | None
    scenario_note: str | None
    assumptions: list[str]


MIN_MONTHS = 12
MAX_HORIZON = 12
MAX_BACKTEST_FOLDS = 6
SEASON_LENGTH = 12
FORECAST_TEMPLATE_ID = "deterministic_monthly_forecast_v1"


def _add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def parse_forecast_request(question: str) -> ForecastRequest:
    text = question.casefold()
    if re.search(r"\b(daily|weekly|quarterly|annual|yearly)\b", text):
        raise ForecastingError("This first forecasting version supports monthly periods; daily, weekly, quarterly, and annual grain are not enabled yet.")
    if re.search(r"\bprofit margin\b", text):
        raise ForecastingError("Gross-profit-margin forecasting is not enabled; ask for gross profit or net product sales instead.")
    if re.search(r"\b(new vs returning|new versus returning|repeat(?:ing)? customers?|customer segments?|cohorts?|sku|suppliers?)\b|\bby\s+products?\b(?!\s+categor(?:y|ies)\b)", text):
        raise ForecastingError("This version forecasts one series or a category, region, or channel breakdown. Customer, cohort, SKU, product-level, and supplier forecasts are not enabled yet.")
    if re.search(r"\b(net|operating) profit\b|\bebitda\b", text):
        raise ForecastingError("Only gross profit is available as a forecast metric; net profit, operating profit, and EBITDA are not present in the approved metric definitions.")
    if re.search(r"\bprofit\b", text) and not re.search(r"\bgross profit\b", text):
        raise ForecastingError("Specify gross profit; the unqualified term 'profit' is ambiguous in the available data.")

    if re.search(r"\bgross profit\b", text):
        metric, metric_label = "gross_profit", "gross profit"
    elif re.search(r"\b(units?|demand|quantity)\b", text):
        metric, metric_label = "units", "units sold"
    elif re.search(r"\b(orders?|order count)\b", text):
        metric, metric_label = "orders", "delivered orders"
    elif re.search(r"\b(sales|revenue|net product sales)\b", text):
        metric, metric_label = "net_sales", "delivered net product sales"
    else:
        raise ForecastingError("Name a supported forecast metric: net product sales, gross profit, delivered orders, or units sold.")

    dimensions = []
    if re.search(r"\b(product categor(?:y|ies)|categor(?:y|ies))\b", text):
        dimensions.append(("category", "product category"))
    if re.search(r"\bregions?\b", text):
        dimensions.append(("region", "region"))
    if re.search(r"\bchannels?\b|\bsales channel\b", text):
        dimensions.append(("channel", "sales channel"))
    if len(dimensions) > 1:
        raise ForecastingError("This first forecasting version can forecast one requested breakdown at a time; ask for category, region, or channel separately.")
    dimension, dimension_label = dimensions[0] if dimensions else (None, "all delivered orders")

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    month_names = "|".join(calendar.month_name[number].lower() for number in range(1, 13))
    explicit_month_range = re.search(
        rf"\b({month_names})\s*(?:through|thru|to|[-–—])\s*({month_names})\s+(20\d{{2}})\b",
        text,
    )
    horizon_match = re.search(r"\b(?:next|coming|upcoming|for)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:months?|periods?)\b", text)
    if horizon_match:
        token = horizon_match.group(1)
        horizon = int(token) if token.isdigit() else words[token]
        assumptions = []
    elif explicit_month_range:
        start_month = list(calendar.month_name).index(explicit_month_range.group(1).capitalize())
        end_month = list(calendar.month_name).index(explicit_month_range.group(2).capitalize())
        horizon = end_month - start_month + 1 if end_month >= start_month else 13 - start_month + end_month
        assumptions = []
    elif re.search(r"\bnext month\b", text):
        horizon, assumptions = 1, []
    elif re.search(r"\bnext quarter\b", text):
        horizon, assumptions = 3, ["A quarter is interpreted as three monthly forecast periods."]
    elif re.search(r"\bnext year\b", text):
        horizon, assumptions = 12, ["A year is interpreted as twelve monthly forecast periods."]
    else:
        horizon, assumptions = 3, ["No forecast horizon was specified; the agent uses three monthly periods."]
    if not 1 <= horizon <= MAX_HORIZON:
        raise ForecastingError(f"Forecast horizon must be between 1 and {MAX_HORIZON} months.")

    origin_match = re.search(
        r"\b(?:as\s+of|forecast\s+origin(?:\s+date)?(?:\s+(?:is|at)|\s*[:=])?)\s*(20\d{2}-\d{2}-\d{2})\b|"
        r"\b(?:treat|use)\s+(20\d{2}-\d{2}-\d{2})\s+as\s+(?:the\s+)?(?:historical\s+)?forecast\s+origin\b|"
        r"\b(20\d{2}-\d{2}-\d{2})\s+as\s+(?:the\s+)?forecast\s+origin\b",
        text,
    )
    forecast_origin = None
    if origin_match:
        try:
            origin_token = next(group for group in origin_match.groups() if group)
            forecast_origin = date.fromisoformat(origin_token)
        except ValueError:
            raise ForecastingError("Forecast origin must be a valid ISO date (YYYY-MM-DD).") from None
        if (forecast_origin + timedelta(days=1)).month == forecast_origin.month:
            raise ForecastingError("A monthly historical forecast origin must be the final calendar day of a month.")

    retrospective = bool(re.search(
        r"\b(after|following)\s+(?:the\s+)?(?:latest|last|final)\s+(?:available\s+|observed\s+|data\s+)?(?:month|period|data)\b|"
        r"\bfrom\s+(?:the\s+)?(?:latest|last)\s+(?:available\s+)?(?:month|period|data)\b",
        text,
    ))
    scenario_pct = None
    scenario_note = None
    if re.search(r"\b(what if|scenario|assume|assuming)\b", text):
        scenario_match = re.search(
            r"\b(?:sales|revenue|demand|units|orders?)\b.{0,48}?\b(increase|increased|grow|grows|rise|rises|decrease|decreased|drop|drops|fall|falls)\b.{0,24}?(\d+(?:\.\d+)?)\s*%",
            text,
        )
        if scenario_match:
            magnitude = float(scenario_match.group(2))
            if magnitude >= 100:
                raise ForecastingError("Scenario percentage must be less than 100% in magnitude.")
            sign = -1 if scenario_match.group(1) in {"decrease", "decreased", "drop", "drops", "fall", "falls"} else 1
            scenario_pct = sign * magnitude
            assumptions.append(f"The scenario mechanically applies a {scenario_pct:+.2f}% change to the base forecast; it is not an estimated causal effect.")
        elif re.search(r"\b(marketing|promotion|discount|price|campaign|advertising)\b", text):
            scenario_note = "The agent cannot estimate the sales impact of a marketing, promotion, or price action without a measured causal relationship; only the base forecast is reported."

    assumptions.append("Only delivered orders are included; line_net_sales excludes shipping and later refunds.")
    if forecast_origin is not None:
        assumptions.append(
            f"Historical forecast origin is fixed at {forecast_origin.isoformat()}; source rows after this date are excluded."
        )
    return ForecastRequest(metric, metric_label, horizon, dimension, dimension_label,
                           retrospective, forecast_origin, scenario_pct, scenario_note, assumptions)


def build_monthly_series_sql(
    request: ForecastRequest, task: Any, allowed_tables: set[str], schemas: list[dict[str, Any]],
    as_of: date | None = None, include_as_of_month: bool = False,
) -> str:
    as_of = as_of or date.today()
    columns = {item.get("table"): set(item.get("columns", [])) for item in schemas}
    needed_tables = {"orders"}
    needed_columns = {"orders": {"order_id", "order_date", "status"}}
    needs_items = request.metric != "orders" or request.dimension == "category"
    if needs_items:
        needed_tables.add("order_items")
        needed_columns["order_items"] = {"order_id"}
        if request.metric == "net_sales":
            needed_columns["order_items"].add("line_net_sales")
        elif request.metric == "gross_profit":
            needed_columns["order_items"].add("line_gross_profit")
        elif request.metric == "units":
            needed_columns["order_items"].add("quantity")
    if request.dimension == "category":
        needed_columns["order_items"].add("product_id")
        needed_tables.add("products")
        needed_columns["products"] = {"product_id", "category"}
    elif request.dimension == "region":
        needed_columns["orders"].add("region")
    elif request.dimension == "channel":
        needed_columns["orders"].add("channel")

    missing_tables = needed_tables - allowed_tables
    missing_columns = [f"{table}.{column}" for table, required in needed_columns.items()
                       for column in required - columns.get(table, set())]
    if missing_tables or missing_columns:
        missing = sorted(missing_tables) + missing_columns
        raise ForecastingError("Approved data is missing required forecast fields: " + ", ".join(missing) + ".")

    measure = {
        "net_sales": "SUM(oi.line_net_sales)",
        "gross_profit": "SUM(oi.line_gross_profit)",
        "units": "SUM(oi.quantity)",
        "orders": "COUNT(DISTINCT o.order_id)",
    }[request.metric]
    if request.dimension == "category":
        dim_expr, dim_alias, history_months = "p.category", "category", 18
        joins = "JOIN order_items AS oi ON oi.order_id = o.order_id JOIN products AS p ON p.product_id = oi.product_id"
    elif request.dimension == "region":
        dim_expr, dim_alias, history_months = "o.region", "region", 24
        joins = "JOIN order_items AS oi ON oi.order_id = o.order_id" if needs_items else ""
    elif request.dimension == "channel":
        dim_expr, dim_alias, history_months = "o.channel", "channel", 24
        joins = "JOIN order_items AS oi ON oi.order_id = o.order_id" if needs_items else ""
    else:
        dim_expr, dim_alias, history_months = "'All'", "all_series", 36
        joins = "JOIN order_items AS oi ON oi.order_id = o.order_id" if needs_items else ""
    select_dimension = f"COALESCE(CAST({dim_expr} AS TEXT), '(unknown)') AS series_label"
    group_clause = f", {dim_expr}" if request.dimension else ""
    upper_bound = (
        (as_of + timedelta(days=1)).isoformat()
        if include_as_of_month else as_of.replace(day=1).isoformat()
    )
    return f"""WITH latest_complete_month AS (
    SELECT date(MAX(order_date), 'start of month') AS latest_month
    FROM orders
    WHERE status = 'delivered' AND order_date < '{upper_bound}'
), monthly AS (
    SELECT date(o.order_date, 'start of month') AS period_start,
           {select_dimension},
           {measure} AS value,
           COUNT(DISTINCT o.order_id) AS delivered_order_count
    FROM orders AS o
    CROSS JOIN latest_complete_month AS lm
    {joins}
    WHERE o.status = 'delivered'
      AND date(o.order_date, 'start of month') >= date(lm.latest_month, '-{history_months - 1} months')
      AND date(o.order_date, 'start of month') <= lm.latest_month
    GROUP BY date(o.order_date, 'start of month'){group_clause}
)
SELECT period_start, series_label, value, delivered_order_count
FROM monthly
ORDER BY period_start, series_label"""


def _month_start(value: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10]).replace(day=1)
    except (ValueError, TypeError):
        raise ForecastingError("The time-series query returned a non-date monthly period.") from None


def _predict(values: list[float], model: str, horizon: int, allow_negative: bool = False) -> list[float]:
    n = len(values)
    clip = (lambda value: value) if allow_negative else (lambda value: max(0.0, value))
    if model == "naive":
        return [clip(values[-1])] * horizon
    if model == "trend":
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / n
        denominator = sum((index - mean_x) ** 2 for index in range(n))
        slope = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator if denominator else 0.0
        intercept = mean_y - slope * mean_x
        return [clip(intercept + slope * (n + step)) for step in range(horizon)]
    if model.startswith("ses_"):
        alpha = float(model.split("_", 1)[1])
        level = values[0]
        for value in values[1:]:
            level = alpha * value + (1 - alpha) * level
        return [clip(level)] * horizon
    if model == "seasonal_naive":
        return [clip(values[-SEASON_LENGTH + (step % SEASON_LENGTH)]) for step in range(horizon)]
    raise ForecastingError("Unknown forecast candidate model.")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def forecast_one_series(
    observations: list[tuple[date, float]], horizon: int, metric: str,
) -> dict[str, Any]:
    if len(observations) < MIN_MONTHS:
        raise ForecastingError(f"Only {len(observations)} monthly observations are available; at least {MIN_MONTHS} are required.")
    periods = [item[0] for item in observations]
    for previous, current in zip(periods, periods[1:]):
        if _add_months(previous, 1) != current:
            raise ForecastingError(f"Monthly history has a gap between {previous:%b %Y} and {current:%b %Y}; missing periods are not silently treated as zero.")
    values = [float(item[1]) for item in observations]
    if any(not math.isfinite(value) for value in values) or (
        metric != "gross_profit" and any(value < 0 for value in values)
    ):
        raise ForecastingError("Monthly measures must be finite; only gross profit may be negative.")

    seasonal_allowed = len(values) >= 2 * SEASON_LENGTH + horizon + 3
    min_train = 2 * SEASON_LENGTH if seasonal_allowed else max(6, min(12, len(values) - horizon - 3))
    first_origin = max(min_train, len(values) - horizon - MAX_BACKTEST_FOLDS + 1)
    origins = list(range(first_origin, len(values) - horizon + 1))
    if len(origins) < 3:
        raise ForecastingError(f"Only {len(origins)} rolling backtest folds fit this history and horizon; at least 3 are required.")
    models = ["naive", "trend", "ses_0.2", "ses_0.5", "ses_0.8"]
    if seasonal_allowed:
        models.append("seasonal_naive")
    scores: dict[str, dict[str, Any]] = {}
    errors_by_step: dict[str, list[list[float]]] = {}
    for model in models:
        errors = [[] for _ in range(horizon)]
        absolute_errors = []
        actual_total = 0.0
        for origin in origins:
            predictions = _predict(values[:origin], model, horizon, allow_negative=(metric == "gross_profit"))
            for step, prediction in enumerate(predictions):
                actual = values[origin + step]
                errors[step].append(actual - prediction)
                absolute_errors.append(abs(actual - prediction))
                actual_total += abs(actual)
        mae = sum(absolute_errors) / len(absolute_errors)
        wape = sum(absolute_errors) / actual_total if actual_total else None
        scores[model] = {"mae": round(mae, 4), "wape": round(wape, 6) if wape is not None else None, "folds": len(origins)}
        errors_by_step[model] = errors
    score_key = lambda name: scores[name]["wape"] if scores[name]["wape"] is not None else scores[name]["mae"]
    chosen = min(models, key=score_key)
    point_values = _predict(values, chosen, horizon, allow_negative=(metric == "gross_profit"))
    intervals = []
    for step, point in enumerate(point_values):
        residuals = errors_by_step[chosen][step]
        lower = point + _quantile(residuals, 0.10)
        upper = point + _quantile(residuals, 0.90)
        if metric != "gross_profit":
            lower = max(0.0, lower)
        lower, upper = min(lower, point), max(upper, point)
        intervals.append({"point": round(point, 2), "lower_80": round(lower, 2), "upper_80": round(upper, 2)})
    periods_out = [_add_months(periods[-1], step) for step in range(1, horizon + 1)]
    return {
        "model": chosen,
        "model_comparison": scores,
        "backtest_folds": len(origins),
        "score_name": "WAPE" if scores[chosen]["wape"] is not None else "MAE",
        "score": scores[chosen]["wape"] if scores[chosen]["wape"] is not None else scores[chosen]["mae"],
        "periods": periods_out,
        "forecast": intervals,
    }


def _aggregate_observations(
    grouped: dict[str, list[tuple[date, float]]],
) -> list[tuple[date, float]]:
    """Sum aligned segment histories into the overall series used for reconciliation."""
    if not grouped:
        raise ForecastingError("No segment histories are available for reconciliation.")
    by_label = {label: dict(observations) for label, observations in grouped.items()}
    periods = set(next(iter(by_label.values())))
    if any(set(observations) != periods for observations in by_label.values()):
        raise ForecastingError(
            "Segment histories do not cover identical months; a reconciled forecast cannot be produced."
        )
    return [
        (period, sum(observations[period] for observations in by_label.values()))
        for period in sorted(periods)
    ]


def _reconcile_group_outputs(
    series_outputs: dict[str, dict[str, Any]],
    aggregate_output: dict[str, Any],
    scenario_change_pct: float | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Adjust group point forecasts so they add exactly to the aggregate forecast.

    Each group's rolling-backtest model remains visible. Any difference between
    the sum of those forecasts and the forecast of the summed history is
    allocated by the groups' absolute standalone forecast shares. Group bands
    are shifted with their point estimates; they are not a jointly estimated
    aggregate prediction interval.
    """
    labels = sorted(series_outputs)
    if not labels:
        raise ForecastingError("No segment forecasts are available for reconciliation.")
    copied = {
        label: {
            **output,
            "forecast": [dict(estimate) for estimate in output.get("forecast", [])],
        }
        for label, output in series_outputs.items()
    }
    periods_by_label = {label: output.get("periods", []) for label, output in copied.items()}
    first_periods = periods_by_label[labels[0]]
    if any(periods != first_periods for periods in periods_by_label.values()) or len(first_periods) != len(aggregate_output.get("forecast", [])):
        raise ForecastingError("Segment and aggregate forecast periods do not align.")

    aggregate_rows = []
    for step, target in enumerate(aggregate_output["forecast"]):
        target_point = round(float(target["point"]), 2)
        raw_points = {
            label: float(copied[label]["forecast"][step]["point"])
            for label in labels
        }
        denominator = sum(abs(value) for value in raw_points.values())
        weights = (
            {label: abs(raw_points[label]) / denominator for label in labels}
            if denominator > 1e-12
            else {label: 1.0 / len(labels) for label in labels}
        )
        point_sum = sum(raw_points.values())
        provisional = {
            label: round(raw_points[label] + (target_point - point_sum) * weights[label], 2)
            for label in labels
        }
        # Put any rounding cents on the largest-share group for an exact rollup.
        recipient = max(labels, key=lambda label: (weights[label], label))
        provisional[recipient] = round(
            target_point - sum(value for label, value in provisional.items() if label != recipient), 2
        )
        for label in labels:
            estimate = copied[label]["forecast"][step]
            standalone = raw_points[label]
            adjustment = round(provisional[label] - standalone, 2)
            estimate["standalone_point"] = round(standalone, 2)
            estimate["reconciliation_adjustment"] = adjustment
            estimate["point"] = provisional[label]
            estimate["lower_80"] = min(
                estimate["point"], round(float(estimate["lower_80"]) + adjustment, 2)
            )
            estimate["upper_80"] = max(
                estimate["point"], round(float(estimate["upper_80"]) + adjustment, 2)
            )

        aggregate_row = {
            "period_start": first_periods[step].isoformat(),
            "forecast": target_point,
            "lower_80": round(float(target["lower_80"]), 2),
            "upper_80": round(float(target["upper_80"]), 2),
        }
        if scenario_change_pct is not None:
            factor = 1.0 + float(scenario_change_pct) / 100.0
            scenario_points = {
                label: round(float(copied[label]["forecast"][step]["point"]) * factor, 2)
                for label in labels
            }
            scenario_target = round(target_point * factor, 2)
            scenario_recipient = max(labels, key=lambda label: (weights[label], label))
            scenario_points[scenario_recipient] = round(
                scenario_target - sum(
                    value for label, value in scenario_points.items()
                    if label != scenario_recipient
                ),
                2,
            )
            for label in labels:
                copied[label]["forecast"][step]["scenario_point"] = scenario_points[label]
                copied[label]["forecast"][step]["scenario_reconciliation_adjustment"] = round(
                    scenario_points[label]
                    - round(float(copied[label]["forecast"][step]["point"]) * factor, 2),
                    2,
                )
            aggregate_row.update({
                "scenario_assumption_pct": float(scenario_change_pct),
                "scenario_forecast": scenario_target,
                "scenario_lower_80": round(aggregate_row["lower_80"] * factor, 2),
                "scenario_upper_80": round(aggregate_row["upper_80"] * factor, 2),
            })
        aggregate_rows.append(aggregate_row)

    return copied, aggregate_rows


def _unavailable(task: Any, question: str, reason: str, history: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    forecast = {
        "evidence_id": task.task_id + ":forecast",
        "task_id": task.task_id,
        "agent": "forecasting_agent",
        "objective": task.objective,
        "action": "forecast",
        "evidence_type": "forecast",
        "forecast_template": FORECAST_TEMPLATE_ID,
        "forecast_status": "unavailable",
        "history_evidence_id": history.get("evidence_id") if history else None,
        "reason": reason,
        "rows": [],
        "chart_rows": [],
        "limitations": ["No forecast values were generated because the readiness check failed."],
        "question": question,
    }
    return ([history] if history else []) + [forecast]


class ForecastingAgent:
    """Monthly business forecaster backed by approved read-only SQL and rolling backtests."""

    def __init__(self, query_executor: Callable[[str, Any], dict[str, Any]]):
        self.query_executor = query_executor

    @trace_component("Forecasting Agent", run_type="chain")
    def run_task(self, question: str, task: Any, allowed_tables: set[str], schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            request = parse_forecast_request(question)
            as_of = date.today()
            source_cutoff = request.forecast_origin or as_of
            sql = build_monthly_series_sql(
                request, task, allowed_tables, schemas, as_of=source_cutoff,
                include_as_of_month=request.forecast_origin is not None,
            )
        except ForecastingError as exc:
            return _unavailable(task, question, str(exc))
        result = self.query_executor(sql, task)
        history_id = task.task_id + ":history"
        history = {
            **result,
            "evidence_id": history_id,
            "task_id": task.task_id,
            "agent": "forecasting_agent",
            "objective": task.objective,
            "action": "forecast_history_query",
            "evidence_type": "forecast_history",
            "specialist_sql": result.get("sql"),
        }
        if result.get("truncated"):
            return _unavailable(task, question, "The monthly series query was truncated; forecast calculations require the complete requested history.", history)
        grouped: dict[str, list[tuple[date, float]]] = {}
        for row in result.get("rows", []):
            if not isinstance(row, dict):
                return _unavailable(task, question, "The series query returned an invalid row.", history)
            try:
                period = _month_start(row.get("period_start"))
                value = float(row.get("value"))
            except (TypeError, ValueError, ForecastingError):
                return _unavailable(task, question, "The series query returned an invalid period or measure.", history)
            if not math.isfinite(value):
                return _unavailable(task, question, "The series query returned a non-finite measure.", history)
            label = str(row.get("series_label") or "(unknown)")
            grouped.setdefault(label, []).append((period, value))
        if not grouped:
            return _unavailable(task, question, "No monthly history was found for the approved metric and filters.", history)
        latest = max(period for series in grouped.values() for period, _ in series)
        current_month = as_of.replace(day=1)
        previous_complete_month = _add_months(current_month, -1)
        for label, observations in grouped.items():
            if max(period for period, _value in observations) != latest:
                return _unavailable(task, question,
                    f"The {label} series does not include the latest source month ({latest:%B %Y}); a complete comparable forecast cannot be produced.",
                    history)
        if request.forecast_origin is not None and latest < request.forecast_origin.replace(day=1):
            return _unavailable(
                task, question,
                f"The requested forecast origin is {request.forecast_origin.isoformat()}, but approved source data ends in {latest:%B %Y}.",
                history,
            )
        if request.forecast_origin is None and not request.retrospective_from_latest and latest < previous_complete_month:
            return _unavailable(task, question,
                f"The latest complete source month is {latest:%B %Y}, but a current forecast needs data through at least {previous_complete_month:%B %Y}. Refresh the approved data before forecasting upcoming months.",
                history)

        forecast_rows = []
        model_comparison = {}
        reconciliation = None
        try:
            for label, observations in sorted(grouped.items()):
                model_comparison[label] = forecast_one_series(
                    observations, request.horizon, request.metric
                )
            series_outputs = model_comparison
            if request.dimension and len(series_outputs) > 1:
                aggregate_history = _aggregate_observations(grouped)
                aggregate_output = forecast_one_series(
                    aggregate_history, request.horizon, request.metric
                )
                series_outputs, aggregate_rows = _reconcile_group_outputs(
                    series_outputs, aggregate_output, request.scenario_change_pct
                )
                reconciliation = {
                    "method": "additive_group_point_adjustment_to_aggregate_forecast_v1",
                    "allocation": "The aggregate point-forecast difference is allocated by each group's absolute standalone point-forecast share.",
                    "selected_model": aggregate_output["model"],
                    "score_name": aggregate_output["score_name"],
                    "score": aggregate_output["score"],
                    "folds": aggregate_output["backtest_folds"],
                    "candidates": aggregate_output["model_comparison"],
                    "rows": aggregate_rows,
                }
            for label, output in sorted(series_outputs.items()):
                model_comparison[label] = {
                    "selected_model": output["model"],
                    "score_name": output["score_name"],
                    "score": output["score"],
                    "folds": output["backtest_folds"],
                    "candidates": output["model_comparison"],
                }
                for period, estimate in zip(output["periods"], output["forecast"]):
                    row = {
                        "series_label": label,
                        "period_start": period.isoformat(),
                        "forecast": estimate["point"],
                        "lower_80": estimate["lower_80"],
                        "upper_80": estimate["upper_80"],
                    }
                    if reconciliation:
                        row["standalone_forecast"] = estimate["standalone_point"]
                        row["reconciliation_adjustment"] = estimate["reconciliation_adjustment"]
                    if request.scenario_change_pct is not None:
                        factor = 1.0 + request.scenario_change_pct / 100.0
                        row.update({
                            "scenario_assumption_pct": request.scenario_change_pct,
                            "scenario_forecast": estimate.get("scenario_point", round(estimate["point"] * factor, 2)),
                            "scenario_lower_80": round(estimate["lower_80"] * factor, 2),
                            "scenario_upper_80": round(estimate["upper_80"] * factor, 2),
                        })
                        if reconciliation:
                            row["scenario_reconciliation_adjustment"] = estimate.get(
                                "scenario_reconciliation_adjustment", 0.0
                            )
                    forecast_rows.append(row)
        except ForecastingError as exc:
            return _unavailable(task, question, str(exc), history)

        status = "retrospective" if (
            request.forecast_origin is not None
            or request.retrospective_from_latest and _month_start(forecast_rows[-1]["period_start"]) <= current_month
        ) else "available"
        caveats = [
            "The 80% forecast band is empirical: it uses the 10th and 90th percentiles of rolling-origin forecast errors; it is not a guarantee.",
            "Forecasts are predictive extrapolations and do not establish causes or estimate an action's causal impact.",
            "The current calendar month is excluded. The demo source has no ingestion watermark, so completeness through the latest recorded month is an assumption.",
        ]
        if status == "retrospective":
            caveats.append("The forecast periods are already past as of the current date; this is a retrospective model demonstration, not a current outlook.")
        if reconciliation:
            caveats.append(
                "For a dimension breakdown, group point forecasts are additively adjusted to the forecast of the summed history. Each group interval is shifted by its point adjustment; group intervals are not a jointly estimated aggregate confidence interval and must not be summed."
            )
        if request.scenario_note:
            caveats.append(request.scenario_note)
        forecast_evidence = {
            "evidence_id": task.task_id + ":forecast",
            "task_id": task.task_id,
            "agent": "forecasting_agent",
            "objective": task.objective,
            "action": "forecast",
            "evidence_type": "forecast",
            "forecast_template": FORECAST_TEMPLATE_ID,
            "forecast_status": status,
            "history_evidence_id": history_id,
            "metric": request.metric,
            "metric_label": request.metric_label,
            "dimension": request.dimension,
            "dimension_label": request.dimension_label,
            "retrospective_from_latest": request.retrospective_from_latest,
            "forecast_origin_date": request.forecast_origin.isoformat() if request.forecast_origin else None,
            "grain": "month",
            "horizon": request.horizon,
            "scenario_change_pct": request.scenario_change_pct,
            "scenario_note": request.scenario_note,
            "assumptions": request.assumptions,
            "model_comparison": model_comparison,
            "reconciliation": reconciliation,
            "rows": forecast_rows,
            "limitations": caveats,
            "as_of_date": as_of.isoformat(),
            "source_cutoff_exclusive": (
                (request.forecast_origin + timedelta(days=1)).isoformat()
                if request.forecast_origin else as_of.replace(day=1).isoformat()
            ),
            "forecast_method": "Select the lowest rolling-origin backtest error among naive, linear trend, and simple exponential-smoothing candidates; include seasonal-naive only when two years plus backtest holdouts are available. For multi-series breakdowns, reconcile group point forecasts additively to a forecast of the summed series.",
        }
        chart_rows = []
        for label, observations in sorted(grouped.items()):
            for period, value in observations:
                chart_rows.append({"series_label": label, "period_start": period.isoformat(), "actual": round(value, 2), "forecast": None, "lower_80": None, "upper_80": None})
            for row in forecast_rows:
                if row["series_label"] == label:
                    chart_rows.append({"series_label": label, "period_start": row["period_start"], "actual": None,
                                       "forecast": row["forecast"], "lower_80": row["lower_80"], "upper_80": row["upper_80"]})
        forecast_evidence["chart_rows"] = chart_rows
        forecast_evidence["chart_suggestion"] = {
            "type": "line", "x": "period_start", "y": ["actual", "forecast", "lower_80", "upper_80"],
            "group_column": "series_label",
        }
        return [history, forecast_evidence]


def validate_forecast_evidence(evidence: list[dict[str, Any]]) -> list[str]:
    forecasts = [item for item in evidence if item.get("forecast_template") == FORECAST_TEMPLATE_ID]
    if not forecasts:
        return []
    issues = []
    for forecast in forecasts:
        if forecast.get("forecast_status") == "unavailable":
            if forecast.get("rows"):
                issues.append("An unavailable forecast must not contain forecast values.")
            continue
        history = next((item for item in evidence if item.get("evidence_id") == forecast.get("history_evidence_id")), None)
        if not history or not history.get("specialist_sql") or history.get("truncated"):
            issues.append("Forecast is missing complete SQL history evidence.")
            continue
        by_series: dict[str, list[tuple[date, float]]] = {}
        for row in history.get("rows", []):
            try:
                by_series.setdefault(str(row["series_label"]), []).append((_month_start(row["period_start"]), float(row["value"])))
            except (KeyError, TypeError, ValueError, ForecastingError):
                issues.append("Forecast history contains an invalid observation.")
                break
        if issues:
            continue
        if forecast.get("metric") not in {"net_sales", "gross_profit", "orders", "units"}:
            issues.append("Forecast metric is unsupported.")
            continue
        horizon = forecast.get("horizon")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
            issues.append("Forecast horizon is invalid.")
            continue
        scenario_pct = forecast.get("scenario_change_pct")
        if scenario_pct is not None and (
            not isinstance(scenario_pct, (int, float)) or abs(float(scenario_pct)) >= 100
        ):
            issues.append("Forecast scenario assumption is invalid.")
            continue
        expected = {}
        aggregate_calculation = None
        expected_reconciliation_rows = None
        should_reconcile = forecast.get("dimension") in {"category", "region", "channel"} and len(by_series) > 1
        try:
            for label, observations in by_series.items():
                expected[label] = forecast_one_series(observations, horizon, forecast["metric"])
            if should_reconcile:
                aggregate_history = _aggregate_observations(by_series)
                aggregate_calculation = forecast_one_series(
                    aggregate_history, horizon, forecast["metric"]
                )
                expected, expected_reconciliation_rows = _reconcile_group_outputs(
                    expected, aggregate_calculation, scenario_pct
                )
        except (ForecastingError, KeyError, TypeError, ValueError) as exc:
            issues.append("Independent forecast recalculation failed: " + str(exc)[:260])
            continue
        reconciliation = forecast.get("reconciliation")
        if should_reconcile:
            if not isinstance(reconciliation, dict) or reconciliation.get("method") != "additive_group_point_adjustment_to_aggregate_forecast_v1":
                issues.append("A dimension forecast is missing its supported aggregate reconciliation evidence.")
            elif aggregate_calculation is not None and expected_reconciliation_rows is not None:
                if reconciliation.get("selected_model") != aggregate_calculation["model"]:
                    issues.append("Aggregate reconciliation model does not match the summed-history backtest.")
                if reconciliation.get("folds") != aggregate_calculation["backtest_folds"] or reconciliation.get("score_name") != aggregate_calculation["score_name"]:
                    issues.append("Aggregate reconciliation backtest metadata is inconsistent.")
                if not isinstance(reconciliation.get("score"), (int, float)) or abs(float(reconciliation["score"]) - float(aggregate_calculation["score"])) > 0.000011:
                    issues.append("Aggregate reconciliation score does not match the summed-history backtest.")
                reported_candidates = reconciliation.get("candidates", {})
                if set(reported_candidates) != set(aggregate_calculation["model_comparison"]):
                    issues.append("Aggregate reconciliation candidate models are incomplete or unexpected.")
                else:
                    for candidate, expected_score in aggregate_calculation["model_comparison"].items():
                        reported_score = reported_candidates.get(candidate, {})
                        if reported_score.get("folds") != expected_score["folds"] or any(
                            not isinstance(reported_score.get(key), (int, float)) or abs(float(reported_score[key]) - float(expected_score[key])) > 0.000011
                            for key in ("mae",) + (("wape",) if expected_score["wape"] is not None else ())
                        ):
                            issues.append("Aggregate reconciliation candidate metrics do not match the summed-history backtest.")
                            break
                actual_reconciliation_rows = reconciliation.get("rows", [])
                if not isinstance(actual_reconciliation_rows, list) or len(actual_reconciliation_rows) != len(expected_reconciliation_rows):
                    issues.append("Aggregate reconciliation rows do not match the requested forecast horizon.")
                else:
                    expected_by_period = {row["period_start"]: row for row in expected_reconciliation_rows}
                    reported_by_period = {row.get("period_start"): row for row in actual_reconciliation_rows if isinstance(row, dict)}
                    if set(reported_by_period) != set(expected_by_period):
                        issues.append("Aggregate reconciliation periods are incomplete or duplicated.")
                    for period, target in expected_by_period.items():
                        reported = reported_by_period.get(period, {})
                        for field in ("forecast", "lower_80", "upper_80"):
                            if not isinstance(reported.get(field), (int, float)) or abs(float(reported[field]) - float(target[field])) > 0.011:
                                issues.append(f"Aggregate reconciliation {field} is not independently verified for {period}.")
                        if scenario_pct is not None:
                            for field in ("scenario_forecast", "scenario_lower_80", "scenario_upper_80"):
                                if not isinstance(reported.get(field), (int, float)) or abs(float(reported[field]) - float(target[field])) > 0.011:
                                    issues.append(f"Aggregate scenario {field} is not independently verified for {period}.")
                for period in (row["period_start"] for row in expected_reconciliation_rows):
                    segment_total = sum(
                        float(row.get("forecast", 0)) for row in forecast.get("rows", [])
                        if isinstance(row, dict) and row.get("period_start") == period
                    )
                    target = next(row["forecast"] for row in expected_reconciliation_rows if row["period_start"] == period)
                    if abs(segment_total - target) > 0.011:
                        issues.append(f"Group point forecasts do not reconcile to the aggregate for {period}.")
                    if scenario_pct is not None:
                        scenario_total = sum(
                            float(row.get("scenario_forecast", 0)) for row in forecast.get("rows", [])
                            if isinstance(row, dict) and row.get("period_start") == period
                        )
                        scenario_target = next(row["scenario_forecast"] for row in expected_reconciliation_rows if row["period_start"] == period)
                        if abs(scenario_total - scenario_target) > 0.001:
                            issues.append(f"Group scenario forecasts do not reconcile to the aggregate for {period}.")
        elif reconciliation is not None:
            issues.append("An unreconciled forecast contains unexpected reconciliation metadata.")
        rows = forecast.get("rows")
        if not isinstance(rows, list):
            issues.append("Forecast rows are invalid.")
            continue
        actual_keys = {(row.get("series_label"), row.get("period_start")): row for row in rows if isinstance(row, dict)}
        expected_count = sum(len(value["forecast"]) for value in expected.values())
        if len(actual_keys) != expected_count:
            issues.append("Forecast row count does not match the requested series and horizon.")
            continue
        for label, calculation in expected.items():
            metadata = forecast.get("model_comparison", {}).get(label, {})
            if metadata.get("selected_model") != calculation["model"]:
                issues.append(f"Selected model for {label} does not match backtest results.")
            if metadata.get("folds") != calculation["backtest_folds"]:
                issues.append(f"Backtest fold count for {label} is incorrect.")
            if metadata.get("score_name") != calculation["score_name"] or not isinstance(metadata.get("score"), (int, float)) or abs(float(metadata["score"]) - float(calculation["score"])) > 0.000011:
                issues.append(f"Backtest score for {label} does not match the independent calculation.")
            candidates = metadata.get("candidates", {})
            if set(candidates) != set(calculation["model_comparison"]):
                issues.append(f"Candidate model set for {label} does not match the backtest.")
            else:
                for candidate, expected_score in calculation["model_comparison"].items():
                    reported_score = candidates.get(candidate, {})
                    if reported_score.get("folds") != expected_score["folds"] or any(
                        not isinstance(reported_score.get(key), (int, float)) or abs(float(reported_score[key]) - float(expected_score[key])) > 0.000011
                        for key in ("mae",) + (("wape",) if expected_score["wape"] is not None else ())
                    ):
                        issues.append(f"Candidate model metrics for {label} do not match the independent backtest.")
                        break
            for period, estimate in zip(calculation["periods"], calculation["forecast"]):
                row = actual_keys.get((label, period.isoformat()))
                if not row:
                    issues.append(f"Forecast period missing for {label}: {period.isoformat()}.")
                    continue
                for field, value in (("forecast", estimate["point"]), ("lower_80", estimate["lower_80"]), ("upper_80", estimate["upper_80"])):
                    if not isinstance(row.get(field), (int, float)) or abs(float(row[field]) - value) > 0.011:
                        issues.append(f"{field} for {label} does not match the independent calculation.")
                if should_reconcile:
                    for field, expected_value in (
                        ("standalone_forecast", estimate.get("standalone_point")),
                        ("reconciliation_adjustment", estimate.get("reconciliation_adjustment")),
                    ):
                        if not isinstance(row.get(field), (int, float)) or not isinstance(expected_value, (int, float)) or abs(float(row[field]) - float(expected_value)) > 0.011:
                            issues.append(f"{field} for {label} does not match the independent reconciliation.")
                lower, point, upper = row.get("lower_80"), row.get("forecast"), row.get("upper_80")
                if all(isinstance(value, (int, float)) for value in (lower, point, upper)) and not lower <= point <= upper:
                    issues.append(f"Forecast interval bounds are invalid for {label}.")
                if scenario_pct is not None:
                    factor = 1.0 + float(scenario_pct) / 100.0
                    for field, base_value in (("scenario_forecast", point), ("scenario_lower_80", lower), ("scenario_upper_80", upper)):
                        expected_scenario = round(float(base_value) * factor, 2) if isinstance(base_value, (int, float)) else None
                        if field == "scenario_forecast" and should_reconcile and isinstance(expected_scenario, (int, float)):
                            expected_scenario += float(estimate.get("scenario_reconciliation_adjustment", 0.0))
                            expected_scenario = round(expected_scenario, 2)
                        if not isinstance(row.get(field), (int, float)) or expected_scenario is None or abs(float(row[field]) - expected_scenario) > 0.001:
                            issues.append(f"Scenario value {field} for {label} does not match its stated assumption.")
                elif any(field in row for field in ("scenario_forecast", "scenario_lower_80", "scenario_upper_80")):
                    issues.append(f"{label} contains unrequested scenario values.")
                if should_reconcile and scenario_pct is not None:
                    expected_scenario_adjustment = estimate.get("scenario_reconciliation_adjustment")
                    if (
                        not isinstance(row.get("scenario_reconciliation_adjustment"), (int, float))
                        or not isinstance(expected_scenario_adjustment, (int, float))
                        or abs(float(row["scenario_reconciliation_adjustment"]) - float(expected_scenario_adjustment)) > 0.001
                    ):
                        issues.append(f"scenario_reconciliation_adjustment for {label} does not match the independent reconciliation.")
        try:
            as_of = date.fromisoformat(str(forecast.get("as_of_date")))
            latest_history = max(period for observations in by_series.values() for period, _value in observations)
            current_month = as_of.replace(day=1)
            previous_complete_month = _add_months(current_month, -1)
            forecast_periods = [
                _month_start(row["period_start"]) for row in rows if isinstance(row, dict)
            ]
            first_forecast = min(forecast_periods)
            last_forecast = max(forecast_periods)
            status = forecast.get("forecast_status")
            origin_text = forecast.get("forecast_origin_date")
            origin = date.fromisoformat(origin_text) if isinstance(origin_text, str) else None
            if origin is not None:
                origin_month = origin.replace(day=1)
                if latest_history > origin_month:
                    issues.append("Historical forecast history includes data after its declared origin.")
                if first_forecast != _add_months(origin_month, 1):
                    issues.append("Historical forecast periods do not begin in the month after the declared origin.")
                expected_cutoff = (origin + timedelta(days=1)).isoformat()
                if forecast.get("source_cutoff_exclusive") != expected_cutoff:
                    issues.append("Historical forecast SQL cutoff does not match its declared origin date.")
            if status == "available" and latest_history < previous_complete_month:
                issues.append("A current forecast was marked available using stale source data.")
            if status == "retrospective" and (
                (forecast.get("retrospective_from_latest") is not True and origin is None)
                or last_forecast > current_month
            ):
                issues.append("Retrospective forecast status or target periods are inconsistent.")
            if status not in {"available", "retrospective"}:
                issues.append("Forecast status is invalid.")
        except (TypeError, ValueError, ForecastingError):
            issues.append("Forecast freshness metadata is invalid.")
        category_domains = [item for item in evidence if item.get("evidence_type") == "dimension_domain" and item.get("dimension") == "product_category"]
        if category_domains and forecast.get("dimension") == "category":
            domain = {row.get("dimension_value") for row in category_domains[0].get("rows", []) if isinstance(row, dict)}
            if domain != set(by_series):
                issues.append("Forecast does not cover the full recorded product-category domain.")
    return issues[:8]


def deterministic_forecast_findings(
    question: str, evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    forecast = next((item for item in evidence if item.get("forecast_template") == FORECAST_TEMPLATE_ID), None)
    if forecast is None:
        return None
    history = next((item for item in evidence if item.get("evidence_id") == forecast.get("history_evidence_id")), None)
    refs = [forecast["evidence_id"]]
    if history and history.get("specialist_sql"):
        refs.append(history["evidence_id"])
    if forecast.get("forecast_status") == "unavailable":
        reason = str(forecast.get("reason", "Required forecast data is unavailable."))
        return {"findings": [{
            "claim": "No forecast was generated. " + reason,
            "evidence_ids": [forecast["evidence_id"]] + ([history["evidence_id"]] if history and history.get("specialist_sql") else []),
            "calculation": "The agent stopped at the data-readiness check and emitted no forecast values.",
            "limitations": ["Refresh or clarify the approved data before using this forecast."],
        }]}
    rows = forecast.get("rows", [])
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_series.setdefault(str(row.get("series_label", "All")), []).append(row)
    findings = []
    reconciliation = forecast.get("reconciliation")
    if isinstance(reconciliation, dict):
        aggregate_rows = reconciliation.get("rows", [])
        aggregate_estimates = "; ".join(
            f"{date.fromisoformat(row['period_start']):%b %Y}: {forecast['metric_label']} {row['forecast']:,.2f} "
            f"(empirical 80% band {row['lower_80']:,.2f}–{row['upper_80']:,.2f})"
            + (f"; assumed scenario {forecast['scenario_change_pct']:+.2f}%: {row['scenario_forecast']:,.2f} "
               f"(band {row['scenario_lower_80']:,.2f}–{row['scenario_upper_80']:,.2f})" if forecast.get("scenario_change_pct") is not None else "")
            for row in aggregate_rows
        )
        aggregate_score = reconciliation.get("score")
        aggregate_score_text = (
            f"{float(aggregate_score) * 100:.2f}% WAPE"
            if reconciliation.get("score_name") == "WAPE" and isinstance(aggregate_score, (int, float))
            else f"MAE {float(aggregate_score):,.2f}"
            if isinstance(aggregate_score, (int, float)) else "backtest score unavailable"
        )
        total_status = " This is a retrospective forecast; its target months are already past." if forecast.get("forecast_status") == "retrospective" else ""
        findings.append({
            "claim": f"Across all {forecast['dimension_label']} groups, the reconciled {forecast['metric_label']} forecast is: {aggregate_estimates}.{total_status}",
            "evidence_ids": refs,
            "calculation": (
                f"Summed the grouped monthly SQL history, then selected {reconciliation.get('selected_model', 'unknown')} "
                f"using the lowest aggregate {aggregate_score_text} across {reconciliation.get('folds', 0)} rolling-origin folds. "
                "Group point forecasts are adjusted to reconcile to this aggregate forecast."
            ),
            "limitations": list(forecast.get("limitations", []))[:6],
        })
    for label, series_rows in sorted(by_series.items()):
        estimates = "; ".join(
            f"{date.fromisoformat(row['period_start']):%b %Y}: {forecast['metric_label']} {row['forecast']:,.2f} "
            f"(empirical 80% band {row['lower_80']:,.2f}–{row['upper_80']:,.2f})"
            + (f"; assumed scenario {forecast['scenario_change_pct']:+.2f}%: {row['scenario_forecast']:,.2f} "
               f"(band {row['scenario_lower_80']:,.2f}–{row['scenario_upper_80']:,.2f})" if forecast.get("scenario_change_pct") is not None else "")
            for row in series_rows
        )
        metadata = forecast.get("model_comparison", {}).get(label, {})
        method = metadata.get("selected_model", "unknown")
        score_name = metadata.get("score_name", "backtest error")
        score = metadata.get("score")
        score_text = f"{float(score) * 100:.2f}% WAPE" if score_name == "WAPE" and isinstance(score, (float, int)) else f"MAE {float(score):,.2f}" if isinstance(score, (float, int)) else "backtest score unavailable"
        status_note = " This is a retrospective forecast; its target months are already past." if forecast.get("forecast_status") == "retrospective" else ""
        claim = f"For {label}, the {forecast['metric_label']} forecast is: {estimates}.{status_note}"
        calc = f"Selected {method} using the lowest {score_text} across {metadata.get('folds', 0)} rolling-origin folds. Band: empirical 10th–90th percentiles of horizon-matched backtest residuals."
        if reconciliation:
            calc += " Group point estimates and interval bounds were shifted by the recorded reconciliation adjustment so group point forecasts add to the aggregate forecast."
        limitations = list(forecast.get("limitations", []))[:6]
        findings.append({"claim": claim, "evidence_ids": refs, "calculation": calc, "limitations": limitations})
    if forecast.get("scenario_change_pct") is not None and findings:
        pct = float(forecast["scenario_change_pct"])
        findings.append({
            "claim": f"A mechanical {pct:+.2f}% scenario was applied to the forecast values as requested; it is an assumption-based sensitivity, not a predicted or causal effect.",
            "evidence_ids": refs,
            "calculation": "Scenario value = base forecast × (1 + assumed percentage change).",
            "limitations": ["No causal relationship between an action and the assumed sales change was estimated."],
        })
    return {"findings": findings[:12]}
