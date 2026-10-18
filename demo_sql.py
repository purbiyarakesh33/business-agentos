from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import calendar
import re
from typing import Any


DEMO_SOURCE_IDS = frozenset({
    "customers", "products", "orders", "order_items", "inventory_snapshots",
    "marketing_campaigns", "fulfillment", "returns", "reviews",
})


@dataclass(frozen=True)
class DemoQuery:
    template_id: str
    sql: str
    limitations: tuple[str, ...] = ()
    rag_query: str | None = None


def is_demo_task(task: Any) -> bool:
    sources = getattr(task, "data_sources", None)
    return bool(sources) and set(sources).issubset(DEMO_SOURCE_IDS)


def _explicit_named_periods(
    question: str, minimum: int = 2, maximum: int = 3,
) -> list[tuple[int, str, str, str]] | None:
    names = {calendar.month_name[number].lower(): number for number in range(1, 13)}
    pattern = r"\b(" + "|".join(names) + r")\b"
    unique_years = {int(value) for value in re.findall(r"\b(20\d{2})\b", question)}
    periods: list[tuple[int, str, str, str]] = []
    seen: set[str] = set()
    candidates = []
    for match in re.finditer(pattern, question, flags=re.IGNORECASE):
        month = names[match.group(1).lower()]
        trailing_year = re.match(r"\s+(20\d{2})\b", question[match.end():])
        if trailing_year:
            year = int(trailing_year.group(1))
        elif len(unique_years) == 1:
            # Support ordinary phrasing such as "compare September and August 2025";
            # a single explicit year unambiguously qualifies the listed months.
            year = next(iter(unique_years))
        else:
            continue
        candidates.append((match.start(), month, year))
    for _position, month, year in sorted(candidates):
        start = date(year, month, 1)
        next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        key = start.strftime("%Y-%m")
        if key in seen:
            continue
        seen.add(key)
        periods.append((len(periods) + 1, start.strftime("%B %Y"), start.isoformat(), next_month.isoformat()))
    return periods if minimum <= len(periods) <= maximum else None


def _explicit_months(question: str) -> list[tuple[int, str, str, str]] | None:
    """Parse the three-period comparison used by the original business templates."""
    periods = _explicit_named_periods(question, 3, 3)
    return periods


def _explicit_half_year(question: str) -> tuple[str, str] | None:
    """Resolve an explicitly named H1/H2 calendar-year window."""
    text = question.casefold()
    match = re.search(r"\b(?:first|second)\s+half(?:\s+of)?\s+(20\d{2})\b", text)
    if not match:
        match = re.search(r"\bh([12])\s+(20\d{2})\b", text)
        if match:
            half, year = int(match.group(1)), int(match.group(2))
        else:
            range_match = re.search(
                r"\b(july)\s+(?:through|to)\s+(december)\s+(20\d{2})\b", text
            )
            if not range_match:
                return None
            half, year = 2, int(range_match.group(3))
    else:
        half = 1 if match.group(0).startswith("first") else 2
        year = int(match.group(1))
    start = f"{year}-{1 if half == 1 else 7:02d}-01"
    end = f"{year}-{7 if half == 1 else 13:02d}-01"
    if half == 2:
        end = f"{year + 1}-01-01"
    return start, end


def _gross_profit_category_window_sql(start: str, end: str) -> str:
    return f"""WITH category_domain AS (
    SELECT DISTINCT COALESCE(category, 'unknown') AS category FROM products
), period_lines AS (
    SELECT o.order_id, COALESCE(p.category, 'unknown') AS category,
           COALESCE(oi.line_gross_profit, 0.0) AS line_gross_profit
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    JOIN products AS p ON p.product_id = oi.product_id
    WHERE o.status = 'delivered'
      AND o.order_date >= '{start}' AND o.order_date < '{end}'
), category_values AS (
    SELECT d.category,
           ROUND(COALESCE(SUM(l.line_gross_profit), 0.0), 2) AS gross_profit_usd,
           COUNT(DISTINCT l.order_id) AS delivered_orders
    FROM category_domain AS d
    LEFT JOIN period_lines AS l ON l.category = d.category
    GROUP BY d.category
), result_rows AS (
    SELECT 'product_category' AS dimension_type, category AS dimension_value,
           gross_profit_usd, delivered_orders
    FROM category_values
    UNION ALL
    SELECT 'total', 'All product categories',
           ROUND(COALESCE(SUM(line_gross_profit), 0.0), 2), COUNT(DISTINCT order_id)
    FROM period_lines
)
SELECT dimension_type, dimension_value, gross_profit_usd, delivered_orders
FROM result_rows
ORDER BY CASE dimension_type WHEN 'total' THEN 0 ELSE 1 END,
         gross_profit_usd DESC, dimension_value"""


def _sales_channel_two_period_sql(
    periods: list[tuple[int, str, str, str]],
) -> str:
    first, second = periods
    start = min(first[2], second[2])
    end = max(first[3], second[3])
    channel_select = f"""SELECT 'sales_channel' AS dimension_type,
           COALESCE(channel, 'unknown') AS dimension_value,
           ROUND(SUM(CASE WHEN order_date >= '{first[2]}' AND order_date < '{first[3]}' THEN order_sales ELSE 0 END), 2) AS period_1_sales,
           ROUND(SUM(CASE WHEN order_date >= '{second[2]}' AND order_date < '{second[3]}' THEN order_sales ELSE 0 END), 2) AS period_2_sales,
           COUNT(DISTINCT CASE WHEN order_date >= '{first[2]}' AND order_date < '{first[3]}' THEN order_id END) AS period_1_orders,
           COUNT(DISTINCT CASE WHEN order_date >= '{second[2]}' AND order_date < '{second[3]}' THEN order_id END) AS period_2_orders
    FROM per_order GROUP BY COALESCE(channel, 'unknown')"""
    total_select = f"""SELECT 'total', 'All delivered orders',
           ROUND(SUM(CASE WHEN order_date >= '{first[2]}' AND order_date < '{first[3]}' THEN order_sales ELSE 0 END), 2),
           ROUND(SUM(CASE WHEN order_date >= '{second[2]}' AND order_date < '{second[3]}' THEN order_sales ELSE 0 END), 2),
           COUNT(DISTINCT CASE WHEN order_date >= '{first[2]}' AND order_date < '{first[3]}' THEN order_id END),
           COUNT(DISTINCT CASE WHEN order_date >= '{second[2]}' AND order_date < '{second[3]}' THEN order_id END)
    FROM per_order"""
    return f"""WITH per_order AS (
    SELECT o.order_id, o.order_date, o.channel,
           COALESCE(SUM(oi.line_net_sales), 0.0) AS order_sales
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.status = 'delivered'
      AND o.order_date >= '{start}' AND o.order_date < '{end}'
    GROUP BY o.order_id, o.order_date, o.channel
), base_rows AS (
    {channel_select}
    UNION ALL
    {total_select}
)
SELECT dimension_type, dimension_value,
       '{first[1]}' AS period_1_label, '{second[1]}' AS period_2_label,
       period_1_sales, period_2_sales,
       ROUND(period_1_sales - period_2_sales, 2) AS change_vs_period_2_usd,
       ROUND(100.0 * (period_1_sales - period_2_sales) / NULLIF(period_2_sales, 0), 2) AS change_vs_period_2_pct,
       period_1_orders, period_2_orders
FROM base_rows
ORDER BY CASE dimension_type WHEN 'total' THEN 0 ELSE 1 END, dimension_value"""


def _periods_cte(periods: list[tuple[int, str, str, str]]) -> str:
    rows = []
    for number, label, start, end in periods:
        month_key = start[:7]
        rows.append(
            f"({number}, '{label}', '{start}', '{end}', '{month_key}')"
        )
    return (
        "periods(period_number, period_label, start_date, end_date, month_key) AS (VALUES\n    "
        + ",\n    ".join(rows)
        + "\n)"
    )


def _has_schema(schemas: list[dict[str, Any]], required: dict[str, set[str]]) -> bool:
    by_table = {item.get("table"): set(item.get("columns", [])) for item in schemas}
    return all(table in by_table and columns.issubset(by_table[table]) for table, columns in required.items())


def _overall_sales_sql(periods: list[tuple[int, str, str, str]]) -> str:
    return f"""WITH {_periods_cte(periods)},
per_order AS (
    SELECT o.order_id, o.order_date, COALESCE(SUM(oi.line_net_sales), 0.0) AS order_sales
    FROM orders AS o
    LEFT JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.status = 'delivered'
      AND o.order_date >= (SELECT MIN(start_date) FROM periods)
      AND o.order_date < (SELECT MAX(end_date) FROM periods)
    GROUP BY o.order_id, o.order_date
),
period_totals AS (
    SELECT p.period_number, p.period_label,
           COUNT(o.order_id) AS order_count,
           ROUND(COALESCE(SUM(o.order_sales), 0.0), 2) AS sales,
           CASE WHEN COUNT(o.order_id) = 0 THEN NULL
                ELSE SUM(o.order_sales) * 1.0 / COUNT(o.order_id) END AS sales_per_order
    FROM periods AS p
    LEFT JOIN per_order AS o
      ON o.order_date >= p.start_date AND o.order_date < p.end_date
    GROUP BY p.period_number, p.period_label
),
pairs AS (
    SELECT 1 AS comparison_number, current.period_label AS current_period,
           prior.period_label AS prior_period, current.sales AS current_sales,
           prior.sales AS prior_sales, current.order_count AS current_order_count,
           prior.order_count AS prior_order_count,
           current.sales_per_order AS current_sales_per_order,
           prior.sales_per_order AS prior_sales_per_order
    FROM period_totals AS current
    JOIN period_totals AS prior ON prior.period_number = 2
    WHERE current.period_number = 1
    UNION ALL
    SELECT 2, current.period_label, prior.period_label, current.sales,
           prior.sales, current.order_count, prior.order_count,
           current.sales_per_order, prior.sales_per_order
    FROM period_totals AS current
    JOIN period_totals AS prior ON prior.period_number = 3
    WHERE current.period_number = 1
)
SELECT current_period, prior_period,
       current_sales, prior_sales,
       current_order_count, prior_order_count,
       current_sales_per_order, prior_sales_per_order,
       ROUND(current_sales - prior_sales, 2) AS change_usd,
       ROUND(100.0 * (current_sales - prior_sales) / NULLIF(prior_sales, 0), 2) AS change_pct,
       CASE WHEN prior_sales_per_order IS NULL THEN NULL
            ELSE ROUND((current_order_count - prior_order_count) * prior_sales_per_order, 2) END
           AS order_count_effect_usd,
       CASE WHEN current_sales_per_order IS NULL OR prior_sales_per_order IS NULL THEN NULL
            ELSE ROUND((current_sales_per_order - prior_sales_per_order) * current_order_count, 2) END
           AS sales_per_order_effect_usd
FROM pairs
ORDER BY comparison_number"""


def _segment_sales_sql(periods: list[tuple[int, str, str, str]]) -> str:
    return f"""WITH {_periods_cte(periods)},
order_values AS (
    SELECT o.order_id, o.order_date, o.region, o.channel, o.customer_id,
           CASE WHEN c.first_order_date IS NULL THEN 'unknown'
                WHEN strftime('%Y-%m', o.order_date) = strftime('%Y-%m', c.first_order_date)
                     THEN 'new'
                ELSE 'returning' END AS customer_type,
           COALESCE(SUM(oi.line_net_sales), 0.0) AS order_sales
    FROM orders AS o
    LEFT JOIN customers AS c ON c.customer_id = o.customer_id
    LEFT JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.status = 'delivered'
      AND o.order_date >= (SELECT MIN(start_date) FROM periods)
      AND o.order_date < (SELECT MAX(end_date) FROM periods)
    GROUP BY o.order_id, o.order_date, o.region, o.channel, o.customer_id, c.first_order_date
),
category_order_values AS (
    SELECT o.order_id, o.order_date, p.category,
           SUM(oi.line_net_sales) AS category_sales
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    JOIN products AS p ON p.product_id = oi.product_id
    WHERE o.status = 'delivered'
      AND o.order_date >= (SELECT MIN(start_date) FROM periods)
      AND o.order_date < (SELECT MAX(end_date) FROM periods)
    GROUP BY o.order_id, o.order_date, p.category
),
segment_rows AS (
    SELECT 'total' AS dimension_type, 'All delivered orders' AS dimension_value,
           strftime('%Y-%m', order_date) AS month_key, COUNT(*) AS order_count,
           SUM(order_sales) AS sales
    FROM order_values GROUP BY strftime('%Y-%m', order_date)
    UNION ALL
    SELECT 'region', COALESCE(region, 'unknown'), strftime('%Y-%m', order_date),
           COUNT(*), SUM(order_sales)
    FROM order_values GROUP BY COALESCE(region, 'unknown'), strftime('%Y-%m', order_date)
    UNION ALL
    SELECT 'sales_channel', COALESCE(channel, 'unknown'), strftime('%Y-%m', order_date),
           COUNT(*), SUM(order_sales)
    FROM order_values GROUP BY COALESCE(channel, 'unknown'), strftime('%Y-%m', order_date)
    UNION ALL
    SELECT 'new_vs_returning', customer_type, strftime('%Y-%m', order_date),
           COUNT(*), SUM(order_sales)
    FROM order_values GROUP BY customer_type, strftime('%Y-%m', order_date)
    UNION ALL
    SELECT 'product_category', COALESCE(category, 'unknown'), strftime('%Y-%m', order_date),
           COUNT(*), SUM(category_sales)
    FROM category_order_values GROUP BY COALESCE(category, 'unknown'), strftime('%Y-%m', order_date)
),
monthly AS (
    SELECT dimension_type, dimension_value,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 1) THEN sales ELSE 0 END) AS period_1_sales,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 2) THEN sales ELSE 0 END) AS period_2_sales,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 3) THEN sales ELSE 0 END) AS period_3_sales,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 1) THEN order_count ELSE 0 END) AS period_1_orders,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 2) THEN order_count ELSE 0 END) AS period_2_orders,
           SUM(CASE WHEN month_key = (SELECT month_key FROM periods WHERE period_number = 3) THEN order_count ELSE 0 END) AS period_3_orders
    FROM segment_rows
    GROUP BY dimension_type, dimension_value
),
changes AS (
    SELECT dimension_type, dimension_value,
           ROUND(period_1_sales, 2) AS period_1_sales,
           ROUND(period_2_sales, 2) AS period_2_sales,
           ROUND(period_3_sales, 2) AS period_3_sales,
           period_1_orders, period_2_orders, period_3_orders,
           ROUND(period_1_sales - period_2_sales, 2) AS change_vs_period_2_usd,
           ROUND(100.0 * (period_1_sales - period_2_sales) / NULLIF(period_2_sales, 0), 2) AS change_vs_period_2_pct,
           ROUND(period_1_sales - period_3_sales, 2) AS change_vs_period_3_usd,
           ROUND(100.0 * (period_1_sales - period_3_sales) / NULLIF(period_3_sales, 0), 2) AS change_vs_period_3_pct
    FROM monthly
)
SELECT changes.*,
       CASE WHEN dimension_type <> 'total' AND change_vs_period_2_usd < 0 THEN ROW_NUMBER() OVER (
           PARTITION BY dimension_type ORDER BY change_vs_period_2_usd ASC, dimension_value ASC
       ) END AS decline_rank_vs_period_2,
       CASE WHEN dimension_type <> 'total' AND change_vs_period_3_usd < 0 THEN ROW_NUMBER() OVER (
           PARTITION BY dimension_type ORDER BY change_vs_period_3_usd ASC, dimension_value ASC
       ) END AS decline_rank_vs_period_3
FROM changes
ORDER BY dimension_type, change_vs_period_2_usd ASC"""


def _marketing_sql(periods: list[tuple[int, str, str, str]]) -> str:
    month_literals = ", ".join("'" + start[:7] + "'" for _, _, start, _ in periods)
    return f"""SELECT month AS period,
       sales_channel,
       campaign_type,
       COUNT(DISTINCT campaign_id) AS campaign_count,
       ROUND(SUM(spend), 2) AS paid_search_spend_usd,
       SUM(impressions) AS impressions,
       SUM(clicks) AS clicks,
       ROUND(100.0 * SUM(clicks) / NULLIF(SUM(impressions), 0), 2) AS click_through_rate_pct,
       SUM(attributed_orders) AS attributed_orders
FROM marketing_campaigns
WHERE substr(month, 1, 7) IN ({month_literals})
  AND lower(campaign_type) LIKE '%paid%search%'
GROUP BY month, sales_channel, campaign_type
ORDER BY month, sales_channel"""


def _review_sql(periods: list[tuple[int, str, str, str]]) -> str:
    return f"""WITH {_periods_cte(periods)},
review_rows AS (
    SELECT p.period_number, p.period_label, product.category, review.rating
    FROM periods AS p
    JOIN orders AS o
      ON o.order_date >= p.start_date AND o.order_date < p.end_date
    JOIN reviews AS review ON review.order_id = o.order_id
    JOIN products AS product ON product.product_id = review.product_id
    WHERE o.status = 'delivered'
)
SELECT period_number, period_label, COALESCE(category, 'unknown') AS product_category,
       COUNT(*) AS review_count,
       ROUND(AVG(rating), 2) AS average_rating,
       SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
       ROUND(100.0 * SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS low_rating_pct,
       SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS high_rating_count,
       ROUND(100.0 * SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS high_rating_pct
FROM review_rows
GROUP BY period_number, period_label, COALESCE(category, 'unknown')
ORDER BY period_number, product_category"""


def _product_performance_sql(periods: list[tuple[int, str, str, str]]) -> str:
    """Product-grain performance with full period grids for safe comparisons."""
    return f"""WITH {_periods_cte(periods)},
product_dim AS (
    SELECT product_id, COALESCE(product_name, 'unknown') AS product_name,
           COALESCE(category, 'unknown') AS category
    FROM products
),
period_lines AS (
    SELECT p.period_number, p.period_label, o.order_id, oi.product_id,
           COALESCE(oi.quantity, 0) AS quantity,
           COALESCE(oi.unit_price, 0.0) AS unit_price,
           COALESCE(oi.discount_amount, 0.0) AS discount_amount,
           COALESCE(oi.unit_cost, 0.0) AS unit_cost,
           COALESCE(oi.line_net_sales, 0.0) AS net_sales,
           COALESCE(oi.line_gross_profit, 0.0) AS gross_profit
    FROM periods AS p
    JOIN orders AS o
      ON o.order_date >= p.start_date AND o.order_date < p.end_date
     AND o.status = 'delivered'
    JOIN order_items AS oi ON oi.order_id = o.order_id
),
sku_metrics AS (
    SELECT period_number, product_id, COUNT(DISTINCT order_id) AS delivered_orders,
           SUM(quantity) AS units_sold,
           ROUND(SUM(quantity * unit_price), 2) AS gross_product_sales_usd,
           ROUND(SUM(discount_amount), 2) AS line_discounts_usd,
           ROUND(SUM(net_sales), 2) AS net_product_sales_usd,
           ROUND(SUM(quantity * unit_cost), 2) AS cogs_usd,
           ROUND(SUM(gross_profit), 2) AS gross_profit_usd
    FROM period_lines GROUP BY period_number, product_id
),
category_metrics AS (
    SELECT l.period_number, d.category, COUNT(DISTINCT l.order_id) AS delivered_orders,
           SUM(l.quantity) AS units_sold,
           ROUND(SUM(l.quantity * l.unit_price), 2) AS gross_product_sales_usd,
           ROUND(SUM(l.discount_amount), 2) AS line_discounts_usd,
           ROUND(SUM(l.net_sales), 2) AS net_product_sales_usd,
           ROUND(SUM(l.quantity * l.unit_cost), 2) AS cogs_usd,
           ROUND(SUM(l.gross_profit), 2) AS gross_profit_usd
    FROM period_lines AS l JOIN product_dim AS d ON d.product_id = l.product_id
    GROUP BY l.period_number, d.category
),
total_metrics AS (
    SELECT period_number, COUNT(DISTINCT order_id) AS delivered_orders,
           SUM(quantity) AS units_sold,
           ROUND(SUM(quantity * unit_price), 2) AS gross_product_sales_usd,
           ROUND(SUM(discount_amount), 2) AS line_discounts_usd,
           ROUND(SUM(net_sales), 2) AS net_product_sales_usd,
           ROUND(SUM(quantity * unit_cost), 2) AS cogs_usd,
           ROUND(SUM(gross_profit), 2) AS gross_profit_usd
    FROM period_lines GROUP BY period_number
),
sku_rows AS (
    SELECT p.period_number, p.period_label, 'product' AS dimension_type,
           CAST(d.product_id AS TEXT) || ' - ' || d.product_name AS dimension_value,
           COALESCE(m.delivered_orders, 0) AS delivered_orders,
           COALESCE(m.units_sold, 0) AS units_sold,
           COALESCE(m.gross_product_sales_usd, 0.0) AS gross_product_sales_usd,
           COALESCE(m.line_discounts_usd, 0.0) AS line_discounts_usd,
           COALESCE(m.net_product_sales_usd, 0.0) AS net_product_sales_usd,
           COALESCE(m.cogs_usd, 0.0) AS cogs_usd,
           COALESCE(m.gross_profit_usd, 0.0) AS gross_profit_usd
    FROM periods AS p CROSS JOIN product_dim AS d
    LEFT JOIN sku_metrics AS m ON m.period_number = p.period_number AND m.product_id = d.product_id
),
category_rows AS (
    SELECT p.period_number, p.period_label, 'product_category' AS dimension_type,
           d.category AS dimension_value,
           COALESCE(m.delivered_orders, 0) AS delivered_orders,
           COALESCE(m.units_sold, 0) AS units_sold,
           COALESCE(m.gross_product_sales_usd, 0.0) AS gross_product_sales_usd,
           COALESCE(m.line_discounts_usd, 0.0) AS line_discounts_usd,
           COALESCE(m.net_product_sales_usd, 0.0) AS net_product_sales_usd,
           COALESCE(m.cogs_usd, 0.0) AS cogs_usd,
           COALESCE(m.gross_profit_usd, 0.0) AS gross_profit_usd
    FROM periods AS p CROSS JOIN (SELECT DISTINCT category FROM product_dim) AS d
    LEFT JOIN category_metrics AS m ON m.period_number = p.period_number AND m.category = d.category
),
total_rows AS (
    SELECT p.period_number, p.period_label, 'total' AS dimension_type,
           'All products' AS dimension_value,
           COALESCE(m.delivered_orders, 0) AS delivered_orders,
           COALESCE(m.units_sold, 0) AS units_sold,
           COALESCE(m.gross_product_sales_usd, 0.0) AS gross_product_sales_usd,
           COALESCE(m.line_discounts_usd, 0.0) AS line_discounts_usd,
           COALESCE(m.net_product_sales_usd, 0.0) AS net_product_sales_usd,
           COALESCE(m.cogs_usd, 0.0) AS cogs_usd,
           COALESCE(m.gross_profit_usd, 0.0) AS gross_profit_usd
    FROM periods AS p LEFT JOIN total_metrics AS m ON m.period_number = p.period_number
),
all_rows AS (
    SELECT * FROM total_rows
    UNION ALL SELECT * FROM category_rows
    UNION ALL SELECT * FROM sku_rows
)
SELECT period_number, period_label, dimension_type, dimension_value,
       delivered_orders, units_sold, gross_product_sales_usd, line_discounts_usd,
       net_product_sales_usd, cogs_usd, gross_profit_usd,
       CASE WHEN net_product_sales_usd = 0 THEN NULL
            ELSE ROUND(100.0 * gross_profit_usd / net_product_sales_usd, 2) END AS gross_margin_pct,
       CASE WHEN gross_product_sales_usd = 0 THEN NULL
            ELSE ROUND(100.0 * line_discounts_usd / gross_product_sales_usd, 2) END AS discount_rate_pct,
       CASE WHEN units_sold = 0 THEN NULL
            ELSE ROUND(1.0 * gross_product_sales_usd / units_sold, 2) END AS average_unit_price_usd
FROM all_rows
ORDER BY CASE dimension_type WHEN 'total' THEN 0 WHEN 'product_category' THEN 1 ELSE 2 END,
         period_number, dimension_value"""


def _operations_sql(
    periods: list[tuple[int, str, str, str]],
    allowed: set[str],
    schemas: list[dict[str, Any]],
    request_text: str = "",
    question: str = "",
) -> tuple[str | None, tuple[str, ...]]:
    sections: list[str] = []
    limitations: list[str] = []
    period_cte = _periods_cte(periods)
    focus = request_text.casefold()
    question_text = question.casefold()
    wants_inventory = bool(re.search(r"inventory|stock|warehouse", focus))
    wants_delivery = bool(re.search(r"deliver|fulfillment|carrier|on.?time|late", focus))
    wants_returns = bool(re.search(r"return|refund", focus))
    focused = wants_inventory or wants_delivery or wants_returns
    if not focused:
        wants_inventory = wants_delivery = wants_returns = True

    if wants_inventory and {"inventory_snapshots", "products"}.issubset(allowed) and _has_schema(schemas, {
        "inventory_snapshots": {"snapshot_id", "product_id", "snapshot_date", "on_hand_units"},
        "products": {"product_id", "category"},
    }):
        sections.append(f"""SELECT 'inventory_snapshot_availability' AS metric_type,
       'product_category' AS dimension_type, COALESCE(product.category, 'unknown') AS dimension_value,
       p.period_label,
       SUM(CASE WHEN inventory.on_hand_units > 0 THEN 1 ELSE 0 END) AS numerator,
       COUNT(inventory.snapshot_id) AS denominator,
       ROUND(100.0 * SUM(CASE WHEN inventory.on_hand_units > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(inventory.snapshot_id), 0), 2) AS rate_pct,
       ROUND(AVG(inventory.on_hand_units), 2) AS supporting_value,
       'average_on_hand_units_per_snapshot' AS supporting_unit
FROM periods AS p
JOIN inventory_snapshots AS inventory ON substr(inventory.snapshot_date, 1, 7) = p.month_key
JOIN products AS product ON product.product_id = inventory.product_id
GROUP BY p.period_number, p.period_label, product.category""")
    elif wants_inventory:
        limitations.append("Inventory availability was not queried because the approved plan lacked inventory snapshots or product-category mappings.")

    schemas_by_table = {item.get("table"): set(item.get("columns", [])) for item in schemas}
    delivery_by_carrier = (
        "carrier" in question_text and "carrier" in schemas_by_table.get("fulfillment", set())
    )
    if wants_delivery and {"fulfillment", "orders"}.issubset(allowed) and _has_schema(schemas, {
        "fulfillment": {"order_id", "ship_date", "delivered_date", "promised_delivery_date"},
        "orders": {"order_id", "order_date", "status"},
    }) and (delivery_by_carrier or "region" in schemas_by_table.get("orders", set())):
        dimension_type = "carrier" if delivery_by_carrier else "region"
        dimension_value = "COALESCE(CAST(f.carrier AS TEXT), 'unknown')" if delivery_by_carrier else "COALESCE(CAST(o.region AS TEXT), 'unknown')"
        dimension_source = "f.carrier" if delivery_by_carrier else "o.region"
        on_time = bool(re.search(r"on.?time", focus + " " + question_text))
        delivery_condition = "f.delivered_date <= f.promised_delivery_date" if on_time else "f.delivered_date > f.promised_delivery_date"
        metric = "on_time_delivery_rate" if on_time else "late_delivery_rate"
        numerator = f"COUNT(DISTINCT CASE WHEN {delivery_condition} THEN o.order_id END)"
        support = (
            "COUNT(DISTINCT CASE WHEN f.delivered_date > f.promised_delivery_date THEN o.order_id END)"
            if on_time else "NULL"
        )
        support_unit = "late_delivered_orders" if on_time else "not_applicable"
        sections.append(f"""SELECT '{metric}' AS metric_type,
       '{dimension_type}' AS dimension_type, {dimension_value} AS dimension_value,
       p.period_label,
       {numerator} AS numerator,
       COUNT(DISTINCT o.order_id) AS denominator,
       ROUND(100.0 * {numerator} / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS rate_pct,
       {support} AS supporting_value,
       '{support_unit}' AS supporting_unit
FROM periods AS p
JOIN fulfillment AS f ON f.ship_date >= p.start_date AND f.ship_date < p.end_date
JOIN orders AS o ON f.order_id = o.order_id AND o.status = 'delivered'
WHERE f.delivered_date IS NOT NULL AND f.promised_delivery_date IS NOT NULL
GROUP BY p.period_number, p.period_label, {dimension_source}""")
    elif wants_delivery:
        limitations.append("Late-delivery rates were not queried because the approved plan lacked fulfillment records or order dates.")

    return_core_available = {"returns", "order_items", "orders"}.issubset(allowed) and _has_schema(schemas, {
        "returns": {"order_item_id", "quantity_returned", "refund_amount", "reason"},
        "order_items": {"order_id", "order_item_id", "quantity"},
        "orders": {"order_id", "order_date", "status"},
    })
    group_returns_by_reason = bool(re.search(r"reason", question_text + " " + focus)) or "products" not in allowed
    category_available = "products" in allowed and _has_schema(schemas, {
        "products": {"product_id", "category"},
        "order_items": {"product_id"},
    })
    if wants_returns and return_core_available and group_returns_by_reason:
        sections.append(f"""SELECT 'return_rate_by_units' AS metric_type,
       'total' AS dimension_type, 'All delivered orders' AS dimension_value,
       p.period_label,
       COALESCE((SELECT SUM(ret.quantity_returned)
                 FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
                 JOIN returns AS ret ON ret.order_item_id = oi.order_item_id
                 WHERE o.status = 'delivered' AND o.order_date >= p.start_date AND o.order_date < p.end_date), 0) AS numerator,
       (SELECT COALESCE(SUM(oi.quantity), 0)
        FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
        WHERE o.status = 'delivered' AND o.order_date >= p.start_date AND o.order_date < p.end_date) AS denominator,
       ROUND(100.0 * COALESCE((SELECT SUM(ret.quantity_returned)
                 FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
                 JOIN returns AS ret ON ret.order_item_id = oi.order_item_id
                 WHERE o.status = 'delivered' AND o.order_date >= p.start_date AND o.order_date < p.end_date), 0)
          / NULLIF((SELECT SUM(oi.quantity)
        FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
        WHERE o.status = 'delivered' AND o.order_date >= p.start_date AND o.order_date < p.end_date), 0), 2) AS rate_pct,
       COALESCE((SELECT SUM(ret.refund_amount)
                 FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id
                 JOIN returns AS ret ON ret.order_item_id = oi.order_item_id
                 WHERE o.status = 'delivered' AND o.order_date >= p.start_date AND o.order_date < p.end_date), 0.0) AS supporting_value,
       'observed_refund_amount_usd' AS supporting_unit
FROM periods AS p
UNION ALL
SELECT 'return_rate_by_units' AS metric_type,
       'return_reason' AS dimension_type, COALESCE(CAST(ret.reason AS TEXT), 'unknown') AS dimension_value,
       p.period_label,
       SUM(ret.quantity_returned) AS numerator,
       (SELECT COALESCE(SUM(oi.quantity), 0)
        FROM orders AS od JOIN order_items AS oi ON oi.order_id = od.order_id
        WHERE od.status = 'delivered' AND od.order_date >= p.start_date AND od.order_date < p.end_date) AS denominator,
       ROUND(100.0 * SUM(ret.quantity_returned) / NULLIF((SELECT SUM(oi.quantity)
        FROM orders AS od JOIN order_items AS oi ON oi.order_id = od.order_id
        WHERE od.status = 'delivered' AND od.order_date >= p.start_date AND od.order_date < p.end_date), 0), 2) AS rate_pct,
       SUM(ret.refund_amount) AS supporting_value,
       'observed_refund_amount_usd' AS supporting_unit
FROM periods AS p
JOIN orders AS o ON o.order_date >= p.start_date AND o.order_date < p.end_date AND o.status = 'delivered'
JOIN order_items AS oi ON oi.order_id = o.order_id
JOIN returns AS ret ON ret.order_item_id = oi.order_item_id
GROUP BY p.period_number, p.period_label, ret.reason""")
    elif wants_returns and return_core_available and category_available:
        sections.append(f"""SELECT 'return_rate_by_units' AS metric_type,
       'product_category' AS dimension_type, COALESCE(s.category, 'unknown') AS dimension_value,
       s.period_label, COALESCE(r.returned_units, 0) AS numerator, s.sold_units AS denominator,
       ROUND(100.0 * COALESCE(r.returned_units, 0) / NULLIF(s.sold_units, 0), 2) AS rate_pct,
       COALESCE(r.refund_amount_usd, 0.0) AS supporting_value,
       'observed_refund_amount_usd' AS supporting_unit
FROM (
    SELECT p.period_number, p.period_label, product.category,
           SUM(oi.quantity) AS sold_units
    FROM periods AS p
    JOIN orders AS o ON o.order_date >= p.start_date AND o.order_date < p.end_date AND o.status = 'delivered'
    JOIN order_items AS oi ON oi.order_id = o.order_id
    JOIN products AS product ON product.product_id = oi.product_id
    GROUP BY p.period_number, p.period_label, product.category
) AS s
LEFT JOIN (
    SELECT p.period_number, product.category,
           SUM(ret.quantity_returned) AS returned_units,
           ROUND(SUM(ret.refund_amount), 2) AS refund_amount_usd
    FROM periods AS p
    JOIN orders AS o ON o.order_date >= p.start_date AND o.order_date < p.end_date AND o.status = 'delivered'
    JOIN order_items AS oi ON oi.order_id = o.order_id
    JOIN products AS product ON product.product_id = oi.product_id
    JOIN returns AS ret ON ret.order_item_id = oi.order_item_id
    GROUP BY p.period_number, product.category
) AS r ON r.period_number = s.period_number AND r.category = s.category""")
    elif wants_returns:
        limitations.append("Return-unit rates were not queried because the approved plan lacked returns, order items, orders, or product mappings.")

    if not sections:
        return None, tuple(limitations)
    sql = "WITH " + period_cte + "\n" + "\nUNION ALL\n".join(sections) + "\nORDER BY metric_type, dimension_type, dimension_value, period_label"
    if wants_inventory:
        limitations.append("Inventory readings are snapshots, not continuous stock histories; September 2025 has extra snapshots for selected SKUs, so availability is a partial indicator.")
    if wants_returns:
        limitations.append("Return rates use returned units recorded for delivered-order cohorts in the available extract; different return-maturation windows may affect period comparisons.")
        if group_returns_by_reason:
            limitations.append("For each reason, its numerator is divided by all delivered units in that cohort; reason-specific rates therefore contribute to the overall return-unit rate.")
    if wants_delivery:
        limitations.append("Delivery rates include only fulfilled delivered orders with both promised and actual dates.")
    return sql, tuple(limitations)


def _customer_lifecycle_sql(periods: list[tuple[int, str, str, str]]) -> str:
    """Customer-grain lifecycle metrics for three explicit calendar months."""
    return f"""WITH {_periods_cte(periods)},
order_values AS (
    SELECT p.period_number, p.period_label, p.month_key,
           o.order_id, o.customer_id,
           CASE
             WHEN NULLIF(TRIM(c.first_order_date), '') IS NULL THEN 'unknown'
             WHEN strftime('%Y-%m', c.first_order_date) = p.month_key THEN 'new'
             WHEN c.first_order_date < p.start_date THEN 'returning'
             ELSE 'unknown'
           END AS customer_type,
           SUM(oi.line_net_sales) AS order_net_product_sales
    FROM periods AS p
    JOIN orders AS o
      ON o.order_date >= p.start_date AND o.order_date < p.end_date
    LEFT JOIN customers AS c ON c.customer_id = o.customer_id
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.status = 'delivered'
    GROUP BY p.period_number, p.period_label, p.month_key,
             o.order_id, o.customer_id, c.first_order_date
),
per_customer AS (
    SELECT period_number, period_label, customer_type, customer_id,
           COUNT(DISTINCT order_id) AS delivered_orders,
           ROUND(SUM(order_net_product_sales), 2) AS net_product_sales
    FROM order_values
    GROUP BY period_number, period_label, customer_type, customer_id
),
customer_types(customer_type) AS (
    VALUES ('all'), ('new'), ('returning'), ('unknown')
),
aggregates AS (
    SELECT period_number, 'all' AS customer_type,
           COUNT(*) AS active_customers,
           SUM(delivered_orders) AS delivered_orders,
           ROUND(SUM(net_product_sales), 2) AS net_product_sales,
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers
    FROM per_customer
    GROUP BY period_number
    UNION ALL
    SELECT period_number, customer_type,
           COUNT(*) AS active_customers,
           SUM(delivered_orders) AS delivered_orders,
           ROUND(SUM(net_product_sales), 2) AS net_product_sales,
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers
    FROM per_customer
    GROUP BY period_number, customer_type
)
SELECT p.period_number, p.period_label, t.customer_type,
       COALESCE(a.active_customers, 0) AS active_customers,
       COALESCE(a.delivered_orders, 0) AS delivered_orders,
       COALESCE(a.net_product_sales, 0.0) AS net_product_sales,
       COALESCE(a.repeat_customers, 0) AS repeat_customers,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(100.0 * a.repeat_customers / a.active_customers, 2) END AS repeat_customer_pct,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(1.0 * a.delivered_orders / a.active_customers, 4) END
            AS purchase_frequency_orders_per_customer,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(1.0 * a.net_product_sales / a.active_customers, 2) END
            AS net_sales_per_active_customer
FROM periods AS p
CROSS JOIN customer_types AS t
LEFT JOIN aggregates AS a
  ON a.period_number = p.period_number AND a.customer_type = t.customer_type
ORDER BY p.period_number,
         CASE t.customer_type WHEN 'all' THEN 0 WHEN 'new' THEN 1
              WHEN 'returning' THEN 2 ELSE 3 END"""


def _customer_cohort_retention_sql(periods: list[tuple[int, str, str, str]]) -> str:
    """Monthly delivered-purchase retention for three recorded first-order cohorts."""
    return f"""WITH {_periods_cte(periods)},
offsets(month_number) AS (VALUES (0), (1), (2), (3)),
cohort_members AS (
    SELECT p.period_number AS cohort_number, p.period_label AS cohort_label,
           p.start_date AS cohort_start, c.customer_id
    FROM periods AS p
    JOIN customers AS c
      ON c.first_order_date >= p.start_date AND c.first_order_date < p.end_date
    WHERE c.customer_id IS NOT NULL AND TRIM(CAST(c.customer_id AS TEXT)) <> ''
),
cohort_sizes AS (
    SELECT p.period_number AS cohort_number, p.period_label AS cohort_label,
           p.start_date AS cohort_start, COUNT(DISTINCT cm.customer_id) AS cohort_size
    FROM periods AS p
    LEFT JOIN cohort_members AS cm ON cm.cohort_number = p.period_number
    GROUP BY p.period_number, p.period_label, p.start_date
),
coverage AS (
    SELECT MAX(order_date) AS observed_through FROM orders
),
cohort_grid AS (
    SELECT c.cohort_number, c.cohort_label, c.cohort_size,
           offsets.month_number,
           date(c.cohort_start, '+' || offsets.month_number || ' months') AS activity_month_start,
           date(c.cohort_start, '+' || (offsets.month_number + 1) || ' months') AS activity_month_end,
           coverage.observed_through,
           CASE WHEN coverage.observed_through IS NOT NULL
                  AND date(c.cohort_start, '+' || (offsets.month_number + 1) || ' months')
                      <= date(coverage.observed_through, 'start of month', '+1 month')
                THEN 1 ELSE 0 END AS observation_available
    FROM cohort_sizes AS c
    CROSS JOIN offsets
    CROSS JOIN coverage
)
SELECT cohort_number, cohort_label, cohort_size, month_number,
       strftime('%Y-%m', activity_month_start) AS activity_month,
       observed_through, observation_available,
       CASE WHEN observation_available = 0 THEN NULL ELSE (
           SELECT COUNT(DISTINCT cm.customer_id)
           FROM cohort_members AS cm
           JOIN orders AS o ON o.customer_id = cm.customer_id
           WHERE cm.cohort_number = cohort_grid.cohort_number
             AND o.status = 'delivered'
             AND o.order_date >= cohort_grid.activity_month_start
             AND o.order_date < cohort_grid.activity_month_end
       ) END AS retained_customers,
       CASE WHEN observation_available = 0 OR cohort_size = 0 THEN NULL
            ELSE ROUND(100.0 * (
                SELECT COUNT(DISTINCT cm.customer_id)
                FROM cohort_members AS cm
                JOIN orders AS o ON o.customer_id = cm.customer_id
                WHERE cm.cohort_number = cohort_grid.cohort_number
                  AND o.status = 'delivered'
                  AND o.order_date >= cohort_grid.activity_month_start
                  AND o.order_date < cohort_grid.activity_month_end
            ) / cohort_size, 2) END AS retention_pct
FROM cohort_grid
ORDER BY cohort_number, month_number"""


def _customer_segment_sql(periods: list[tuple[int, str, str, str]]) -> str:
    """Aggregate lifecycle measures by recorded customer segment dimensions."""
    return f"""WITH {_periods_cte(periods)},
order_values AS (
    SELECT p.period_number, p.period_label, o.order_id, o.customer_id,
           COALESCE(NULLIF(TRIM(c.segment), ''), 'unknown') AS customer_segment,
           COALESCE(NULLIF(TRIM(c.region), ''), 'unknown') AS customer_region,
           COALESCE(NULLIF(TRIM(c.acquisition_channel), ''), 'unknown') AS acquisition_channel,
           SUM(oi.line_net_sales) AS order_net_product_sales
    FROM periods AS p
    JOIN orders AS o
      ON o.order_date >= p.start_date AND o.order_date < p.end_date
    LEFT JOIN customers AS c ON c.customer_id = o.customer_id
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.status = 'delivered'
    GROUP BY p.period_number, p.period_label, o.order_id, o.customer_id,
             customer_segment, customer_region, acquisition_channel
),
per_customer AS (
    SELECT period_number, period_label, customer_id, customer_segment,
           customer_region, acquisition_channel,
           COUNT(DISTINCT order_id) AS delivered_orders,
           SUM(order_net_product_sales) AS net_product_sales
    FROM order_values
    GROUP BY period_number, period_label, customer_id, customer_segment,
             customer_region, acquisition_channel
),
dimension_values AS (
    SELECT 'total' AS dimension_type, 'All customers' AS dimension_value
    UNION SELECT 'customer_segment', customer_segment FROM per_customer
    UNION SELECT 'customer_region', customer_region FROM per_customer
    UNION SELECT 'acquisition_channel', acquisition_channel FROM per_customer
),
aggregates AS (
    SELECT period_number, 'total' AS dimension_type, 'All customers' AS dimension_value,
           COUNT(DISTINCT customer_id) AS active_customers,
           SUM(delivered_orders) AS delivered_orders,
           ROUND(SUM(net_product_sales), 2) AS net_product_sales,
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers
    FROM per_customer GROUP BY period_number
    UNION ALL
    SELECT period_number, 'customer_segment', customer_segment,
           COUNT(DISTINCT customer_id), SUM(delivered_orders),
           ROUND(SUM(net_product_sales), 2),
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END)
    FROM per_customer GROUP BY period_number, customer_segment
    UNION ALL
    SELECT period_number, 'customer_region', customer_region,
           COUNT(DISTINCT customer_id), SUM(delivered_orders),
           ROUND(SUM(net_product_sales), 2),
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END)
    FROM per_customer GROUP BY period_number, customer_region
    UNION ALL
    SELECT period_number, 'acquisition_channel', acquisition_channel,
           COUNT(DISTINCT customer_id), SUM(delivered_orders),
           ROUND(SUM(net_product_sales), 2),
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END)
    FROM per_customer GROUP BY period_number, acquisition_channel
)
SELECT p.period_number, p.period_label, d.dimension_type, d.dimension_value,
       COALESCE(a.active_customers, 0) AS active_customers,
       COALESCE(a.delivered_orders, 0) AS delivered_orders,
       COALESCE(a.net_product_sales, 0.0) AS net_product_sales,
       COALESCE(a.repeat_customers, 0) AS repeat_customers,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(100.0 * a.repeat_customers / a.active_customers, 2) END AS repeat_customer_pct,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(1.0 * a.delivered_orders / a.active_customers, 4) END AS purchase_frequency_orders_per_customer,
       CASE WHEN COALESCE(a.active_customers, 0) = 0 THEN NULL
            ELSE ROUND(1.0 * a.net_product_sales / a.active_customers, 2) END AS net_sales_per_active_customer
FROM periods AS p
CROSS JOIN dimension_values AS d
LEFT JOIN aggregates AS a
  ON a.period_number = p.period_number
 AND a.dimension_type = d.dimension_type
 AND a.dimension_value = d.dimension_value
ORDER BY CASE d.dimension_type WHEN 'total' THEN 0 WHEN 'customer_segment' THEN 1
               WHEN 'customer_region' THEN 2 ELSE 3 END,
         d.dimension_value, p.period_number"""


def _customer_repeat_window_sql(start: str, end: str) -> str:
    """Customer-grain within-window repeat purchase rates by current segment."""
    return f"""WITH per_customer AS (
    SELECT c.customer_id,
           COALESCE(NULLIF(TRIM(c.segment), ''), 'unknown') AS segment,
           COUNT(DISTINCT o.order_id) AS delivered_orders
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    WHERE o.status = 'delivered'
      AND o.order_date >= '{start}' AND o.order_date < '{end}'
      AND c.customer_id IS NOT NULL
      AND TRIM(CAST(c.customer_id AS TEXT)) <> ''
    GROUP BY c.customer_id, COALESCE(NULLIF(TRIM(c.segment), ''), 'unknown')
), segment_rates AS (
    SELECT segment,
           COUNT(*) AS total_customers_with_delivered_orders,
           SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers,
           ROUND(1.0 * SUM(CASE WHEN delivered_orders >= 2 THEN 1 ELSE 0 END)
                 / NULLIF(COUNT(*), 0), 4) AS repeat_purchase_rate
    FROM per_customer
    GROUP BY segment
)
SELECT segment, total_customers_with_delivered_orders, repeat_customers,
       repeat_purchase_rate
FROM segment_rates
ORDER BY repeat_purchase_rate ASC, segment"""


def _required_for_kind(kind: str) -> dict[str, set[str]]:
    return {
        "sales_overall": {
            "orders": {"order_id", "order_date", "status"},
            "order_items": {"order_id", "line_net_sales"},
        },
        "sales_segments": {
            "orders": {"order_id", "order_date", "customer_id", "region", "channel", "status"},
            "order_items": {"order_id", "product_id", "line_net_sales"},
            "products": {"product_id", "category"},
            "customers": {"customer_id", "first_order_date"},
        },
        "marketing": {
            "marketing_campaigns": {"month", "campaign_id", "campaign_type", "sales_channel", "spend", "impressions", "clicks", "attributed_orders"},
        },
        "reviews": {
            "reviews": {"order_id", "product_id", "rating"},
            "orders": {"order_id", "order_date", "status"},
            "products": {"product_id", "category"},
        },
        "customer_lifecycle": {
            "customers": {"customer_id", "first_order_date"},
            "orders": {"order_id", "order_date", "customer_id", "status"},
            "order_items": {"order_id", "line_net_sales"},
        },
        "customer_segment": {
            "customers": {"customer_id", "segment", "region", "acquisition_channel"},
            "orders": {"order_id", "order_date", "customer_id", "status"},
            "order_items": {"order_id", "line_net_sales"},
        },
        "customer_cohort_retention": {
            "customers": {"customer_id", "first_order_date"},
            "orders": {"order_id", "order_date", "customer_id", "status"},
        },
    }.get(kind, {})


def build_demo_query(question: str, task: Any, allowed: set[str], schemas: list[dict[str, Any]]) -> DemoQuery | None:
    """Select one fixed SQL template for supported demo-company analysis tasks."""
    agent = str(getattr(task, "agent", ""))
    source_ids = set(getattr(task, "data_sources", []))
    objective = str(getattr(task, "objective", "")).lower()
    task_id = str(getattr(task, "task_id", "")).lower()
    routing_text = f"{task_id} {objective}"
    operation_text = re.sub(r"[_-]+", " ", routing_text)
    operation_request = bool(re.search(
        r"\binventory\b|\bstock\b|\bwarehouse\b|\bfulfillment\b|\bdeliver(?:y|ies|ed)?\b|\breturns?\b|\brefunds?\b|\bcarrier\b",
        operation_text,
    ))
    excluded_methods = re.search(
        r"\b(clean|cleaning|deduplicat|significance|significant|hypothesis|p[- ]?value|statistical test)\b",
        question,
        flags=re.IGNORECASE,
    )
    gross_profit_window = _explicit_half_year(question)
    if (
        agent == "data_analyst" and not excluded_methods and gross_profit_window
        and "gross profit" in question.casefold()
        and re.search(r"\bcategor(?:y|ies)\b", question, flags=re.IGNORECASE)
        and {"orders", "order_items", "products"}.issubset(source_ids)
        and _has_schema(schemas, {
            "orders": {"order_id", "order_date", "status"},
            "order_items": {"order_id", "product_id", "line_gross_profit"},
            "products": {"product_id", "category"},
        })
    ):
        start, end = gross_profit_window
        return DemoQuery(
            "demo_gross_profit_category_window_v1",
            _gross_profit_category_window_sql(start, end),
            (
                "Gross profit sums order_items.line_gross_profit for delivered orders in the explicitly requested half-year window.",
                "Gross profit is before returns, fulfillment, marketing, and overhead; zero-sales categories are included as zero.",
                "The category ranking is descriptive and does not establish why category gross profit differs.",
            ),
        )
    repeat_window = _explicit_half_year(question)
    if (
        agent == "customer_analyst" and not excluded_methods and repeat_window
        and re.search(r"\brepeat(?:[- ]purchase)?\b", routing_text)
        and re.search(r"\bsegment", routing_text + " " + question, flags=re.IGNORECASE)
        and {"customers", "orders"}.issubset(source_ids)
        and _has_schema(schemas, {
            "customers": {"customer_id", "segment"},
            "orders": {"order_id", "order_date", "customer_id", "status"},
        })
    ):
        start, end = repeat_window
        return DemoQuery(
            "demo_customer_repeat_window_v1",
            _customer_repeat_window_sql(start, end),
            (
                "Repeat customers have at least two distinct delivered orders in the explicitly requested window; the denominator is customers with at least one delivered order in that window.",
                "Customer segments use the current customers-table value, not a historical segment snapshot.",
                "This within-window repeat rate is descriptive and does not establish churn, lifetime returning status, or a cause.",
            ),
        )
    channel_periods = _explicit_named_periods(question, 2, 2)
    if (
        agent == "data_analyst" and not excluded_methods and channel_periods
        and re.search(r"\bnet\s+(?:product\s+)?sales\b", question, flags=re.IGNORECASE)
        and re.search(r"\b(?:sales\s+channel|channels?)\b", question, flags=re.IGNORECASE)
        and {"orders", "order_items"}.issubset(source_ids)
        and _has_schema(schemas, {
            "orders": {"order_id", "order_date", "channel", "status"},
            "order_items": {"order_id", "line_net_sales"},
        })
    ):
        return DemoQuery(
            "demo_sales_channel_two_period_v1",
            _sales_channel_two_period_sql(channel_periods),
            (
                "Delivered net product sales sum order_items.line_net_sales and exclude shipping and later refunds.",
                "Sales by channel are descriptive; they do not establish why channel results changed.",
            ),
        )
    if agent in {"product_operations_analyst", "data_analyst"} and operation_request and not excluded_methods:
        operation_periods = _explicit_named_periods(question, 2, 3)
        if operation_periods is not None:
            sql, limitations = _operations_sql(
                operation_periods, allowed, schemas, request_text=operation_text, question=question,
            )
            if sql is not None:
                return DemoQuery("demo_operations_period_comparison_v1", sql, limitations)

    periods = _explicit_months(question)
    if periods is None or excluded_methods:
        return None

    kind: str | None = None

    if agent == "product_operations_analyst":
        operation_terms = ("inventory", "stock", "warehouse", "fulfillment", "delivery", "return")
        product_terms = ("sku", "product sales", "product performance", "gross margin", "gross profit", "discount", "price")
        operation_request = any(term in objective for term in operation_terms)
        product_request = any(term in objective for term in product_terms)
        if product_request and not operation_request and {"orders", "order_items", "products"}.issubset(source_ids):
            required = {
                "orders": {"order_id", "order_date", "status"},
                "order_items": {"order_id", "product_id", "quantity", "unit_price", "discount_amount", "unit_cost", "line_net_sales", "line_gross_profit"},
                "products": {"product_id", "product_name", "category"},
            }
            if not _has_schema(schemas, required):
                return None
            return DemoQuery(
                "demo_product_performance_period_comparison_v1",
                _product_performance_sql(periods),
                (
                    "Product comparisons use delivered orders and order-time line values; shipping and later refunds are excluded.",
                    "Gross profit is before returns, fulfillment, marketing, and overhead. Product and category order counts can overlap.",
                    "Product-level results are descriptive and do not establish why performance changed.",
                ),
            )
        if source_ids & {"inventory_snapshots", "fulfillment", "returns"}:
            sql, limitations = _operations_sql(periods, allowed, schemas)
            if sql is None:
                return None
            return DemoQuery("demo_operations_period_comparison_v1", sql, limitations)

    if agent == "customer_analyst" and "reviews" in source_ids and any(word in objective for word in ("review", "sentiment", "rating")):
        kind = "reviews"
    elif agent == "customer_analyst" and {"customers", "orders"}.issubset(source_ids):
        if "cohort" in routing_text or "retention" in routing_text:
            kind = "customer_cohort_retention"
        elif {"customers", "orders", "order_items"}.issubset(source_ids) and any(word in routing_text for word in (
            "segment", "acquisition", "customer_region", "customer region",
        )):
            kind = "customer_segment"
        elif {"customers", "orders", "order_items"}.issubset(source_ids) and any(word in routing_text for word in (
            "customer", "lifecycle", "first-time", "returning", "repeat", "frequency",
        )):
            kind = "customer_lifecycle"
    elif agent == "data_analyst" and "marketing_campaigns" in source_ids and any(word in objective for word in ("paid-search", "paid search", "campaign", "marketing")):
        kind = "marketing"
    elif agent == "data_analyst" and {"orders", "order_items"}.issubset(source_ids):
        if any(word in routing_text for word in (
            "segment", "break down", "breakdown", "region", "category", "channel", "returning",
        )):
            kind = "sales_segments"
        elif any(word in routing_text for word in (
            "decompos", "overall", "total", "sales-per-order", "sales", "revenue",
        )):
            # Planner task wording varies; task ID and objective both identify the
            # requested work. A simple sales task over orders + order items is the
            # supported overall comparison unless it explicitly requests dimensions.
            kind = "sales_overall"

    if kind is None or not _has_schema(schemas, _required_for_kind(kind)):
        return None

    limitations: tuple[str, ...] = ()
    if kind == "sales_overall":
        sql = _overall_sales_sql(periods)
        limitations = (
            "Net product sales sum order_items.line_net_sales for delivered orders; shipping and later refunds are excluded.",
            "The order-count/sales-per-order decomposition is an accounting identity, not a causal explanation.",
        )
    elif kind == "sales_segments":
        sql = _segment_sales_sql(periods)
        limitations = (
            "New versus returning is defined by whether order month matches customers.first_order_date month; missing first-order dates are shown as unknown.",
            "Sales reconcile additively across each dimension. Order counts across product categories can overlap when one order contains multiple categories.",
        )
    elif kind == "marketing":
        sql = _marketing_sql(periods)
        limitations = (
            "The marketing table has no product-category field, so paid-search metrics cannot be attributed to product categories.",
            "Marketing attribution is simulated; spend, clicks, and attributed orders are descriptive associations, not proof of incremental impact.",
        )
    elif kind == "customer_lifecycle":
        sql = _customer_lifecycle_sql(periods)
        limitations = (
            "Active customers have at least one delivered order in each calendar month; repeat customers have at least two distinct delivered orders in that month.",
            "New versus returning uses customers.first_order_date month in the available demo history; missing dates are reported as unknown.",
            "Net product sales use order_items.line_net_sales for delivered orders and exclude shipping and later refunds; monthly associations do not establish causes.",
        )
    elif kind == "customer_cohort_retention":
        sql = _customer_cohort_retention_sql(periods)
        limitations = (
            "Cohort membership is based on the recorded month in customers.first_order_date; it is not guaranteed to be the customer's first-ever purchase.",
            "Retention means at least one delivered order in the stated month offset. Month 0 is measured the same way and can be below 100%.",
            "A period is marked observable from the latest order date's calendar month; without a source-completeness watermark, this does not prove that every event for that month was loaded.",
            "Cohort retention is descriptive and does not establish why customers returned or stopped purchasing.",
        )
    elif kind == "customer_segment":
        sql = _customer_segment_sql(periods)
        limitations = (
            "Customer segment, region, and acquisition channel use the values recorded on the customer row; they are not time-varying historical attributes.",
            "Repeat customers have at least two distinct delivered orders in the selected month; rates and customer-value averages are descriptive.",
            "Segment, region, and acquisition-channel totals are checked independently; each dimension is mutually exclusive only under the source's one-row-per-customer key.",
            "These comparisons show associations and do not establish causes.",
        )
    else:
        sql = _review_sql(periods)
        limitations = (
            "Reviews are a subset of orders; rating distributions are indicative and may not represent all customers.",
            "Ratings are used as a sentiment proxy; review text themes require semantic retrieval and are not inferred from ratings.",
        )

    rag_query = None
    if kind == "reviews":
        labels = ", ".join(label for _, label, _, _ in periods)
        rag_query = (
            "Representative customer review wording about product quality, fit, durability, "
            "and delivery; compare the requested periods: " + labels
        )
    return DemoQuery("demo_" + kind + "_period_comparison_v1", sql, limitations, rag_query)
