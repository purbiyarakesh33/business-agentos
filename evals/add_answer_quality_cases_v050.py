"""Add answer-quality gold cases computed directly from the synthetic SQLite data."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = HERE / "golden_dataset.json"
DB = ROOT / "demo_company" / "data" / "business_demo.sqlite3"


def money(value: float) -> str:
    return f"{value:,.2f}"


def change(current: float, prior: float) -> tuple[float, float]:
    delta = round(current - prior, 2)
    return delta, round(100 * delta / prior, 2) if prior else 0.0


def build_cases() -> list[dict]:
    with sqlite3.connect(DB) as conn:
        channel_sql = """SELECT o.channel,
            SUM(CASE WHEN o.order_date >= '2025-08-01' AND o.order_date < '2025-09-01' THEN oi.line_net_sales ELSE 0 END) AS august_sales,
            SUM(CASE WHEN o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01' THEN oi.line_net_sales ELSE 0 END) AS september_sales
          FROM orders o JOIN order_items oi ON oi.order_id=o.order_id
          WHERE o.status='delivered' AND o.order_date >= '2025-08-01' AND o.order_date < '2025-10-01'
          GROUP BY o.channel ORDER BY o.channel"""
        channels = [(name, round(aug or 0, 2), round(sep or 0, 2))
                    for name, aug, sep in conn.execute(channel_sql)]
        if {name for name, _, _ in channels} != {"Marketplace", "Retail Partner", "Web"}:
            raise SystemExit("Channel fixture changed; manually review and relabel this gold case.")

        gross_sql = """SELECT p.category, ROUND(SUM(oi.line_gross_profit), 2) AS gross_profit
          FROM orders o JOIN order_items oi ON oi.order_id=o.order_id
          JOIN products p ON p.product_id=oi.product_id
          WHERE o.status='delivered' AND o.order_date >= '2025-07-01' AND o.order_date < '2026-01-01'
          GROUP BY p.category ORDER BY gross_profit DESC, p.category"""
        gross = [(name, float(value)) for name, value in conn.execute(gross_sql)]
        if [name for name, _ in gross] != ["Storage", "Decor", "Bedding", "Lighting", "Kitchen"]:
            raise SystemExit("Gross-profit fixture changed; manually review and relabel this gold case.")

    channel_claims, channel_numeric = [], []
    august_total = september_total = 0.0
    for name, august, september in channels:
        delta, pct = change(september, august)
        channel_claims.append(
            f"{name} delivered net product sales changed from USD {money(august)} in August 2025 "
            f"to USD {money(september)} in September 2025 (USD {money(delta)}; {pct:.2f}%)."
        )
        channel_numeric.extend([august, september, delta, pct])
        august_total += august
        september_total += september
    total_delta, total_pct = change(rounded_september := round(september_total, 2),
                                   rounded_august := round(august_total, 2))
    channel_claims.extend([
        f"Across channels, delivered net product sales changed from USD {money(rounded_august)} "
        f"to USD {money(rounded_september)}, a change of USD {money(total_delta)} ({total_pct:.2f}%).",
        "The three channel changes reconcile to the overall change; the observed breakdown is descriptive and does not establish a cause.",
    ])
    channel_numeric.extend([rounded_august, rounded_september, total_delta, total_pct])

    gross_total = round(sum(value for _, value in gross), 2)
    gross_claims = [
        f"Delivered gross profit in July–December 2025 totaled USD {money(gross_total)} across the five product categories."
    ]
    gross_claims.extend(
        f"Rank {rank}: {name}, USD {money(value)} in delivered gross profit."
        for rank, (name, value) in enumerate(gross, 1)
    )
    gross_claims.extend([
        "Storage ranked first and Kitchen ranked fifth by delivered gross profit.",
        "This is a descriptive ranking; it does not explain why category gross profit differs.",
    ])
    gross_numeric = [gross_total, *(value for _, value in gross)]

    return [
        {
            "case_id": "report_sales_channel_compare_v050",
            "split": "development",
            "component": "answer_quality",
            "family_id": "planner-sales-channel-compare",
            "query": "Compare delivered net sales by sales channel for September and August 2025.",
            "gold_calculation_sql": channel_sql,
            "expected": {
                "required_claims": channel_claims,
                "required_caveats": [
                    "The measure is sum of order_items.line_net_sales for delivered orders, filtered by orders.order_date; shipping and later refunds are excluded.",
                    "The data is synthetic demo data and the comparison is descriptive, not causal.",
                ],
                "forbidden_claims": ["A sales channel caused the decline without causal evidence."],
                "relevant_sources": ["orders", "order_items"],
                "numeric_facts": channel_numeric,
            },
        },
        {
            "case_id": "report_gross_profit_category_h2_v050",
            "split": "development",
            "component": "answer_quality",
            "family_id": "planner-profit-category-rank",
            "query": "Rank product categories by gross profit for the second half of 2025.",
            "gold_calculation_sql": gross_sql,
            "expected": {
                "required_claims": gross_claims,
                "required_caveats": [
                    "Gross profit is sum of order_items.line_gross_profit for delivered orders from July 1 through December 31, 2025.",
                    "Gross profit is before returns, fulfillment, marketing, and overhead; categories are current product-row categories.",
                    "The data is synthetic demo data; the ranking is descriptive and does not establish a cause.",
                ],
                "forbidden_claims": ["The category ranking proves the cause of the profit differences."],
                "relevant_sources": ["orders", "order_items", "products"],
                "numeric_facts": gross_numeric,
            },
        },
    ]


if __name__ == "__main__":
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    if data.get("dataset_version") != "0.5.0":
        raise SystemExit("Expected golden dataset v0.5.0.")
    additions = build_cases()
    existing = {item["case_id"] for item in data["cases"]}
    if existing.intersection(item["case_id"] for item in additions):
        raise SystemExit("At least one answer-quality case already exists; refusing duplicate insertion.")
    data["cases"].extend(additions)
    GOLD.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Added", len(additions), "answer-quality gold cases from independently queried SQLite results.")
    for item in additions:
        print(item["case_id"], "numeric facts:", len(item["expected"]["numeric_facts"]))
