"""Generate Cedar & Stone's deterministic, fully synthetic demo tables."""
from __future__ import annotations

import csv
import json
import random
import sqlite3
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
SEED = 20261006
RNG = random.Random(SEED)

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Web", "Marketplace", "Retail Partner"]
CATEGORIES = ["Storage", "Lighting", "Kitchen", "Bedding", "Decor"]
LOW_INVENTORY_PRODUCTS = {"P003", "P008", "P013", "P018"}


def write_csv(name: str, rows: list[dict]) -> None:
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_database(tables: dict[str, list[dict]]) -> None:
    db_path = DATA / "business_demo.sqlite3"
    if db_path.exists():
        db_path.unlink()
    integer_fields = {"quantity", "quantity_returned", "on_hand_units", "rating",
                      "impressions", "clicks", "attributed_orders"}
    real_fields = {"unit_price", "unit_cost", "subtotal", "discount_amount",
                   "shipping_amount", "order_total", "line_net_sales",
                   "line_gross_profit", "spend", "refund_amount"}
    primary_keys = {
        "customers": "customer_id", "products": "product_id", "orders": "order_id",
        "order_items": "order_item_id", "inventory_snapshots": "snapshot_id",
        "marketing_campaigns": "campaign_id", "fulfillment": "fulfillment_id",
        "returns": "return_id", "reviews": "review_id",
    }
    foreign_keys = {
        "orders": [("customer_id", "customers", "customer_id")],
        "order_items": [("order_id", "orders", "order_id"), ("product_id", "products", "product_id")],
        "inventory_snapshots": [("product_id", "products", "product_id")],
        "fulfillment": [("order_id", "orders", "order_id")],
        "returns": [("order_item_id", "order_items", "order_item_id")],
        "reviews": [("order_id", "orders", "order_id"), ("product_id", "products", "product_id")],
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for table_name, rows in tables.items():
            columns = list(rows[0]) if rows else []
            definitions = []
            for column in columns:
                kind = "INTEGER" if column in integer_fields else "REAL" if column in real_fields else "TEXT"
                suffix = " PRIMARY KEY" if primary_keys.get(table_name) == column else ""
                definitions.append(f'"{column}" {kind}{suffix}')
            for field_name, target, target_field in foreign_keys.get(table_name, []):
                definitions.append(f'FOREIGN KEY ("{field_name}") REFERENCES "{target}"("{target_field}")')
            connection.execute(f'CREATE TABLE "{table_name}" ({", ".join(definitions)})')
            placeholders = ",".join("?" for _ in columns)
            quoted_columns = ",".join(f'"{column}"' for column in columns)
            values = []
            for row in rows:
                converted = []
                for column in columns:
                    value = row[column]
                    if value == "":
                        converted.append(None)
                    elif column in integer_fields:
                        converted.append(int(value))
                    elif column in real_fields:
                        converted.append(float(value))
                    elif column == "active":
                        converted.append(1 if value.lower() == "true" else 0)
                    else:
                        converted.append(value)
                values.append(tuple(converted))
            connection.executemany(f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})', values)


def iso(d: date | datetime) -> str:
    return d.isoformat()


def month_starts() -> list[date]:
    return [date(year, month, 1) for year in (2024, 2025) for month in range(1, 13)]


def build_products() -> list[dict]:
    names = ["Stackable Bin", "Wall Sconce", "Prep Bowl Set", "Cotton Duvet", "Ceramic Vase",
             "Drawer Organizer", "Desk Lamp", "Spice Rack", "Linen Sheet Set", "Table Runner",
             "Woven Basket", "Pendant Light", "Chef Knife", "Quilted Coverlet", "Stoneware Tray",
             "Underbed Box", "Reading Lamp", "Pantry Canister", "Bath Towel Set", "Candle Holder"]
    rows = []
    for index, name in enumerate(names, start=1):
        price = round(RNG.uniform(18, 145), 2)
        rows.append({
            "product_id": f"P{index:03d}", "product_name": name,
            "category": CATEGORIES[(index - 1) // 4],
            "unit_price": f"{price:.2f}", "unit_cost": f"{price * RNG.uniform(.38, .62):.2f}",
            "supplier_id": f"SUP{((index - 1) % 5) + 1:02d}", "active": "true",
        })
    return rows


def build_customers() -> list[dict]:
    rows = []
    for index in range(1, 801):
        rows.append({
            "customer_id": f"C{index:04d}", "region": RNG.choice(REGIONS),
            "segment": RNG.choices(["value", "core", "premium"], [0.35, 0.5, 0.15])[0],
            "first_order_date": "", "acquisition_channel": RNG.choice(CHANNELS),
        })
    return rows


def generate() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    products = build_products()
    customers = build_customers()
    product_by_id = {p["product_id"]: p for p in products}
    customer_by_id = {c["customer_id"]: c for c in customers}
    orders: list[dict] = []
    items: list[dict] = []
    fulfillment: list[dict] = []
    returns: list[dict] = []
    reviews: list[dict] = []
    inventory: list[dict] = []
    campaigns: list[dict] = []
    first_seen: dict[str, date] = {}

    month_weights = [0.86, 0.82, 0.92, 0.96, 1.00, 0.98, 1.05, 1.08, 0.94, 1.00, 1.30, 1.42]
    order_num = item_num = return_num = review_num = fulfill_num = campaign_num = 0
    order_item_index: dict[str, list[dict]] = {}
    for month_index, month in enumerate(month_starts()):
        growth = 1.0 if month.year == 2024 else 1.12
        expected = 105 * month_weights[month.month - 1] * growth
        if month == date(2025, 9, 1):
            expected *= 0.72  # planted demand softness; no causal claim implied
        order_count = max(40, round(expected + RNG.uniform(-7, 7)))
        for _ in range(order_count):
            order_num += 1
            customer = RNG.choice(customers)
            customer_id = customer["customer_id"]
            order_day = RNG.randint(1, monthrange(month.year, month.month)[1])
            order_date = date(month.year, month.month, order_day)
            first_seen[customer_id] = min(order_date, first_seen.get(customer_id, order_date))
            region = customer["region"]
            channel = RNG.choices(CHANNELS, [0.64, 0.29, 0.07])[0]
            order_id = f"ORD{order_num:06d}"
            status = RNG.choices(["delivered", "cancelled", "processing"], [0.94, 0.035, 0.025])[0]
            line_count = RNG.choices([1, 2, 3], [0.52, 0.36, 0.12])[0]
            item_rows: list[dict] = []
            for _line in range(line_count):
                choices = products
                if month == date(2025, 9, 1):
                    weights = [0.15 if p["product_id"] in LOW_INVENTORY_PRODUCTS else 1.0 for p in products]
                    product = RNG.choices(choices, weights)[0]
                else:
                    product = RNG.choice(choices)
                quantity = RNG.choices([1, 2, 3], [0.78, 0.19, 0.03])[0]
                unit_price = float(product["unit_price"])
                unit_cost = float(product["unit_cost"])
                discount = round(unit_price * quantity * RNG.choice([0, 0, 0, .05, .10]), 2)
                net_sales = round(unit_price * quantity - discount, 2)
                item_num += 1
                item = {
                    "order_item_id": f"ITM{item_num:07d}", "order_id": order_id,
                    "product_id": product["product_id"], "quantity": quantity,
                    "unit_price": f"{unit_price:.2f}", "discount_amount": f"{discount:.2f}",
                    "unit_cost": f"{unit_cost:.2f}", "line_net_sales": f"{net_sales:.2f}",
                    "line_gross_profit": f"{net_sales - unit_cost * quantity:.2f}",
                }
                items.append(item)
                item_rows.append(item)
            order_item_index[order_id] = item_rows
            subtotal = round(sum(float(i["unit_price"]) * int(i["quantity"]) for i in item_rows), 2)
            discount_total = round(sum(float(i["discount_amount"]) for i in item_rows), 2)
            shipping = 0.0 if subtotal >= 100 else round(RNG.choice([4.99, 6.99, 8.99]), 2)
            orders.append({
                "order_id": order_id, "order_date": iso(order_date), "customer_id": customer_id,
                "region": region, "channel": channel, "status": status,
                "subtotal": f"{subtotal:.2f}", "discount_amount": f"{discount_total:.2f}",
                "shipping_amount": f"{shipping:.2f}",
                "order_total": f"{subtotal - discount_total + shipping:.2f}",
            })
            if status == "delivered":
                ship_date = order_date + timedelta(days=RNG.randint(0, 2))
                promised = ship_date + timedelta(days=RNG.randint(3, 6))
                is_late = RNG.random() < (0.23 if month == date(2025, 9, 1) else 0.12)
                delivered = promised + timedelta(days=RNG.randint(1, 3)) if is_late else promised - timedelta(days=RNG.randint(0, 2))
                fulfill_num += 1
                fulfillment.append({
                    "fulfillment_id": f"FUL{fulfill_num:06d}", "order_id": order_id,
                    "ship_date": iso(ship_date), "promised_delivery_date": iso(promised),
                    "delivered_date": iso(delivered), "delivery_status": "delivered",
                    "carrier": RNG.choice(["ParcelNorth", "SwiftPost", "GroundLine"]),
                })
                for item in item_rows:
                    if RNG.random() < 0.055:
                        quantity_returned = RNG.randint(1, int(item["quantity"]))
                        return_num += 1
                        per_unit_net = float(item["line_net_sales"]) / int(item["quantity"])
                        returns.append({
                            "return_id": f"RET{return_num:06d}", "order_item_id": item["order_item_id"],
                            "return_date": iso(delivered + timedelta(days=RNG.randint(3, 30))),
                            "quantity_returned": quantity_returned,
                            "reason": RNG.choice(["damaged", "not_as_expected", "changed_mind", "wrong_item"]),
                            "refund_amount": f"{per_unit_net * quantity_returned:.2f}",
                        })
                if RNG.random() < 0.36:
                    review_num += 1
                    review_product = RNG.choice(item_rows)["product_id"]
                    rating = RNG.choices([1, 2, 3, 4, 5], [0.04, 0.06, 0.15, 0.35, 0.40])[0]
                    texts = {
                        1: ["Arrived damaged and did not match the listing.", "Very disappointed with the quality."],
                        2: ["The item feels cheaper than expected.", "Delivery took too long for this order."],
                        3: ["It is fine, but nothing special.", "Works as described; average experience."],
                        4: ["Good quality and arrived on time.", "Nice product and easy to use."],
                        5: ["Excellent quality; I would buy it again.", "Beautiful finish and fast delivery."],
                    }
                    reviews.append({
                        "review_id": f"REV{review_num:06d}", "order_id": order_id,
                        "product_id": review_product, "rating": rating,
                        "review_text": RNG.choice(texts[rating]),
                        "review_date": iso(delivered + timedelta(days=RNG.randint(0, 12))),
                    })

    for customer in customers:
        if customer["customer_id"] in first_seen:
            customer["first_order_date"] = iso(first_seen[customer["customer_id"]])

    # Monthly snapshots for each SKU, plus within-month snapshots on selected low-inventory SKUs.
    for month in month_starts():
        month_end = date(month.year, month.month, monthrange(month.year, month.month)[1])
        for product in products:
            pid = product["product_id"]
            stock = RNG.randint(18, 140)
            inventory.append({"snapshot_id": f"INV{len(inventory)+1:06d}", "product_id": pid,
                              "snapshot_date": iso(month_end), "on_hand_units": stock,
                              "warehouse": "WH-CENTRAL"})
        if month == date(2025, 9, 1):
            for day, stock in [(1, 12), (8, 3), (15, 0), (22, 0)]:
                for pid in sorted(LOW_INVENTORY_PRODUCTS):
                    inventory.append({"snapshot_id": f"INV{len(inventory)+1:06d}", "product_id": pid,
                                      "snapshot_date": iso(date(2025, 9, day)), "on_hand_units": stock,
                                      "warehouse": "WH-CENTRAL"})

    for month in month_starts():
        for channel in CHANNELS:
            campaign_num += 1
            base_spend = {"Web": 3800, "Marketplace": 2100, "Retail Partner": 650}[channel]
            spend = base_spend * (1.0 + RNG.uniform(-0.12, 0.12))
            if month == date(2025, 9, 1) and channel == "Web":
                spend *= 0.65
            campaign_type = {"Web": "Paid Search", "Marketplace": "Marketplace Ads",
                             "Retail Partner": "Partner Promotion"}[channel]
            impressions = round(spend * RNG.uniform(18, 34))
            clicks = round(impressions * RNG.uniform(.018, .045))
            attributed = round(clicks * RNG.uniform(.025, .075))
            campaigns.append({
                "campaign_id": f"CAM{campaign_num:04d}", "month": iso(month),
                "campaign_type": campaign_type, "sales_channel": channel,
                "spend": f"{spend:.2f}", "impressions": impressions, "clicks": clicks,
                "attributed_orders": attributed,
            })

    write_csv("customers.csv", customers)
    write_csv("products.csv", products)
    write_csv("orders.csv", orders)
    write_csv("order_items.csv", items)
    write_csv("inventory_snapshots.csv", inventory)
    write_csv("marketing_campaigns.csv", campaigns)
    write_csv("fulfillment.csv", fulfillment)
    write_csv("returns.csv", returns)
    write_csv("reviews.csv", reviews)

    tables = {
        "customers": customers, "products": products, "orders": orders,
        "order_items": items, "inventory_snapshots": inventory,
        "marketing_campaigns": campaigns, "fulfillment": fulfillment,
        "returns": returns, "reviews": reviews,
    }
    write_database(tables)

    source_specs = [
        ("customers", "customers.csv", "One row per customer", customers),
        ("products", "products.csv", "One row per product", products),
        ("orders", "orders.csv", "One row per order", orders),
        ("order_items", "order_items.csv", "One row per order item", items),
        ("inventory_snapshots", "inventory_snapshots.csv", "One row per product per snapshot", inventory),
        ("marketing_campaigns", "marketing_campaigns.csv", "One row per month and campaign type/sales channel", campaigns),
        ("fulfillment", "fulfillment.csv", "One row per shipped order", fulfillment),
        ("returns", "returns.csv", "One row per returned order item", returns),
        ("reviews", "reviews.csv", "One row per review", reviews),
    ]
    catalog = []
    for source_id, filename, grain, rows in source_specs:
        catalog.append({"id": source_id, "name": filename, "type": "synthetic_csv_and_sqlite",
                        "description": grain, "fields": list(rows[0]) if rows else [],
                        "row_count": len(rows), "sqlite_table": source_id, "synthetic": True})
    (DOCS / "source_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    truth = {
        "synthetic": True,
        "seed": SEED,
        "planted_period": "2025-09",
        "observations_to_check": [
            "Order volume is reduced by the generator for September 2025.",
            "Paid-search spend for the Web sales channel is reduced for September 2025.",
            "Selected products have low/zero on-hand inventory snapshots during September 2025.",
            "The late-delivery probability is higher in September 2025.",
        ],
        "low_inventory_product_ids": sorted(LOW_INVENTORY_PRODUCTS),
        "causality_status": "No causal relationships are established by the fixture. Treat co-occurring changes as observations or hypotheses.",
    }
    (DOCS / "scenario_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Generated {len(source_specs)} CSV tables in {DATA}")


if __name__ == "__main__":
    generate()
