# Data dictionary

All currency amounts are USD. IDs are stable within this generated fixture. Blank cells mean unavailable/not applicable; they are not zero. Order dates cover January 2024 through December 2025; later return, delivery, or review events may fall in early 2026 because they follow December orders.

| File | Grain and primary key | Important fields | Joins / caveats |
|---|---|---|---|
| `customers.csv` | One row per `customer_id` | `region`, `segment`, `first_order_date`, `acquisition_channel` | Synthetic customer segments; one customer can have many orders. |
| `products.csv` | One row per `product_id` | `category`, `unit_price`, `unit_cost`, `supplier_id`, `active` | Cost and price are synthetic list values; order-time amounts are in line items. |
| `orders.csv` | One row per `order_id` | `order_date`, `customer_id`, `region`, `channel`, `status`, `subtotal`, `discount_amount`, `shipping_amount`, `order_total` | `order_total` is pre-return and includes shipping. Use delivered orders for realized sales. |
| `order_items.csv` | One row per `order_item_id` | `order_id`, `product_id`, `quantity`, `unit_price`, `discount_amount`, `unit_cost`, `line_net_sales`, `line_gross_profit` | Join to orders; item revenue is `quantity * unit_price - discount_amount`. |
| `inventory_snapshots.csv` | One row per product and snapshot date | `snapshot_id`, `product_id`, `snapshot_date`, `on_hand_units`, `warehouse` | Monthly snapshots plus extra September 2025 snapshots for selected SKUs; snapshots do not record every inventory movement. |
| `marketing_campaigns.csv` | One row per month and campaign type/sales channel | `campaign_id`, `month`, `campaign_type`, `sales_channel`, `spend`, `impressions`, `clicks`, `attributed_orders` | Attribution is simulated and not proof of incremental impact. |
| `fulfillment.csv` | One row per shipped order | `fulfillment_id`, `order_id`, `ship_date`, `promised_delivery_date`, `delivered_date`, `delivery_status`, `carrier` | Canceled orders have no fulfillment record. Delay is based on delivered vs promised date. |
| `returns.csv` | One row per returned order item | `return_id`, `order_item_id`, `return_date`, `quantity_returned`, `reason`, `refund_amount` | Return sample is synthetic. Link to order item and order before aggregation. |
| `reviews.csv` | One row per review | `review_id`, `order_id`, `product_id`, `rating`, `review_text`, `review_date` | Text and ratings are synthetic; review coverage is a subset of orders. |

## Status and inclusion rules

- `orders.status` can be `delivered`, `cancelled`, or `processing`.
- Delivered revenue is calculated from delivered orders only. Canceled and processing orders should be reported separately.
- Returns are recorded after the original sale; do not subtract them twice if using `line_net_sales` and `refund_amount` together.
- `order_total` includes shipping and is not the same as product revenue.
- Product cost is a synthetic per-unit cost. Gross profit in line items is before shipping, fulfillment, marketing, overhead, and later refunds.

## Evidence keys

Use `order_id`, `order_item_id`, `product_id`, `customer_id`, `snapshot_id`, `campaign_id`, `fulfillment_id`, `return_id`, and `review_id` as row-level evidence identifiers. The SQL file provides reproducible aggregates whose source rows can be traced through these keys.
