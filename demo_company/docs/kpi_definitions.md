# KPI definitions

Use these definitions consistently in the demo. If a question uses another definition, state it before calculating.

- **Delivered orders:** count of distinct `order_id` where `orders.status = 'delivered'`.
- **Gross product sales:** sum of `quantity * unit_price` for line items on delivered orders, before discounts and returns.
- **Discounts:** sum of line-item `discount_amount` for delivered orders.
- **Net product sales:** gross product sales minus line-item discounts. Excludes shipping and later refunds.
- **Net order value:** sum of `orders.order_total` for delivered orders, after order-level discounts and including shipping. This field is pre-return.
- **Average order value (AOV):** net order value divided by delivered distinct orders. State whether shipping is included; the default above includes shipping.
- **COGS:** sum of `quantity * unit_cost` on delivered order items.
- **Gross profit:** net product sales minus COGS, before returns, fulfillment, marketing, and overhead.
- **Gross margin rate:** gross profit divided by net product sales; undefined when net sales are zero.
- **Return rate by units:** returned units divided by sold units, for the same order cohort and with an explicitly stated observation window.
- **On-time delivery rate:** fulfilled delivered orders delivered on or before `promised_delivery_date`, divided by fulfilled delivered orders with both dates present.
- **Repeat customer:** a customer with at least two delivered orders in the selected analysis window. Do not confuse with lifetime returning status.
- **Recorded acquisition cohort:** customers whose `customers.first_order_date` falls in the same calendar month. This is the first-order date recorded in the available source, not independently verified first-ever purchase.
- **Monthly cohort retention:** distinct cohort customers with at least one delivered order in offset month M+n, divided by the recorded cohort size. M+0 is measured like every other month and can be below 100%. A zero cohort has an undefined rate.
- **Customer churn:** not a defined company KPI until the company supplies an inactivity threshold, an eligible-customer rule, an as-of date, and a source-completeness watermark. Do not relabel non-purchase in a single month as churn.
- **Customer lifetime value:** not available as a forecast from this fixture. Historical net sales per customer are observed sales over the captured period and must not be called predicted lifetime value.
- **Monthly growth:** `(current period value - comparison period value) / comparison period value`; comparison period must be stated.
- **Campaign ROAS:** attributed revenue divided by spend is not directly calculable because this fixture does not include attributed revenue. Do not derive ROAS from attributed order counts without an order-to-campaign attribution table.

## Time comparisons

Default monthly comparisons use complete calendar months. For year-over-year comparisons, compare the same calendar month. Explicitly handle the 2024 leap year only if daily-rate normalization is relevant.

Cohort offsets beyond the latest observed order-date month are unavailable. The latest observed event date does not by itself prove that the source extract is complete through that date.
