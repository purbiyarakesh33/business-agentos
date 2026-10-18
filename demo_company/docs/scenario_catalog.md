# Scenario catalog and expected reasoning

These are designed-in fixture properties, not real-company findings. Run the actual supplied data through queries before stating numeric results.

## Scenario A — September 2025 sales softness

The generator reduces order volume in September 2025 relative to the surrounding seasonal baseline. Selected product SKUs also have low/zero on-hand inventory snapshots during September, and the paid-search campaign for the Web sales channel has reduced spend. These events are deliberately co-occurring.

Expected behavior:

- Data Analyst quantifies month-over-month and year-over-year changes using delivered orders and net sales, with exact periods and formulas.
- Product & Operations checks item/category contributions and inventory snapshots; zero on-hand observations do not quantify lost demand.
- Customer Analyst checks new/repeat order mix and purchase frequency.
- Do not say stockouts or lower campaign spend caused the sales decline. The fixture has no randomized holdout, attribution join, or counterfactual proving causal impact.

## Scenario B — delivery and returns

Some fulfillment records are late and some line items are returned. Compare delivery status and return reason by period/product only after joining at the correct grain. Multiple items can belong to an order; avoid counting the same order multiple times.

## Scenario C — customer and forecast limits

The fixture contains two years of monthly history, enough for a prototype forecast demonstration but not a reliable commercial forecast. Forecasts should show uncertainty and note the short synthetic history. Customer IDs are pseudonymous and only support behavioral grouping.

## Known limitations to preserve

- No web sessions, conversion funnel, competitor prices, supplier lead-time history, actual advertising attribution, or experimental treatment data.
- Inventory snapshots are sparse and do not capture every stock movement.
- Reviews are synthetic and not a validated sentiment benchmark.
- The dataset is generated with planted structure, so a model must not present it as an independently observed company dataset.
