# Company profile: Cedar & Stone

## Fictional business

Cedar & Stone is a fictional US online retailer of home goods: storage, lighting, kitchen, bedding, and decor. It sells through its own web shop, marketplace listings, and a small retail-partner channel. It ships from one simulated fulfillment network and serves four regions: North, South, East, and West.

## Reporting setup

- Currency: USD.
- Dates: January 2024 through December 2025.
- Reporting timezone: UTC for timestamps; daily and monthly reporting uses the date portion as stored.
- Sales grain: one row per order in `orders.csv`; item grain in `order_items.csv`.
- Customer IDs are generated pseudonyms and do not identify people.
- All records are generated from a fixed random seed. No external company data or real customer data is included.

## Operational context represented

The fixture includes orders and line items, product prices and costs, customer segments, inventory snapshots, campaign activity, delivery outcomes, returns, and review text. Some operational signals are deliberately incomplete, as they would be in a real integration.

## Use limits

This fixture supports prototype behavior, schema mapping, evidence citation, workflow routing, and demonstration of analysis caveats. It is not a statistically representative retail benchmark, business forecast, or source of real-world conclusions. Synthetic co-occurrence must not be described as proven causality.
