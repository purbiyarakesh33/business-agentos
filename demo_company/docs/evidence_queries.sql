-- SQLite-friendly starter queries. Evidence IDs should include the relevant row IDs.

-- Monthly delivered order count, net order value, and net product sales.
WITH item_sales AS (
  SELECT order_id, SUM(line_net_sales) AS net_product_sales
  FROM order_items GROUP BY order_id
)
SELECT strftime('%Y-%m', o.order_date) AS month,
       COUNT(*) AS delivered_orders,
       ROUND(SUM(o.order_total), 2) AS net_order_value_including_shipping,
       ROUND(SUM(i.net_product_sales), 2) AS net_product_sales
FROM orders o
JOIN item_sales i USING (order_id)
WHERE o.status = 'delivered'
GROUP BY 1
ORDER BY 1;

-- Product/category contribution; cite product IDs and source order/item IDs.
SELECT p.category, i.product_id,
       ROUND(SUM(i.line_net_sales), 2) AS net_product_sales,
       SUM(i.quantity) AS units_sold,
       COUNT(DISTINCT o.order_id) AS delivered_orders
FROM orders o
JOIN order_items i USING (order_id)
JOIN products p USING (product_id)
WHERE o.status = 'delivered'
  AND o.order_date >= '2025-09-01' AND o.order_date < '2025-10-01'
GROUP BY p.category, i.product_id
ORDER BY net_product_sales DESC;

-- Delivery timeliness by month; one row per fulfillment/order.
SELECT strftime('%Y-%m', delivered_date) AS month,
       COUNT(*) AS delivered_shipments,
       SUM(CASE WHEN date(delivered_date) <= date(promised_delivery_date) THEN 1 ELSE 0 END) AS on_time_shipments
FROM fulfillment
WHERE delivery_status = 'delivered'
GROUP BY 1
ORDER BY 1;

-- Return units by product. Join through order_items to identify original order rows.
SELECT i.product_id, SUM(r.quantity_returned) AS units_returned,
       ROUND(SUM(r.refund_amount), 2) AS refund_amount
FROM returns r
JOIN order_items i USING (order_item_id)
GROUP BY i.product_id
ORDER BY units_returned DESC;

-- Repeat-customer analysis requires an explicit selected window.
WITH customer_orders AS (
  SELECT customer_id, COUNT(DISTINCT order_id) AS delivered_orders,
         SUM(order_total) AS net_order_value
  FROM orders
  WHERE status = 'delivered'
    AND order_date >= '2025-09-01' AND order_date < '2025-10-01'
  GROUP BY customer_id
)
SELECT CASE WHEN delivered_orders >= 2 THEN 'repeat_in_window' ELSE 'one_order_in_window' END AS segment,
       COUNT(*) AS customers, SUM(delivered_orders) AS orders,
       ROUND(SUM(net_order_value), 2) AS net_order_value
FROM customer_orders
GROUP BY 1;
