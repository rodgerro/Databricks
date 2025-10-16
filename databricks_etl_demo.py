# Databricks notebook source
# MAGIC %md
# MAGIC # Mini ETL Demo (Bronze ➜ Silver ➜ Gold)
# MAGIC 
# MAGIC **What this shows**
# MAGIC - Ingest tiny raw data with **PySpark**
# MAGIC - Create **Delta tables** for bronze/silver/gold
# MAGIC - Use **Spark SQL** for transformations
# MAGIC - Simple quality checks and an example **MERGE** (upsert)
# MAGIC 
# MAGIC **How to use**
# MAGIC 1. Import this file into Databricks (Workspace ➜ Import ➜ File).
# MAGIC 2. Attach a cluster and run cells top-to-bottom.
# MAGIC 3. Change the `catalog_name`, `schema_name`, and `base_path` widgets as needed.

# COMMAND ----------

# MAGIC %python
# Setup widgets (edit these for your workspace)
dbutils.widgets.text("catalog_name", "hive_metastore", "Catalog")
dbutils.widgets.text("schema_name", "etl_demo", "Schema")
dbutils.widgets.text("base_path", "/tmp/etl_demo", "Base Path")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
base_path = dbutils.widgets.get("base_path")

print(f"Using catalog={catalog_name}, schema={schema_name}, base_path={base_path}")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create and use the target schema (database)
# MAGIC CREATE SCHEMA IF NOT EXISTS ${catalog_name}.${schema_name};
# MAGIC USE ${catalog_name}.${schema_name};

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create tiny raw datasets (in-memory) and land to **Bronze**

# COMMAND ----------

# MAGIC %python
from pyspark.sql import functions as F, types as T

# --- Tiny raw datasets (simulating a landing/raw zone) ---
raw_customers = [
    {"customer_id": 1, "first_name": "Alice", "last_name": "Brown", "email": "alice@example.com", "state": "TX"},
    {"customer_id": 2, "first_name": "Bob", "last_name": "Smith", "email": "bob@example.com", "state": "UT"},
    {"customer_id": 3, "first_name": "Carlos", "last_name": "Diaz", "email": "carlos@example.com", "state": None},   # bad state to fix
]

raw_orders = [
    {"order_id": 101, "customer_id": 1, "order_ts": "2025-10-01T09:15:00Z", "item": "Widget", "qty": 2, "unit_price": "19.99"},
    {"order_id": 102, "customer_id": 1, "order_ts": "2025-10-02T12:30:00Z", "item": "Gadget", "qty": 1, "unit_price": "29.50"},
    {"order_id": 103, "customer_id": 2, "order_ts": "2025-10-03T18:05:00Z", "item": "Widget", "qty": 5, "unit_price": "19.99"},
    {"order_id": 104, "customer_id": 3, "order_ts": "2025-10-03T20:20:00Z", "item": "Thingamajig", "qty": 1, "unit_price": "9.00"},
]

customers_df = spark.createDataFrame(raw_customers)
orders_df = spark.createDataFrame(raw_orders)

# Add ingestion metadata
ingestion_ts = F.current_timestamp()
customers_bronze = customers_df.withColumn("ingestion_ts", ingestion_ts)
orders_bronze = (
    orders_df
    .withColumn("ingestion_ts", ingestion_ts)
    .withColumn("source_file", F.lit("mock_json"))
)

# Write Bronze as Delta
bronze_cust_tbl = f"{catalog_name}.{schema_name}.bronze_customers"
bronze_ord_tbl  = f"{catalog_name}.{schema_name}.bronze_orders"

customers_bronze.write.mode("overwrite").format("delta").saveAsTable(bronze_cust_tbl)
orders_bronze.write.mode("overwrite").format("delta").saveAsTable(bronze_ord_tbl)

display(spark.table(bronze_cust_tbl).limit(10))
display(spark.table(bronze_ord_tbl).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform to **Silver** (type casting, derived cols, light cleansing)

# COMMAND ----------

# MAGIC %sql
-- Ensure we're in the right schema
USE ${catalog_name}.${schema_name};

-- Silver customers: fix null states, trim, basic profiling flags
CREATE OR REPLACE TABLE silver_customers AS
SELECT
  customer_id,
  trim(first_name) AS first_name,
  trim(last_name)  AS last_name,
  lower(email)     AS email,
  COALESCE(state, 'UNKNOWN') AS state,
  ingestion_ts,
  -- basic data quality flags
  CASE WHEN email RLIKE '^[^@]+@[^@]+\.[^@]+$' THEN true ELSE false END AS email_is_valid
FROM bronze_customers;

SELECT * FROM silver_customers;

# COMMAND ----------

# MAGIC %python
# Silver orders: cast types, add total_amount, normalize timestamp
bronze_orders = spark.table(bronze_ord_tbl)

silver_orders = (
    bronze_orders
    .withColumn("order_ts", F.to_timestamp("order_ts"))
    .withColumn("unit_price", F.col("unit_price").cast("decimal(10,2)"))
    .withColumn("qty", F.col("qty").cast("int"))
    .withColumn("total_amount", F.col("qty") * F.col("unit_price"))
    .withColumn("order_date", F.to_date("order_ts"))
)

silver_ord_tbl = f"{catalog_name}.{schema_name}.silver_orders"
silver_orders.write.mode("overwrite").format("delta").saveAsTable(silver_ord_tbl)

display(spark.table(silver_ord_tbl))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a small **Gold** star-schema style output

# COMMAND ----------

# MAGIC %sql
-- Dimensions
CREATE OR REPLACE TABLE dim_customer AS
SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  c.email,
  c.state
FROM silver_customers c;

-- Facts (daily sales by customer)
CREATE OR REPLACE TABLE fct_daily_sales AS
SELECT
  o.order_date,
  o.customer_id,
  SUM(o.qty)          AS total_qty,
  SUM(o.total_amount) AS total_sales
FROM silver_orders o
GROUP BY o.order_date, o.customer_id;

-- A simple reporting view joining facts + dims
CREATE OR REPLACE VIEW v_daily_sales AS
SELECT
  f.order_date,
  d.customer_id,
  concat(d.first_name, ' ', d.last_name) AS customer_name,
  d.state,
  f.total_qty,
  f.total_sales
FROM fct_daily_sales f
JOIN dim_customer d ON d.customer_id = f.customer_id;

SELECT * FROM v_daily_sales ORDER BY order_date, customer_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example **MERGE** (upsert) to Silver
# MAGIC Simulate a new order and an update to an existing order, then MERGE into `silver_orders`.

# COMMAND ----------

# MAGIC %python
from pyspark.sql import Row

updates = [
    # New order (insert)
    {"order_id": 105, "customer_id": 2, "order_ts": "2025-10-04T11:00:00", "item": "Gadget", "qty": 2, "unit_price": "29.50"},
    # Update existing order 104 (change qty)
    {"order_id": 104, "customer_id": 3, "order_ts": "2025-10-03T20:20:00", "item": "Thingamajig", "qty": 3, "unit_price": "9.00"},
]

updates_df = spark.createDataFrame(updates)     .withColumn("order_ts", F.to_timestamp("order_ts"))     .withColumn("unit_price", F.col("unit_price").cast("decimal(10,2)"))     .withColumn("qty", F.col("qty").cast("int"))     .withColumn("total_amount", F.col("qty") * F.col("unit_price"))     .withColumn("order_date", F.to_date("order_ts"))

updates_df.createOrReplaceTempView("updates_src")

# COMMAND ----------

# MAGIC %sql
MERGE INTO ${catalog_name}.${schema_name}.silver_orders AS tgt
USING updates_src AS src
ON tgt.order_id = src.order_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

SELECT * FROM silver_orders ORDER BY order_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild Gold after the upsert (incremental pattern would handle only changes)

# COMMAND ----------

# MAGIC %sql
CREATE OR REPLACE TABLE fct_daily_sales AS
SELECT
  o.order_date,
  o.customer_id,
  SUM(o.qty)          AS total_qty,
  SUM(o.total_amount) AS total_sales
FROM silver_orders o
GROUP BY o.order_date, o.customer_id;

SELECT * FROM v_daily_sales ORDER BY order_date, customer_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Quality Checks

# COMMAND ----------

# MAGIC %sql
-- 1) No null customer_ids in silver_orders
SELECT COUNT(*) AS null_customer_ids FROM silver_orders WHERE customer_id IS NULL;

-- 2) Any invalid emails in silver_customers
SELECT COUNT(*) AS invalid_emails FROM silver_customers WHERE email_is_valid = false;

-- 3) Row counts
SELECT 'bronze_customers' AS table_name, COUNT(*) AS cnt FROM bronze_customers
UNION ALL
SELECT 'bronze_orders' AS table_name, COUNT(*) FROM bronze_orders
UNION ALL
SELECT 'silver_customers' AS table_name, COUNT(*) FROM silver_customers
UNION ALL
SELECT 'silver_orders' AS table_name, COUNT(*) FROM silver_orders
UNION ALL
SELECT 'dim_customer' AS table_name, COUNT(*) FROM dim_customer
UNION ALL
SELECT 'fct_daily_sales' AS table_name, COUNT(*) FROM fct_daily_sales;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: handy ad-hoc queries

# COMMAND ----------

# MAGIC %sql
-- Total sales by state and day
SELECT order_date, state, SUM(total_sales) AS sales
FROM v_daily_sales
GROUP BY order_date, state
ORDER BY order_date, state;

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ **Done.** You now have a tiny, end-to-end ETL with PySpark + SQL and Delta tables.
