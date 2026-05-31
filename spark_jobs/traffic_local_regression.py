from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "traffic_db")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "airflow")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "airflow")

SOURCE_TABLE = os.getenv("SOURCE_TABLE", "measurements_history_demo")
WINDOW_SIZE = os.getenv("WINDOW_SIZE", "1 hour")


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("traffic-demo-local-regression")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )


def get_clickhouse_jdbc_url() -> str:
    return f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DATABASE}"


def read_measurements(spark: SparkSession):
    jdbc_url = get_clickhouse_jdbc_url()

    return (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", SOURCE_TABLE)
        .option("user", CLICKHOUSE_USER)
        .option("password", CLICKHOUSE_PASSWORD)
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
        .load()
        .select("sensor_id", "measured_at", "value")
    )


def write_to_clickhouse(df, table_name: str) -> None:
    jdbc_url = get_clickhouse_jdbc_url()

    (
        df.coalesce(1)
        .write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", CLICKHOUSE_USER)
        .option("password", CLICKHOUSE_PASSWORD)
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
        .mode("append")
        .save()
    )


def calculate_metrics(df):
    prepared = (
        df
        .withColumn("measured_at", F.col("measured_at").cast("timestamp"))
        .withColumn("value", F.col("value").cast("double"))
        .filter(F.col("measured_at").isNotNull())
        .filter(F.col("value").isNotNull())
        .filter(F.col("value") >= 0)
        .withColumn("traffic_window", F.window(F.col("measured_at"), WINDOW_SIZE))
        .withColumn("window_start", F.col("traffic_window.start"))
        .withColumn("window_end", F.col("traffic_window.end"))
        .withColumn(
            "x",
            F.unix_timestamp("measured_at").cast("double")
            - F.unix_timestamp("window_start").cast("double"),
        )
        .withColumn("y", F.col("value"))
        .withColumn("xy", F.col("x") * F.col("y"))
        .withColumn("xx", F.col("x") * F.col("x"))
        .withColumn("yy", F.col("y") * F.col("y"))
    )

    grouped = (
        prepared
        .groupBy("sensor_id", "window_start", "window_end")
        .agg(
            F.count("*").cast("long").alias("records_count"),
            F.avg("value").alias("avg_cars_per_hour"),
            F.min("value").alias("min_cars_per_hour"),
            F.max("value").alias("max_cars_per_hour"),
            F.sum("x").alias("sum_x"),
            F.sum("y").alias("sum_y"),
            F.sum("xy").alias("sum_xy"),
            F.sum("xx").alias("sum_xx"),
            F.sum("yy").alias("sum_yy"),
        )
    )

    n = F.col("records_count").cast("double")
    numerator = n * F.col("sum_xy") - F.col("sum_x") * F.col("sum_y")
    denominator = n * F.col("sum_xx") - F.col("sum_x") * F.col("sum_x")

    r2_denominator = (
        (n * F.col("sum_xx") - F.col("sum_x") * F.col("sum_x"))
        *
        (n * F.col("sum_yy") - F.col("sum_y") * F.col("sum_y"))
    )

    metrics = (
        grouped
        .withColumn(
            "slope_per_second",
            F.when((F.col("records_count") > 1) & (denominator != 0), numerator / denominator)
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "intercept",
            (F.col("sum_y") - F.col("slope_per_second") * F.col("sum_x")) / n,
        )
        .withColumn(
            "r2_score",
            F.when(
                (F.col("records_count") > 1) & (r2_denominator > 0),
                (numerator * numerator) / r2_denominator,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "trend_label",
            F.when(F.col("slope_per_second") > 0.01, F.lit("growing"))
            .when(F.col("slope_per_second") < -0.01, F.lit("falling"))
            .otherwise(F.lit("stable")),
        )
        .cache()
    )

    aggregates_df = (
        metrics
        .select(
            F.col("sensor_id").cast("int").alias("sensor_id"),
            "window_start",
            "window_end",
            F.col("records_count").cast("long").alias("records_count"),
            "avg_cars_per_hour",
            "min_cars_per_hour",
            "max_cars_per_hour",
            F.lit(0.0).alias("stddev_cars_per_hour"),
            F.current_timestamp().alias("calculated_at"),
        )
    )

    regression_df = (
        metrics
        .select(
            F.col("sensor_id").cast("int").alias("sensor_id"),
            "window_start",
            "window_end",
            F.col("records_count").cast("long").alias("records_count"),
            "slope_per_second",
            "intercept",
            "r2_score",
            "trend_label",
            F.current_timestamp().alias("calculated_at"),
        )
    )

    return aggregates_df, regression_df


def main() -> None:
    spark = build_spark_session()

    try:
        source_df = read_measurements(spark)
        source_df = source_df.repartition(2, "sensor_id")

        aggregates_df, regression_df = calculate_metrics(source_df)

        write_to_clickhouse(aggregates_df, "traffic_window_aggregates")
        write_to_clickhouse(regression_df, "traffic_local_regression")

        print("Spark demo traffic processing finished successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()