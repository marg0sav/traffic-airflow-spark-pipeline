from __future__ import annotations

import os
from typing import Dict, List

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# -----------------------------
# ClickHouse connection settings
# -----------------------------

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "traffic_db")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "airflow")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "airflow")

SOURCE_TABLE = os.getenv("SOURCE_TABLE", "measurements_history_demo")

AGGREGATES_TABLE = os.getenv("AGGREGATES_TABLE", "traffic_window_aggregates")
REGRESSION_TABLE = os.getenv("REGRESSION_TABLE", "traffic_local_regression")

WINDOW_SIZE = os.getenv("WINDOW_SIZE", "1 hour")

# Для сервера можно ставить больше партиций, чем физических worker-ов.
# Например, при 3 worker-ах нормально 6 или 12 shuffle partitions.
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "12"))
SPARK_DEFAULT_PARALLELISM = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "12"))
SPARK_READ_PARTITIONS = int(os.getenv("SPARK_READ_PARTITIONS", "12"))
SPARK_WRITE_PARTITIONS = int(os.getenv("SPARK_WRITE_PARTITIONS", "3"))

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "1g")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "1g")

TREND_SLOPE_THRESHOLD = float(os.getenv("TREND_SLOPE_THRESHOLD", "0.01"))


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("traffic-local-regression")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE_PARTITIONS))
        .config("spark.default.parallelism", str(SPARK_DEFAULT_PARALLELISM))
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def get_clickhouse_jdbc_url() -> str:
    return f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DATABASE}"


def get_jdbc_properties() -> Dict[str, str]:
    return {
        "user": CLICKHOUSE_USER,
        "password": CLICKHOUSE_PASSWORD,
        "driver": "com.clickhouse.jdbc.ClickHouseDriver",
    }


def get_sensor_predicates(spark: SparkSession) -> List[str]:
    """
    Получает список sensor_id и формирует JDBC predicates.
    Так Spark сможет читать данные из ClickHouse параллельно:
    отдельный JDBC-запрос на каждый sensor_id.
    """
    jdbc_url = get_clickhouse_jdbc_url()
    properties = get_jdbc_properties()

    sensors_query = f"(SELECT DISTINCT sensor_id FROM {SOURCE_TABLE}) AS sensors"

    sensor_rows = (
        spark.read
        .jdbc(url=jdbc_url, table=sensors_query, properties=properties)
        .select("sensor_id")
        .dropna()
        .distinct()
        .orderBy("sensor_id")
        .collect()
    )

    sensor_ids = [int(row["sensor_id"]) for row in sensor_rows]

    if not sensor_ids:
        raise RuntimeError(f"No sensors found in source table: {SOURCE_TABLE}")

    print(f"Found sensors: {sensor_ids}")

    return [f"sensor_id = {sensor_id}" for sensor_id in sensor_ids]


def read_measurements(spark: SparkSession) -> DataFrame:
    """
    Читает исходные измерения из ClickHouse.
    Чтение выполняется параллельно по sensor_id через JDBC predicates.
    """
    jdbc_url = get_clickhouse_jdbc_url()
    properties = get_jdbc_properties()
    predicates = get_sensor_predicates(spark)

    df = (
        spark.read
        .jdbc(
            url=jdbc_url,
            table=SOURCE_TABLE,
            predicates=predicates,
            properties=properties,
        )
        .select("sensor_id", "measured_at", "value")
    )

    return df


def write_to_clickhouse(df: DataFrame, table_name: str) -> None:
    """
    Записывает DataFrame в ClickHouse через JDBC.
    Не используем coalesce(1), чтобы не собирать результат в одну партицию.
    """
    jdbc_url = get_clickhouse_jdbc_url()

    (
        df.repartition(SPARK_WRITE_PARTITIONS, "sensor_id")
        .write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", CLICKHOUSE_USER)
        .option("password", CLICKHOUSE_PASSWORD)
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
        .option("batchsize", "10000")
        .mode("append")
        .save()
    )


def calculate_metrics(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Для каждого датчика и каждого временного окна рассчитывает:
    - агрегаты транспортного потока;
    - параметры локальной линейной регрессии value = slope * x + intercept;
    - R^2;
    - текстовую метку тренда.
    """
    prepared = (
        df
        .withColumn("sensor_id", F.col("sensor_id").cast("int"))
        .withColumn("measured_at", F.col("measured_at").cast("timestamp"))
        .withColumn("value", F.col("value").cast("double"))
        .filter(F.col("sensor_id").isNotNull())
        .filter(F.col("measured_at").isNotNull())
        .filter(F.col("value").isNotNull())
        .filter(F.col("value") >= 0)
        .withColumn("traffic_window", F.window(F.col("measured_at"), WINDOW_SIZE))
        .withColumn("window_start", F.col("traffic_window.start"))
        .withColumn("window_end", F.col("traffic_window.end"))
        # x — номер секунды внутри текущего окна.
        # Это лучше, чем использовать большой unix timestamp напрямую.
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
            F.stddev("value").alias("stddev_cars_per_hour"),
            F.sum("x").alias("sum_x"),
            F.sum("y").alias("sum_y"),
            F.sum("xy").alias("sum_xy"),
            F.sum("xx").alias("sum_xx"),
            F.sum("yy").alias("sum_yy"),
        )
    )

    n = F.col("records_count").cast("double")

    regression_numerator = (
        n * F.col("sum_xy")
        - F.col("sum_x") * F.col("sum_y")
    )

    regression_denominator = (
        n * F.col("sum_xx")
        - F.col("sum_x") * F.col("sum_x")
    )

    r2_denominator = (
        (n * F.col("sum_xx") - F.col("sum_x") * F.col("sum_x"))
        *
        (n * F.col("sum_yy") - F.col("sum_y") * F.col("sum_y"))
    )

    metrics = (
        grouped
        .withColumn(
            "slope_per_second",
            F.when(
                (F.col("records_count") > 1) & (regression_denominator != 0),
                regression_numerator / regression_denominator,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "intercept",
            F.when(
                n > 0,
                (F.col("sum_y") - F.col("slope_per_second") * F.col("sum_x")) / n,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "r2_score",
            F.when(
                (F.col("records_count") > 1) & (r2_denominator > 0),
                (regression_numerator * regression_numerator) / r2_denominator,
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "trend_label",
            F.when(F.col("slope_per_second") > TREND_SLOPE_THRESHOLD, F.lit("growing"))
            .when(F.col("slope_per_second") < -TREND_SLOPE_THRESHOLD, F.lit("falling"))
            .otherwise(F.lit("stable")),
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    aggregates_df = (
        metrics
        .select(
            F.col("sensor_id").cast("int").alias("sensor_id"),
            "window_start",
            "window_end",
            F.col("records_count").cast("long").alias("records_count"),
            F.col("avg_cars_per_hour").cast("double").alias("avg_cars_per_hour"),
            F.col("min_cars_per_hour").cast("double").alias("min_cars_per_hour"),
            F.col("max_cars_per_hour").cast("double").alias("max_cars_per_hour"),
            F.coalesce(
                F.col("stddev_cars_per_hour").cast("double"),
                F.lit(0.0),
            ).alias("stddev_cars_per_hour"),
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
            F.col("slope_per_second").cast("double").alias("slope_per_second"),
            F.col("intercept").cast("double").alias("intercept"),
            F.col("r2_score").cast("double").alias("r2_score"),
            F.col("trend_label").cast("string").alias("trend_label"),
            F.current_timestamp().alias("calculated_at"),
        )
    )

    return aggregates_df, regression_df


def main() -> None:
    spark = build_spark_session()

    print("Starting traffic local regression Spark job")
    print(f"Source table: {SOURCE_TABLE}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Aggregates target table: {AGGREGATES_TABLE}")
    print(f"Regression target table: {REGRESSION_TABLE}")
    print(f"Shuffle partitions: {SPARK_SHUFFLE_PARTITIONS}")
    print(f"Read partitions target: {SPARK_READ_PARTITIONS}")
    print(f"Write partitions: {SPARK_WRITE_PARTITIONS}")

    metrics_df = None

    try:
        source_df = read_measurements(spark)

        source_df = (
            source_df
            .repartition(SPARK_READ_PARTITIONS, "sensor_id")
        )

        aggregates_df, regression_df = calculate_metrics(source_df)

        # Сохраняем ссылку на общий cached DataFrame, чтобы потом unpersist.
        # Он является родителем для обоих результирующих DataFrame.
        metrics_df = aggregates_df

        print(f"Writing aggregates to ClickHouse table: {AGGREGATES_TABLE}")
        write_to_clickhouse(aggregates_df, AGGREGATES_TABLE)

        print(f"Writing regression results to ClickHouse table: {REGRESSION_TABLE}")
        write_to_clickhouse(regression_df, REGRESSION_TABLE)

        print("Spark traffic local regression job finished successfully")

    finally:
        if metrics_df is not None:
            try:
                metrics_df.unpersist()
            except Exception:
                pass

        spark.stop()


if __name__ == "__main__":
    main()