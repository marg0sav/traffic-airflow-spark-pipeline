from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import clickhouse_connect
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "traffic_db")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")


def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )


def check_source_data_exists() -> None:
    """
    Проверяет, что в исходной таблице есть данные.
    """
    client = get_clickhouse_client()

    result = client.query(
        """
        SELECT count()
        FROM measurements_history
        """
    )

    rows_count = result.result_rows[0][0]

    if rows_count == 0:
        client.command(
            """
            INSERT INTO pipeline_quality_checks
            (check_name, check_status, details)
            VALUES
            ('source_data_exists', 'failed', 'measurements_history is empty')
            """
        )
        raise ValueError("Source table measurements_history is empty")

    client.command(
        f"""
        INSERT INTO pipeline_quality_checks
        (check_name, check_status, details)
        VALUES
        ('source_data_exists', 'success', 'rows_count={rows_count}')
        """
    )


def check_source_data_quality() -> None:
    """
    Базовая проверка качества данных:
    - value не должен быть NULL;
    - value не должен быть отрицательным;
    - measured_at не должен быть NULL.
    """
    client = get_clickhouse_client()

    result = client.query(
        """
        SELECT
            count() AS total_rows,
            countIf(value < 0) AS negative_values,
            countIf(isNull(measured_at)) AS null_timestamps
        FROM measurements_history
        """
    )

    total_rows, negative_values, null_timestamps = result.result_rows[0]

    details = (
        f"total_rows={total_rows}, "
        f"negative_values={negative_values}, "
        f"null_timestamps={null_timestamps}"
    )

    if negative_values > 0 or null_timestamps > 0:
        client.command(
            f"""
            INSERT INTO pipeline_quality_checks
            (check_name, check_status, details)
            VALUES
            ('source_data_quality', 'failed', '{details}')
            """
        )
        raise ValueError(f"Data quality check failed: {details}")

    client.command(
        f"""
        INSERT INTO pipeline_quality_checks
        (check_name, check_status, details)
        VALUES
        ('source_data_quality', 'success', '{details}')
        """
    )


def unstable_task_for_retry_demo() -> None:
    """
    Специальная демонстрационная задача.
    Иногда падает, чтобы показать retry в Airflow.
    Для отчёта можно заскринить состояние up_for_retry.
    """
    fail_probability = 0.5

    if random.random() < fail_probability:
        raise RuntimeError("Demo failure: task failed intentionally to test Airflow retry")

    print("Retry demo task finished successfully")


def check_results_written() -> None:
    """
    Проверяет, что Spark job записал результаты в целевые таблицы.
    """
    client = get_clickhouse_client()

    agg_count = client.query(
        """
        SELECT count()
        FROM traffic_window_aggregates
        """
    ).result_rows[0][0]

    regression_count = client.query(
        """
        SELECT count()
        FROM traffic_local_regression
        """
    ).result_rows[0][0]

    details = f"aggregates={agg_count}, regression_rows={regression_count}"

    if agg_count == 0 or regression_count == 0:
        client.command(
            f"""
            INSERT INTO pipeline_quality_checks
            (check_name, check_status, details)
            VALUES
            ('spark_results_written', 'failed', '{details}')
            """
        )
        raise ValueError(f"Spark results were not written: {details}")

    client.command(
        f"""
        INSERT INTO pipeline_quality_checks
        (check_name, check_status, details)
        VALUES
        ('spark_results_written', 'success', '{details}')
        """
    )


default_args = {
    "owner": "margo",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="traffic_airflow_spark_pipeline",
    description="Traffic data processing pipeline with Airflow, Spark and ClickHouse",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["traffic", "spark", "clickhouse", "coursework"],
) as dag:

    check_data_exists = PythonOperator(
        task_id="check_source_data_exists",
        python_callable=check_source_data_exists,
    )

    check_data_quality = PythonOperator(
        task_id="check_source_data_quality",
        python_callable=check_source_data_quality,
    )

    retry_demo = PythonOperator(
        task_id="unstable_task_for_retry_demo",
        python_callable=unstable_task_for_retry_demo,
        retries=3,
        retry_delay=timedelta(seconds=30),
    )

    run_spark_processing = BashOperator(
        task_id="run_spark_local_regression_job",
        bash_command=f"""
        spark-submit \
          --master {SPARK_MASTER_URL} \
          --packages com.clickhouse:clickhouse-jdbc:0.7.1 \
          /opt/airflow/spark_jobs/traffic_local_regression.py
        """,
    )

    check_results = PythonOperator(
        task_id="check_results_written",
        python_callable=check_results_written,
    )

    (
        check_data_exists
        >> check_data_quality
        >> retry_demo
        >> run_spark_processing
        >> check_results
    )