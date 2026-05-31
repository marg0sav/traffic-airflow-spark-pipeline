from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a reduced traffic dataset from a large CSV file. "
            "The script selects a five-month period without outage zeros "
            "and maps each month to a separate virtual sensor."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("traffic_measurements_history.csv"),
        help="Path to source CSV file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("traffic_measurements_history_5_months_virtual_sensors.csv"),
        help="Path to output CSV file",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date of the selected normal period, for example: 2023-08-01",
    )

    parser.add_argument(
        "--months",
        type=int,
        default=5,
        help="Number of months to select",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Number of rows per chunk",
    )

    parser.add_argument(
        "--min-value",
        type=float,
        default=1.0,
        help="Minimum value to keep. Use 1.0 to remove outage zeros.",
    )

    return parser.parse_args()


def month_start(date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=date.year, month=date.month, day=1)


def add_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
    return date + pd.DateOffset(months=months)


def prepare_chunk(
    chunk: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_value: float,
) -> pd.DataFrame:
    required_columns = {"sensor_id", "measured_at", "value"}
    missing_columns = required_columns - set(chunk.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in source CSV: {missing_columns}")

    chunk = chunk[["sensor_id", "measured_at", "value"]].copy()

    chunk["measured_at"] = pd.to_datetime(
        chunk["measured_at"],
        errors="coerce",
        utc=False,
    )

    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    chunk = chunk.dropna(subset=["measured_at", "value"])

    chunk = chunk[
        (chunk["measured_at"] >= start_date)
        & (chunk["measured_at"] < end_date)
        & (chunk["value"] >= min_value)
    ]

    if chunk.empty:
        return chunk

    # Нормализуем месяц до первого дня месяца.
    chunk["month"] = chunk["measured_at"].dt.to_period("M").dt.to_timestamp()

    # Каждый месяц превращаем в отдельный виртуальный датчик:
    # первый месяц -> 1, второй -> 2, ...
    chunk["sensor_id"] = (
        (chunk["month"].dt.year - start_date.year) * 12
        + (chunk["month"].dt.month - start_date.month)
        + 1
    )

    chunk = chunk[
        (chunk["sensor_id"] >= 1)
        & (chunk["sensor_id"] <= 5)
    ]

    if chunk.empty:
        return chunk

    # Чтобы разные виртуальные датчики были сопоставимы,
    # переносим время каждого месяца к общей временной оси.
    #
    # Например:
    # sensor_id=1: 2023-08-01 00:00:00 -> 2022-01-01 00:00:00
    # sensor_id=2: 2023-09-01 00:00:00 -> 2022-01-01 00:00:00
    #
    # Так у каждого виртуального датчика получается свой ряд за один "условный месяц".
    month_index = chunk["sensor_id"] - 1
    virtual_month_start = start_date + pd.to_timedelta(month_index * 0, unit="D")

    original_month_start = chunk["month"]
    time_offset = chunk["measured_at"] - original_month_start

    chunk["measured_at"] = start_date + time_offset

    result = chunk[["sensor_id", "measured_at", "value"]].copy()
    result["sensor_id"] = result["sensor_id"].astype("int32")
    result["value"] = result["value"].astype("float64")

    return result


def main() -> None:
    args = parse_args()

    start_date = month_start(pd.Timestamp(args.start_date))
    end_date = add_months(start_date, args.months)

    if args.months != 5:
        print(
            "Warning: this script is designed for 5 virtual sensors. "
            f"Current months value: {args.months}"
        )

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Selected period: {start_date} <= measured_at < {end_date}")
    print(f"Rows with value < {args.min_value} will be removed")
    print(f"Chunk size: {args.chunksize}")

    if args.output.exists():
        args.output.unlink()

    total_input_rows = 0
    total_output_rows = 0
    is_first_chunk = True

    sensor_counts = {}

    reader = pd.read_csv(args.input, chunksize=args.chunksize)

    for chunk_number, chunk in enumerate(reader, start=1):
        total_input_rows += len(chunk)

        prepared = prepare_chunk(
            chunk=chunk,
            start_date=start_date,
            end_date=end_date,
            min_value=args.min_value,
        )

        if not prepared.empty:
            prepared.to_csv(
                args.output,
                mode="w" if is_first_chunk else "a",
                header=is_first_chunk,
                index=False,
            )

            is_first_chunk = False
            total_output_rows += len(prepared)

            counts = prepared.groupby("sensor_id").size().to_dict()
            for sensor_id, count in counts.items():
                sensor_counts[sensor_id] = sensor_counts.get(sensor_id, 0) + count

        print(
            f"Chunk {chunk_number}: "
            f"input_rows={len(chunk)}, "
            f"saved_rows={len(prepared)}, "
            f"total_saved={total_output_rows}"
        )

    print()
    print("Finished")
    print(f"Total input rows read: {total_input_rows}")
    print(f"Total output rows saved: {total_output_rows}")
    print("Rows by virtual sensor:")

    for sensor_id in sorted(sensor_counts):
        print(f"sensor_id={sensor_id}: {sensor_counts[sensor_id]} rows")

    if total_output_rows == 0:
        print()
        print("Warning: output is empty. Check --start-date or source CSV date range.")


if __name__ == "__main__":
    main()