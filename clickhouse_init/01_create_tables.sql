CREATE DATABASE IF NOT EXISTS traffic_db;

CREATE TABLE IF NOT EXISTS traffic_db.measurements_history
(
    sensor_id UInt32,
    measured_at DateTime,
    value Float64
)
ENGINE = MergeTree
ORDER BY (sensor_id, measured_at);

CREATE TABLE IF NOT EXISTS traffic_db.traffic_window_aggregates
(
    sensor_id UInt32,
    window_start DateTime,
    window_end DateTime,

    records_count UInt64,
    avg_cars_per_hour Float64,
    min_cars_per_hour Float64,
    max_cars_per_hour Float64,
    stddev_cars_per_hour Float64,

    calculated_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (sensor_id, window_start);

CREATE TABLE IF NOT EXISTS traffic_db.traffic_local_regression
(
    sensor_id UInt32,
    window_start DateTime,
    window_end DateTime,

    records_count UInt64,
    slope_per_second Float64,
    intercept Float64,
    r2_score Float64,

    trend_label String,
    calculated_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (sensor_id, window_start);

CREATE TABLE IF NOT EXISTS traffic_db.pipeline_quality_checks
(
    check_name String,
    check_status String,
    details String,
    checked_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY checked_at;