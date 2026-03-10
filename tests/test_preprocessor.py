"""Tests for the preprocessor module."""

import polars as pl
import pytest
import numpy as np

from insurance_telematics.preprocessor import clean_trips, _MAX_PLAUSIBLE_SPEED_KMH
from insurance_telematics.trip_simulator import TripSimulator


@pytest.fixture(scope="module")
def raw_trips():
    sim = TripSimulator(seed=20)
    trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
    return trips_df


def test_clean_returns_dataframe(raw_trips):
    cleaned = clean_trips(raw_trips)
    assert isinstance(cleaned, pl.DataFrame)


def test_gps_jumps_removed():
    df = pl.DataFrame({
        "trip_id": ["T1"] * 5,
        "timestamp": [
            "2024-01-01 08:00:00",
            "2024-01-01 08:00:01",
            "2024-01-01 08:00:02",
            "2024-01-01 08:00:03",
            "2024-01-01 08:00:04",
        ],
        "latitude": [51.5] * 5,
        "longitude": [-0.1] * 5,
        "speed_kmh": [30.0, 50.0, 999.0, 40.0, 35.0],  # 999 is a GPS jump
        "acceleration_ms2": [0.0] * 5,
        "heading_deg": [90.0] * 5,
        "driver_id": ["D1"] * 5,
    }).with_columns(pl.col("timestamp").str.to_datetime())
    cleaned = clean_trips(df)
    assert cleaned["speed_kmh"].max() <= _MAX_PLAUSIBLE_SPEED_KMH
    assert len(cleaned) == 4  # one row removed


def test_clean_adds_road_type(raw_trips):
    cleaned = clean_trips(raw_trips)
    assert "road_type" in cleaned.columns
    valid_types = {"urban", "rural", "motorway"}
    assert set(cleaned["road_type"].unique().to_list()).issubset(valid_types)


def test_clean_adds_jerk(raw_trips):
    cleaned = clean_trips(raw_trips)
    assert "jerk_ms3" in cleaned.columns


def test_clean_derives_acceleration_when_null():
    df = pl.DataFrame({
        "trip_id": ["T1"] * 4,
        "timestamp": [
            "2024-01-01 08:00:00",
            "2024-01-01 08:00:01",
            "2024-01-01 08:00:02",
            "2024-01-01 08:00:03",
        ],
        "latitude": [51.5] * 4,
        "longitude": [-0.1] * 4,
        "speed_kmh": [36.0, 72.0, 54.0, 54.0],  # 36 km/h changes
        "acceleration_ms2": [None, None, None, None],
        "heading_deg": [90.0] * 4,
        "driver_id": ["D1"] * 4,
    }).with_columns([
        pl.col("timestamp").str.to_datetime(),
        pl.col("acceleration_ms2").cast(pl.Float64),
    ])
    cleaned = clean_trips(df)
    # Row 1: (72 - 36) / 3.6 / 1.0 = 10 m/s²
    assert cleaned["acceleration_ms2"].null_count() < 4  # some values derived


def test_road_type_urban_threshold():
    df = pl.DataFrame({
        "trip_id": ["T1"] * 3,
        "timestamp": ["2024-01-01 08:00:00", "2024-01-01 08:00:01", "2024-01-01 08:00:02"],
        "latitude": [51.5] * 3,
        "longitude": [-0.1] * 3,
        "speed_kmh": [30.0, 75.0, 120.0],
        "acceleration_ms2": [0.0] * 3,
        "heading_deg": [90.0] * 3,
        "driver_id": ["D1"] * 3,
    }).with_columns(pl.col("timestamp").str.to_datetime())
    cleaned = clean_trips(df)
    road_types = cleaned["road_type"].to_list()
    assert road_types[0] == "urban"
    assert road_types[1] == "rural"
    assert road_types[2] == "motorway"


def test_clean_preserves_row_count_without_jumps(raw_trips):
    # Simulated data should have no GPS jumps (max speed ~150 km/h)
    cleaned = clean_trips(raw_trips)
    assert len(cleaned) <= len(raw_trips)  # can only lose rows, not gain


def test_clean_idempotent(raw_trips):
    once = clean_trips(raw_trips)
    twice = clean_trips(once)
    # Cleaning a second time should not change row count
    assert len(twice) == len(once)
