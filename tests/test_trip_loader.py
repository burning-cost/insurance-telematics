"""Tests for trip_loader."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from insurance_telematics.trip_loader import (
    load_trips,
    load_trips_from_dataframe,
    REQUIRED_COLUMNS,
)
from insurance_telematics.trip_simulator import TripSimulator


@pytest.fixture(scope="module")
def sample_trips_df():
    sim = TripSimulator(seed=10)
    trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
    return trips_df


def test_load_from_csv(sample_trips_df, tmp_path):
    csv_path = tmp_path / "trips.csv"
    sample_trips_df.write_csv(csv_path)
    loaded = load_trips(csv_path)
    assert loaded.shape[0] == sample_trips_df.shape[0]
    for col in REQUIRED_COLUMNS:
        assert col in loaded.columns


def test_load_from_parquet(sample_trips_df, tmp_path):
    pq_path = tmp_path / "trips.parquet"
    sample_trips_df.write_parquet(pq_path)
    loaded = load_trips(pq_path)
    assert loaded.shape[0] == sample_trips_df.shape[0]


def test_load_from_directory_of_parquets(sample_trips_df, tmp_path):
    # Split into two parquet files
    half = len(sample_trips_df) // 2
    sample_trips_df[:half].write_parquet(tmp_path / "part1.parquet")
    sample_trips_df[half:].write_parquet(tmp_path / "part2.parquet")
    loaded = load_trips(tmp_path)
    assert loaded.shape[0] == sample_trips_df.shape[0]


def test_missing_required_column_raises(sample_trips_df, tmp_path):
    bad_df = sample_trips_df.drop("speed_kmh")
    pq_path = tmp_path / "bad.parquet"
    bad_df.write_parquet(pq_path)
    with pytest.raises(ValueError, match="speed_kmh"):
        load_trips(pq_path)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_trips("/nonexistent/path/trips.csv")


def test_unsupported_format_raises(tmp_path):
    bad_path = tmp_path / "trips.xlsx"
    bad_path.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_trips(bad_path)


def test_schema_rename(sample_trips_df, tmp_path):
    # Rename speed_kmh to gps_speed in the file, then use schema to map it back
    renamed = sample_trips_df.rename({"speed_kmh": "gps_speed"})
    pq_path = tmp_path / "renamed.parquet"
    renamed.write_parquet(pq_path)
    loaded = load_trips(pq_path, schema={"gps_speed": "speed_kmh"})
    assert "speed_kmh" in loaded.columns


def test_missing_optional_columns_added_as_null(tmp_path):
    # DataFrame without optional columns
    minimal_df = pl.DataFrame({
        "trip_id": ["T1", "T1", "T2"],
        "timestamp": ["2024-01-01 08:00:00", "2024-01-01 08:00:01", "2024-01-01 09:00:00"],
        "latitude": [51.5, 51.5, 51.6],
        "longitude": [-0.1, -0.1, -0.2],
        "speed_kmh": [30.0, 31.0, 50.0],
    })
    pq_path = tmp_path / "minimal.parquet"
    minimal_df.write_parquet(pq_path)
    loaded = load_trips(pq_path)
    assert "acceleration_ms2" in loaded.columns
    assert "heading_deg" in loaded.columns
    assert "driver_id" in loaded.columns
    # driver_id should be "unknown"
    assert (loaded["driver_id"] == "unknown").all()


def test_load_from_dataframe_validates():
    bad_df = pl.DataFrame({"trip_id": ["T1"], "speed_kmh": [50.0]})
    with pytest.raises(ValueError, match="latitude"):
        load_trips_from_dataframe(bad_df)


def test_sorted_by_trip_and_timestamp(sample_trips_df):
    loaded = load_trips_from_dataframe(sample_trips_df)
    trip_ids = loaded["trip_id"].to_list()
    # Check that trip_ids are non-decreasing
    for i in range(1, len(trip_ids)):
        assert trip_ids[i] >= trip_ids[i - 1]
