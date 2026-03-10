"""Tests for TripSimulator."""

import polars as pl
import pytest
from insurance_telematics.trip_simulator import TripSimulator


def test_simulate_returns_tuple():
    sim = TripSimulator(seed=1)
    result = sim.simulate(n_drivers=3, trips_per_driver=5)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_trips_dataframe_schema():
    sim = TripSimulator(seed=1)
    trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
    required_cols = {
        "driver_id", "trip_id", "timestamp", "latitude", "longitude",
        "speed_kmh", "acceleration_ms2", "heading_deg",
    }
    assert required_cols.issubset(set(trips_df.columns))


def test_claims_dataframe_schema():
    sim = TripSimulator(seed=1)
    _, claims_df = sim.simulate(n_drivers=3, trips_per_driver=5)
    required_cols = {"driver_id", "n_claims", "exposure_years", "aggressive_fraction"}
    assert required_cols.issubset(set(claims_df.columns))


def test_n_drivers_matches():
    n = 7
    sim = TripSimulator(seed=2)
    trips_df, claims_df = sim.simulate(n_drivers=n, trips_per_driver=3)
    assert claims_df["driver_id"].n_unique() == n
    assert trips_df["driver_id"].n_unique() == n


def test_trip_count_per_driver():
    sim = TripSimulator(seed=3)
    trips_per_driver = 10
    trips_df, _ = sim.simulate(n_drivers=4, trips_per_driver=trips_per_driver)
    counts = trips_df.group_by("driver_id").agg(pl.col("trip_id").n_unique())
    assert (counts["trip_id"] == trips_per_driver).all()


def test_speed_non_negative():
    sim = TripSimulator(seed=4)
    trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=5)
    assert (trips_df["speed_kmh"] >= 0).all()


def test_reproducibility():
    sim1 = TripSimulator(seed=99)
    sim2 = TripSimulator(seed=99)
    df1, _ = sim1.simulate(n_drivers=3, trips_per_driver=5)
    df2, _ = sim2.simulate(n_drivers=3, trips_per_driver=5)
    assert df1["speed_kmh"].sum() == df2["speed_kmh"].sum()


def test_different_seeds_differ():
    sim1 = TripSimulator(seed=1)
    sim2 = TripSimulator(seed=2)
    df1, _ = sim1.simulate(n_drivers=3, trips_per_driver=5)
    df2, _ = sim2.simulate(n_drivers=3, trips_per_driver=5)
    assert df1["speed_kmh"].sum() != df2["speed_kmh"].sum()


def test_aggressive_fraction_in_01():
    sim = TripSimulator(seed=5)
    _, claims_df = sim.simulate(n_drivers=10, trips_per_driver=5)
    assert (claims_df["aggressive_fraction"] >= 0).all()
    assert (claims_df["aggressive_fraction"] <= 1).all()


def test_claims_non_negative():
    sim = TripSimulator(seed=6)
    _, claims_df = sim.simulate(n_drivers=10, trips_per_driver=5)
    assert (claims_df["n_claims"] >= 0).all()


def test_timestamp_dtype():
    sim = TripSimulator(seed=7)
    trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
    assert trips_df["timestamp"].dtype in (
        pl.Datetime("us", "UTC"), pl.Datetime("ns", "UTC"), pl.Datetime
    )


def test_trip_ids_globally_unique():
    sim = TripSimulator(seed=8)
    trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=4)
    n_trips = 3 * 4
    assert trips_df["trip_id"].n_unique() == n_trips


def test_custom_duration_bounds():
    sim = TripSimulator(seed=9)
    trips_df, _ = sim.simulate(
        n_drivers=2,
        trips_per_driver=5,
        min_trip_duration_s=60,
        max_trip_duration_s=120,
    )
    # Each trip should have between 60 and 120 rows
    trip_lengths = trips_df.group_by("trip_id").agg(pl.len().alias("n"))
    assert (trip_lengths["n"] >= 60).all()
    assert (trip_lengths["n"] <= 120).all()
