"""Tests for feature_extractor."""

import polars as pl
import pytest
import numpy as np

from insurance_telematics.feature_extractor import extract_trip_features
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.preprocessor import clean_trips


@pytest.fixture(scope="module")
def trip_features_df():
    sim = TripSimulator(seed=30)
    trips_df, _ = sim.simulate(n_drivers=5, trips_per_driver=10)
    cleaned = clean_trips(trips_df)
    return extract_trip_features(cleaned)


def test_returns_one_row_per_trip(trip_features_df):
    sim = TripSimulator(seed=31)
    trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=7)
    cleaned = clean_trips(trips_df)
    features = extract_trip_features(cleaned)
    expected_n_trips = trips_df["trip_id"].n_unique()
    assert len(features) == expected_n_trips


def test_required_output_columns(trip_features_df):
    expected = {
        "trip_id", "distance_km", "duration_min", "mean_speed_kmh",
        "p95_speed_kmh", "speed_variation_coeff", "harsh_braking_rate",
        "harsh_accel_rate", "harsh_cornering_rate", "speeding_fraction",
        "night_driving_fraction", "urban_fraction",
    }
    assert expected.issubset(set(trip_features_df.columns))


def test_driver_id_preserved(trip_features_df):
    assert "driver_id" in trip_features_df.columns


def test_distance_km_positive(trip_features_df):
    assert (trip_features_df["distance_km"] >= 0).all()


def test_duration_min_positive(trip_features_df):
    assert (trip_features_df["duration_min"] > 0).all()


def test_fractions_in_0_1(trip_features_df):
    for col in ["speeding_fraction", "night_driving_fraction", "urban_fraction"]:
        vals = trip_features_df[col].drop_nulls()
        assert (vals >= 0).all(), f"{col} has values < 0"
        assert (vals <= 1).all(), f"{col} has values > 1"


def test_rates_non_negative(trip_features_df):
    for col in ["harsh_braking_rate", "harsh_accel_rate", "harsh_cornering_rate"]:
        assert (trip_features_df[col] >= 0).all(), f"{col} has negative values"


def test_speed_variation_coeff_non_negative(trip_features_df):
    assert (trip_features_df["speed_variation_coeff"] >= 0).all()


def test_p95_speed_ge_mean(trip_features_df):
    """95th percentile speed must be >= mean speed."""
    p95 = trip_features_df["p95_speed_kmh"]
    mean = trip_features_df["mean_speed_kmh"]
    assert (p95 >= mean - 0.01).all()  # small tolerance for floating point


def test_missing_road_type_raises():
    df = pl.DataFrame({
        "trip_id": ["T1"],
        "timestamp": ["2024-01-01 08:00:00"],
        "speed_kmh": [50.0],
        "acceleration_ms2": [0.0],
        # road_type missing
    }).with_columns(pl.col("timestamp").str.to_datetime())
    with pytest.raises(ValueError, match="road_type"):
        extract_trip_features(df)


def test_high_speed_driver_has_lower_urban_fraction():
    """An aggressive (high-speed) driver should spend less time in urban band."""
    sim_slow = TripSimulator(seed=100)
    # Force cautious regime by using a custom simulation
    # We test directionally: moderate vs fast speed correlates with urban fraction
    sim = TripSimulator(seed=40)
    trips_df, _ = sim.simulate(n_drivers=10, trips_per_driver=20)
    cleaned = clean_trips(trips_df)
    features = extract_trip_features(cleaned)
    # Drivers with higher mean speed should have lower urban fraction
    corr = features.select([
        pl.col("mean_speed_kmh"),
        pl.col("urban_fraction"),
    ]).to_pandas().corr()
    # Correlation should be negative
    assert corr.loc["mean_speed_kmh", "urban_fraction"] < 0
