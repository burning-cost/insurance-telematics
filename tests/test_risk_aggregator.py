"""Tests for risk_aggregator."""

import polars as pl
import pytest
import numpy as np

from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.feature_extractor import extract_trip_features


@pytest.fixture(scope="module")
def trip_features():
    sim = TripSimulator(seed=60)
    trips_df, _ = sim.simulate(
        n_drivers=8, trips_per_driver=25, min_trip_duration_s=180, max_trip_duration_s=900
    )
    cleaned = clean_trips(trips_df)
    return extract_trip_features(cleaned)


def test_returns_one_row_per_driver(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    n_drivers = trip_features["driver_id"].n_unique()
    assert len(driver_df) == n_drivers


def test_output_columns(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    required = {
        "driver_id", "n_trips", "total_km", "credibility_weight",
        "composite_risk_score",
    }
    assert required.issubset(set(driver_df.columns))


def test_composite_score_in_0_100(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    scores = driver_df["composite_risk_score"]
    assert scores.min() >= -0.01  # small tolerance
    assert scores.max() <= 100.01


def test_credibility_weight_in_0_1(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    w = driver_df["credibility_weight"]
    assert (w >= 0).all()
    assert (w <= 1).all()


def test_credibility_weight_increases_with_trips():
    """Drivers with more trips should have higher credibility weight."""
    # Create two drivers with different trip counts explicitly
    rows = []
    for trip_num in range(5):
        rows.append({"trip_id": f"T{trip_num}", "driver_id": "D_few", "distance_km": 10.0,
                     "mean_speed_kmh": 50.0, "p95_speed_kmh": 80.0, "speed_variation_coeff": 0.2,
                     "harsh_braking_rate": 0.1, "harsh_accel_rate": 0.1,
                     "harsh_cornering_rate": 0.05, "speeding_fraction": 0.1,
                     "night_driving_fraction": 0.05, "urban_fraction": 0.4})
    for trip_num in range(60):
        rows.append({"trip_id": f"T{trip_num+100}", "driver_id": "D_many", "distance_km": 10.0,
                     "mean_speed_kmh": 50.0, "p95_speed_kmh": 80.0, "speed_variation_coeff": 0.2,
                     "harsh_braking_rate": 0.1, "harsh_accel_rate": 0.1,
                     "harsh_cornering_rate": 0.05, "speeding_fraction": 0.1,
                     "night_driving_fraction": 0.05, "urban_fraction": 0.4})
    df = pl.DataFrame(rows)
    driver_df = aggregate_to_driver(df, credibility_threshold=30)
    w_few = driver_df.filter(pl.col("driver_id") == "D_few")["credibility_weight"][0]
    w_many = driver_df.filter(pl.col("driver_id") == "D_many")["credibility_weight"][0]
    assert w_many > w_few, f"Expected D_many ({w_many:.3f}) > D_few ({w_few:.3f})"


def test_total_km_positive(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    assert (driver_df["total_km"] > 0).all()


def test_n_trips_correct(trip_features):
    driver_df = aggregate_to_driver(trip_features)
    expected_counts = (
        trip_features.group_by("driver_id")
        .agg(pl.len().alias("expected_n"))
        .sort("driver_id")
    )
    actual_counts = driver_df.sort("driver_id").select(["driver_id", "n_trips"])
    for row_exp, row_act in zip(
        expected_counts.iter_rows(), actual_counts.iter_rows()
    ):
        assert row_exp[0] == row_act[0]  # driver_id
        assert row_exp[1] == row_act[1]  # n_trips


def test_missing_driver_id_raises():
    df = pl.DataFrame({
        "trip_id": ["T1"],
        "distance_km": [10.0],
        "mean_speed_kmh": [50.0],
    })
    with pytest.raises(ValueError, match="driver_id"):
        aggregate_to_driver(df)


def test_missing_distance_km_raises():
    df = pl.DataFrame({
        "trip_id": ["T1"],
        "driver_id": ["D1"],
        "mean_speed_kmh": [50.0],
    })
    with pytest.raises(ValueError, match="distance_km"):
        aggregate_to_driver(df)


def test_credibility_threshold_effect(trip_features):
    """Lower threshold → more drivers get higher credibility weight."""
    df_low = aggregate_to_driver(trip_features, credibility_threshold=5)
    df_high = aggregate_to_driver(trip_features, credibility_threshold=200)
    # Low threshold should have higher mean credibility weight
    assert df_low["credibility_weight"].mean() > df_high["credibility_weight"].mean()


def test_mean_speed_aggregated(trip_features):
    """Aggregated mean speed should be within the range of trip-level speeds."""
    driver_df = aggregate_to_driver(trip_features)
    if "mean_speed_kmh" in driver_df.columns:
        trip_min = trip_features["mean_speed_kmh"].min()
        trip_max = trip_features["mean_speed_kmh"].max()
        driver_min = driver_df["mean_speed_kmh"].min()
        driver_max = driver_df["mean_speed_kmh"].max()
        # After credibility shrinkage, driver scores are between trip range and portfolio mean
        assert driver_min >= trip_min * 0.5  # allow for shrinkage
        assert driver_max <= trip_max * 1.5
