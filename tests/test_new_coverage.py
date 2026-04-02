"""
Comprehensive coverage tests for insurance-telematics.

Targets gaps in existing coverage:
- feature_extractor: night hours, speeding thresholds by road type, single-row,
  multi-trip correctness, distance computation
- preprocessor: exact acceleration values, jerk computation, boundary speeds,
  acceleration when column present but fully null, no-acceleration-column path
- hmm_model: DrivingStateHMM 4-state, state ordering math, predict_state_probs
  reordering, driver_state_features single-trip driver, transition count
- risk_aggregator: distance-weighted mean correctness, credibility formula,
  shrinkage direction, two-driver boundary
- trip_loader: multi-column schema mapping, load_trips_from_dataframe with schema,
  sorted output guarantee, .pq extension
- trip_simulator: heading bounded, lat/lon are floats, claims fraction columns
- scoring_pipeline: glm_feature_subset, glm_features before fit, score_trips alias
- ContinuousTimeHMM: _transition_matrix properties, _emission_log_prob shape,
  two-state boundary
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from insurance_telematics.feature_extractor import (
    extract_trip_features,
    _HARSH_BRAKE_THRESHOLD,
    _HARSH_ACCEL_THRESHOLD,
    _SPEED_LIMIT_URBAN,
    _SPEED_LIMIT_RURAL,
    _SPEED_LIMIT_MOTORWAY,
    _NIGHT_HOURS,
)
from insurance_telematics.hmm_model import (
    DrivingStateHMM,
    ContinuousTimeHMM,
    _logsumexp,
)
from insurance_telematics.preprocessor import (
    clean_trips,
    _URBAN_MAX_KMH,
    _RURAL_MAX_KMH,
    _MAX_PLAUSIBLE_SPEED_KMH,
)
from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.scoring_pipeline import TelematicsScoringPipeline, score_trips
from insurance_telematics.trip_loader import (
    load_trips,
    load_trips_from_dataframe,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    ALL_SCHEMA_COLUMNS,
)
from insurance_telematics.trip_simulator import TripSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cleaned_trip(
    trip_id: str,
    speed_values: list[float],
    road_type: str = "urban",
    driver_id: str = "D1",
    hour: int = 10,
) -> pl.DataFrame:
    """Minimal already-cleaned trip for feature extraction."""
    from datetime import timedelta
    n = len(speed_values)
    base_ts = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_ts + timedelta(seconds=i) for i in range(n)]
    return pl.DataFrame({
        "trip_id": [trip_id] * n,
        "timestamp": timestamps,
        "speed_kmh": speed_values,
        "acceleration_ms2": [0.0] * n,
        "road_type": [road_type] * n,
        "driver_id": [driver_id] * n,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _make_raw_trip(
    trip_id: str,
    speed_values: list[float],
    driver_id: str = "D1",
    accel_values: list[float] | None = None,
) -> pl.DataFrame:
    """Minimal raw trip for preprocessor tests."""
    n = len(speed_values)
    timestamps = [
        datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
        for i in range(n)
    ]
    data = {
        "trip_id": [trip_id] * n,
        "timestamp": timestamps,
        "latitude": [51.5] * n,
        "longitude": [-0.1] * n,
        "speed_kmh": speed_values,
        "heading_deg": [90.0] * n,
        "driver_id": [driver_id] * n,
    }
    if accel_values is not None:
        data["acceleration_ms2"] = accel_values
    return pl.DataFrame(data).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


def _make_trip_features(
    driver_id: str,
    n_trips: int,
    mean_speed: float = 50.0,
    harsh_braking_rate: float = 0.1,
    speeding_fraction: float = 0.1,
    distance_km: float = 10.0,
) -> list[dict]:
    return [
        {
            "trip_id": f"{driver_id}_T{i}",
            "driver_id": driver_id,
            "distance_km": distance_km,
            "mean_speed_kmh": mean_speed,
            "p95_speed_kmh": mean_speed * 1.3,
            "speed_variation_coeff": 0.2,
            "harsh_braking_rate": harsh_braking_rate,
            "harsh_accel_rate": harsh_braking_rate * 0.8,
            "harsh_cornering_rate": 0.05,
            "speeding_fraction": speeding_fraction,
            "night_driving_fraction": 0.05,
            "urban_fraction": 0.4,
        }
        for i in range(n_trips)
    ]


# ===========================================================================
# 1. feature_extractor — night driving detection
# ===========================================================================

class TestNightDrivingDetection:
    """Night hours are {23, 0, 1, 2, 3, 4}."""

    def test_night_hours_constant_is_correct(self):
        assert _NIGHT_HOURS == {23, 0, 1, 2, 3, 4}

    def test_pure_night_trip_has_fraction_one(self):
        # All observations at 01:00 UTC
        df = _make_cleaned_trip("T1", [50.0] * 30, hour=1)
        feats = extract_trip_features(df)
        assert feats["night_driving_fraction"][0] == pytest.approx(1.0)

    def test_pure_day_trip_has_fraction_zero(self):
        # All observations at 14:00 UTC
        df = _make_cleaned_trip("T1", [50.0] * 30, hour=14)
        feats = extract_trip_features(df)
        assert feats["night_driving_fraction"][0] == pytest.approx(0.0)

    def test_midnight_hour_is_night(self):
        # hour=0 is in _NIGHT_HOURS
        df = _make_cleaned_trip("T1", [50.0] * 20, hour=0)
        feats = extract_trip_features(df)
        assert feats["night_driving_fraction"][0] == pytest.approx(1.0)

    def test_hour_23_is_night(self):
        df = _make_cleaned_trip("T1", [50.0] * 20, hour=23)
        feats = extract_trip_features(df)
        assert feats["night_driving_fraction"][0] == pytest.approx(1.0)

    def test_hour_5_is_day(self):
        # 05:00 is not in _NIGHT_HOURS
        df = _make_cleaned_trip("T1", [50.0] * 20, hour=5)
        feats = extract_trip_features(df)
        assert feats["night_driving_fraction"][0] == pytest.approx(0.0)


# ===========================================================================
# 2. feature_extractor — road-type-specific speeding thresholds
# ===========================================================================

class TestSpeedingThresholds:
    """Each road type has a different speed limit for flagging speeding."""

    def test_urban_speeding_above_threshold(self):
        # Speed just above urban limit
        speed = _SPEED_LIMIT_URBAN + 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="urban")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(1.0)

    def test_urban_not_speeding_below_threshold(self):
        speed = _SPEED_LIMIT_URBAN - 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="urban")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(0.0)

    def test_rural_speeding_above_threshold(self):
        speed = _SPEED_LIMIT_RURAL + 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="rural")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(1.0)

    def test_rural_not_speeding_at_threshold(self):
        speed = _SPEED_LIMIT_RURAL - 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="rural")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(0.0)

    def test_motorway_speeding_above_threshold(self):
        speed = _SPEED_LIMIT_MOTORWAY + 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="motorway")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(1.0)

    def test_motorway_not_speeding_at_limit(self):
        speed = _SPEED_LIMIT_MOTORWAY - 1.0
        df = _make_cleaned_trip("T1", [speed] * 30, road_type="motorway")
        feats = extract_trip_features(df)
        assert feats["speeding_fraction"][0] == pytest.approx(0.0)

    def test_mixed_road_types_partial_speeding(self):
        # Half urban (speeding), half rural (not speeding at same speed)
        n = 40
        # Urban speed 40 km/h > 35 (urban limit) but < 96 (rural limit)
        speed = 40.0
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        road_types = ["urban"] * (n // 2) + ["rural"] * (n // 2)
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [speed] * n,
            "acceleration_ms2": [0.0] * n,
            "road_type": road_types,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        # Half the rows are urban-speeding (40 > 35)
        assert feats["speeding_fraction"][0] == pytest.approx(0.5)


# ===========================================================================
# 3. feature_extractor — harsh event thresholds
# ===========================================================================

class TestHarshEventThresholds:
    """Exactly at threshold should NOT trigger; one unit past should."""

    def test_harsh_braking_threshold_exact(self):
        # Exactly at threshold → not flagged
        accel = _HARSH_BRAKE_THRESHOLD  # exactly -3.5
        n = 30
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [accel] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        # == threshold is NOT < threshold, so not flagged
        assert feats["harsh_braking_rate"][0] == pytest.approx(0.0)

    def test_harsh_braking_below_threshold(self):
        accel = _HARSH_BRAKE_THRESHOLD - 0.1  # worse than threshold
        n = 30
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [accel] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        assert feats["harsh_braking_rate"][0] > 0.0

    def test_harsh_accel_threshold_exact(self):
        accel = _HARSH_ACCEL_THRESHOLD  # exactly 3.5
        n = 30
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [accel] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        # == threshold is NOT > threshold, so not flagged
        assert feats["harsh_accel_rate"][0] == pytest.approx(0.0)

    def test_harsh_accel_above_threshold(self):
        accel = _HARSH_ACCEL_THRESHOLD + 0.1
        n = 30
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [accel] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        assert feats["harsh_accel_rate"][0] > 0.0


# ===========================================================================
# 4. feature_extractor — distance computation
# ===========================================================================

class TestDistanceComputation:
    """Distance = sum(speed_kmh) / 3600 km at 1Hz."""

    def test_constant_speed_distance(self):
        speed = 72.0  # km/h
        n = 100  # seconds
        expected_km = speed * n / 3600.0
        df = _make_cleaned_trip("T1", [speed] * n)
        feats = extract_trip_features(df)
        assert feats["distance_km"][0] == pytest.approx(expected_km, rel=1e-6)

    def test_zero_speed_distance_is_zero(self):
        df = _make_cleaned_trip("T1", [0.0] * 50)
        feats = extract_trip_features(df)
        assert feats["distance_km"][0] == pytest.approx(0.0, abs=1e-10)

    def test_duration_min_equals_n_obs_over_60(self):
        n = 120  # seconds
        df = _make_cleaned_trip("T1", [50.0] * n)
        feats = extract_trip_features(df)
        assert feats["duration_min"][0] == pytest.approx(n / 60.0, rel=1e-6)


# ===========================================================================
# 5. feature_extractor — multiple trips, no driver_id
# ===========================================================================

class TestMultiTripNoDriverId:
    def test_no_driver_id_column_excluded_from_output(self):
        n = 30
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["rural"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feats = extract_trip_features(df)
        assert "driver_id" not in feats.columns

    def test_multiple_trips_sorted_by_trip_id(self):
        dfs = []
        for trip_id in ["T3", "T1", "T2"]:
            dfs.append(_make_cleaned_trip(trip_id, [50.0] * 30))
        combined = pl.concat(dfs)
        feats = extract_trip_features(combined)
        trip_ids = feats["trip_id"].to_list()
        assert trip_ids == sorted(trip_ids)

    def test_urban_fraction_pure_urban(self):
        df = _make_cleaned_trip("T1", [30.0] * 40, road_type="urban")
        feats = extract_trip_features(df)
        assert feats["urban_fraction"][0] == pytest.approx(1.0)

    def test_urban_fraction_pure_motorway(self):
        df = _make_cleaned_trip("T1", [120.0] * 40, road_type="motorway")
        feats = extract_trip_features(df)
        assert feats["urban_fraction"][0] == pytest.approx(0.0)


# ===========================================================================
# 6. feature_extractor — missing column errors
# ===========================================================================

class TestFeatureExtractorErrors:
    def test_missing_trip_id_raises(self):
        n = 10
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["urban"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        with pytest.raises(ValueError, match="trip_id"):
            extract_trip_features(df)

    def test_missing_speed_kmh_raises(self):
        n = 10
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["urban"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        with pytest.raises(ValueError, match="speed_kmh"):
            extract_trip_features(df)

    def test_missing_timestamp_raises(self):
        n = 10
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["urban"] * n,
        })
        with pytest.raises(ValueError, match="timestamp"):
            extract_trip_features(df)


# ===========================================================================
# 7. preprocessor — exact acceleration derivation
# ===========================================================================

class TestAccelerationDerivation:
    """
    From 1Hz speed differences: dv_ms2 = (v2 - v1) km/h / 3.6
    """

    def test_acceleration_computed_from_speed_when_absent(self):
        # 36 km/h → 72 km/h in 1s → accel = (72-36)/3.6 = 10.0 m/s²
        df = _make_raw_trip("T1", [36.0, 72.0])
        cleaned = clean_trips(df)
        # Row index 1 should have derived accel ≈ 10.0 m/s²
        accels = cleaned["acceleration_ms2"].to_list()
        # Row 0 is NaN (diff undefined at first row), row 1 should be ~10
        assert accels[1] == pytest.approx(10.0, abs=0.01)

    def test_existing_accel_preserved_when_not_null(self):
        # Provided acceleration values should be kept as-is
        accels = [2.0, 1.5, -0.5, -1.0]
        df = _make_raw_trip("T1", [30.0, 40.0, 50.0, 48.0], accel_values=accels)
        cleaned = clean_trips(df)
        # Non-null provided values should be preserved
        result_accels = cleaned["acceleration_ms2"].drop_nulls().to_list()
        for v in accels:
            assert v in result_accels

    def test_null_accel_replaced_by_derived(self):
        # Provide accel column with nulls; should be filled with derived values
        df = _make_raw_trip("T1", [36.0, 72.0, 108.0], accel_values=[None, None, None])
        df = df.with_columns(pl.col("acceleration_ms2").cast(pl.Float64))
        cleaned = clean_trips(df)
        non_null_count = cleaned["acceleration_ms2"].drop_nulls().shape[0]
        # Rows 1 and 2 should have derived values (row 0 stays null from diff)
        assert non_null_count >= 2


# ===========================================================================
# 8. preprocessor — jerk computation
# ===========================================================================

class TestJerkComputation:
    def test_jerk_column_present(self):
        df = _make_raw_trip("T1", [30.0, 40.0, 50.0, 60.0])
        cleaned = clean_trips(df)
        assert "jerk_ms3" in cleaned.columns

    def test_jerk_is_diff_of_acceleration(self):
        # Linear speed increase → constant accel → zero jerk after first diff
        speeds = [0.0, 36.0, 72.0, 108.0, 144.0]  # +36 km/h each second
        df = _make_raw_trip("T1", speeds)
        cleaned = clean_trips(df)
        # acceleration ≈ 10.0 m/s² constant → jerk ≈ 0.0 after first 2 rows
        jerks = cleaned["jerk_ms3"].drop_nulls().to_list()
        # All interior jerks should be near zero (constant accel)
        for j in jerks[1:]:  # skip first (diff from null)
            assert abs(j) < 0.5, f"Expected near-zero jerk for constant accel, got {j}"

    def test_jerk_first_row_is_null(self):
        df = _make_raw_trip("T1", [30.0] * 5)
        cleaned = clean_trips(df)
        # jerk = diff of accel, so first row is null
        assert cleaned["jerk_ms3"][0] is None


# ===========================================================================
# 9. preprocessor — road type boundary speeds
# ===========================================================================

class TestRoadTypeBoundaries:
    def test_exactly_urban_max_is_rural(self):
        # speed == 50.0 → NOT < 50, so rural
        df = _make_raw_trip("T1", [_URBAN_MAX_KMH] * 3)
        cleaned = clean_trips(df)
        assert all(rt == "rural" for rt in cleaned["road_type"].to_list())

    def test_just_below_urban_max_is_urban(self):
        df = _make_raw_trip("T1", [_URBAN_MAX_KMH - 0.1] * 3)
        cleaned = clean_trips(df)
        assert all(rt == "urban" for rt in cleaned["road_type"].to_list())

    def test_exactly_rural_max_is_rural(self):
        # speed == 100.0 → <= 100, so rural
        df = _make_raw_trip("T1", [_RURAL_MAX_KMH] * 3)
        cleaned = clean_trips(df)
        assert all(rt == "rural" for rt in cleaned["road_type"].to_list())

    def test_just_above_rural_max_is_motorway(self):
        df = _make_raw_trip("T1", [_RURAL_MAX_KMH + 0.1] * 3)
        cleaned = clean_trips(df)
        assert all(rt == "motorway" for rt in cleaned["road_type"].to_list())

    def test_zero_speed_is_urban(self):
        df = _make_raw_trip("T1", [0.0] * 3)
        cleaned = clean_trips(df)
        assert all(rt == "urban" for rt in cleaned["road_type"].to_list())

    def test_gps_jump_exactly_at_limit_is_kept(self):
        # _MAX_PLAUSIBLE_SPEED_KMH (250) should be retained (≤ 250)
        df = _make_raw_trip("T1", [_MAX_PLAUSIBLE_SPEED_KMH] * 3)
        cleaned = clean_trips(df)
        assert len(cleaned) == 3

    def test_gps_jump_above_limit_removed(self):
        df = _make_raw_trip("T1", [_MAX_PLAUSIBLE_SPEED_KMH + 0.1] * 3)
        cleaned = clean_trips(df)
        assert len(cleaned) == 0


# ===========================================================================
# 10. preprocessor — multiple trips in same DataFrame
# ===========================================================================

class TestPreprocessorMultipleTrips:
    def test_road_type_computed_per_trip(self):
        """Road type assigned independently per row — trip identity doesn't matter."""
        df1 = _make_raw_trip("T1", [30.0] * 5)
        df2 = _make_raw_trip("T2", [120.0] * 5)
        combined = pl.concat([df1, df2])
        cleaned = clean_trips(combined)
        t1_types = cleaned.filter(pl.col("trip_id") == "T1")["road_type"].unique().to_list()
        t2_types = cleaned.filter(pl.col("trip_id") == "T2")["road_type"].unique().to_list()
        assert t1_types == ["urban"]
        assert t2_types == ["motorway"]

    def test_acceleration_derived_per_trip(self):
        """diff().over(trip_id) means first row of each trip is null."""
        df1 = _make_raw_trip("T1", [36.0, 72.0])
        df2 = _make_raw_trip("T2", [50.0, 86.0])
        combined = pl.concat([df1, df2])
        cleaned = clean_trips(combined)
        # Each trip's first row should be null for jerk
        t1_clean = cleaned.filter(pl.col("trip_id") == "T1")
        t2_clean = cleaned.filter(pl.col("trip_id") == "T2")
        assert t1_clean["jerk_ms3"][0] is None
        assert t2_clean["jerk_ms3"][0] is None


# ===========================================================================
# 11. risk_aggregator — distance-weighted mean correctness
# ===========================================================================

class TestDistanceWeightedMean:
    def test_distance_weighted_mean_is_correct(self):
        """
        With two trips of different distances, the weighted mean should weight
        by distance, not give equal weight to each trip.
        """
        rows = [
            {
                "trip_id": "T1", "driver_id": "D1", "distance_km": 100.0,
                "mean_speed_kmh": 100.0, "p95_speed_kmh": 130.0,
                "speed_variation_coeff": 0.1, "harsh_braking_rate": 0.0,
                "harsh_accel_rate": 0.0, "harsh_cornering_rate": 0.0,
                "speeding_fraction": 0.0, "night_driving_fraction": 0.0,
                "urban_fraction": 0.0,
            },
            {
                "trip_id": "T2", "driver_id": "D1", "distance_km": 10.0,
                "mean_speed_kmh": 10.0, "p95_speed_kmh": 15.0,
                "speed_variation_coeff": 0.3, "harsh_braking_rate": 1.0,
                "harsh_accel_rate": 1.0, "harsh_cornering_rate": 0.5,
                "speeding_fraction": 0.5, "night_driving_fraction": 0.3,
                "urban_fraction": 0.8,
            },
        ]
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=1)
        # After credibility shrinkage with threshold=1 and n=2,
        # weight = 2/(2+1) = 0.667, but correctness of weighted mean before shrinkage
        # Weighted mean (without shrinkage): (100*100 + 10*10) / (100+10) = 10100/110 ≈ 91.8
        weighted_mean = (100.0 * 100.0 + 10.0 * 10.0) / 110.0
        # After shrinkage, value is between weighted_mean and portfolio_mean
        # Since there's only one driver, portfolio_mean IS the weighted_mean
        result_speed = driver_df["mean_speed_kmh"][0]
        # With single driver, portfolio_mean = result, shrinkage doesn't change value
        assert result_speed == pytest.approx(weighted_mean, rel=0.01)

    def test_total_km_is_sum(self):
        rows = _make_trip_features("D1", 5, distance_km=20.0)
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df)
        assert driver_df["total_km"][0] == pytest.approx(100.0)

    def test_n_trips_exact(self):
        rows = _make_trip_features("D1", 7)
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df)
        assert driver_df["n_trips"][0] == 7


# ===========================================================================
# 12. risk_aggregator — Bühlmann-Straub credibility formula
# ===========================================================================

class TestCredibilityFormula:
    def test_credibility_formula_n_over_n_plus_k(self):
        """credibility_weight = n / (n + k) for credibility_threshold k."""
        rows = _make_trip_features("D1", 10)
        df = pl.DataFrame(rows)
        k = 30
        driver_df = aggregate_to_driver(df, credibility_threshold=k)
        expected_weight = 10 / (10 + k)
        assert driver_df["credibility_weight"][0] == pytest.approx(expected_weight, rel=1e-6)

    def test_credibility_weight_zero_trips_impossible(self):
        """With n > 0, weight is always > 0."""
        rows = _make_trip_features("D1", 1)
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=100)
        assert driver_df["credibility_weight"][0] > 0.0

    def test_high_credibility_threshold_shrinks_towards_mean(self):
        """With very high threshold, driver score is close to portfolio mean."""
        rows = (
            _make_trip_features("D_LOW", 5, mean_speed=20.0, harsh_braking_rate=0.0)
            + _make_trip_features("D_HIGH", 5, mean_speed=120.0, harsh_braking_rate=5.0)
        )
        df = pl.DataFrame(rows)

        # Very high threshold → almost full shrinkage
        driver_df_high = aggregate_to_driver(df, credibility_threshold=10000)
        speeds = driver_df_high["mean_speed_kmh"].to_list()
        # Both drivers should be close to portfolio mean (~70 km/h)
        for speed in speeds:
            assert 30.0 < speed < 110.0, f"Expected speed near portfolio mean, got {speed}"

    def test_low_credibility_threshold_preserves_driver_signal(self):
        """With threshold=1, most data-rich drivers get near-full credibility."""
        rows = (
            _make_trip_features("D_LOW", 100, mean_speed=20.0)
            + _make_trip_features("D_HIGH", 100, mean_speed=120.0)
        )
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=1)
        speeds = {
            r["driver_id"]: r["mean_speed_kmh"]
            for r in driver_df.iter_rows(named=True)
        }
        # With near-full credibility, gap between drivers should be large
        assert speeds["D_HIGH"] > speeds["D_LOW"] + 50.0


# ===========================================================================
# 13. trip_loader — schema mapping and edge cases
# ===========================================================================

class TestTripLoaderSchemaMapping:
    def _make_sample_df(self) -> pl.DataFrame:
        sim = TripSimulator(seed=200)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
        return trips_df

    def test_multi_column_schema_rename(self, tmp_path):
        df = self._make_sample_df()
        renamed = df.rename({"speed_kmh": "gps_speed", "latitude": "lat"})
        pq_path = tmp_path / "data.parquet"
        renamed.write_parquet(pq_path)
        loaded = load_trips(
            pq_path, schema={"gps_speed": "speed_kmh", "lat": "latitude"}
        )
        assert "speed_kmh" in loaded.columns
        assert "latitude" in loaded.columns

    def test_pq_extension_works(self, tmp_path):
        df = self._make_sample_df()
        pq_path = tmp_path / "data.pq"
        df.write_parquet(pq_path)
        loaded = load_trips(pq_path)
        assert len(loaded) == len(df)

    def test_output_sorted_by_trip_then_timestamp(self, tmp_path):
        df = self._make_sample_df()
        # Write in random order
        shuffled = df.sample(fraction=1.0, seed=42)
        pq_path = tmp_path / "shuffled.parquet"
        shuffled.write_parquet(pq_path)
        loaded = load_trips(pq_path)
        trip_ids = loaded["trip_id"].to_list()
        timestamps = loaded["timestamp"].to_list()
        # Check sorted by trip_id, then timestamp within trip
        for i in range(1, len(trip_ids)):
            if trip_ids[i] == trip_ids[i - 1]:
                assert timestamps[i] >= timestamps[i - 1]
            else:
                assert trip_ids[i] >= trip_ids[i - 1]

    def test_driver_id_defaults_to_unknown(self, tmp_path):
        # Minimal DF without driver_id
        minimal = pl.DataFrame({
            "trip_id": ["T1", "T1"],
            "timestamp": ["2024-01-01 08:00:00", "2024-01-01 08:00:01"],
            "latitude": [51.5, 51.5],
            "longitude": [-0.1, -0.1],
            "speed_kmh": [30.0, 35.0],
        })
        pq_path = tmp_path / "minimal.parquet"
        minimal.write_parquet(pq_path)
        loaded = load_trips(pq_path)
        assert "driver_id" in loaded.columns
        assert (loaded["driver_id"] == "unknown").all()

    def test_all_schema_columns_defined(self):
        """ALL_SCHEMA_COLUMNS = REQUIRED + OPTIONAL."""
        assert set(ALL_SCHEMA_COLUMNS) == set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)

    def test_load_from_dataframe_with_schema(self):
        df = pl.DataFrame({
            "trip_id": ["T1"],
            "ts": ["2024-01-01 08:00:00"],
            "latitude": [51.5],
            "longitude": [-0.1],
            "speed_kmh": [30.0],
        }).with_columns(pl.col("ts").str.to_datetime())
        with pytest.raises(ValueError, match="timestamp"):
            load_trips_from_dataframe(df)

    def test_load_from_dataframe_with_schema_rename(self):
        df = pl.DataFrame({
            "trip_id": ["T1"],
            "ts": ["2024-01-01 08:00:00"],
            "latitude": [51.5],
            "longitude": [-0.1],
            "speed_kmh": [30.0],
        }).with_columns(pl.col("ts").str.to_datetime())
        loaded = load_trips_from_dataframe(df, schema={"ts": "timestamp"})
        assert "timestamp" in loaded.columns


# ===========================================================================
# 14. trip_simulator — output properties
# ===========================================================================

class TestTripSimulatorOutputProperties:
    def test_heading_bounded_0_360(self):
        sim = TripSimulator(seed=11)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
        headings = trips_df["heading_deg"]
        assert (headings >= 0.0).all()
        assert (headings <= 360.0).all()  # simulator rounds to 1dp; 360.0 is valid edge

    def test_lat_lon_are_floats(self):
        sim = TripSimulator(seed=12)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
        assert trips_df["latitude"].dtype == pl.Float64
        assert trips_df["longitude"].dtype == pl.Float64

    def test_claims_df_has_fraction_columns(self):
        sim = TripSimulator(seed=13)
        _, claims_df = sim.simulate(n_drivers=5, trips_per_driver=5)
        for col in ["cautious_fraction", "normal_fraction", "aggressive_fraction"]:
            assert col in claims_df.columns

    def test_fractions_sum_to_one(self):
        sim = TripSimulator(seed=14)
        _, claims_df = sim.simulate(n_drivers=10, trips_per_driver=3)
        total = (
            claims_df["cautious_fraction"]
            + claims_df["normal_fraction"]
            + claims_df["aggressive_fraction"]
        )
        assert (total - 1.0).abs().max() < 2e-4  # round(4) precision

    def test_exposure_years_positive(self):
        sim = TripSimulator(seed=15)
        _, claims_df = sim.simulate(n_drivers=5, trips_per_driver=10)
        assert (claims_df["exposure_years"] > 0).all()

    def test_annual_km_positive(self):
        sim = TripSimulator(seed=16)
        _, claims_df = sim.simulate(n_drivers=5, trips_per_driver=5)
        assert (claims_df["annual_km"] > 0).all()

    def test_trip_id_format(self):
        sim = TripSimulator(seed=17)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
        for tid in trips_df["trip_id"].unique().to_list():
            assert tid.startswith("TRP"), f"Expected TRP prefix, got {tid}"

    def test_driver_id_format(self):
        sim = TripSimulator(seed=18)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=2)
        for did in trips_df["driver_id"].unique().to_list():
            assert did.startswith("DRV"), f"Expected DRV prefix, got {did}"

    def test_single_driver(self):
        sim = TripSimulator(seed=19)
        trips_df, claims_df = sim.simulate(n_drivers=1, trips_per_driver=5)
        assert claims_df["driver_id"].n_unique() == 1
        assert trips_df["driver_id"].n_unique() == 1


# ===========================================================================
# 15. DrivingStateHMM — state ordering and edge cases
# ===========================================================================

class TestDrivingStateHMMStateOrdering:
    @pytest.fixture(scope="class")
    def fitted_3state(self):
        sim = TripSimulator(seed=300)
        trips_df, _ = sim.simulate(
            n_drivers=10, trips_per_driver=20,
            min_trip_duration_s=120, max_trip_duration_s=600,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(feats)
        return model, feats

    def test_state_rank_is_permutation_of_0_to_n(self, fitted_3state):
        model, _ = fitted_3state
        assert set(model._state_rank.tolist()) == {0, 1, 2}

    def test_state_order_is_permutation_of_0_to_n(self, fitted_3state):
        model, _ = fitted_3state
        assert set(model._state_order.tolist()) == {0, 1, 2}

    def test_state_rank_state_order_are_inverse(self, fitted_3state):
        """state_rank[state_order[k]] should equal k."""
        model, _ = fitted_3state
        for k in range(3):
            assert model._state_rank[model._state_order[k]] == k

    def test_four_state_model_valid_states(self):
        sim = TripSimulator(seed=301)
        trips_df, _ = sim.simulate(
            n_drivers=8, trips_per_driver=20,
            min_trip_duration_s=120, max_trip_duration_s=600,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)
        model = DrivingStateHMM(n_states=4, random_state=42)
        model.fit(feats)
        states = model.predict_states(feats)
        assert states.min() >= 0
        assert states.max() <= 3
        assert len(states) == len(feats)

    def test_predict_state_probs_reordered(self, fitted_3state):
        """Probability columns should be in ordered state space, not raw hmmlearn order."""
        model, feats = fitted_3state
        probs = model.predict_state_probs(feats)
        # Shape is correct
        assert probs.shape == (len(feats), 3)
        # Each row sums to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        # All values in [0, 1]
        assert probs.min() >= -1e-9
        assert probs.max() <= 1.0 + 1e-9

    def test_driver_state_features_single_trip_driver(self, fitted_3state):
        """A driver with only 1 trip: no transitions → transition_rate = 0."""
        model, feats = fitted_3state
        # Create a minimal single-trip driver feature row
        one_trip = feats[:1].with_columns(pl.lit("SOLO_DRIVER").alias("driver_id"))
        states = model.predict_states(one_trip)
        driver_df = model.driver_state_features(one_trip, states)
        assert len(driver_df) == 1
        assert driver_df["mean_transition_rate"][0] == pytest.approx(0.0)

    def test_driver_features_requires_driver_id(self, fitted_3state):
        model, feats = fitted_3state
        states = model.predict_states(feats)
        feats_no_id = feats.drop("driver_id")
        with pytest.raises(ValueError, match="driver_id"):
            model.driver_state_features(feats_no_id, states)

    def test_standardise_uses_fit_mean_std(self, fitted_3state):
        """After fit, _mean and _std attributes exist and are correct shape."""
        model, feats = fitted_3state
        assert hasattr(model, "_mean")
        assert hasattr(model, "_std")
        assert len(model._mean) == len(model.features)
        assert len(model._std) == len(model.features)
        assert (model._std > 0).all()


# ===========================================================================
# 16. ContinuousTimeHMM — internal math correctness
# ===========================================================================

class TestCTHMMInternalMath:
    @pytest.fixture(scope="class")
    def minimal_model(self):
        """Inject known parameters into a CTHMM for deterministic testing."""
        Q = np.array([[-0.5, 0.5], [0.3, -0.3]])
        means = np.array([[0.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.5, 0.5]])
        covars = np.ones((2, 4))
        pi = np.array([0.5, 0.5])

        model = ContinuousTimeHMM(n_states=2, n_iter=1, random_state=0)
        model.Q_ = Q.copy()
        model.means_ = means.copy()
        model.covars_ = covars.copy()
        model.pi_ = pi.copy()
        model._mean = np.zeros(4)
        model._std = np.ones(4)
        model.is_fitted = True
        return model

    def test_transition_matrix_is_stochastic(self, minimal_model):
        """P(dt) rows sum to 1 for any dt."""
        for dt in [0.1, 1.0, 5.0, 60.0]:
            P = minimal_model._transition_matrix(dt)
            np.testing.assert_allclose(
                P.sum(axis=1), 1.0, atol=1e-6,
                err_msg=f"Transition matrix rows don't sum to 1 at dt={dt}"
            )

    def test_transition_matrix_non_negative(self, minimal_model):
        P = minimal_model._transition_matrix(1.0)
        assert P.min() >= -1e-9

    def test_transition_matrix_dt_zero_is_identity(self, minimal_model):
        """At dt→0, P(dt) should approach identity matrix."""
        P = minimal_model._transition_matrix(1e-8)
        np.testing.assert_allclose(P, np.eye(2), atol=1e-4)

    def test_emission_log_prob_shape(self, minimal_model):
        """_emission_log_prob returns (n_obs, n_states)."""
        X = np.random.randn(50, 4)
        log_probs = minimal_model._emission_log_prob(X)
        assert log_probs.shape == (50, 2)

    def test_emission_log_prob_all_finite(self, minimal_model):
        X = np.random.randn(20, 4)
        log_probs = minimal_model._emission_log_prob(X)
        assert np.isfinite(log_probs).all()

    def test_init_generator_valid(self):
        """Initialised Q should be a valid generator matrix."""
        model = ContinuousTimeHMM(n_states=3, random_state=42)
        rng = np.random.default_rng(42)
        Q = model._init_generator(rng)
        assert Q.shape == (3, 3)
        # Off-diagonal non-negative
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert Q[i, j] >= 0.0
        # Row sums zero
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)

    def test_predict_before_fit_raises(self):
        model = ContinuousTimeHMM(n_states=2)
        n = 10
        df = pl.DataFrame({
            "mean_speed_kmh": [50.0] * n,
            "speed_variation_coeff": [0.1] * n,
            "harsh_braking_rate": [0.0] * n,
            "harsh_accel_rate": [0.0] * n,
        })
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_states(df)

    def test_missing_feature_raises(self):
        model = ContinuousTimeHMM(n_states=2, features=["nonexistent"])
        df = pl.DataFrame({"mean_speed_kmh": [50.0] * 10})
        with pytest.raises(ValueError, match="nonexistent"):
            model.fit(df)

    def test_two_state_fit_and_predict(self):
        """End-to-end two-state CTHMM with minimal data."""
        sim = TripSimulator(seed=400)
        trips_df, _ = sim.simulate(
            n_drivers=4, trips_per_driver=10,
            min_trip_duration_s=120, max_trip_duration_s=300,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)
        model = ContinuousTimeHMM(n_states=2, n_iter=5, random_state=42)
        model.fit(feats)
        states = model.predict_states(feats)
        assert states.shape == (len(feats),)
        assert states.min() >= 0
        assert states.max() <= 1

    def test_cthmm_driver_state_features_fractions_sum_to_one(self):
        sim = TripSimulator(seed=401)
        trips_df, _ = sim.simulate(
            n_drivers=4, trips_per_driver=10,
            min_trip_duration_s=120, max_trip_duration_s=300,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)
        model = ContinuousTimeHMM(n_states=3, n_iter=5, random_state=42)
        model.fit(feats)
        states = model.predict_states(feats)
        driver_df = model.driver_state_features(feats, states)
        frac_cols = [f"state_{k}_fraction" for k in range(3)]
        row_sums = driver_df.select(frac_cols).sum_horizontal()
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_cthmm_state_entropy_non_negative(self):
        sim = TripSimulator(seed=402)
        trips_df, _ = sim.simulate(
            n_drivers=4, trips_per_driver=10,
            min_trip_duration_s=120, max_trip_duration_s=300,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)
        model = ContinuousTimeHMM(n_states=3, n_iter=5, random_state=42)
        model.fit(feats)
        states = model.predict_states(feats)
        driver_df = model.driver_state_features(feats, states)
        assert (driver_df["state_entropy"] >= -1e-9).all()


# ===========================================================================
# 17. _logsumexp — additional cases
# ===========================================================================

class TestLogsumexpAdditional:
    def test_uniform_values(self):
        """logsumexp([0, 0, 0]) = log(3)."""
        a = np.array([0.0, 0.0, 0.0])
        result = _logsumexp(a)
        assert result == pytest.approx(math.log(3), rel=1e-6)

    def test_dominated_by_max(self):
        """When one value is much larger, result ≈ that value."""
        a = np.array([-1000.0, 500.0, -1000.0])
        result = _logsumexp(a)
        assert result == pytest.approx(500.0, abs=1e-6)

    def test_two_equal_values(self):
        """logsumexp([x, x]) = x + log(2)."""
        x = 5.0
        a = np.array([x, x])
        result = _logsumexp(a)
        assert result == pytest.approx(x + math.log(2), rel=1e-6)

    def test_negative_values(self):
        """Works with negative inputs."""
        a = np.array([-3.0, -2.0, -1.0])
        import scipy.special
        expected = scipy.special.logsumexp(a)
        assert _logsumexp(a) == pytest.approx(float(expected), rel=1e-6)


# ===========================================================================
# 18. scoring_pipeline — additional paths
# ===========================================================================

class TestScoringPipelineAdditionalPaths:
    @pytest.fixture(scope="class")
    def pipeline_and_data(self):
        sim = TripSimulator(seed=500)
        trips_df, claims_df = sim.simulate(
            n_drivers=15, trips_per_driver=25,
            min_trip_duration_s=180, max_trip_duration_s=900,
        )
        pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
        pipe.fit(trips_df, claims_df)
        return pipe, trips_df, claims_df

    def test_glm_features_after_fit_includes_state_features(self, pipeline_and_data):
        pipe, trips_df, _ = pipeline_and_data
        feats = pipe.glm_features(trips_df)
        # HMM state fractions should be in the output
        assert any("state_" in c for c in feats.columns), (
            f"Expected state_ columns; got: {feats.columns}"
        )

    def test_score_trips_alias_matches_predict(self, pipeline_and_data):
        pipe, trips_df, _ = pipeline_and_data
        direct = pipe.predict(trips_df)
        via_alias = score_trips(trips_df, pipe)
        # Both should have same driver_ids and same predictions
        direct_sorted = direct.sort("driver_id")
        alias_sorted = via_alias.sort("driver_id")
        np.testing.assert_allclose(
            direct_sorted["predicted_claim_frequency"].to_numpy(),
            alias_sorted["predicted_claim_frequency"].to_numpy(),
            rtol=1e-6,
        )

    def test_glm_feature_subset_parameter(self):
        """glm_feature_subset limits which features reach the GLM."""
        sim = TripSimulator(seed=501)
        trips_df, claims_df = sim.simulate(
            n_drivers=15, trips_per_driver=25,
            min_trip_duration_s=180, max_trip_duration_s=900,
        )
        subset = ["mean_speed_kmh", "speeding_fraction"]
        pipe = TelematicsScoringPipeline(
            n_hmm_states=2,
            glm_feature_subset=subset,
            random_state=42,
        )
        pipe.fit(trips_df, claims_df)
        preds = pipe.predict(trips_df)
        assert (preds["predicted_claim_frequency"] >= 0).all()

    def test_pipeline_fit_stores_glm_feature_names(self, pipeline_and_data):
        pipe, _, _ = pipeline_and_data
        assert len(pipe._glm_feature_names) >= 1

    def test_pipeline_credibility_threshold_propagates(self):
        sim = TripSimulator(seed=502)
        trips_df, claims_df = sim.simulate(
            n_drivers=10, trips_per_driver=15,
            min_trip_duration_s=120, max_trip_duration_s=600,
        )
        pipe = TelematicsScoringPipeline(
            n_hmm_states=2,
            credibility_threshold=5,
            random_state=42,
        )
        pipe.fit(trips_df, claims_df)
        assert pipe.credibility_threshold == 5


# ===========================================================================
# 19. DrivingStateHMM — fit/predict with single-feature list
# ===========================================================================

class TestDrivingStateHMMSingleFeature:
    def test_single_feature_fit_and_predict(self):
        sim = TripSimulator(seed=600)
        trips_df, _ = sim.simulate(
            n_drivers=5, trips_per_driver=15,
            min_trip_duration_s=120, max_trip_duration_s=400,
        )
        from insurance_telematics.preprocessor import clean_trips
        from insurance_telematics.feature_extractor import extract_trip_features
        cleaned = clean_trips(trips_df)
        feats = extract_trip_features(cleaned)

        model = DrivingStateHMM(
            n_states=2, features=["speeding_fraction"], random_state=42
        )
        model.fit(feats)
        states = model.predict_states(feats)
        probs = model.predict_state_probs(feats)
        assert states.shape == (len(feats),)
        assert probs.shape == (len(feats), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ===========================================================================
# 20. aggregate_to_driver — edge cases
# ===========================================================================

class TestAggregateEdgeCases:
    def test_composite_score_min_is_zero(self):
        """The minimum composite score across the portfolio is always 0."""
        rows = (
            _make_trip_features("D1", 5, harsh_braking_rate=0.0, speeding_fraction=0.0)
            + _make_trip_features("D2", 5, harsh_braking_rate=1.0, speeding_fraction=0.5)
            + _make_trip_features("D3", 5, harsh_braking_rate=3.0, speeding_fraction=0.9)
        )
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=1)
        assert driver_df["composite_risk_score"].min() == pytest.approx(0.0, abs=1e-3)

    def test_composite_score_max_is_100(self):
        """The maximum composite score across the portfolio is always 100."""
        rows = (
            _make_trip_features("D1", 5, harsh_braking_rate=0.0, speeding_fraction=0.0)
            + _make_trip_features("D2", 5, harsh_braking_rate=5.0, speeding_fraction=1.0)
        )
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=1)
        assert driver_df["composite_risk_score"].max() == pytest.approx(100.0, abs=1e-3)

    def test_two_identical_drivers_get_same_score(self):
        """Identical trip histories → identical risk scores."""
        rows = (
            _make_trip_features("D1", 10, mean_speed=60.0, harsh_braking_rate=0.2)
            + _make_trip_features("D2", 10, mean_speed=60.0, harsh_braking_rate=0.2)
        )
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df, credibility_threshold=5)
        scores = driver_df.sort("driver_id")["composite_risk_score"].to_list()
        assert abs(scores[0] - scores[1]) < 1e-6

    def test_result_sorted_by_driver_id(self):
        rows = (
            _make_trip_features("DRV_Z", 5)
            + _make_trip_features("DRV_A", 5)
            + _make_trip_features("DRV_M", 5)
        )
        df = pl.DataFrame(rows)
        driver_df = aggregate_to_driver(df)
        ids = driver_df["driver_id"].to_list()
        assert ids == sorted(ids)
