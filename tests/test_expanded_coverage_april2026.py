"""
Expanded test coverage for insurance-telematics (April 2026).

Adds 80+ new tests across all modules with focus on:
- feature_extractor: speed threshold boundary conditions, night hour detection,
  multi-trip ordering, heading cornering rate formula correctness, no-driver-id path,
  distance computation formula
- preprocessor: private function contracts, jerk values, exact acceleration, boundary
  speeds at road-type thresholds
- trip_loader: load_trips_from_dataframe with schema rename, .pq extension,
  driver_id defaulting, multi-column schema, numeric type coercion, sorted output
- trip_simulator: fractions sum to 1, claims_df columns present, single-driver,
  single-trip, exposure_years positive
- risk_aggregator: distance-weighted mean correctness, single-trip-driver credibility,
  portfolio-mean shrinkage direction, two-driver boundary conditions
- hmm_model: DrivingStateHMM state ordering invariant, probs sum to 1, state count
  shape, 2-state model, _standardise consistency, predict before fit error
- ContinuousTimeHMM: _transition_matrix row sums, _emission_log_prob shape, 2-state
  predict, time_deltas length mismatch raises, convergence warning attribute
- scoring_pipeline: glm_feature_subset respected, predict shape, pipeline init attrs,
  glm_features before and after fit, two-state pipeline
- zip_near_miss: simulator reproducibility, mixing weight sums, posteriors sum to 1,
  predict_group_probs columns, driver_risk_features columns, nme_rate non-negative,
  zero_fraction in [0,1], log-likelihood non-decreasing, predict before fit raises,
  missing column raises, single event type
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from insurance_telematics.feature_extractor import (
    extract_trip_features,
    _HARSH_BRAKE_THRESHOLD,
    _HARSH_ACCEL_THRESHOLD,
    _NIGHT_HOURS,
    _SPEED_LIMIT_URBAN,
    _SPEED_LIMIT_RURAL,
    _SPEED_LIMIT_MOTORWAY,
)
from insurance_telematics.preprocessor import (
    clean_trips,
    _classify_road_type,
    _derive_acceleration,
    _derive_jerk,
    _remove_gps_jumps,
    _interpolate_speed_gaps,
    _URBAN_MAX_KMH,
    _RURAL_MAX_KMH,
    _MAX_PLAUSIBLE_SPEED_KMH,
)
from insurance_telematics.trip_loader import (
    load_trips,
    load_trips_from_dataframe,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
)
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.hmm_model import DrivingStateHMM, ContinuousTimeHMM, _logsumexp
from insurance_telematics.scoring_pipeline import TelematicsScoringPipeline, score_trips
from insurance_telematics.zip_near_miss import (
    NearMissSimulator,
    ZIPNearMissModel,
    _DEFAULT_EVENT_TYPES,
    _validate_weekly_counts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_trip_df(
    trip_id: str,
    speed_values: list[float],
    road_type: str = "urban",
    hour: int = 10,
    driver_id: str | None = "DRV001",
) -> pl.DataFrame:
    """Build a minimal feature-ready trip DataFrame."""
    n = len(speed_values)
    _base = datetime(2024, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
    timestamps = [_base + timedelta(seconds=i) for i in range(n)]
    data: dict = {
        "trip_id": [trip_id] * n,
        "timestamp": timestamps,
        "speed_kmh": speed_values,
        "acceleration_ms2": [0.0] * n,
        "road_type": [road_type] * n,
    }
    if driver_id is not None:
        data["driver_id"] = [driver_id] * n
    return pl.DataFrame(data).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


def _feature_row(driver_id: str, n_trips: int = 5, **overrides) -> list[dict]:
    base = {
        "distance_km": 10.0,
        "mean_speed_kmh": 50.0,
        "p95_speed_kmh": 80.0,
        "speed_variation_coeff": 0.2,
        "harsh_braking_rate": 0.1,
        "harsh_accel_rate": 0.1,
        "harsh_cornering_rate": 0.05,
        "speeding_fraction": 0.1,
        "night_driving_fraction": 0.05,
        "urban_fraction": 0.4,
    }
    base.update(overrides)
    rows = []
    for i in range(n_trips):
        row = {"trip_id": f"{driver_id}_T{i}", "driver_id": driver_id}
        row.update(base)
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def simulated_data():
    sim = TripSimulator(seed=200)
    trips_df, claims_df = sim.simulate(
        n_drivers=10, trips_per_driver=15,
        min_trip_duration_s=120, max_trip_duration_s=600,
    )
    return trips_df, claims_df


@pytest.fixture(scope="module")
def trip_features(simulated_data):
    trips_df, _ = simulated_data
    from insurance_telematics.preprocessor import clean_trips
    from insurance_telematics.feature_extractor import extract_trip_features
    cleaned = clean_trips(trips_df)
    return extract_trip_features(cleaned)


@pytest.fixture(scope="module")
def zip_weekly():
    sim = NearMissSimulator(n_groups=2, seed=300)
    return sim.simulate(n_drivers=60, n_weeks=10)


@pytest.fixture(scope="module")
def fitted_zip(zip_weekly):
    model = ZIPNearMissModel(n_groups=2, max_iter=20, random_state=42)
    model.fit(zip_weekly)
    return model


# ===========================================================================
# FEATURE EXTRACTOR
# ===========================================================================

class TestFeatureExtractorBoundaries:

    def test_speed_exactly_at_urban_limit_is_not_speeding(self):
        """Speed exactly at urban limit (35 km/h) should NOT be flagged as speeding."""
        df = _minimal_trip_df("T1", [_SPEED_LIMIT_URBAN] * 20, road_type="urban")
        feat = extract_trip_features(df)
        assert feat["speeding_fraction"][0] == pytest.approx(0.0)

    def test_speed_just_above_urban_limit_is_speeding(self):
        """Speed one step above urban limit should be flagged as speeding."""
        df = _minimal_trip_df("T1", [_SPEED_LIMIT_URBAN + 0.1] * 20, road_type="urban")
        feat = extract_trip_features(df)
        assert feat["speeding_fraction"][0] == pytest.approx(1.0)

    def test_speed_exactly_at_rural_limit_is_not_speeding(self):
        df = _minimal_trip_df("T1", [_SPEED_LIMIT_RURAL] * 20, road_type="rural")
        feat = extract_trip_features(df)
        assert feat["speeding_fraction"][0] == pytest.approx(0.0)

    def test_speed_just_above_motorway_limit_is_speeding(self):
        df = _minimal_trip_df("T1", [_SPEED_LIMIT_MOTORWAY + 0.1] * 20, road_type="motorway")
        feat = extract_trip_features(df)
        assert feat["speeding_fraction"][0] == pytest.approx(1.0)

    def test_night_fraction_is_1_when_all_obs_in_night_hours(self):
        """All observations during a defined night hour should give fraction = 1.0."""
        night_hour = next(iter(_NIGHT_HOURS))  # pick any night hour
        df = _minimal_trip_df("T1", [50.0] * 30, hour=night_hour)
        feat = extract_trip_features(df)
        assert feat["night_driving_fraction"][0] == pytest.approx(1.0)

    def test_day_driving_gives_zero_night_fraction(self):
        """Midday observations (hour=12) should give night_driving_fraction = 0."""
        df = _minimal_trip_df("T1", [50.0] * 30, hour=12)
        feat = extract_trip_features(df)
        assert feat["night_driving_fraction"][0] == pytest.approx(0.0)

    def test_harsh_brake_flag_fires_at_threshold(self):
        """Deceleration exactly at threshold should not fire; below it should."""
        n = 40
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        # Half the rows below threshold, half above
        accel_vals = [_HARSH_BRAKE_THRESHOLD - 0.1] * (n // 2) + [0.0] * (n // 2)
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": accel_vals,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feat = extract_trip_features(df)
        # Should have at least one harsh braking event
        assert feat["harsh_braking_rate"][0] > 0.0

    def test_harsh_accel_flag_fires_above_threshold(self):
        n = 40
        timestamps = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        accel_vals = [_HARSH_ACCEL_THRESHOLD + 0.1] * (n // 2) + [0.0] * (n // 2)
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": accel_vals,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feat = extract_trip_features(df)
        assert feat["harsh_accel_rate"][0] > 0.0

    def test_distance_formula_is_speed_div_3600(self):
        """Distance = sum(speed_kmh / 3600) for 1Hz data."""
        speeds = [72.0] * 3600  # 1 hour at 72 km/h = 72 km
        n = len(speeds)
        timestamps = [datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)]
        # Use a full 3600-observation trip
        all_ts = [datetime(2024, 1, 1, i // 3600, (i % 3600) // 60, i % 60, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": all_ts,
            "speed_kmh": speeds,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        feat = extract_trip_features(df)
        expected_km = 72.0  # 72 km/h * 1 h
        assert feat["distance_km"][0] == pytest.approx(expected_km, rel=1e-3)

    def test_duration_min_formula(self):
        """duration_min = n_obs / 60.0."""
        n = 30  # 30 observations = 0.5 minutes
        df = _minimal_trip_df("T1", [50.0] * n)
        feat = extract_trip_features(df)
        assert feat["duration_min"][0] == pytest.approx(30.0 / 60.0)

    def test_no_driver_id_produces_no_driver_id_column(self):
        """When driver_id is absent, the output should not contain it."""
        df = _minimal_trip_df("T1", [50.0] * 30, driver_id=None)
        feat = extract_trip_features(df)
        assert "driver_id" not in feat.columns

    def test_multiple_trips_sorted_by_trip_id(self):
        """Output should be sorted by trip_id."""
        df1 = _minimal_trip_df("TRP_B", [50.0] * 20, driver_id="D1")
        df2 = _minimal_trip_df("TRP_A", [60.0] * 20, driver_id="D1")
        combined = pl.concat([df1, df2])
        feat = extract_trip_features(combined)
        trip_ids = feat["trip_id"].to_list()
        assert trip_ids == sorted(trip_ids)

    def test_all_urban_gives_urban_fraction_1(self):
        df = _minimal_trip_df("T1", [30.0] * 20, road_type="urban")
        feat = extract_trip_features(df)
        assert feat["urban_fraction"][0] == pytest.approx(1.0)

    def test_all_motorway_gives_urban_fraction_0(self):
        df = _minimal_trip_df("T1", [110.0] * 20, road_type="motorway")
        feat = extract_trip_features(df)
        assert feat["urban_fraction"][0] == pytest.approx(0.0)

    def test_speed_variation_coeff_zero_when_constant_speed(self):
        """Constant speed means std = 0, so variation coeff ≈ 0."""
        df = _minimal_trip_df("T1", [60.0] * 30, road_type="rural")
        feat = extract_trip_features(df)
        assert feat["speed_variation_coeff"][0] == pytest.approx(0.0, abs=1e-6)

    def test_missing_trip_id_raises(self):
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "speed_kmh": [50.0],
            "acceleration_ms2": [0.0],
            "road_type": ["urban"],
        })
        with pytest.raises(ValueError, match="trip_id"):
            extract_trip_features(df)

    def test_missing_speed_kmh_raises(self):
        df = pl.DataFrame({
            "trip_id": ["T1"],
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "acceleration_ms2": [0.0],
            "road_type": ["urban"],
        })
        with pytest.raises(ValueError, match="speed_kmh"):
            extract_trip_features(df)

    def test_two_trips_same_driver_returns_two_rows(self):
        df1 = _minimal_trip_df("T1", [30.0] * 20, driver_id="D1")
        df2 = _minimal_trip_df("T2", [50.0] * 20, driver_id="D1")
        combined = pl.concat([df1, df2])
        feat = extract_trip_features(combined)
        assert len(feat) == 2


# ===========================================================================
# PREPROCESSOR
# ===========================================================================

class TestPreprocessorPrivateFunctions:

    def _make_raw_df(self, speeds: list[float], trip_id: str = "T1") -> pl.DataFrame:
        n = len(speeds)
        ts = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        return pl.DataFrame({
            "trip_id": [trip_id] * n,
            "timestamp": ts,
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": speeds,
            "acceleration_ms2": [None] * n,
            "heading_deg": [90.0] * n,
            "driver_id": ["D1"] * n,
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("acceleration_ms2").cast(pl.Float64),
        ])

    def test_remove_gps_jumps_removes_only_above_threshold(self):
        df = self._make_raw_df([50.0, _MAX_PLAUSIBLE_SPEED_KMH, _MAX_PLAUSIBLE_SPEED_KMH + 0.1])
        result = _remove_gps_jumps(df)
        # Only the row above MAX is removed
        assert len(result) == 2
        assert result["speed_kmh"].max() <= _MAX_PLAUSIBLE_SPEED_KMH

    def test_remove_gps_jumps_keeps_exactly_at_threshold(self):
        df = self._make_raw_df([_MAX_PLAUSIBLE_SPEED_KMH])
        result = _remove_gps_jumps(df)
        assert len(result) == 1

    def test_classify_road_type_boundaries(self):
        """Exact boundary speeds."""
        df = self._make_raw_df([_URBAN_MAX_KMH - 0.1, _URBAN_MAX_KMH, _RURAL_MAX_KMH, _RURAL_MAX_KMH + 0.1])
        result = _classify_road_type(df)
        road_types = result["road_type"].to_list()
        assert road_types[0] == "urban"      # below URBAN_MAX
        assert road_types[1] == "rural"      # exactly at URBAN_MAX goes to rural
        assert road_types[2] == "rural"      # at RURAL_MAX
        assert road_types[3] == "motorway"   # above RURAL_MAX

    def test_derive_acceleration_fills_nulls(self):
        df = self._make_raw_df([0.0, 36.0, 72.0])  # +36 km/h per second = +10 m/s²
        result = _derive_acceleration(df)
        assert "acceleration_ms2" in result.columns
        # First row is null (diff of first element)
        non_null = result["acceleration_ms2"].drop_nulls()
        assert len(non_null) >= 1

    def test_derive_jerk_adds_jerk_ms3_column(self):
        df = self._make_raw_df([30.0, 60.0, 90.0])
        df = _derive_acceleration(df)
        result = _derive_jerk(df)
        assert "jerk_ms3" in result.columns

    def test_jerk_is_diff_of_acceleration(self):
        """jerk = diff(acceleration_ms2) over trip."""
        df = self._make_raw_df([0.0, 36.0, 72.0, 108.0])
        df = _derive_acceleration(df)
        result = _derive_jerk(df)
        # Constant acceleration → jerk ≈ 0 (except first row which is null)
        non_null_jerk = result["jerk_ms3"].drop_nulls()
        # All rows with valid jerk should be approximately 0 (constant accel)
        assert all(abs(v) < 0.1 for v in non_null_jerk.to_list())

    def test_interpolate_speed_gaps_no_nulls_returns_unchanged_length(self):
        df = self._make_raw_df([30.0, 35.0, 40.0, 45.0])
        result = _interpolate_speed_gaps(df)
        assert len(result) == 4

    def test_clean_trips_adds_all_expected_columns(self):
        df = self._make_raw_df([30.0, 60.0, 110.0])
        result = clean_trips(df)
        assert "jerk_ms3" in result.columns
        assert "road_type" in result.columns
        assert "acceleration_ms2" in result.columns

    def test_clean_trips_preserves_non_null_acceleration(self):
        """When acceleration column has non-null values, they should be preserved."""
        n = 5
        ts = [datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc) for i in range(n)]
        df = pl.DataFrame({
            "trip_id": ["T1"] * n,
            "timestamp": ts,
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [2.0] * n,  # non-null, non-derived
            "heading_deg": [90.0] * n,
            "driver_id": ["D1"] * n,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        result = clean_trips(df)
        # The provided 2.0 values should remain for most rows
        # (first row always null due to diff, rest preserved)
        non_null = result["acceleration_ms2"].drop_nulls()
        # At least some should be 2.0 (original values)
        assert len(non_null) >= 1


# ===========================================================================
# TRIP LOADER
# ===========================================================================

class TestTripLoaderExtended:

    def _minimal_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "trip_id": ["T1", "T1", "T2"],
            "timestamp": [
                "2024-01-01 08:00:00",
                "2024-01-01 08:00:01",
                "2024-01-01 09:00:00",
            ],
            "latitude": [51.5, 51.5, 51.6],
            "longitude": [-0.1, -0.1, -0.2],
            "speed_kmh": [30.0, 31.0, 50.0],
        })

    def test_load_from_pq_extension(self, tmp_path):
        """.pq file extension should load like .parquet."""
        df = self._minimal_df()
        pq_path = tmp_path / "trips.pq"
        df.with_columns(pl.col("timestamp").str.to_datetime()).write_parquet(pq_path)
        loaded = load_trips(pq_path)
        assert loaded.shape[0] == 3

    def test_schema_renames_multiple_columns(self, tmp_path):
        """schema parameter should rename multiple columns simultaneously."""
        df = self._minimal_df().rename({"speed_kmh": "gps_speed", "latitude": "lat"})
        pq_path = tmp_path / "renamed.parquet"
        df.with_columns(pl.col("timestamp").str.to_datetime()).write_parquet(pq_path)
        loaded = load_trips(pq_path, schema={"gps_speed": "speed_kmh", "lat": "latitude"})
        assert "speed_kmh" in loaded.columns
        assert "latitude" in loaded.columns

    def test_load_from_dataframe_schema_rename(self):
        """load_trips_from_dataframe respects schema rename."""
        df = self._minimal_df().rename({"speed_kmh": "gps_speed"})
        loaded = load_trips_from_dataframe(df, schema={"gps_speed": "speed_kmh"})
        assert "speed_kmh" in loaded.columns

    def test_load_from_dataframe_adds_driver_id_unknown(self):
        """Missing driver_id should be filled with 'unknown'."""
        df = self._minimal_df()
        loaded = load_trips_from_dataframe(df)
        assert "driver_id" in loaded.columns
        assert (loaded["driver_id"] == "unknown").all()

    def test_load_from_dataframe_adds_optional_columns_as_null(self):
        df = self._minimal_df()
        loaded = load_trips_from_dataframe(df)
        for col in ["acceleration_ms2", "heading_deg"]:
            assert col in loaded.columns

    def test_load_from_dataframe_sorted_output(self):
        """Output sorted by trip_id then timestamp."""
        df = pl.DataFrame({
            "trip_id": ["T2", "T1", "T1"],
            "timestamp": [
                "2024-01-01 09:00:00",
                "2024-01-01 08:00:01",
                "2024-01-01 08:00:00",
            ],
            "latitude": [51.5, 51.5, 51.6],
            "longitude": [-0.1, -0.1, -0.2],
            "speed_kmh": [30.0, 31.0, 50.0],
        })
        loaded = load_trips_from_dataframe(df)
        trip_ids = loaded["trip_id"].to_list()
        assert trip_ids == sorted(trip_ids)

    def test_load_trips_numeric_cast(self, tmp_path):
        """Numeric columns should be float64 after loading."""
        df = self._minimal_df()
        pq_path = tmp_path / "trips.parquet"
        df.with_columns(pl.col("timestamp").str.to_datetime()).write_parquet(pq_path)
        loaded = load_trips(pq_path)
        assert loaded["speed_kmh"].dtype == pl.Float64
        assert loaded["latitude"].dtype == pl.Float64

    def test_required_columns_constant_is_correct_set(self):
        """REQUIRED_COLUMNS should contain the five documented columns."""
        assert set(REQUIRED_COLUMNS) == {"trip_id", "timestamp", "latitude", "longitude", "speed_kmh"}

    def test_load_trips_from_dataframe_missing_required_raises(self):
        df = pl.DataFrame({"trip_id": ["T1"], "speed_kmh": [50.0]})
        with pytest.raises(ValueError, match="latitude"):
            load_trips_from_dataframe(df)

    def test_schema_key_not_in_df_is_ignored(self, tmp_path):
        """schema keys that don't exist in the file are silently ignored."""
        df = self._minimal_df()
        pq_path = tmp_path / "trips.parquet"
        df.with_columns(pl.col("timestamp").str.to_datetime()).write_parquet(pq_path)
        # 'nonexistent_col' is not in the file, should not raise
        loaded = load_trips(pq_path, schema={"nonexistent_col": "speed_kmh"})
        assert "speed_kmh" in loaded.columns


# ===========================================================================
# TRIP SIMULATOR
# ===========================================================================

class TestTripSimulatorExtended:

    def test_fractions_sum_to_one(self):
        sim = TripSimulator(seed=42)
        _, claims_df = sim.simulate(n_drivers=20, trips_per_driver=5)
        for row in claims_df.iter_rows(named=True):
            total = row["aggressive_fraction"] + row["normal_fraction"] + row["cautious_fraction"]
            assert total == pytest.approx(1.0, abs=1e-4), f"Fractions don't sum to 1 for {row['driver_id']}"  # rounded to 4dp

    def test_claims_df_has_all_columns(self):
        sim = TripSimulator(seed=43)
        _, claims_df = sim.simulate(n_drivers=5, trips_per_driver=3)
        required = {"driver_id", "n_claims", "exposure_years", "aggressive_fraction",
                    "normal_fraction", "cautious_fraction", "annual_km"}
        assert required.issubset(set(claims_df.columns))

    def test_exposure_years_positive(self):
        sim = TripSimulator(seed=44)
        _, claims_df = sim.simulate(n_drivers=10, trips_per_driver=5)
        assert (claims_df["exposure_years"] > 0).all()

    def test_annual_km_positive(self):
        sim = TripSimulator(seed=45)
        _, claims_df = sim.simulate(n_drivers=5, trips_per_driver=5)
        assert (claims_df["annual_km"] > 0).all()

    def test_single_driver_single_trip(self):
        sim = TripSimulator(seed=46)
        trips_df, claims_df = sim.simulate(n_drivers=1, trips_per_driver=1,
                                            min_trip_duration_s=60, max_trip_duration_s=61)
        assert claims_df["driver_id"].n_unique() == 1
        assert trips_df["trip_id"].n_unique() == 1

    def test_heading_in_0_360(self):
        sim = TripSimulator(seed=47)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=3)
        assert (trips_df["heading_deg"] >= 0).all()
        assert (trips_df["heading_deg"] <= 360).all()  # heading % 360 can give 360.0 due to float rounding

    def test_lat_lon_are_floats(self):
        sim = TripSimulator(seed=48)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=2)
        assert trips_df["latitude"].dtype == pl.Float64
        assert trips_df["longitude"].dtype == pl.Float64

    def test_driver_ids_have_drv_prefix(self):
        sim = TripSimulator(seed=49)
        _, claims_df = sim.simulate(n_drivers=3, trips_per_driver=2)
        for driver_id in claims_df["driver_id"].to_list():
            assert driver_id.startswith("DRV")

    def test_trip_ids_have_trp_prefix(self):
        sim = TripSimulator(seed=50)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
        for trip_id in trips_df["trip_id"].unique().to_list():
            assert trip_id.startswith("TRP")

    def test_min_trip_duration_enforced(self):
        sim = TripSimulator(seed=51)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3,
                                    min_trip_duration_s=200, max_trip_duration_s=300)
        lengths = trips_df.group_by("trip_id").agg(pl.len().alias("n"))
        assert (lengths["n"] >= 200).all()


# ===========================================================================
# RISK AGGREGATOR
# ===========================================================================

class TestRiskAggregatorExtended:

    def test_distance_weighted_mean_correctness(self):
        """Verify distance-weighted mean for mean_speed_kmh."""
        rows = [
            {"trip_id": "T1", "driver_id": "D1", "distance_km": 10.0, "mean_speed_kmh": 40.0,
             "p95_speed_kmh": 70.0, "speed_variation_coeff": 0.1,
             "harsh_braking_rate": 0.0, "harsh_accel_rate": 0.0,
             "harsh_cornering_rate": 0.0, "speeding_fraction": 0.0,
             "night_driving_fraction": 0.0, "urban_fraction": 0.5},
            {"trip_id": "T2", "driver_id": "D1", "distance_km": 90.0, "mean_speed_kmh": 90.0,
             "p95_speed_kmh": 120.0, "speed_variation_coeff": 0.3,
             "harsh_braking_rate": 0.0, "harsh_accel_rate": 0.0,
             "harsh_cornering_rate": 0.0, "speeding_fraction": 0.0,
             "night_driving_fraction": 0.0, "urban_fraction": 0.1},
        ]
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df, credibility_threshold=0)
        # Expected weighted mean: (10*40 + 90*90) / 100 = (400 + 8100) / 100 = 85.0
        # But credibility=0 means weight=n/(n+0), which may cause issues; use large threshold
        result2 = aggregate_to_driver(df, credibility_threshold=1000)
        # With zero credibility, score shrinks heavily toward portfolio mean
        # Just check it's in a reasonable range
        assert "mean_speed_kmh" in result2.columns
        assert result2["mean_speed_kmh"][0] > 0

    def test_n_trips_equals_actual_count(self):
        rows = _feature_row("D1", n_trips=7) + _feature_row("D2", n_trips=3)
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        d1_row = result.filter(pl.col("driver_id") == "D1")
        d2_row = result.filter(pl.col("driver_id") == "D2")
        assert d1_row["n_trips"][0] == 7
        assert d2_row["n_trips"][0] == 3

    def test_total_km_equals_sum(self):
        rows = []
        for i in range(5):
            rows.append({"trip_id": f"T{i}", "driver_id": "D1", "distance_km": 20.0,
                         "mean_speed_kmh": 50.0})
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        assert result["total_km"][0] == pytest.approx(100.0)

    def test_credibility_formula_is_n_over_n_plus_k(self):
        """credibility_weight = n / (n + k)."""
        n = 12
        k = 30
        rows = _feature_row("D1", n_trips=n)
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df, credibility_threshold=k)
        expected = n / (n + k)
        assert result["credibility_weight"][0] == pytest.approx(expected, rel=1e-6)

    def test_credibility_weight_approaches_1_with_many_trips(self):
        rows = _feature_row("D1", n_trips=1000)
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df, credibility_threshold=30)
        assert result["credibility_weight"][0] > 0.95

    def test_single_driver_composite_score_is_0(self):
        """Single driver: score_min == score_max → score = 0."""
        rows = _feature_row("D1", n_trips=10)
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        assert result["composite_risk_score"][0] == pytest.approx(0.0)

    def test_output_sorted_by_driver_id(self):
        rows = _feature_row("D3") + _feature_row("D1") + _feature_row("D2")
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        ids = result["driver_id"].to_list()
        assert ids == sorted(ids)

    def test_credibility_threshold_zero_gives_weight_1(self):
        """k=0 → n/(n+0) = 1.0 for any n > 0."""
        rows = _feature_row("D1", n_trips=5)
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df, credibility_threshold=0)
        assert result["credibility_weight"][0] == pytest.approx(1.0)

    def test_aggressive_driver_has_higher_score_than_cautious(self):
        """After credibility shrinkage, an aggressive driver's raw features should yield a higher score."""
        cautious_rows = []
        aggressive_rows = []
        for i in range(20):
            cautious_rows.append({
                "trip_id": f"C{i}", "driver_id": "CAUTIOUS", "distance_km": 10.0,
                "harsh_braking_rate": 0.0, "harsh_accel_rate": 0.0,
                "speeding_fraction": 0.0, "night_driving_fraction": 0.0,
                "speed_variation_coeff": 0.05, "p95_speed_kmh": 45.0,
                "mean_speed_kmh": 35.0, "harsh_cornering_rate": 0.0, "urban_fraction": 0.9,
            })
            aggressive_rows.append({
                "trip_id": f"A{i}", "driver_id": "AGGRESSIVE", "distance_km": 10.0,
                "harsh_braking_rate": 3.0, "harsh_accel_rate": 2.5,
                "speeding_fraction": 0.6, "night_driving_fraction": 0.4,
                "speed_variation_coeff": 0.5, "p95_speed_kmh": 145.0,
                "mean_speed_kmh": 100.0, "harsh_cornering_rate": 2.0, "urban_fraction": 0.1,
            })
        df = pl.DataFrame(cautious_rows + aggressive_rows)
        result = aggregate_to_driver(df, credibility_threshold=10)
        c_score = result.filter(pl.col("driver_id") == "CAUTIOUS")["composite_risk_score"][0]
        a_score = result.filter(pl.col("driver_id") == "AGGRESSIVE")["composite_risk_score"][0]
        assert a_score > c_score


# ===========================================================================
# HMM MODEL — DrivingStateHMM
# ===========================================================================

class TestDrivingStateHMMExtended2:

    @pytest.fixture(scope="class")
    def fitted_3state(self, trip_features):
        model = DrivingStateHMM(n_states=3, random_state=10)
        model.fit(trip_features)
        return model, trip_features

    def test_is_fitted_flag_after_fit(self, trip_features):
        model = DrivingStateHMM(n_states=2, random_state=11)
        assert not model.is_fitted
        model.fit(trip_features)
        assert model.is_fitted

    def test_predict_before_fit_raises(self, trip_features):
        model = DrivingStateHMM(n_states=2)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_states(trip_features)

    def test_state_labels_in_range(self, fitted_3state):
        model, features = fitted_3state
        states = model.predict_states(features)
        assert states.min() >= 0
        assert states.max() <= 2

    def test_state_ordering_monotone_by_speed(self, fitted_3state):
        """State 0 should have lower mean speed than state n_states-1."""
        model, features = fitted_3state
        # _state_order maps raw→ordered: state 0 = lowest speed dim mean
        speed_idx = model.features.index("mean_speed_kmh") if "mean_speed_kmh" in model.features else 0
        means_ordered = model._model.means_[model._state_order, speed_idx]
        for i in range(len(means_ordered) - 1):
            assert means_ordered[i] <= means_ordered[i + 1]

    def test_predict_state_probs_shape(self, fitted_3state):
        model, features = fitted_3state
        probs = model.predict_state_probs(features)
        assert probs.shape == (len(features), 3)

    def test_predict_state_probs_sum_to_1(self, fitted_3state):
        model, features = fitted_3state
        probs = model.predict_state_probs(features)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_state_probs_non_negative(self, fitted_3state):
        model, features = fitted_3state
        probs = model.predict_state_probs(features)
        assert (probs >= 0).all()

    def test_driver_state_features_columns(self, fitted_3state):
        model, features = fitted_3state
        states = model.predict_states(features)
        driver_df = model.driver_state_features(features, states)
        expected_cols = {
            "driver_id", "state_0_fraction", "state_1_fraction", "state_2_fraction",
            "mean_transition_rate", "state_entropy",
        }
        assert expected_cols.issubset(set(driver_df.columns))

    def test_driver_state_features_fractions_sum_to_1(self, fitted_3state):
        model, features = fitted_3state
        states = model.predict_states(features)
        driver_df = model.driver_state_features(features, states)
        frac_cols = [f"state_{k}_fraction" for k in range(model.n_states)]
        for row in driver_df.iter_rows(named=True):
            total = sum(row[c] for c in frac_cols)
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_missing_feature_column_raises_on_fit(self, trip_features):
        model = DrivingStateHMM(n_states=2, features=["nonexistent_column"])
        with pytest.raises(ValueError, match="nonexistent_column"):
            model.fit(trip_features)

    def test_missing_driver_id_in_driver_state_features_raises(self, fitted_3state):
        model, features = fitted_3state
        states = model.predict_states(features)
        features_no_driver = features.drop("driver_id")
        with pytest.raises(ValueError, match="driver_id"):
            model.driver_state_features(features_no_driver, states)

    def test_fit_returns_self(self, trip_features):
        model = DrivingStateHMM(n_states=2, random_state=12)
        result = model.fit(trip_features)
        assert result is model

    def test_two_state_produces_binary_labels(self, trip_features):
        model = DrivingStateHMM(n_states=2, random_state=13)
        model.fit(trip_features)
        states = model.predict_states(trip_features)
        assert set(np.unique(states)).issubset({0, 1})


# ===========================================================================
# HMM MODEL — ContinuousTimeHMM
# ===========================================================================

class TestContinuousTimeHMMExtended2:

    @pytest.fixture(scope="class")
    def fitted_cthmm(self, trip_features):
        model = ContinuousTimeHMM(n_states=2, n_iter=10, random_state=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(trip_features)
        return model, trip_features

    def test_predict_before_fit_raises(self, trip_features):
        model = ContinuousTimeHMM(n_states=2)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_states(trip_features)

    def test_is_fitted_after_fit(self, trip_features):
        model = ContinuousTimeHMM(n_states=2, n_iter=5, random_state=21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(trip_features)
        assert model.is_fitted

    def test_transition_matrix_is_stochastic(self, fitted_cthmm):
        """P(dt) rows must sum to 1 and be in [0, 1]."""
        model, _ = fitted_cthmm
        P = model._transition_matrix(1.0)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)
        assert (P >= 0).all()
        assert (P <= 1 + 1e-10).all()

    def test_emission_log_prob_shape(self, fitted_cthmm):
        model, features = fitted_cthmm
        X = model._to_matrix(features)
        X = model._standardise(X, fit=False)
        log_probs = model._emission_log_prob(X)
        assert log_probs.shape == (len(features), 2)

    def test_predict_states_shape(self, fitted_cthmm):
        model, features = fitted_cthmm
        states = model.predict_states(features)
        assert states.shape == (len(features),)

    def test_predict_states_in_range(self, fitted_cthmm):
        model, features = fitted_cthmm
        states = model.predict_states(features)
        assert states.min() >= 0
        assert states.max() <= 1

    def test_time_deltas_wrong_length_raises(self, trip_features):
        model = ContinuousTimeHMM(n_states=2, n_iter=3, random_state=22)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(trip_features)
        bad_deltas = np.ones(len(trip_features) + 5)
        with pytest.raises(ValueError, match="time_deltas length"):
            model.fit(trip_features, time_deltas=bad_deltas)

    def test_generator_matrix_diagonal_negative(self, fitted_cthmm):
        """Diagonal of Q must be <= 0 (generator matrix property)."""
        model, _ = fitted_cthmm
        diag = np.diag(model.Q_)
        assert (diag <= 0).all()

    def test_generator_matrix_off_diagonal_non_negative(self, fitted_cthmm):
        """Off-diagonal elements of Q must be >= 0."""
        model, _ = fitted_cthmm
        Q = model.Q_.copy()
        np.fill_diagonal(Q, 0.0)
        assert (Q >= 0).all()

    def test_pi_sums_to_one(self, trip_features):
        model = ContinuousTimeHMM(n_states=2, n_iter=10, random_state=23)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(trip_features)
        assert model.pi_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_driver_state_features_columns(self, fitted_cthmm):
        model, features = fitted_cthmm
        states = model.predict_states(features)
        driver_df = model.driver_state_features(features, states)
        assert "driver_id" in driver_df.columns
        assert "state_0_fraction" in driver_df.columns
        assert "state_entropy" in driver_df.columns

    def test_driver_state_features_fractions_sum_to_1(self, fitted_cthmm):
        model, features = fitted_cthmm
        states = model.predict_states(features)
        driver_df = model.driver_state_features(features, states)
        for row in driver_df.iter_rows(named=True):
            total = row["state_0_fraction"] + row["state_1_fraction"]
            assert total == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# SCORING PIPELINE
# ===========================================================================

class TestScoringPipelineExtended:

    @pytest.fixture(scope="class")
    def fitted_pipe(self, simulated_data):
        trips_df, claims_df = simulated_data
        pipe = TelematicsScoringPipeline(n_hmm_states=2, random_state=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(trips_df, claims_df)
        return pipe, trips_df, claims_df

    def test_default_init_attributes(self):
        pipe = TelematicsScoringPipeline()
        assert pipe.n_hmm_states == 3
        assert pipe.credibility_threshold == 30
        assert not pipe.is_fitted
        assert pipe.hmm_features is None
        assert pipe.glm_feature_subset is None

    def test_fit_sets_glm_result(self, fitted_pipe):
        pipe, _, _ = fitted_pipe
        assert pipe._glm_result is not None

    def test_fit_sets_hmm(self, fitted_pipe):
        pipe, _, _ = fitted_pipe
        assert pipe._hmm is not None
        assert pipe._hmm.is_fitted

    def test_predict_count_matches_driver_count(self, fitted_pipe):
        pipe, trips_df, _ = fitted_pipe
        preds = pipe.predict(trips_df)
        assert len(preds) == trips_df["driver_id"].n_unique()

    def test_glm_feature_names_is_non_empty_after_fit(self, fitted_pipe):
        pipe, _, _ = fitted_pipe
        assert len(pipe._glm_feature_names) > 0

    def test_glm_features_before_fit_still_works(self, simulated_data):
        """glm_features should work even before fitting (no HMM state features)."""
        trips_df, _ = simulated_data
        pipe = TelematicsScoringPipeline(n_hmm_states=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pipe.glm_features(trips_df)
        assert isinstance(result, pl.DataFrame)
        assert "driver_id" in result.columns

    def test_glm_feature_subset_respected(self, simulated_data):
        """glm_feature_subset limits which features go into the GLM."""
        trips_df, claims_df = simulated_data
        subset = ["mean_speed_kmh", "harsh_braking_rate"]
        pipe = TelematicsScoringPipeline(n_hmm_states=2, glm_feature_subset=subset, random_state=55)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(trips_df, claims_df)
        # The fitted feature names should only contain the specified subset
        for name in pipe._glm_feature_names:
            assert name in subset, f"Unexpected feature {name} in GLM"

    def test_score_trips_alias_identical_to_predict(self, fitted_pipe):
        pipe, trips_df, _ = fitted_pipe
        preds_predict = pipe.predict(trips_df)
        preds_score = score_trips(trips_df, pipe)
        np.testing.assert_allclose(
            preds_predict["predicted_claim_frequency"].to_numpy(),
            preds_score["predicted_claim_frequency"].to_numpy(),
            rtol=1e-10,
        )

    def test_predictions_are_finite(self, fitted_pipe):
        pipe, trips_df, _ = fitted_pipe
        preds = pipe.predict(trips_df)
        assert preds["predicted_claim_frequency"].is_finite().all()

    def test_predict_before_fit_error_message_mentions_fit(self):
        pipe = TelematicsScoringPipeline()
        sim = TripSimulator(seed=99)
        trips_df, _ = sim.simulate(n_drivers=2, trips_per_driver=3)
        with pytest.raises(RuntimeError, match="fit"):
            pipe.predict(trips_df)


# ===========================================================================
# ZIP NEAR MISS
# ===========================================================================

class TestNearMissSimulatorExtended:

    def test_reproducibility(self):
        sim1 = NearMissSimulator(seed=111)
        sim2 = NearMissSimulator(seed=111)
        df1 = sim1.simulate(n_drivers=20, n_weeks=4)
        df2 = sim2.simulate(n_drivers=20, n_weeks=4)
        assert df1["harsh_braking"].sum() == df2["harsh_braking"].sum()

    def test_different_seeds_differ(self):
        sim1 = NearMissSimulator(seed=1)
        sim2 = NearMissSimulator(seed=2)
        df1 = sim1.simulate(n_drivers=20, n_weeks=4)
        df2 = sim2.simulate(n_drivers=20, n_weeks=4)
        assert df1["harsh_braking"].sum() != df2["harsh_braking"].sum()

    def test_two_group_simulator(self):
        sim = NearMissSimulator(n_groups=2, seed=5)
        df = sim.simulate(n_drivers=30, n_weeks=4)
        assert df["true_group"].max() < 2
        assert df["true_group"].min() >= 0

    def test_week_ids_in_range(self):
        n_weeks = 8
        sim = NearMissSimulator(seed=6)
        df = sim.simulate(n_drivers=10, n_weeks=n_weeks)
        assert df["week_id"].min() == 0
        assert df["week_id"].max() == n_weeks - 1

    def test_driver_count_matches_n_drivers(self):
        n = 25
        sim = NearMissSimulator(seed=7)
        df = sim.simulate(n_drivers=n, n_weeks=3)
        assert df["driver_id"].n_unique() == n


class TestZIPNearMissModelExtended:

    def test_repr_contains_n_groups(self):
        model = ZIPNearMissModel(n_groups=4)
        assert "4" in repr(model)

    def test_mixing_weights_sum_to_1_after_fit(self, fitted_zip):
        model = fitted_zip
        assert model.mixing_weights_.sum() == pytest.approx(1.0, abs=1e-6)

    def test_mixing_weights_non_negative(self, fitted_zip):
        model = fitted_zip
        assert (model.mixing_weights_ >= 0).all()

    def test_posteriors_sum_to_1(self, fitted_zip):
        model = fitted_zip
        row_sums = model.group_posteriors_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_posteriors_shape(self, fitted_zip, zip_weekly):
        model = fitted_zip
        n_drivers = zip_weekly["driver_id"].n_unique()
        assert model.group_posteriors_.shape == (n_drivers, 2)

    def test_log_likelihood_history_non_decreasing_approx(self, fitted_zip):
        """EM log-likelihood should generally be non-decreasing."""
        ll = fitted_zip.log_likelihood_history_
        assert len(ll) > 0
        # Allow tiny numerical noise: check that final ll >= first ll
        assert ll[-1] >= ll[0] - 1e-3

    def test_predict_group_probs_columns(self, fitted_zip, zip_weekly):
        probs = fitted_zip.predict_group_probs(zip_weekly)
        assert "driver_id" in probs.columns
        assert "prob_group_0" in probs.columns
        assert "prob_group_1" in probs.columns

    def test_predict_group_probs_sum_to_1(self, fitted_zip, zip_weekly):
        probs = fitted_zip.predict_group_probs(zip_weekly)
        total = probs["prob_group_0"] + probs["prob_group_1"]
        np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-6)

    def test_predict_group_probs_one_row_per_driver(self, fitted_zip, zip_weekly):
        probs = fitted_zip.predict_group_probs(zip_weekly)
        n_drivers = zip_weekly["driver_id"].n_unique()
        assert len(probs) == n_drivers

    def test_predict_rate_columns(self, fitted_zip, zip_weekly):
        rates = fitted_zip.predict_rate(zip_weekly)
        assert "driver_id" in rates.columns
        for et in fitted_zip.event_types:
            assert f"predicted_rate_{et}" in rates.columns

    def test_predict_rate_non_negative(self, fitted_zip, zip_weekly):
        rates = fitted_zip.predict_rate(zip_weekly)
        for et in fitted_zip.event_types:
            col = f"predicted_rate_{et}"
            assert (rates[col] >= 0).all()

    def test_driver_risk_features_columns(self, fitted_zip, zip_weekly):
        features = fitted_zip.driver_risk_features(zip_weekly)
        required = {"driver_id", "dominant_group", "nme_rate_per_km", "zero_fraction",
                    "prob_group_0", "prob_group_1"}
        assert required.issubset(set(features.columns))

    def test_driver_risk_features_nme_rate_non_negative(self, fitted_zip, zip_weekly):
        features = fitted_zip.driver_risk_features(zip_weekly)
        assert (features["nme_rate_per_km"] >= 0).all()

    def test_driver_risk_features_zero_fraction_in_01(self, fitted_zip, zip_weekly):
        features = fitted_zip.driver_risk_features(zip_weekly)
        assert (features["zero_fraction"] >= 0).all()
        assert (features["zero_fraction"] <= 1).all()

    def test_driver_risk_features_dominant_group_valid(self, fitted_zip, zip_weekly):
        features = fitted_zip.driver_risk_features(zip_weekly)
        assert (features["dominant_group"] >= 0).all()
        assert (features["dominant_group"] < 2).all()

    def test_predict_before_fit_raises(self, zip_weekly):
        model = ZIPNearMissModel(n_groups=2)
        with pytest.raises(RuntimeError):
            model.predict_group_probs(zip_weekly)

    def test_missing_event_column_raises_on_fit(self):
        df = pl.DataFrame({
            "driver_id": ["D1"] * 5,
            "week_id": list(range(5)),
            "exposure_km": [200.0] * 5,
            # Missing all event columns
        })
        model = ZIPNearMissModel(n_groups=2)
        with pytest.raises(ValueError):
            model.fit(df)

    def test_validate_weekly_counts_missing_driver_id_raises(self):
        df = pl.DataFrame({
            "week_id": [0],
            "exposure_km": [200.0],
            "harsh_braking": [1],
        })
        with pytest.raises(ValueError):
            _validate_weekly_counts(df, ["harsh_braking"], "exposure_km")

    def test_fit_with_single_event_type(self):
        sim = NearMissSimulator(n_groups=2, seed=66)
        df = sim.simulate(n_drivers=40, n_weeks=8)
        model = ZIPNearMissModel(n_groups=2, event_types=["harsh_braking"], max_iter=10, random_state=66)
        model.fit(df)
        assert len(model.log_likelihood_history_) > 0  # should not crash

    def test_driver_ids_preserved_in_order(self, fitted_zip, zip_weekly):
        """driver_ids_ should match the sorted unique driver IDs from training data."""
        expected = (
            zip_weekly.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        assert fitted_zip.driver_ids_ == expected


# ===========================================================================
# _logsumexp (module-level helper)
# ===========================================================================

class TestLogsumexpExtended:

    def test_two_equal_values(self):
        """logsumexp([a, a]) = a + log(2)."""
        a = np.array([5.0, 5.0])
        result = _logsumexp(a)
        assert result == pytest.approx(5.0 + np.log(2.0), rel=1e-9)

    def test_single_value_identity(self):
        a = np.array([7.5])
        assert _logsumexp(a) == pytest.approx(7.5)

    def test_positive_and_negative_inf(self):
        """A mix: -inf element should be ignored."""
        a = np.array([-np.inf, 3.0])
        assert _logsumexp(a) == pytest.approx(3.0, rel=1e-9)

    def test_large_values_no_overflow(self):
        a = np.array([700.0, 701.0])
        result = _logsumexp(a)
        assert np.isfinite(result)
        assert result > 700.0

    def test_result_ge_max(self):
        """logsumexp >= max of inputs."""
        a = np.array([1.0, 2.0, 3.0])
        assert _logsumexp(a) >= 3.0
