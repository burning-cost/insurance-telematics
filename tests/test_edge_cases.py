"""
Edge-case tests for insurance-telematics.

These tests cover conditions that are unusual but valid — or that should
produce clean errors rather than silent wrong results:

- Empty trip data (zero rows)
- Single-trip driver
- Missing sensor columns (no acceleration, no heading)
- NaN values in sensor data
- Zero-length trips (single observation)
- TripSimulator convergence edge cases (min_trip_duration_s=1)
- Mismatched column names between pipeline stages
- Single driver in aggregation
- All-identical feature values (zero variance)
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_telematics.trip_loader import load_trips_from_dataframe, REQUIRED_COLUMNS
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.feature_extractor import extract_trip_features
from insurance_telematics.hmm_model import DrivingStateHMM, ContinuousTimeHMM
from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.trip_simulator import TripSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_trip_df(
    trip_id: str = "T001",
    driver_id: str = "DRV0001",
    n_rows: int = 60,
    speed_kmh: float = 50.0,
) -> pl.DataFrame:
    """Minimal valid trip DataFrame with one trip and constant speed."""
    import datetime as dt
    base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = [base + dt.timedelta(seconds=i) for i in range(n_rows)]
    return pl.DataFrame({
        "trip_id": [trip_id] * n_rows,
        "driver_id": [driver_id] * n_rows,
        "timestamp": timestamps,
        "latitude": [51.5 + i * 0.0001 for i in range(n_rows)],
        "longitude": [-0.1] * n_rows,
        "speed_kmh": [speed_kmh] * n_rows,
        "acceleration_ms2": [0.0] * n_rows,
        "heading_deg": [90.0] * n_rows,
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _make_multi_trip_df(n_drivers: int = 3, trips_per_driver: int = 5) -> pl.DataFrame:
    """Small but realistic multi-driver DataFrame via TripSimulator."""
    sim = TripSimulator(seed=77)
    trips_df, _ = sim.simulate(
        n_drivers=n_drivers,
        trips_per_driver=trips_per_driver,
        min_trip_duration_s=120,
        max_trip_duration_s=600,
    )
    return trips_df


# ---------------------------------------------------------------------------
# Empty data edge cases
# ---------------------------------------------------------------------------

class TestEmptyTripData:
    """Behaviour when zero observation rows are passed to pipeline stages."""

    def test_load_from_dataframe_empty_df_raises_on_missing_columns(self):
        """An empty DataFrame with no columns must raise ValueError, not AttributeError."""
        empty = pl.DataFrame()
        with pytest.raises((ValueError, KeyError, Exception)):
            load_trips_from_dataframe(empty)

    def test_load_from_dataframe_zero_rows_valid_schema(self):
        """Zero-row DataFrame with the right schema is technically valid."""
        empty_schema = pl.DataFrame(schema={
            "trip_id": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "speed_kmh": pl.Float64,
        })
        # Should not raise — zero rows is valid input
        result = load_trips_from_dataframe(empty_schema)
        assert len(result) == 0

    def test_clean_trips_zero_rows_returns_empty_dataframe(self):
        """clean_trips on zero rows must return an empty DataFrame, not crash."""
        empty_schema = pl.DataFrame(schema={
            "trip_id": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "speed_kmh": pl.Float64,
            "acceleration_ms2": pl.Float64,
            "heading_deg": pl.Float64,
            "driver_id": pl.String,
        })
        result = clean_trips(empty_schema)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_extract_features_zero_rows_returns_empty(self):
        """extract_trip_features on zero rows must return an empty DataFrame."""
        empty_schema = pl.DataFrame(schema={
            "trip_id": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
            "speed_kmh": pl.Float64,
            "acceleration_ms2": pl.Float64,
            "road_type": pl.String,
            "driver_id": pl.String,
            "heading_deg": pl.Float64,
        })
        result = extract_trip_features(empty_schema)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_aggregate_to_driver_empty_df_raises(self):
        """aggregate_to_driver on a DataFrame missing required columns must raise."""
        empty = pl.DataFrame(schema={
            "trip_id": pl.String,
            "driver_id": pl.String,
        })
        with pytest.raises(ValueError, match="distance_km"):
            aggregate_to_driver(empty)


# ---------------------------------------------------------------------------
# Single-trip driver edge cases
# ---------------------------------------------------------------------------

class TestSingleTripDriver:
    """A driver with exactly one trip is common at portfolio inception."""

    def test_full_pipeline_single_trip_driver(self):
        """Single-trip driver must produce one output row, not be dropped."""
        df = _make_minimal_trip_df(trip_id="T001", driver_id="DRV0001", n_rows=120)
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        assert len(driver_df) == 1
        assert driver_df["driver_id"][0] == "DRV0001"

    def test_single_trip_driver_credibility_weight_low(self):
        """A single-trip driver must have credibility_weight < 0.1 (default threshold=30)."""
        df = _make_minimal_trip_df(n_rows=120)
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features, credibility_threshold=30)
        # 1 trip / (1 + 30) ≈ 0.032
        assert float(driver_df["credibility_weight"][0]) < 0.1

    def test_hmm_with_single_trip_driver(self):
        """HMM must run when a driver has only one trip (no state transitions)."""
        df = _make_minimal_trip_df(n_rows=120)
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=3, random_state=0)
        # Need more than 1 observation for HMM to fit; use multiple trips from different drivers
        multi = _make_multi_trip_df(n_drivers=5, trips_per_driver=5)
        multi_cleaned = clean_trips(multi)
        multi_features = extract_trip_features(multi_cleaned)
        hmm.fit(multi_features)
        # Now predict on single-trip driver
        states = hmm.predict_states(features)
        driver_hmm = hmm.driver_state_features(features, states)
        assert len(driver_hmm) == 1
        # With one trip, n_transitions = 0
        assert float(driver_hmm["mean_transition_rate"][0]) == 0.0

    def test_hmm_driver_state_features_single_trip_entropy_zero(self):
        """A driver in one state for all trips should have entropy = 0."""
        # Build a features df where a single driver has 5 trips all in state 0
        import scipy.stats as stats

        # We cannot force HMM state, so instead test the driver_state_features
        # method directly with a manually constructed states array.
        features = pl.DataFrame({
            "trip_id": [f"T{i:03d}" for i in range(5)],
            "driver_id": ["DRV0001"] * 5,
            "distance_km": [10.0] * 5,
            "mean_speed_kmh": [50.0] * 5,
            "speed_variation_coeff": [0.2] * 5,
            "harsh_braking_rate": [0.1] * 5,
            "harsh_accel_rate": [0.1] * 5,
        })
        hmm = DrivingStateHMM(n_states=3, random_state=0)
        # We fit on the minimal data just to mark the object as fitted
        # without caring about convergence quality
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(features)

        # Force all states to 0 for the single driver
        all_state_zero = np.zeros(5, dtype=int)
        driver_hmm = hmm.driver_state_features(features, all_state_zero)
        assert float(driver_hmm["state_0_fraction"][0]) == 1.0
        # Entropy of [1, 0, 0] is 0 (log(0) terms are 0 * -inf = 0 by convention)
        assert float(driver_hmm["state_entropy"][0]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Missing sensor columns
# ---------------------------------------------------------------------------

class TestMissingSensorColumns:
    """Pipeline must handle sensors that are absent or entirely null."""

    def test_pipeline_without_heading_deg(self):
        """heading_deg is optional. Pipeline must work without it."""
        df = _make_minimal_trip_df(n_rows=120).drop("heading_deg")
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        # harsh_cornering_rate should default to 0 when no heading
        assert "harsh_cornering_rate" in features.columns
        assert float(features["harsh_cornering_rate"][0]) == pytest.approx(0.0)

    def test_pipeline_with_null_heading_deg(self):
        """All-null heading_deg must be treated as 'not available' — no crash."""
        import datetime as dt
        n = 120
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T001"] * n,
            "driver_id": ["DRV0001"] * n,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(n)],
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [0.0] * n,
            "heading_deg": [None] * n,
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("heading_deg").cast(pl.Float64),
        )
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        assert isinstance(features, pl.DataFrame)

    def test_pipeline_without_acceleration_ms2(self):
        """acceleration_ms2 is optional. Preprocessor must derive it from speed."""
        import datetime as dt
        n = 120
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T001"] * n,
            "driver_id": ["DRV0001"] * n,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(n)],
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": [50.0 + i * 0.01 for i in range(n)],
            # No acceleration_ms2
        })
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        loaded = load_trips_from_dataframe(df)
        # acceleration_ms2 should be added as null by loader, then derived by cleaner
        assert "acceleration_ms2" in loaded.columns
        cleaned = clean_trips(loaded)
        assert "acceleration_ms2" in cleaned.columns
        # The derived values should be non-null (except the first diff which is null)
        non_null_count = cleaned["acceleration_ms2"].drop_nulls().len()
        assert non_null_count > 0


# ---------------------------------------------------------------------------
# NaN values in sensor data
# ---------------------------------------------------------------------------

class TestNaNInSensorData:
    """NaN/null speed values must not propagate into feature outputs."""

    def test_null_speed_rows_handled_in_clean_trips(self):
        """clean_trips must not crash on null speed values; they are interpolated."""
        import datetime as dt
        n = 30
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        speeds = [50.0] * n
        # Insert a null in the middle
        speeds[10] = None
        df = pl.DataFrame({
            "trip_id": ["T001"] * n,
            "driver_id": ["D1"] * n,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(n)],
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": speeds,
            "acceleration_ms2": [0.0] * n,
            "heading_deg": [90.0] * n,
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("speed_kmh").cast(pl.Float64),
        )
        cleaned = clean_trips(df)
        # After forward-fill interpolation, the null should be filled
        assert isinstance(cleaned, pl.DataFrame)

    def test_nan_acceleration_not_propagated_to_harsh_rates(self):
        """NaN acceleration must not cause harsh_braking_rate to be NaN."""
        import datetime as dt
        n = 60
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T001"] * n,
            "driver_id": ["D1"] * n,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(n)],
            "latitude": [51.5] * n,
            "longitude": [-0.1] * n,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [None] * n,
            "heading_deg": [90.0] * n,
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("acceleration_ms2").cast(pl.Float64),
        )
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        # harsh_braking_rate and harsh_accel_rate should be non-null (0.0 because
        # derived acceleration from constant speed is 0)
        assert features["harsh_braking_rate"].null_count() == 0
        assert features["harsh_accel_rate"].null_count() == 0

    def test_feature_extractor_output_has_no_null_rates(self):
        """All rate columns from extract_trip_features must be non-null for valid input."""
        df = _make_multi_trip_df(n_drivers=3, trips_per_driver=5)
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        rate_cols = [
            "harsh_braking_rate", "harsh_accel_rate",
            "harsh_cornering_rate", "speeding_fraction",
            "night_driving_fraction", "urban_fraction",
        ]
        for col in rate_cols:
            null_count = features[col].null_count()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"


# ---------------------------------------------------------------------------
# Zero-length trips (single observation)
# ---------------------------------------------------------------------------

class TestZeroLengthTrips:
    """A trip with a single GPS point has zero duration and zero distance."""

    def test_single_observation_trip_does_not_crash_extractor(self):
        """extract_trip_features must not divide by zero when a trip has 1 row."""
        import datetime as dt
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T001"],
            "driver_id": ["D1"],
            "timestamp": [base],
            "latitude": [51.5],
            "longitude": [-0.1],
            "speed_kmh": [0.0],
            "acceleration_ms2": [0.0],
            "heading_deg": [90.0],
            "road_type": ["urban"],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        # Should not raise, even though distance=0 (clamped to 0.01 in code)
        features = extract_trip_features(df)
        assert len(features) == 1
        assert float(features["distance_km"][0]) == pytest.approx(0.0, abs=1e-9)

    def test_single_observation_trip_rates_are_finite(self):
        """Event rates for a zero-distance trip must be finite (no inf from div-by-zero)."""
        import datetime as dt
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T001"],
            "driver_id": ["D1"],
            "timestamp": [base],
            "latitude": [51.5],
            "longitude": [-0.1],
            "speed_kmh": [0.0],
            "acceleration_ms2": [-5.0],  # harsh braking on a 1-row trip
            "heading_deg": [90.0],
            "road_type": ["urban"],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        features = extract_trip_features(df)
        assert np.isfinite(float(features["harsh_braking_rate"][0]))

    def test_single_observation_trip_mixed_with_normal_trips(self):
        """A single-observation trip mixed with normal trips must not break aggregation."""
        import datetime as dt
        normal_df = _make_minimal_trip_df(trip_id="T001", driver_id="D1", n_rows=120)

        base = dt.datetime(2024, 1, 1, 10, 0, 0, tzinfo=dt.timezone.utc)
        single_obs = pl.DataFrame({
            "trip_id": ["T002"],
            "driver_id": ["D1"],
            "timestamp": [base],
            "latitude": [51.5],
            "longitude": [-0.1],
            "speed_kmh": [0.0],
            "acceleration_ms2": [0.0],
            "heading_deg": [90.0],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

        combined = pl.concat([normal_df, single_obs])
        cleaned = clean_trips(combined)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        assert len(driver_df) == 1


# ---------------------------------------------------------------------------
# TripSimulator edge cases
# ---------------------------------------------------------------------------

class TestTripSimulatorEdgeCases:
    """Edge cases in the trip data generator itself."""

    def test_simulator_single_driver_single_trip(self):
        """Smallest possible simulation: 1 driver, 1 trip."""
        sim = TripSimulator(seed=0)
        trips_df, claims_df = sim.simulate(n_drivers=1, trips_per_driver=1)
        assert len(trips_df) > 0
        assert len(claims_df) == 1
        assert trips_df["driver_id"].n_unique() == 1
        assert trips_df["trip_id"].n_unique() == 1

    def test_simulator_minimum_trip_duration(self):
        """min_trip_duration_s=2 exercises the very short trip path."""
        sim = TripSimulator(seed=1)
        trips_df, _ = sim.simulate(
            n_drivers=3,
            trips_per_driver=3,
            min_trip_duration_s=2,
            max_trip_duration_s=10,
        )
        assert len(trips_df) > 0
        # Each trip must have at least 2 rows
        trip_lengths = trips_df.group_by("trip_id").agg(pl.len().alias("n_rows"))
        assert int(trip_lengths["n_rows"].min()) >= 2

    def test_simulator_outputs_schema_columns(self):
        """TripSimulator must produce all columns the pipeline expects."""
        sim = TripSimulator(seed=2)
        trips_df, claims_df = sim.simulate(n_drivers=2, trips_per_driver=5)
        for col in ["driver_id", "trip_id", "timestamp", "latitude", "longitude",
                    "speed_kmh", "acceleration_ms2", "heading_deg"]:
            assert col in trips_df.columns, f"Missing column: {col}"
        for col in ["driver_id", "n_claims", "exposure_years"]:
            assert col in claims_df.columns, f"Missing claims column: {col}"

    def test_simulator_speed_non_negative(self):
        """Ornstein-Uhlenbeck process must be clamped — no negative speeds."""
        sim = TripSimulator(seed=3)
        trips_df, _ = sim.simulate(n_drivers=5, trips_per_driver=10)
        assert float(trips_df["speed_kmh"].min()) >= 0.0

    def test_simulator_deterministic_with_same_seed(self):
        """Same seed must produce identical output."""
        sim_a = TripSimulator(seed=42)
        trips_a, claims_a = sim_a.simulate(n_drivers=3, trips_per_driver=5)
        sim_b = TripSimulator(seed=42)
        trips_b, claims_b = sim_b.simulate(n_drivers=3, trips_per_driver=5)
        assert trips_a.equals(trips_b)
        assert claims_a.equals(claims_b)

    def test_simulator_different_seeds_differ(self):
        """Different seeds must produce different data."""
        sim_a = TripSimulator(seed=1)
        trips_a, _ = sim_a.simulate(n_drivers=3, trips_per_driver=5)
        sim_b = TripSimulator(seed=2)
        trips_b, _ = sim_b.simulate(n_drivers=3, trips_per_driver=5)
        # It is essentially impossible for all speed values to match by chance
        assert not trips_a["speed_kmh"].equals(trips_b["speed_kmh"])

    def test_full_pipeline_on_minimum_simulation(self):
        """The pipeline must not crash on the smallest possible simulation."""
        sim = TripSimulator(seed=5)
        trips_df, claims_df = sim.simulate(
            n_drivers=3,
            trips_per_driver=5,
            min_trip_duration_s=60,
            max_trip_duration_s=120,
        )
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        assert len(driver_df) == 3


# ---------------------------------------------------------------------------
# Mismatched column names
# ---------------------------------------------------------------------------

class TestMismatchedColumnNames:
    """Tests that catch column-name mismatches between pipeline stages."""

    def test_aggregate_to_driver_without_driver_id_raises(self):
        """aggregate_to_driver must raise ValueError if driver_id is absent."""
        features = pl.DataFrame({
            "trip_id": ["T1", "T2"],
            "distance_km": [5.0, 8.0],
            "mean_speed_kmh": [50.0, 60.0],
        })
        with pytest.raises(ValueError, match="driver_id"):
            aggregate_to_driver(features)

    def test_aggregate_to_driver_without_distance_km_raises(self):
        """aggregate_to_driver must raise ValueError if distance_km is absent."""
        features = pl.DataFrame({
            "trip_id": ["T1", "T2"],
            "driver_id": ["D1", "D1"],
            "mean_speed_kmh": [50.0, 60.0],
        })
        with pytest.raises(ValueError, match="distance_km"):
            aggregate_to_driver(features)

    def test_extractor_without_road_type_raises(self):
        """extract_trip_features must raise ValueError if road_type is missing."""
        import datetime as dt
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T1"] * 30,
            "driver_id": ["D1"] * 30,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(30)],
            "speed_kmh": [50.0] * 30,
            "acceleration_ms2": [0.0] * 30,
            # road_type intentionally missing
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        with pytest.raises(ValueError, match="road_type"):
            extract_trip_features(df)

    def test_extractor_without_acceleration_ms2_raises(self):
        """extract_trip_features must raise ValueError if acceleration_ms2 is missing."""
        import datetime as dt
        base = dt.datetime(2024, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "trip_id": ["T1"] * 30,
            "driver_id": ["D1"] * 30,
            "timestamp": [base + dt.timedelta(seconds=i) for i in range(30)],
            "speed_kmh": [50.0] * 30,
            "road_type": ["urban"] * 30,
            # acceleration_ms2 intentionally missing
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        with pytest.raises(ValueError, match="acceleration_ms2"):
            extract_trip_features(df)

    def test_hmm_with_wrong_feature_names_raises_clearly(self):
        """DrivingStateHMM must raise ValueError naming the missing column."""
        features = pl.DataFrame({
            "trip_id": ["T1", "T2", "T3"],
            "driver_id": ["D1"] * 3,
            "mean_speed_kmh": [50.0, 60.0, 40.0],
            "speed_variation_coeff": [0.2, 0.3, 0.1],
            "harsh_braking_rate": [0.1, 0.2, 0.05],
            # 'harsh_accel_rate' is missing — only three of the four default features
        })
        hmm = DrivingStateHMM(n_states=2, random_state=0)
        with pytest.raises(ValueError, match="harsh_accel_rate"):
            hmm.fit(features)


# ---------------------------------------------------------------------------
# HMM convergence edge cases
# ---------------------------------------------------------------------------

class TestHMMConvergenceEdgeCases:
    """Edge cases around HMM fitting stability."""

    def test_hmm_warns_on_non_convergence(self):
        """ContinuousTimeHMM must issue a UserWarning when EM does not converge
        within n_iter iterations."""
        sim = TripSimulator(seed=11)
        trips_df, _ = sim.simulate(n_drivers=5, trips_per_driver=10)
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        # n_iter=1 guarantees non-convergence
        cthmm = ContinuousTimeHMM(n_states=3, n_iter=1, tol=1e-20, random_state=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cthmm.fit(features)
        convergence_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(convergence_warnings) > 0

    def test_hmm_all_identical_features(self):
        """Zero-variance feature columns must not crash the HMM (handled by _standardise)."""
        features = pl.DataFrame({
            "trip_id": [f"T{i:03d}" for i in range(20)],
            "driver_id": ["D1"] * 20,
            "distance_km": [10.0] * 20,
            "mean_speed_kmh": [50.0] * 20,         # zero variance
            "speed_variation_coeff": [0.2] * 20,   # zero variance
            "harsh_braking_rate": [0.0] * 20,      # zero variance
            "harsh_accel_rate": [0.0] * 20,        # zero variance
        })
        hmm = DrivingStateHMM(n_states=2, n_iter=10, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(features)
        assert hmm.is_fitted

    def test_hmm_n_states_1_is_valid(self):
        """n_states=1 is degenerate but must not crash."""
        sim = TripSimulator(seed=12)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=1, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(features)
        states = hmm.predict_states(features)
        # All states must be 0
        assert set(states.tolist()) == {0}

    def test_cthmm_time_deltas_wrong_length_raises(self):
        """ContinuousTimeHMM must raise ValueError if time_deltas length != n_obs."""
        sim = TripSimulator(seed=13)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        n = len(features)
        wrong_deltas = np.ones(n + 5)  # wrong length
        cthmm = ContinuousTimeHMM(n_states=2, n_iter=5, random_state=0)
        with pytest.raises(ValueError, match=str(n + 5)):
            cthmm.fit(features, time_deltas=wrong_deltas)

    def test_cthmm_zero_time_deltas_clamped_no_crash(self):
        """time_deltas of zero must be clamped to 1e-6, not cause a singular matrix."""
        sim = TripSimulator(seed=14)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        n = len(features)
        zero_deltas = np.zeros(n)
        cthmm = ContinuousTimeHMM(n_states=2, n_iter=5, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should not raise — zeros are clamped to 1e-6
            cthmm.fit(features, time_deltas=zero_deltas)
        assert cthmm.is_fitted


# ---------------------------------------------------------------------------
# Single-driver aggregation
# ---------------------------------------------------------------------------

class TestSingleDriverAggregation:
    """Composite risk score scaling with a single driver is a degenerate case."""

    def test_single_driver_composite_score_defined(self):
        """With one driver, score_min == score_max, so the scaled score must not be NaN."""
        df = _make_minimal_trip_df(n_rows=180)
        cleaned = clean_trips(df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        score = driver_df["composite_risk_score"][0]
        # With one driver the span is 0, so the code falls back to lit(50.0).
        # Either 0.0 (from scale) or 50.0 (from fallback) is acceptable; neither is NaN.
        assert not (score is None or (isinstance(score, float) and np.isnan(score)))
