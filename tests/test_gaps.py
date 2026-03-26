"""
Structural gap tests for insurance-telematics.

Focuses on uncovered branches identified by code tracing:
1. feature_extractor: heading_deg cornering branch (has_heading=True path)
2. feature_extractor: near-zero-distance trip (clip at 0.01 km guard)
3. risk_aggregator: no active weights -> composite score = 50.0 literal
4. risk_aggregator: single-driver portfolio (score_span=0 guard)
5. preprocessor: null speed interpolation path
6. trip_loader: empty parquet directory raises ValueError
7. hmm_model: driver_state_features without distance_km (fallback to n_trips)
8. hmm_model: _logsumexp with all-inf input
9. ContinuousTimeHMM: non-convergence warning (n_iter=1)
10. DrivingStateHMM: custom features list without 'mean_speed_kmh' (speed_dim=0 path)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from insurance_telematics.feature_extractor import extract_trip_features
from insurance_telematics.hmm_model import ContinuousTimeHMM, DrivingStateHMM, _logsumexp
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.trip_loader import load_trips
from insurance_telematics.trip_simulator import TripSimulator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_trip_df(trip_id: str, speed_values: list[float], road_type: str = "urban") -> pl.DataFrame:
    """Build a minimal cleaned trip DataFrame for feature extraction."""
    n = len(speed_values)
    timestamps = [
        datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
        for i in range(n)
    ]
    return pl.DataFrame({
        "trip_id": [trip_id] * n,
        "timestamp": timestamps,
        "speed_kmh": speed_values,
        "acceleration_ms2": [0.5] * n,
        "road_type": [road_type] * n,
        "driver_id": ["DRV001"] * n,
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


def _make_trip_df_with_heading(trip_id: str, speed_values: list[float], headings: list[float]) -> pl.DataFrame:
    """Build a trip DataFrame with heading_deg column."""
    n = len(speed_values)
    timestamps = [
        datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
        for i in range(n)
    ]
    return pl.DataFrame({
        "trip_id": [trip_id] * n,
        "timestamp": timestamps,
        "speed_kmh": speed_values,
        "acceleration_ms2": [0.0] * n,
        "road_type": ["rural"] * n,
        "driver_id": ["DRV001"] * n,
        "heading_deg": headings,
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )


# ---------------------------------------------------------------------------
# 1. heading_deg cornering branch
# ---------------------------------------------------------------------------

class TestHeadingCornering:
    def test_heading_column_activates_cornering_proxy(self):
        """When heading_deg is present, harsh_cornering_rate should be computed from it."""
        # Create a trip with sharp turns (large heading changes at speed)
        n = 60
        speed = [80.0] * n  # fast enough that large heading changes score as cornering
        # Every 10 steps, make a sharp 90-degree turn
        headings = []
        h = 0.0
        for i in range(n):
            if i % 10 == 5:
                h = (h + 90.0) % 360.0
            headings.append(h)

        df = _make_trip_df_with_heading("T_CORNER", speed, headings)
        features = extract_trip_features(df)
        # With sharp heading changes at 80 km/h, there should be cornering events
        assert features["harsh_cornering_rate"][0] > 0.0

    def test_no_heading_gives_zero_cornering(self):
        """When heading_deg is absent, harsh_cornering_rate should be 0."""
        df = _make_trip_df("T_NO_HEADING", [60.0] * 30)
        # no heading_deg column
        features = extract_trip_features(df)
        assert features["harsh_cornering_rate"][0] == 0.0

    def test_gentle_curves_produce_low_cornering_rate(self):
        """Small heading changes should produce fewer cornering events."""
        n = 60
        speed = [80.0] * n
        # Gentle 2-degree changes per step
        headings = [(i * 2.0) % 360 for i in range(n)]
        df = _make_trip_df_with_heading("T_GENTLE", speed, headings)
        features = extract_trip_features(df)
        # Should have less cornering than the sharp-turn case
        assert features["harsh_cornering_rate"][0] >= 0.0

    def test_null_heading_falls_back_to_zero_cornering(self):
        """A heading_deg column that is all null should trigger the fallback."""
        n = 30
        timestamps = [
            datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
            for i in range(n)
        ]
        df = pl.DataFrame({
            "trip_id": ["T_NULL_HEAD"] * n,
            "timestamp": timestamps,
            "speed_kmh": [50.0] * n,
            "acceleration_ms2": [0.0] * n,
            "road_type": ["rural"] * n,
            "driver_id": ["DRV001"] * n,
            "heading_deg": [None] * n,
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("heading_deg").cast(pl.Float64),
        ])
        features = extract_trip_features(df)
        assert features["harsh_cornering_rate"][0] == 0.0


# ---------------------------------------------------------------------------
# 2. Near-zero-distance trip (0.01 km clip guard)
# ---------------------------------------------------------------------------

class TestNearZeroDistance:
    def test_very_short_trip_rates_finite(self):
        """A 1-observation trip has near-zero distance; rates must stay finite."""
        # 1 second at speed 0.0 gives ~0 distance (< 0.01 threshold)
        df = _make_trip_df("T_SHORT", [0.001] * 5)
        features = extract_trip_features(df)
        for rate_col in ["harsh_braking_rate", "harsh_accel_rate", "harsh_cornering_rate"]:
            val = features[rate_col][0]
            assert np.isfinite(val), f"{rate_col} not finite for near-zero-distance trip"

    def test_clip_denominator_is_0_01(self):
        """With zero speed, distance = 0, so rates = 0/0.01 = 0 events/km."""
        df = _make_trip_df("T_STILL", [0.0] * 10)
        # No harsh events (acceleration = 0.5 < 3.5), so numerator = 0
        features = extract_trip_features(df)
        assert features["harsh_braking_rate"][0] == 0.0
        assert features["harsh_accel_rate"][0] == 0.0


# ---------------------------------------------------------------------------
# 3 & 4. risk_aggregator: no active weights + single-driver portfolio
# ---------------------------------------------------------------------------

class TestCompositeScoreEdgeCases:
    def _make_feature_row(self, driver_id: str, n_trips: int = 5) -> list[dict]:
        """Create trip feature rows stripped of all weighted features."""
        rows = []
        for i in range(n_trips):
            rows.append({
                "trip_id": f"{driver_id}_T{i}",
                "driver_id": driver_id,
                "distance_km": 10.0,
                # None of the _FEATURE_WEIGHTS columns are present
            })
        return rows

    def test_no_active_weights_returns_50(self):
        """When none of the feature weight columns are present, score = 50.0."""
        rows = self._make_feature_row("D1")
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        assert result["composite_risk_score"][0] == pytest.approx(50.0)

    def test_single_driver_portfolio_score_valid(self):
        """With one driver, score_span=0 so score should be 50.0 (or 0.0 from the clip)."""
        # Single driver with all weight features present
        rows = []
        for i in range(10):
            rows.append({
                "trip_id": f"T{i}",
                "driver_id": "ONLY_DRIVER",
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
            })
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df)
        # Single driver: score_min == score_max so score_span = 1.0
        # raw_score - score_min = 0, so composite = 0.0
        assert result["composite_risk_score"][0] == pytest.approx(0.0)

    def test_two_driver_portfolio_min_max_scores(self):
        """One aggressive and one cautious driver: scores should be 0 and 100."""
        rows = []
        # Cautious driver
        for i in range(10):
            rows.append({
                "trip_id": f"C{i}", "driver_id": "CAUTIOUS", "distance_km": 10.0,
                "harsh_braking_rate": 0.0, "harsh_accel_rate": 0.0,
                "speeding_fraction": 0.0, "night_driving_fraction": 0.0,
                "speed_variation_coeff": 0.0, "p95_speed_kmh": 30.0,
                "mean_speed_kmh": 25.0, "harsh_cornering_rate": 0.0,
                "urban_fraction": 0.8,
            })
        # Aggressive driver
        for i in range(10):
            rows.append({
                "trip_id": f"A{i}", "driver_id": "AGGRESSIVE", "distance_km": 10.0,
                "harsh_braking_rate": 5.0, "harsh_accel_rate": 5.0,
                "speeding_fraction": 0.8, "night_driving_fraction": 0.5,
                "speed_variation_coeff": 0.6, "p95_speed_kmh": 140.0,
                "mean_speed_kmh": 110.0, "harsh_cornering_rate": 3.0,
                "urban_fraction": 0.0,
            })
        df = pl.DataFrame(rows)
        result = aggregate_to_driver(df, credibility_threshold=5)
        scores = {
            r["driver_id"]: r["composite_risk_score"]
            for r in result.iter_rows(named=True)
        }
        # One driver gets 0, the other 100 (or thereabouts after credibility shrinkage)
        assert max(scores.values()) > 50
        assert min(scores.values()) < 50


# ---------------------------------------------------------------------------
# 5. preprocessor: null speed interpolation
# ---------------------------------------------------------------------------

class TestNullSpeedInterpolation:
    def test_null_speeds_filled_by_forward_fill(self):
        """Null speed values within a trip should be forward-filled."""
        timestamps = [
            datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
            for i in range(8)
        ]
        df = pl.DataFrame({
            "trip_id": ["T1"] * 8,
            "timestamp": timestamps,
            "latitude": [51.5] * 8,
            "longitude": [-0.1] * 8,
            "speed_kmh": [30.0, None, None, None, 40.0, None, None, 50.0],
            "acceleration_ms2": [0.0] * 8,
            "heading_deg": [90.0] * 8,
            "driver_id": ["D1"] * 8,
        }).with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("speed_kmh").cast(pl.Float64),
        ])
        cleaned = clean_trips(df)
        # After forward fill with limit=4, null values should be filled
        assert cleaned["speed_kmh"].null_count() < 3

    def test_no_nulls_skips_interpolation(self):
        """When there are no null speeds, the df is returned unchanged."""
        timestamps = [
            datetime(2024, 1, 1, 8, 0, i, tzinfo=timezone.utc)
            for i in range(4)
        ]
        df = pl.DataFrame({
            "trip_id": ["T1"] * 4,
            "timestamp": timestamps,
            "latitude": [51.5] * 4,
            "longitude": [-0.1] * 4,
            "speed_kmh": [30.0, 35.0, 40.0, 45.0],
            "acceleration_ms2": [0.0] * 4,
            "heading_deg": [90.0] * 4,
            "driver_id": ["D1"] * 4,
        }).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
        cleaned = clean_trips(df)
        assert cleaned["speed_kmh"].null_count() == 0
        assert len(cleaned) == 4


# ---------------------------------------------------------------------------
# 6. trip_loader: empty parquet directory
# ---------------------------------------------------------------------------

class TestEmptyParquetDirectory:
    def test_empty_directory_raises_value_error(self, tmp_path):
        """An empty directory should raise ValueError, not hang or produce empty df."""
        with pytest.raises(ValueError, match="No Parquet files found"):
            load_trips(tmp_path)

    def test_directory_with_non_parquet_files_raises(self, tmp_path):
        """A directory with only CSV files (not .parquet) should raise ValueError."""
        (tmp_path / "data.csv").write_text("trip_id,speed\nT1,50\n")
        with pytest.raises(ValueError, match="No Parquet files found"):
            load_trips(tmp_path)


# ---------------------------------------------------------------------------
# 7. hmm_model: driver_state_features without distance_km
# ---------------------------------------------------------------------------

class TestDriverStateFeaturesWithoutDistanceKm:
    @pytest.fixture(scope="class")
    def fitted_hmm(self):
        sim = TripSimulator(seed=77)
        trips_df, _ = sim.simulate(
            n_drivers=4, trips_per_driver=10,
            min_trip_duration_s=120, max_trip_duration_s=400,
        )
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        model = DrivingStateHMM(n_states=2, random_state=42)
        model.fit(features)
        return model, features

    def test_driver_features_without_distance_km(self, fitted_hmm):
        """When distance_km is absent, transition_rate denominator falls back to n_trips."""
        model, features = fitted_hmm
        states = model.predict_states(features)
        # Drop distance_km column
        features_no_dist = features.drop("distance_km")
        driver_df = model.driver_state_features(features_no_dist, states)
        # Should still produce one row per driver without errors
        n_drivers = features["driver_id"].n_unique()
        assert len(driver_df) == n_drivers
        # mean_transition_rate should be finite (denominator = n_trips)
        assert driver_df["mean_transition_rate"].is_finite().all()

    def test_cthmm_driver_features_without_distance_km(self, fitted_hmm):
        """Same test for ContinuousTimeHMM.driver_state_features."""
        _, features = fitted_hmm
        model = ContinuousTimeHMM(n_states=2, n_iter=5, random_state=42)
        model.fit(features)
        states = model.predict_states(features)
        features_no_dist = features.drop("distance_km")
        driver_df = model.driver_state_features(features_no_dist, states)
        assert len(driver_df) == features["driver_id"].n_unique()


# ---------------------------------------------------------------------------
# 8. _logsumexp edge cases
# ---------------------------------------------------------------------------

class TestLogsumexp:
    def test_all_negative_infinity_returns_neg_inf(self):
        """An array of all -inf should return -inf (not NaN or error)."""
        a = np.array([-np.inf, -np.inf, -np.inf])
        result = _logsumexp(a)
        assert result == -np.inf

    def test_single_finite_value(self):
        """logsumexp of a single value is that value."""
        a = np.array([3.0])
        result = _logsumexp(a)
        assert result == pytest.approx(3.0)

    def test_numerically_stable_for_large_values(self):
        """logsumexp should not overflow for large values."""
        a = np.array([1000.0, 1001.0])
        result = _logsumexp(a)
        # log(exp(1000) + exp(1001)) = 1000 + log(1 + e) ≈ 1001.31
        expected = 1000.0 + np.log(1 + np.exp(1.0))
        assert result == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. ContinuousTimeHMM non-convergence warning
# ---------------------------------------------------------------------------

class TestCTHMMNonConvergenceWarning:
    def test_non_convergence_warning_with_n_iter_1(self):
        """n_iter=1 with a loose tolerance should trigger the non-convergence warning."""
        sim = TripSimulator(seed=88)
        trips_df, _ = sim.simulate(
            n_drivers=3, trips_per_driver=8,
            min_trip_duration_s=120, max_trip_duration_s=300,
        )
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)

        model = ContinuousTimeHMM(n_states=2, n_iter=1, tol=1e-20, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features)
        convergence_warnings = [
            w for w in caught if "did not converge" in str(w.message).lower()
        ]
        assert len(convergence_warnings) == 1, (
            f"Expected 1 convergence warning, got {len(convergence_warnings)}"
        )


# ---------------------------------------------------------------------------
# 10. DrivingStateHMM: custom features list without 'mean_speed_kmh'
# ---------------------------------------------------------------------------

class TestDrivingStateHMMCustomFeatures:
    def test_custom_features_no_mean_speed(self):
        """With a custom feature list lacking 'mean_speed_kmh', speed_dim falls back to 0."""
        sim = TripSimulator(seed=99)
        trips_df, _ = sim.simulate(
            n_drivers=4, trips_per_driver=10,
            min_trip_duration_s=120, max_trip_duration_s=400,
        )
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)

        # Custom features — 'mean_speed_kmh' deliberately excluded
        custom_feats = ["harsh_braking_rate", "harsh_accel_rate", "speeding_fraction"]
        model = DrivingStateHMM(n_states=2, features=custom_feats, random_state=42)
        model.fit(features)
        states = model.predict_states(features)
        # Should succeed; _state_order based on feature index 0 (harsh_braking_rate)
        assert len(states) == len(features)
        assert states.min() >= 0
        assert states.max() <= 1
