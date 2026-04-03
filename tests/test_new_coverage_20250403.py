"""
New coverage tests for insurance-telematics (2025-04-03).

Focus areas:
  - DrivingStateHMM: n_states=4, custom features, missing driver_id,
    predict_state_probs column order, covariance_type variants
  - ContinuousTimeHMM: 2-state model, convergence warning, predict before fit,
    driver_state_features without distance_km, degenerate time_deltas
  - _logsumexp edge cases (internal helper)
  - ZIPNearMissModel: custom event_types, custom exposure_col, repr, relabelling,
    two-group model, predict_rate output, driver_risk_features columns,
    _mom_zip edge cases, two-driver edge case, empty-week edge case
  - Feature extractor: with heading_deg, no driver_id, zero-speed trip,
    all-night trip, all-motorway trip, single-observation trip
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_telematics.hmm_model import (
    DrivingStateHMM,
    ContinuousTimeHMM,
    _logsumexp,
)
from insurance_telematics.zip_near_miss import (
    NearMissSimulator,
    ZIPNearMissModel,
    _DEFAULT_EVENT_TYPES,
    _validate_weekly_counts,
)
from insurance_telematics.feature_extractor import extract_trip_features
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.preprocessor import clean_trips


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features_with_driver():
    sim = TripSimulator(seed=77)
    trips_df, _ = sim.simulate(
        n_drivers=8,
        trips_per_driver=20,
        min_trip_duration_s=120,
        max_trip_duration_s=600,
    )
    cleaned = clean_trips(trips_df)
    return extract_trip_features(cleaned)


@pytest.fixture(scope="module")
def small_weekly():
    sim = NearMissSimulator(n_groups=3, seed=99)
    return sim.simulate(n_drivers=50, n_weeks=12)


@pytest.fixture(scope="module")
def fitted_zip_2group():
    sim = NearMissSimulator(n_groups=2, seed=55)
    df = sim.simulate(n_drivers=60, n_weeks=16)
    model = ZIPNearMissModel(n_groups=2, max_iter=25, random_state=0)
    model.fit(df)
    return model, df


# ---------------------------------------------------------------------------
# DrivingStateHMM — additional edge cases
# ---------------------------------------------------------------------------


class TestDrivingStateHMMExtended:
    def test_four_state_model(self, features_with_driver):
        """n_states=4 should produce states in range [0, 3]."""
        model = DrivingStateHMM(n_states=4, random_state=1)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.min() >= 0
        assert states.max() <= 3

    def test_custom_feature_list(self, features_with_driver):
        """Model should work with a subset of features."""
        model = DrivingStateHMM(
            n_states=2,
            features=["mean_speed_kmh", "harsh_braking_rate"],
            random_state=2,
        )
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.shape == (len(features_with_driver),)

    def test_predict_state_probs_columns_match_n_states(self, features_with_driver):
        """predict_state_probs should return n_states columns."""
        for n in [2, 4]:
            model = DrivingStateHMM(n_states=n, random_state=3)
            model.fit(features_with_driver)
            probs = model.predict_state_probs(features_with_driver)
            assert probs.shape[1] == n

    def test_predict_state_probs_non_negative(self, features_with_driver):
        """Posterior probabilities should all be non-negative."""
        model = DrivingStateHMM(n_states=3, random_state=4)
        model.fit(features_with_driver)
        probs = model.predict_state_probs(features_with_driver)
        assert np.all(probs >= 0)

    def test_driver_state_features_missing_driver_id_raises(self, features_with_driver):
        """driver_state_features should raise if driver_id is not in DataFrame."""
        model = DrivingStateHMM(n_states=3, random_state=5)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        # Drop driver_id
        df_no_driver = features_with_driver.drop("driver_id")
        with pytest.raises(ValueError, match="driver_id"):
            model.driver_state_features(df_no_driver, states)

    def test_driver_state_features_entropy_non_negative(self, features_with_driver):
        """Shannon entropy should be >= 0."""
        model = DrivingStateHMM(n_states=3, random_state=6)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        assert (driver_df["state_entropy"] >= 0).all()

    def test_driver_state_features_transition_rate_non_negative(self, features_with_driver):
        """Transition rate per km should be non-negative."""
        model = DrivingStateHMM(n_states=3, random_state=7)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        assert (driver_df["mean_transition_rate"] >= 0).all()

    def test_driver_state_features_without_distance_km(self, features_with_driver):
        """Without distance_km, transition_rate should default to n_trips denominator."""
        model = DrivingStateHMM(n_states=3, random_state=8)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        df_no_dist = features_with_driver.drop("distance_km")
        driver_df = model.driver_state_features(df_no_dist, states)
        assert "mean_transition_rate" in driver_df.columns
        assert (driver_df["mean_transition_rate"] >= 0).all()

    def test_full_covariance_type(self, features_with_driver):
        """covariance_type='full' should fit without error."""
        model = DrivingStateHMM(n_states=2, covariance_type="full", random_state=9)
        model.fit(features_with_driver)
        assert model.is_fitted

    def test_predict_state_probs_before_fit_raises(self, features_with_driver):
        model = DrivingStateHMM(n_states=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_state_probs(features_with_driver)

    def test_n_states_stored_on_model(self, features_with_driver):
        for n in [2, 3, 5]:
            model = DrivingStateHMM(n_states=n, random_state=10)
            model.fit(features_with_driver)
            assert model.n_states == n


# ---------------------------------------------------------------------------
# ContinuousTimeHMM — additional edge cases
# ---------------------------------------------------------------------------


class TestContinuousTimeHMMExtended:
    def test_two_state_model(self, features_with_driver):
        """Two-state CTHMM should converge and produce valid states."""
        model = ContinuousTimeHMM(n_states=2, n_iter=15, random_state=11)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert set(states).issubset({0, 1})

    def test_convergence_warning_emitted(self, features_with_driver):
        """With n_iter=1 the model should warn about non-convergence."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = ContinuousTimeHMM(n_states=3, n_iter=1, tol=1e-20, random_state=12)
            model.fit(features_with_driver)
        messages = [str(ww.message) for ww in w]
        assert any("converge" in m.lower() for m in messages), (
            f"Expected convergence warning. Got: {messages}"
        )

    def test_predict_before_fit_raises(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_states(features_with_driver)

    def test_very_small_time_deltas_clamped(self, features_with_driver):
        """time_deltas=0 should be clamped to 1e-6 without error."""
        model = ContinuousTimeHMM(n_states=2, n_iter=10, random_state=13)
        n = len(features_with_driver)
        dts = np.zeros(n)  # all zeros — should be clamped
        model.fit(features_with_driver, time_deltas=dts)
        states = model.predict_states(features_with_driver, time_deltas=dts)
        assert states.shape == (n,)

    def test_large_time_deltas(self, features_with_driver):
        """Large time_deltas (inter-session gaps) should not crash."""
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=14)
        n = len(features_with_driver)
        dts = np.full(n, 1440.0)  # 24-hour gaps
        model.fit(features_with_driver, time_deltas=dts)
        states = model.predict_states(features_with_driver, time_deltas=dts)
        assert states.shape == (n,)

    def test_driver_state_features_missing_driver_id_raises(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=15)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        df_no_driver = features_with_driver.drop("driver_id")
        with pytest.raises(ValueError, match="driver_id"):
            model.driver_state_features(df_no_driver, states)

    def test_driver_state_features_without_distance_km(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=2, n_iter=10, random_state=16)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        df_no_dist = features_with_driver.drop("distance_km")
        driver_df = model.driver_state_features(df_no_dist, states)
        assert "mean_transition_rate" in driver_df.columns

    def test_missing_feature_column_in_predict_raises(self, features_with_driver):
        model = ContinuousTimeHMM(
            n_states=2,
            features=["mean_speed_kmh", "harsh_braking_rate"],
            n_iter=5,
            random_state=17,
        )
        model.fit(features_with_driver)
        df_bad = features_with_driver.drop("mean_speed_kmh")
        with pytest.raises(ValueError, match="mean_speed_kmh"):
            model.predict_states(df_bad)

    def test_state_order_is_ascending_by_speed(self, features_with_driver):
        """State 0 should have lower mean speed feature than state n-1."""
        model = ContinuousTimeHMM(n_states=3, n_iter=20, random_state=18)
        model.fit(features_with_driver)
        # means_ columns are in feature order; first feature is mean_speed_kmh
        ordered_means = model.means_[model._state_order, 0]
        # Check ascending
        assert ordered_means[0] <= ordered_means[-1], (
            "State ordering should be ascending by first feature (speed proxy)."
        )

    def test_pi_sums_to_one(self, features_with_driver):
        """Initial state distribution should sum to 1."""
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=19)
        model.fit(features_with_driver)
        assert abs(model.pi_.sum() - 1.0) < 1e-6

    def test_custom_features_cthmm(self, features_with_driver):
        model = ContinuousTimeHMM(
            n_states=2,
            features=["mean_speed_kmh", "speed_variation_coeff"],
            n_iter=8,
            random_state=20,
        )
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.shape == (len(features_with_driver),)


# ---------------------------------------------------------------------------
# _logsumexp — internal helper
# ---------------------------------------------------------------------------


class TestLogsumexp:
    def test_all_neg_inf(self):
        a = np.full(5, -np.inf)
        assert _logsumexp(a) == -np.inf

    def test_single_value(self):
        a = np.array([3.0])
        assert abs(_logsumexp(a) - 3.0) < 1e-10

    def test_known_value(self):
        # log(e^0 + e^1) = log(1 + e) ≈ 1.3133
        a = np.array([0.0, 1.0])
        expected = float(np.log(1 + np.e))
        assert abs(_logsumexp(a) - expected) < 1e-8

    def test_large_values_stable(self):
        # Should not overflow
        a = np.array([1000.0, 1001.0])
        result = _logsumexp(a)
        assert np.isfinite(result)
        assert result > 1000.0


# ---------------------------------------------------------------------------
# ZIPNearMissModel — extended coverage
# ---------------------------------------------------------------------------


class TestZIPRepr:
    def test_repr_contains_n_groups(self):
        model = ZIPNearMissModel(n_groups=4)
        assert "4" in repr(model)


class TestZIPCustomEventTypes:
    def test_custom_event_types_fit(self):
        """Model should work with a custom (smaller) event type list."""
        event_types = ["harsh_braking", "harsh_accel"]
        sim = NearMissSimulator(seed=21)
        full_df = sim.simulate(n_drivers=40, n_weeks=10)
        # Subset to relevant columns
        keep = {"driver_id", "week_id", "exposure_km", "true_group"} | set(event_types)
        df = full_df.select([c for c in full_df.columns if c in keep])

        model = ZIPNearMissModel(
            n_groups=2, event_types=event_types, max_iter=15, random_state=22
        )
        model.fit(df)
        assert set(model.zip_params_.keys()) == {0, 1}

    def test_custom_event_types_predict(self):
        event_types = ["harsh_braking"]
        sim = NearMissSimulator(seed=23)
        full_df = sim.simulate(n_drivers=30, n_weeks=8)
        df = full_df.select(["driver_id", "week_id", "exposure_km", "harsh_braking"])
        model = ZIPNearMissModel(
            n_groups=2, event_types=event_types, max_iter=10, random_state=24
        )
        model.fit(df)
        probs = model.predict_group_probs(df)
        assert probs.shape[0] == 30


class TestZIPCustomExposureCol:
    def test_custom_exposure_col(self):
        """Model should accept a custom exposure column name."""
        sim = NearMissSimulator(seed=25)
        df = sim.simulate(n_drivers=30, n_weeks=6)
        # Rename exposure column
        df = df.rename({"exposure_km": "miles_driven"})
        model = ZIPNearMissModel(
            n_groups=2, max_iter=10, exposure_col="miles_driven", random_state=26
        )
        model.fit(df)
        assert model.mixing_weights_.sum() > 0.99


class TestZIPGroupRelabelling:
    def test_group_0_has_lowest_rate(self, small_weekly):
        """After fitting, group 0 should have the lowest lambda_per_km."""
        model = ZIPNearMissModel(n_groups=3, max_iter=30, random_state=27)
        model.fit(small_weekly)
        rates = [model.zip_params_[g]["lambda_per_km"] for g in range(3)]
        assert rates[0] <= rates[1] <= rates[2] + 1e-8, (
            f"Groups not ordered by rate: {rates}"
        )


class TestZIPTwoGroup:
    def test_two_group_posteriors_valid(self, fitted_zip_2group):
        model, df = fitted_zip_2group
        probs = model.predict_group_probs(df)
        n_drivers = df["driver_id"].n_unique()
        assert probs.shape[0] == n_drivers
        row_sums = sum(probs[f"prob_group_{g}"] for g in range(2))
        assert (row_sums - 1.0).abs().max() < 1e-5

    def test_two_group_mixing_weights_valid(self, fitted_zip_2group):
        model, _ = fitted_zip_2group
        assert abs(model.mixing_weights_.sum() - 1.0) < 1e-6
        assert np.all(model.mixing_weights_ >= 0)

    def test_two_group_predict_rate_columns(self, fitted_zip_2group):
        model, df = fitted_zip_2group
        rates = model.predict_rate(df)
        for et in model.event_types:
            assert f"predicted_rate_{et}" in rates.columns

    def test_two_group_driver_risk_features_has_dominant_group(self, fitted_zip_2group):
        model, df = fitted_zip_2group
        features = model.driver_risk_features(df)
        assert "dominant_group" in features.columns
        assert "nme_rate_per_km" in features.columns
        assert "zero_fraction" in features.columns
        # dominant_group should be 0 or 1
        vals = features["dominant_group"].to_list()
        assert all(v in [0, 1] for v in vals)


class TestZIPDriverRiskFeaturesColumns:
    def test_all_expected_columns_present(self, small_weekly):
        model = ZIPNearMissModel(n_groups=3, max_iter=20, random_state=28)
        model.fit(small_weekly)
        features = model.driver_risk_features(small_weekly)
        expected = {
            "driver_id", "dominant_group", "nme_rate_per_km", "zero_fraction",
            "prob_group_0", "prob_group_1", "prob_group_2",
        }
        assert expected.issubset(set(features.columns))

    def test_zero_fraction_in_0_1(self, small_weekly):
        model = ZIPNearMissModel(n_groups=3, max_iter=20, random_state=29)
        model.fit(small_weekly)
        features = model.driver_risk_features(small_weekly)
        zf = features["zero_fraction"].to_numpy()
        assert np.all(zf >= 0) and np.all(zf <= 1)

    def test_nme_rate_non_negative(self, small_weekly):
        model = ZIPNearMissModel(n_groups=3, max_iter=20, random_state=30)
        model.fit(small_weekly)
        features = model.driver_risk_features(small_weekly)
        rates = features["nme_rate_per_km"].to_numpy()
        assert np.all(rates >= 0)


class TestZIPLogLikelihoodHistory:
    def test_ll_history_length_at_least_one(self, small_weekly):
        model = ZIPNearMissModel(n_groups=2, max_iter=20, random_state=31)
        model.fit(small_weekly)
        assert len(model.log_likelihood_history_) >= 1

    def test_ll_history_finite(self, small_weekly):
        model = ZIPNearMissModel(n_groups=2, max_iter=10, random_state=32)
        model.fit(small_weekly)
        for ll in model.log_likelihood_history_:
            assert np.isfinite(ll), f"Non-finite log-likelihood: {ll}"


class TestZIPValidateWeeklyCounts:
    def test_empty_df_raises(self):
        df = pl.DataFrame({
            "driver_id": [], "week_id": [], "exposure_km": [],
            "harsh_braking": [],
        })
        with pytest.raises(ValueError, match="empty"):
            _validate_weekly_counts(df, ["harsh_braking"], "exposure_km")

    def test_missing_event_type_raises(self):
        sim = NearMissSimulator(seed=33)
        df = sim.simulate(n_drivers=10, n_weeks=4)
        with pytest.raises(ValueError, match="missing"):
            _validate_weekly_counts(df, ["nonexistent_event"], "exposure_km")


class TestZIPTwoDriversEdgeCase:
    def test_two_drivers_fit(self):
        """Model must handle as few as 2 drivers without crashing."""
        sim = NearMissSimulator(n_groups=2, seed=34)
        df = sim.simulate(n_drivers=2, n_weeks=20)
        model = ZIPNearMissModel(n_groups=2, max_iter=5, random_state=35)
        model.fit(df)
        probs = model.predict_group_probs(df)
        assert probs.shape[0] == 2


class TestNearMissSimulatorEdges:
    def test_two_group_simulator(self):
        sim = NearMissSimulator(n_groups=2, seed=36)
        df = sim.simulate(n_drivers=20, n_weeks=5)
        assert df["true_group"].max() <= 1
        assert df["true_group"].min() >= 0

    def test_four_group_simulator(self):
        sim = NearMissSimulator(n_groups=4, seed=37)
        df = sim.simulate(n_drivers=40, n_weeks=5)
        assert df["true_group"].max() <= 3
        # All groups should appear (with enough drivers)
        # At least 3 of 4 groups should appear with 40 drivers
        n_groups_seen = df["true_group"].n_unique()
        assert n_groups_seen >= 3

    def test_exposure_within_bounds(self):
        sim = NearMissSimulator(seed=38)
        df = sim.simulate(n_drivers=50, n_weeks=4)
        assert df["exposure_km"].min() >= 20.0
        assert df["exposure_km"].max() <= 800.0


# ---------------------------------------------------------------------------
# Feature extractor — additional edge cases
# ---------------------------------------------------------------------------


def _make_single_trip_df(
    n_obs: int = 60,
    speed: float = 50.0,
    accel: float = 0.0,
    road_type: str = "rural",
    include_driver_id: bool = True,
    include_heading: bool = False,
    hour: int = 12,
) -> pl.DataFrame:
    """Construct a minimal clean trip DataFrame for feature extractor tests."""
    from datetime import datetime, timedelta

    base_ts = datetime(2024, 1, 1, hour, 0, 0)
    timestamps = [base_ts + timedelta(seconds=i) for i in range(n_obs)]

    data: dict = {
        "trip_id": ["TEST_TRIP"] * n_obs,
        "timestamp": timestamps,
        "speed_kmh": [speed] * n_obs,
        "acceleration_ms2": [accel] * n_obs,
        "road_type": [road_type] * n_obs,
    }
    if include_driver_id:
        data["driver_id"] = ["DRV001"] * n_obs
    if include_heading:
        data["heading_deg"] = [float(i % 360) for i in range(n_obs)]

    return pl.DataFrame(data).with_columns(
        pl.col("timestamp").cast(pl.Datetime)
    )


class TestFeatureExtractorEdgeCases:
    def test_no_driver_id_column_works(self):
        """extract_trip_features should work without driver_id column."""
        df = _make_single_trip_df(include_driver_id=False)
        features = extract_trip_features(df)
        assert "trip_id" in features.columns
        assert "driver_id" not in features.columns

    def test_with_heading_deg_column(self):
        """With heading_deg present, harsh_cornering_rate should be computed."""
        df = _make_single_trip_df(include_heading=True, speed=80.0)
        features = extract_trip_features(df)
        assert "harsh_cornering_rate" in features.columns
        # Should be non-negative
        assert features["harsh_cornering_rate"].item() >= 0

    def test_all_night_trip(self):
        """A trip entirely in night hours should have night_driving_fraction=1."""
        df = _make_single_trip_df(hour=1)  # hour=1 is in _NIGHT_HOURS
        features = extract_trip_features(df)
        assert abs(features["night_driving_fraction"].item() - 1.0) < 1e-6

    def test_all_day_trip_night_fraction_zero(self):
        """A trip at noon should have night_driving_fraction=0."""
        df = _make_single_trip_df(hour=12)
        features = extract_trip_features(df)
        assert abs(features["night_driving_fraction"].item() - 0.0) < 1e-6

    def test_urban_trip_urban_fraction_one(self):
        """All-urban trip should have urban_fraction=1."""
        df = _make_single_trip_df(road_type="urban")
        features = extract_trip_features(df)
        assert abs(features["urban_fraction"].item() - 1.0) < 1e-6

    def test_motorway_trip_urban_fraction_zero(self):
        """All-motorway trip should have urban_fraction=0."""
        df = _make_single_trip_df(road_type="motorway")
        features = extract_trip_features(df)
        assert abs(features["urban_fraction"].item() - 0.0) < 1e-6

    def test_speeding_urban_road(self):
        """Speed=50 kmh on urban road exceeds 35 kmh threshold -> speeding_fraction>0."""
        df = _make_single_trip_df(speed=50.0, road_type="urban")
        features = extract_trip_features(df)
        assert features["speeding_fraction"].item() > 0

    def test_no_speeding_below_threshold(self):
        """Speed=20 kmh on urban road should give speeding_fraction=0."""
        df = _make_single_trip_df(speed=20.0, road_type="urban")
        features = extract_trip_features(df)
        assert features["speeding_fraction"].item() == 0.0

    def test_harsh_braking_detected(self):
        """Acceleration < -3.5 m/s2 should register harsh braking events."""
        df = _make_single_trip_df(accel=-5.0, speed=80.0, road_type="rural")
        features = extract_trip_features(df)
        assert features["harsh_braking_rate"].item() > 0

    def test_harsh_accel_detected(self):
        """Acceleration > 3.5 m/s2 should register harsh acceleration events."""
        df = _make_single_trip_df(accel=5.0, speed=30.0, road_type="urban")
        features = extract_trip_features(df)
        assert features["harsh_accel_rate"].item() > 0

    def test_single_observation_trip(self):
        """A one-second trip should not crash."""
        df = _make_single_trip_df(n_obs=1)
        features = extract_trip_features(df)
        assert len(features) == 1
        assert features["duration_min"].item() > 0

    def test_missing_column_raises_with_helpful_message(self):
        """Missing speed_kmh should raise ValueError naming the column."""
        df = _make_single_trip_df()
        df = df.drop("speed_kmh")
        with pytest.raises(ValueError, match="speed_kmh"):
            extract_trip_features(df)

    def test_multiple_trips_sorted_by_trip_id(self):
        """Output should be sorted by trip_id."""
        sim = TripSimulator(seed=50)
        trips_df, _ = sim.simulate(n_drivers=3, trips_per_driver=5)
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        trip_ids = features["trip_id"].to_list()
        assert trip_ids == sorted(trip_ids)

    def test_distance_proportional_to_speed(self):
        """Higher speed should produce greater distance_km for the same duration."""
        df_slow = _make_single_trip_df(speed=30.0)
        df_fast = _make_single_trip_df(speed=100.0)
        dist_slow = extract_trip_features(df_slow)["distance_km"].item()
        dist_fast = extract_trip_features(df_fast)["distance_km"].item()
        assert dist_fast > dist_slow
