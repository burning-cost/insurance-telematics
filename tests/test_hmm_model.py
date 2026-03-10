"""Tests for DrivingStateHMM and ContinuousTimeHMM."""

import numpy as np
import polars as pl
import pytest

from insurance_telematics.hmm_model import DrivingStateHMM, ContinuousTimeHMM
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.feature_extractor import extract_trip_features


@pytest.fixture(scope="module")
def features_with_driver():
    sim = TripSimulator(seed=50)
    trips_df, _ = sim.simulate(
        n_drivers=6, trips_per_driver=15, min_trip_duration_s=120, max_trip_duration_s=600
    )
    cleaned = clean_trips(trips_df)
    return extract_trip_features(cleaned)


# ----- DrivingStateHMM tests -----

class TestDrivingStateHMM:
    def test_fit_returns_self(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        result = model.fit(features_with_driver)
        assert result is model

    def test_is_fitted_after_fit(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        assert model.is_fitted

    def test_predict_states_shape(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.shape == (len(features_with_driver),)

    def test_predict_states_valid_range(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.min() >= 0
        assert states.max() <= 2

    def test_predict_state_probs_shape(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        probs = model.predict_state_probs(features_with_driver)
        assert probs.shape == (len(features_with_driver), 3)

    def test_predict_state_probs_sum_to_one(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        probs = model.predict_state_probs(features_with_driver)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_driver_state_features_returns_one_row_per_driver(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        n_drivers = features_with_driver["driver_id"].n_unique()
        assert len(driver_df) == n_drivers

    def test_driver_state_features_columns(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        for k in range(3):
            assert f"state_{k}_fraction" in driver_df.columns
        assert "mean_transition_rate" in driver_df.columns
        assert "state_entropy" in driver_df.columns

    def test_state_fractions_sum_to_one(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        frac_cols = [f"state_{k}_fraction" for k in range(3)]
        row_sums = driver_df.select(frac_cols).sum_horizontal()
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_predict_before_fit_raises(self, features_with_driver):
        model = DrivingStateHMM(n_states=3)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_states(features_with_driver)

    def test_missing_feature_column_raises(self, features_with_driver):
        model = DrivingStateHMM(n_states=3, features=["nonexistent_col"])
        with pytest.raises(ValueError, match="nonexistent_col"):
            model.fit(features_with_driver)

    def test_two_state_model(self, features_with_driver):
        model = DrivingStateHMM(n_states=2, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.max() <= 1


# ----- ContinuousTimeHMM tests -----

class TestContinuousTimeHMM:
    def test_fit_returns_self(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        result = model.fit(features_with_driver)
        assert result is model

    def test_is_fitted_after_fit(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        model.fit(features_with_driver)
        assert model.is_fitted

    def test_predict_states_shape(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.shape == (len(features_with_driver),)

    def test_predict_states_valid_range(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        assert states.min() >= 0
        assert states.max() <= 2

    def test_generator_matrix_valid(self, features_with_driver):
        """Q should have non-negative off-diagonal and zero row sums."""
        model = ContinuousTimeHMM(n_states=3, n_iter=20, random_state=42)
        model.fit(features_with_driver)
        Q = model.Q_
        n = Q.shape[0]
        # Off-diagonal non-negative
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert Q[i, j] >= -1e-6, f"Q[{i},{j}] = {Q[i,j]} is negative"
        # Row sums near zero
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-6)

    def test_with_time_deltas(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        n = len(features_with_driver)
        dts = np.ones(n) * 5.0  # 5-minute intervals
        model.fit(features_with_driver, time_deltas=dts)
        states = model.predict_states(features_with_driver, time_deltas=dts)
        assert states.shape == (n,)

    def test_wrong_time_delta_length_raises(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=5)
        with pytest.raises(ValueError, match="time_deltas"):
            model.fit(features_with_driver, time_deltas=np.ones(3))

    def test_driver_state_features(self, features_with_driver):
        model = ContinuousTimeHMM(n_states=3, n_iter=10, random_state=42)
        model.fit(features_with_driver)
        states = model.predict_states(features_with_driver)
        driver_df = model.driver_state_features(features_with_driver, states)
        n_drivers = features_with_driver["driver_id"].n_unique()
        assert len(driver_df) == n_drivers
        for k in range(3):
            assert f"state_{k}_fraction" in driver_df.columns
