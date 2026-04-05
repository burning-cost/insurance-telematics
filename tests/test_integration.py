"""
Integration tests for the insurance-telematics scoring pipeline.

These tests exercise the full sequential pipeline:
    trip_loader -> preprocessor -> feature_extractor -> hmm_model -> risk_aggregator

The unit tests for each component are in their own files.  These tests
exist specifically to catch breakage at the *seams* between components —
column renames, schema contract violations, and data that flows correctly
through every stage individually but breaks when composed end-to-end.

Each test group is self-contained: it generates data with TripSimulator
and runs it through the real pipeline code.  No mocking.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.trip_loader import load_trips_from_dataframe
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.feature_extractor import extract_trip_features
from insurance_telematics.hmm_model import DrivingStateHMM
from insurance_telematics.risk_aggregator import aggregate_to_driver
from insurance_telematics.scoring_pipeline import TelematicsScoringPipeline


# ---------------------------------------------------------------------------
# Shared fixture: small but realistic simulation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simulation():
    """10 drivers × 20 trips — fast but enough for the HMM to converge."""
    sim = TripSimulator(seed=99)
    trips_df, claims_df = sim.simulate(
        n_drivers=10,
        trips_per_driver=20,
        min_trip_duration_s=180,
        max_trip_duration_s=900,
    )
    return trips_df, claims_df


# ---------------------------------------------------------------------------
# Stage-to-stage column contract tests
# ---------------------------------------------------------------------------

class TestLoaderToPreprocessorSeam:
    """Columns produced by load_trips_from_dataframe must be accepted by clean_trips."""

    def test_loaded_df_passes_directly_to_cleaner(self, simulation):
        trips_df, _ = simulation
        loaded = load_trips_from_dataframe(trips_df)
        # Should not raise
        cleaned = clean_trips(loaded)
        assert isinstance(cleaned, pl.DataFrame)

    def test_loader_preserves_all_required_columns_for_preprocessor(self, simulation):
        trips_df, _ = simulation
        loaded = load_trips_from_dataframe(trips_df)
        required_for_clean = {"trip_id", "timestamp", "speed_kmh"}
        assert required_for_clean.issubset(set(loaded.columns))

    def test_row_count_survives_loader_then_clean(self, simulation):
        """clean_trips only drops GPS-jump rows. Row count must not silently collapse."""
        trips_df, _ = simulation
        loaded = load_trips_from_dataframe(trips_df)
        cleaned = clean_trips(loaded)
        # TripSimulator never produces >250 km/h, so row count should be preserved.
        assert len(cleaned) == len(loaded)


class TestPreprocessorToExtractorSeam:
    """Columns produced by clean_trips must satisfy extract_trip_features._check_required."""

    def test_cleaned_output_satisfies_extractor_schema(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        # extract_trip_features needs: trip_id, timestamp, speed_kmh, acceleration_ms2, road_type
        required = {"trip_id", "timestamp", "speed_kmh", "acceleration_ms2", "road_type"}
        assert required.issubset(set(cleaned.columns))

    def test_extractor_accepts_cleaner_output_without_error(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        assert isinstance(features, pl.DataFrame)

    def test_trip_ids_consistent_across_stages(self, simulation):
        """Every trip_id in cleaned data must appear in feature output."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        cleaned_ids = set(cleaned["trip_id"].unique().to_list())
        feature_ids = set(features["trip_id"].unique().to_list())
        assert feature_ids == cleaned_ids

    def test_driver_id_propagates_from_cleaner_to_extractor(self, simulation):
        """driver_id must survive the clean -> extract pipeline unchanged."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        assert "driver_id" in features.columns
        original_driver_ids = set(trips_df["driver_id"].unique().to_list())
        feature_driver_ids = set(features["driver_id"].unique().to_list())
        assert feature_driver_ids == original_driver_ids


class TestExtractorToHMMSeam:
    """Columns produced by extract_trip_features must be accepted by DrivingStateHMM."""

    def test_hmm_default_features_present_in_extractor_output(self, simulation):
        """The four default HMM features must be in the extractor's output columns."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm_defaults = [
            "mean_speed_kmh",
            "speed_variation_coeff",
            "harsh_braking_rate",
            "harsh_accel_rate",
        ]
        for col in hmm_defaults:
            assert col in features.columns, (
                f"HMM default feature '{col}' missing from extract_trip_features output. "
                "A rename in feature_extractor.py would silently break the HMM stage."
            )

    def test_hmm_fits_on_extractor_output(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=3, random_state=1)
        hmm.fit(features)
        assert hmm.is_fitted

    def test_hmm_predicts_states_with_extractor_output(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=3, random_state=1)
        hmm.fit(features)
        states = hmm.predict_states(features)
        assert len(states) == len(features)
        assert set(states.tolist()).issubset({0, 1, 2})

    def test_hmm_driver_state_features_column_contract(self, simulation):
        """driver_state_features must produce the columns that risk_aggregator expects
        when the pipeline joins them."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=3, random_state=1)
        hmm.fit(features)
        states = hmm.predict_states(features)
        driver_hmm = hmm.driver_state_features(features, states)
        # Required join key
        assert "driver_id" in driver_hmm.columns
        # State fraction columns
        for k in range(3):
            assert f"state_{k}_fraction" in driver_hmm.columns
        assert "state_entropy" in driver_hmm.columns
        assert "mean_transition_rate" in driver_hmm.columns


class TestExtractorToAggregatorSeam:
    """aggregate_to_driver requires driver_id and distance_km from the extractor."""

    def test_aggregator_accepts_extractor_output(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        assert isinstance(driver_df, pl.DataFrame)

    def test_aggregator_produces_one_row_per_driver(self, simulation):
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        n_drivers = trips_df["driver_id"].n_unique()
        assert len(driver_df) == n_drivers

    def test_distance_km_column_present_for_aggregator(self, simulation):
        """If feature_extractor renames distance_km, the aggregator will fail.
        This test catches that regression."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        assert "distance_km" in features.columns, (
            "'distance_km' missing from feature_extractor output. "
            "aggregate_to_driver requires this column for distance-weighted means."
        )


# ---------------------------------------------------------------------------
# Full end-to-end pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipelineEndToEnd:
    """Run the entire pipeline from raw trip observations to risk scores."""

    @pytest.fixture(scope="class")
    def full_pipeline_run(self, simulation):
        trips_df, claims_df = simulation
        # Stage 1: load
        loaded = load_trips_from_dataframe(trips_df)
        # Stage 2: clean
        cleaned = clean_trips(loaded)
        # Stage 3: features
        features = extract_trip_features(cleaned)
        # Stage 4: HMM
        hmm = DrivingStateHMM(n_states=3, random_state=42)
        hmm.fit(features)
        states = hmm.predict_states(features)
        hmm_driver = hmm.driver_state_features(features, states)
        # Stage 5: aggregate
        driver_risk = aggregate_to_driver(features)
        # Merge HMM features
        driver_risk = driver_risk.join(hmm_driver, on="driver_id", how="left")
        return dict(
            trips_df=trips_df,
            claims_df=claims_df,
            loaded=loaded,
            cleaned=cleaned,
            features=features,
            hmm=hmm,
            states=states,
            hmm_driver=hmm_driver,
            driver_risk=driver_risk,
        )

    def test_final_driver_count_matches_input(self, full_pipeline_run):
        n_input_drivers = full_pipeline_run["trips_df"]["driver_id"].n_unique()
        n_output_drivers = len(full_pipeline_run["driver_risk"])
        assert n_output_drivers == n_input_drivers

    def test_composite_risk_score_in_0_100(self, full_pipeline_run):
        scores = full_pipeline_run["driver_risk"]["composite_risk_score"]
        assert float(scores.min()) >= -0.01  # small float tolerance
        assert float(scores.max()) <= 100.01

    def test_no_null_driver_ids_in_final_output(self, full_pipeline_run):
        driver_ids = full_pipeline_run["driver_risk"]["driver_id"]
        assert driver_ids.null_count() == 0

    def test_hmm_state_fractions_sum_to_one(self, full_pipeline_run):
        driver_hmm = full_pipeline_run["hmm_driver"]
        for k in range(3):
            col = f"state_{k}_fraction"
            assert col in driver_hmm.columns
        frac_sum = (
            driver_hmm["state_0_fraction"]
            + driver_hmm["state_1_fraction"]
            + driver_hmm["state_2_fraction"]
        )
        assert (frac_sum - 1.0).abs().max() < 1e-9

    def test_all_driver_ids_survive_pipeline(self, full_pipeline_run):
        """No driver may be dropped between raw trips and final risk scores."""
        input_ids = set(full_pipeline_run["trips_df"]["driver_id"].unique().to_list())
        output_ids = set(full_pipeline_run["driver_risk"]["driver_id"].unique().to_list())
        lost = input_ids - output_ids
        assert not lost, f"Drivers lost in pipeline: {lost}"

    def test_scoring_pipeline_class_matches_manual_pipeline(self, simulation):
        """TelematicsScoringPipeline.glm_features() should cover the same drivers
        as the manual stage-by-stage pipeline."""
        trips_df, claims_df = simulation
        pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
        pipe.fit(trips_df, claims_df)
        glm_df = pipe.glm_features(trips_df)
        n_drivers = trips_df["driver_id"].n_unique()
        assert len(glm_df) == n_drivers

    def test_fit_predict_roundtrip(self, simulation):
        trips_df, claims_df = simulation
        pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
        pipe.fit(trips_df, claims_df)
        preds = pipe.predict(trips_df)
        # One prediction per driver
        assert len(preds) == trips_df["driver_id"].n_unique()
        # Predictions are non-negative (Poisson frequency must be >= 0)
        assert (preds["predicted_claim_frequency"] >= 0.0).all()


# ---------------------------------------------------------------------------
# Column-contract regression tests
# ---------------------------------------------------------------------------

class TestColumnContractRegressions:
    """
    Tests that would catch a specific class of regression: a rename in one
    module that passes that module's unit tests but breaks a downstream stage.

    Each test enforces the column-name contract at a specific pipeline seam.
    """

    def test_feature_extractor_output_column_names(self, simulation):
        """
        Exact column names from extract_trip_features.  If this test fails
        after a code change, something downstream (HMM, aggregator, GLM) will
        also fail.  Fix the rename here and in the downstream consumer together.
        """
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        expected_cols = {
            "trip_id",
            "distance_km",
            "duration_min",
            "mean_speed_kmh",
            "p95_speed_kmh",
            "speed_variation_coeff",
            "harsh_braking_rate",
            "harsh_accel_rate",
            "harsh_cornering_rate",
            "speeding_fraction",
            "night_driving_fraction",
            "urban_fraction",
            "driver_id",
        }
        actual_cols = set(features.columns)
        missing = expected_cols - actual_cols
        assert not missing, (
            f"extract_trip_features output is missing expected columns: {missing}. "
            "Check for renames that may break HMM or aggregator stages."
        )

    def test_risk_aggregator_output_column_names(self, simulation):
        """aggregate_to_driver column contract."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        driver_df = aggregate_to_driver(features)
        expected_cols = {
            "driver_id",
            "n_trips",
            "total_km",
            "credibility_weight",
            "composite_risk_score",
        }
        actual_cols = set(driver_df.columns)
        missing = expected_cols - actual_cols
        assert not missing, (
            f"aggregate_to_driver output is missing expected columns: {missing}."
        )

    def test_hmm_driver_features_output_column_names(self, simulation):
        """DrivingStateHMM.driver_state_features column contract for n_states=3."""
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        hmm = DrivingStateHMM(n_states=3, random_state=0)
        hmm.fit(features)
        states = hmm.predict_states(features)
        driver_hmm = hmm.driver_state_features(features, states)
        expected_cols = {
            "driver_id",
            "state_0_fraction",
            "state_1_fraction",
            "state_2_fraction",
            "mean_transition_rate",
            "state_entropy",
        }
        actual_cols = set(driver_hmm.columns)
        missing = expected_cols - actual_cols
        assert not missing, (
            f"driver_state_features output is missing expected columns: {missing}."
        )

    def test_mismatched_hmm_feature_names_raise_clearly(self, simulation):
        """
        If a downstream caller passes feature names that do not match the extractor
        output, the error should come from the HMM (not a cryptic Polars error).
        """
        trips_df, _ = simulation
        cleaned = clean_trips(trips_df)
        features = extract_trip_features(cleaned)
        # Simulate a rename scenario: caller uses old name 'mean_speed' instead of
        # 'mean_speed_kmh'
        hmm = DrivingStateHMM(
            n_states=3,
            features=["mean_speed", "speed_variation_coeff"],  # typo in column name
            random_state=0,
        )
        with pytest.raises(ValueError, match="mean_speed"):
            hmm.fit(features)
