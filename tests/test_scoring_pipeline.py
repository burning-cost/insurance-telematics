"""Tests for TelematicsScoringPipeline and score_trips."""

import polars as pl
import pytest
import numpy as np

from insurance_telematics.scoring_pipeline import TelematicsScoringPipeline, score_trips
from insurance_telematics.trip_simulator import TripSimulator


@pytest.fixture(scope="module")
def pipeline_data():
    sim = TripSimulator(seed=70)
    trips_df, claims_df = sim.simulate(
        n_drivers=15,
        trips_per_driver=25,
        min_trip_duration_s=180,
        max_trip_duration_s=1200,
    )
    return trips_df, claims_df


def test_fit_returns_self(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    result = pipe.fit(trips_df, claims_df)
    assert result is pipe


def test_is_fitted_after_fit(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    assert pipe.is_fitted


def test_predict_returns_dataframe(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = pipe.predict(trips_df)
    assert isinstance(preds, pl.DataFrame)


def test_predict_has_required_columns(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = pipe.predict(trips_df)
    assert "driver_id" in preds.columns
    assert "predicted_claim_frequency" in preds.columns


def test_predict_one_row_per_driver(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = pipe.predict(trips_df)
    n_drivers = trips_df["driver_id"].n_unique()
    assert len(preds) == n_drivers


def test_predictions_non_negative(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = pipe.predict(trips_df)
    assert (preds["predicted_claim_frequency"] >= 0).all()


def test_glm_features_returns_dataframe(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    features = pipe.glm_features(trips_df)
    assert isinstance(features, pl.DataFrame)
    assert "driver_id" in features.columns


def test_glm_features_one_row_per_driver(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    features = pipe.glm_features(trips_df)
    n_drivers = trips_df["driver_id"].n_unique()
    assert len(features) == n_drivers


def test_predict_before_fit_raises(pipeline_data):
    trips_df, _ = pipeline_data
    pipe = TelematicsScoringPipeline()
    with pytest.raises(RuntimeError, match="fit"):
        pipe.predict(trips_df)


def test_score_trips_convenience_function(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=3, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = score_trips(trips_df, pipe)
    assert isinstance(preds, pl.DataFrame)
    assert len(preds) == trips_df["driver_id"].n_unique()


def test_two_state_hmm_pipeline(pipeline_data):
    trips_df, claims_df = pipeline_data
    pipe = TelematicsScoringPipeline(n_hmm_states=2, random_state=42)
    pipe.fit(trips_df, claims_df)
    preds = pipe.predict(trips_df)
    assert len(preds) == trips_df["driver_id"].n_unique()
