"""
Shared pytest fixtures for the insurance-telematics test suite.

All fixtures generate data using TripSimulator so no external data files
are required. Small simulations (few drivers, few trips) keep tests fast.
"""

import pytest
import polars as pl
from insurance_telematics.trip_simulator import TripSimulator
from insurance_telematics.preprocessor import clean_trips
from insurance_telematics.feature_extractor import extract_trip_features


@pytest.fixture(scope="session")
def small_simulation():
    """5 drivers × 20 trips. Enough for HMM and aggregation tests."""
    sim = TripSimulator(seed=42)
    trips_df, claims_df = sim.simulate(
        n_drivers=5,
        trips_per_driver=20,
        min_trip_duration_s=120,
        max_trip_duration_s=600,
    )
    return trips_df, claims_df


@pytest.fixture(scope="session")
def medium_simulation():
    """20 drivers × 30 trips. For pipeline and GLM tests."""
    sim = TripSimulator(seed=7)
    trips_df, claims_df = sim.simulate(
        n_drivers=20,
        trips_per_driver=30,
        min_trip_duration_s=180,
        max_trip_duration_s=1200,
    )
    return trips_df, claims_df


@pytest.fixture(scope="session")
def cleaned_trips(small_simulation):
    """Cleaned trip DataFrame from the small simulation."""
    trips_df, _ = small_simulation
    return clean_trips(trips_df)


@pytest.fixture(scope="session")
def trip_features(cleaned_trips):
    """Trip-level feature DataFrame from the small simulation."""
    return extract_trip_features(cleaned_trips)
