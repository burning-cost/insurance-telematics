"""
insurance-telematics
====================

End-to-end pipeline from raw telematics trip data to GLM-compatible risk scores
for UK motor insurance pricing.

Pipeline stages:
    1. Load  — trip_loader.load_trips()
    2. Clean — preprocessor.clean_trips()
    3. Score — feature_extractor.extract_trip_features()
    4. Model — hmm_model.DrivingStateHMM
    5. Aggregate — risk_aggregator.aggregate_to_driver()
    6. Price — scoring_pipeline.TelematicsScoringPipeline

ADAS near-miss event modelling:
    zip_near_miss.NearMissSimulator — synthetic weekly event count generator
    zip_near_miss.ZIPNearMissModel  — Group-Based ZIP EM model

Synthetic data for testing and prototyping:
    trip_simulator.TripSimulator

Academic basis:
    Jiang & Shi (2024), NAAJ 28(4), pp.822-839
    Wüthrich (2017), European Actuarial Journal 7, pp.89-108
    Boucher, Denuit & Guillen (2007), Insurance: Mathematics and Economics 40(3)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-telematics")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .trip_loader import load_trips
from .preprocessor import clean_trips
from .feature_extractor import extract_trip_features
from .hmm_model import DrivingStateHMM, ContinuousTimeHMM
from .risk_aggregator import aggregate_to_driver
from .scoring_pipeline import TelematicsScoringPipeline, score_trips
from .trip_simulator import TripSimulator
from .zip_near_miss import NearMissSimulator, ZIPNearMissModel

__all__ = [
    "load_trips",
    "clean_trips",
    "extract_trip_features",
    "DrivingStateHMM",
    "ContinuousTimeHMM",
    "aggregate_to_driver",
    "TelematicsScoringPipeline",
    "score_trips",
    "TripSimulator",
    "NearMissSimulator",
    "ZIPNearMissModel",
]
