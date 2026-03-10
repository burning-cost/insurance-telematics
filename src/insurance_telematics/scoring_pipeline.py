"""
scoring_pipeline.py — End-to-end scikit-learn Pipeline wrapper.

TelematicsScoringPipeline integrates all library stages into a single
sklearn-compatible estimator:
    clean → extract → HMM → aggregate → Poisson GLM

This makes it possible to embed the telematics pipeline inside a larger
sklearn workflow (cross-validation, grid search, etc.) or call it
standalone as a scoring tool.

The GLM step uses statsmodels Poisson with log-link to produce claim
frequency predictions (E[N] per year per driver). Traditional rating
factors can be concatenated alongside the telematics features before
passing to the GLM.

Academic basis
--------------
Poisson GLM with telematics covariates:
    Guillen, Pérez-Marín & Nielsen (2024), Heliyon 10(17)
    Gao, Wang & Wüthrich (2021), Machine Learning 111
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

from .preprocessor import clean_trips
from .feature_extractor import extract_trip_features
from .hmm_model import DrivingStateHMM
from .risk_aggregator import aggregate_to_driver


class TelematicsScoringPipeline:
    """
    End-to-end pipeline from raw trip data to claim frequency predictions.

    Stages:

    1. ``clean_trips`` — GPS cleaning and kinematics derivation
    2. ``extract_trip_features`` — trip-level scalar features
    3. ``DrivingStateHMM.fit`` — HMM state classification
    4. ``aggregate_to_driver`` — driver-level risk aggregation
    5. Poisson GLM — maps telematics features to claim frequency

    Parameters
    ----------
    n_hmm_states:
        Number of HMM latent states (default 3).
    credibility_threshold:
        Bühlmann-Straub credibility threshold in number of trips (default 30).
    hmm_features:
        Feature columns to pass to the HMM. Defaults to the standard four
        (mean speed, speed variation, harsh braking rate, harsh accel rate).
    glm_features:
        Subset of aggregated driver features to include in the GLM. If None,
        uses all available aggregated features.
    random_state:
        Seed for reproducibility.

    Examples
    --------
    >>> sim = TripSimulator(seed=42)
    >>> trips_df, claims_df = sim.simulate(n_drivers=50, trips_per_driver=40)
    >>> pipe = TelematicsScoringPipeline()
    >>> pipe.fit(trips_df, claims_df)
    >>> predictions = pipe.predict(trips_df)
    """

    def __init__(
        self,
        n_hmm_states: int = 3,
        credibility_threshold: int = 30,
        hmm_features: list[str] | None = None,
        glm_features: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_hmm_states = n_hmm_states
        self.credibility_threshold = credibility_threshold
        self.hmm_features = hmm_features
        self.glm_features = glm_features
        self.random_state = random_state

        self._hmm: Optional[DrivingStateHMM] = None
        self._glm_result = None
        self._glm_feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        trips_df: pl.DataFrame,
        claims_df: pl.DataFrame,
    ) -> "TelematicsScoringPipeline":
        """
        Fit the full pipeline.

        Parameters
        ----------
        trips_df:
            Raw trip observations as returned by :func:`~insurance_telematics.load_trips`
            or :class:`~insurance_telematics.TripSimulator`.
        claims_df:
            Driver-level claims data. Required columns:
            ``driver_id``, ``n_claims``, ``exposure_years``.

        Returns
        -------
        self
        """
        if not _STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for the GLM step. "
                "Install it with: pip install statsmodels"
            )

        # Stage 1-2: clean and extract features
        driver_features = self._extract_features(trips_df)

        # Stage 3: fit HMM and add state features
        self._hmm = DrivingStateHMM(
            n_states=self.n_hmm_states,
            features=self.hmm_features,
            random_state=self.random_state,
        )
        self._hmm.fit(driver_features)
        states = self._hmm.predict_states(driver_features)
        hmm_driver_features = self._hmm.driver_state_features(driver_features, states)

        # Stage 4: aggregate trip features to driver level
        driver_risk = aggregate_to_driver(
            driver_features,
            credibility_threshold=self.credibility_threshold,
        )

        # Merge HMM state features into driver risk
        driver_risk = driver_risk.join(hmm_driver_features, on="driver_id", how="left")

        # Merge with claims data
        model_df = driver_risk.join(claims_df, on="driver_id", how="inner")

        # Stage 5: Poisson GLM
        glm_feature_cols = self._select_glm_features(model_df)
        self._glm_feature_names = glm_feature_cols

        X = model_df.select(glm_feature_cols).to_pandas()
        X = sm.add_constant(X, has_constant="add")

        y = model_df["n_claims"].to_numpy()
        offset = np.log(model_df["exposure_years"].clip(1e-6).to_numpy())

        poisson_model = sm.GLM(
            y,
            X,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=offset,
        )
        self._glm_result = poisson_model.fit(maxiter=100, disp=False)
        self.is_fitted = True
        return self

    def predict(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict annual claim frequency for each driver.

        Parameters
        ----------
        trips_df:
            Raw trip observations. Must contain ``driver_id``.

        Returns
        -------
        pl.DataFrame
            One row per driver with columns ``driver_id`` and
            ``predicted_claim_frequency`` (claims per year).
        """
        self._check_fitted()
        glm_df = self.glm_features(trips_df)
        return self._predict_from_features(glm_df)

    def glm_features(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """
        Produce a driver-level DataFrame of GLM-ready features without
        fitting or running the GLM.

        Parameters
        ----------
        trips_df:
            Raw trip data.

        Returns
        -------
        pl.DataFrame
            One row per driver with telematics-derived GLM covariates.
            Column names follow the controlled vocabulary described in the
            library documentation and suitable for regulatory filings.
        """
        driver_features = self._extract_features(trips_df)

        if self._hmm is not None and self._hmm.is_fitted:
            states = self._hmm.predict_states(driver_features)
            hmm_df = self._hmm.driver_state_features(driver_features, states)
        else:
            hmm_df = None

        driver_risk = aggregate_to_driver(
            driver_features,
            credibility_threshold=self.credibility_threshold,
        )

        if hmm_df is not None:
            driver_risk = driver_risk.join(hmm_df, on="driver_id", how="left")

        return driver_risk

    def _extract_features(self, trips_df: pl.DataFrame) -> pl.DataFrame:
        """Run clean + extract_trip_features."""
        cleaned = clean_trips(trips_df)
        return extract_trip_features(cleaned)

    def _select_glm_features(self, df: pl.DataFrame) -> list[str]:
        """Choose GLM feature columns from available numeric columns."""
        if self.glm_features is not None:
            return [c for c in self.glm_features if c in df.columns]

        # Default: all numeric columns except identifiers and metadata
        exclude = {
            "driver_id",
            "n_claims",
            "exposure_years",
            "n_trips",
            "total_km",
            "credibility_weight",
            "composite_risk_score",
            "aggressive_fraction",
            "normal_fraction",
            "cautious_fraction",
            "annual_km",
        }
        candidates = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]
        return candidates

    def _predict_from_features(self, glm_df: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted GLM to feature DataFrame."""
        import statsmodels.api as sm
        feature_cols = [c for c in self._glm_feature_names if c in glm_df.columns]
        X = glm_df.select(feature_cols).to_pandas()
        X = sm.add_constant(X, has_constant="add")

        # Fill any missing columns with 0
        for col in self._glm_result.model.exog_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._glm_result.model.exog_names]

        predictions = self._glm_result.predict(X)
        return pl.DataFrame(
            {
                "driver_id": glm_df["driver_id"].to_list(),
                "predicted_claim_frequency": predictions.tolist(),
            }
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")


def score_trips(trips_df: pl.DataFrame, model: TelematicsScoringPipeline) -> pl.DataFrame:
    """
    Convenience function: apply a fitted pipeline to raw trip data.

    Parameters
    ----------
    trips_df:
        Raw trip DataFrame.
    model:
        A fitted :class:`TelematicsScoringPipeline` instance.

    Returns
    -------
    pl.DataFrame
        Driver-level predictions with ``driver_id`` and
        ``predicted_claim_frequency``.

    Examples
    --------
    >>> predictions = score_trips(new_trips_df, fitted_pipeline)
    """
    return model.predict(trips_df)
