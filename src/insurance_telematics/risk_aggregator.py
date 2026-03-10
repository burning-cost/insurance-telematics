"""
risk_aggregator.py — Driver-level risk score aggregation.

Aggregates trip-level feature rows to one driver-level row. Applies
Bühlmann-Straub credibility weighting for drivers with few trips so that
a driver with 3 trips does not get a wildly different risk score from the
portfolio mean compared to a driver with 200 trips.

The composite risk score is a weighted linear combination of all aggregated
features, scaled to [0, 100].

Academic basis
--------------
Bühlmann-Straub credibility: Bühlmann & Straub (1970), ASTIN Bulletin 5(3).
Portfolio credibility in telematics: Gao, Wang & Wüthrich (2021), Machine
Learning 111, pp.1787-1827 (Section 4.2 on credibility-weighted risk scores).
"""

from __future__ import annotations

import polars as pl
import numpy as np


# Feature weights for composite score. Higher weight = more predictive of claims.
# Based on the empirical importance ordering in Henckaerts & Antonio (2022).
_FEATURE_WEIGHTS: dict[str, float] = {
    "harsh_braking_rate": 2.5,
    "harsh_accel_rate": 2.0,
    "harsh_cornering_rate": 1.5,
    "speeding_fraction": 2.0,
    "night_driving_fraction": 1.5,
    "speed_variation_coeff": 1.0,
    "p95_speed_kmh": 0.8,
    "mean_speed_kmh": 0.5,
    "urban_fraction": -0.3,  # urban driving is higher-intensity but lower speed
}

# Features to aggregate as distance-weighted means
_MEAN_FEATURES: list[str] = [
    "mean_speed_kmh",
    "p95_speed_kmh",
    "speed_variation_coeff",
    "harsh_braking_rate",
    "harsh_accel_rate",
    "harsh_cornering_rate",
    "speeding_fraction",
    "night_driving_fraction",
    "urban_fraction",
]


def aggregate_to_driver(
    trip_features: pl.DataFrame,
    *,
    credibility_threshold: int = 30,
) -> pl.DataFrame:
    """
    Aggregate trip-level features to one row per driver.

    Features are aggregated as distance-weighted means. Drivers below the
    credibility threshold have their scores shrunk towards the portfolio
    mean using Bühlmann-Straub weights.

    Parameters
    ----------
    trip_features:
        Trip-level DataFrame from :func:`~insurance_telematics.extract_trip_features`.
        Must contain ``driver_id`` and ``distance_km``.
    credibility_threshold:
        Number of trips above which a driver is given full credibility
        (weight = 1.0). Below this, the Bühlmann weight is n / (n + k)
        where k is the credibility parameter (default 30).

    Returns
    -------
    pl.DataFrame
        One row per driver with columns:

        - All aggregated feature columns (distance-weighted means)
        - ``n_trips`` — number of trips
        - ``total_km`` — total distance driven
        - ``credibility_weight`` — Bühlmann-Straub weight ∈ (0, 1]
        - ``composite_risk_score`` — weighted composite score, scaled 0-100
        - ``driver_id``

    Raises
    ------
    ValueError
        If ``driver_id`` or ``distance_km`` are not in ``trip_features``.
    """
    _validate_input(trip_features)

    # Compute distance-weighted mean of each feature per driver
    available_mean_features = [
        c for c in _MEAN_FEATURES if c in trip_features.columns
    ]

    agg_exprs = [
        pl.len().alias("n_trips"),
        pl.col("distance_km").sum().alias("total_km"),
    ]

    for feat in available_mean_features:
        # Weighted mean: sum(feat * distance) / sum(distance)
        agg_exprs.append(
            (
                (pl.col(feat) * pl.col("distance_km")).sum()
                / pl.col("distance_km").sum().clip(0.01)
            ).alias(feat)
        )

    driver_df = trip_features.group_by("driver_id").agg(agg_exprs)

    # Bühlmann-Straub credibility weighting
    k = float(credibility_threshold)
    driver_df = driver_df.with_columns(
        (pl.col("n_trips") / (pl.col("n_trips") + k)).alias("credibility_weight")
    )

    # Compute portfolio means (for shrinkage)
    portfolio_means = {}
    for feat in available_mean_features:
        portfolio_means[feat] = float(driver_df[feat].mean())

    # Apply credibility shrinkage toward portfolio mean
    shrink_exprs = []
    for feat in available_mean_features:
        pm = portfolio_means[feat]
        shrink_exprs.append(
            (
                pl.col("credibility_weight") * pl.col(feat)
                + (1.0 - pl.col("credibility_weight")) * pm
            ).alias(feat)
        )
    driver_df = driver_df.with_columns(shrink_exprs)

    # Composite risk score
    driver_df = _compute_composite_score(driver_df, available_mean_features)

    return driver_df.sort("driver_id")


def _compute_composite_score(
    df: pl.DataFrame,
    available_features: list[str],
) -> pl.DataFrame:
    """
    Compute a composite risk score as a weighted sum of features, scaled
    to the range [0, 100] across the portfolio.

    The sign of each weight reflects directionality: positive weights
    mean "more of this feature = higher risk". The raw score is linearly
    scaled so the minimum portfolio score maps to 0 and the maximum to 100.
    """
    # Build weighted sum expression using only available features
    active_weights = {
        feat: w for feat, w in _FEATURE_WEIGHTS.items()
        if feat in available_features
    }
    if not active_weights:
        return df.with_columns(pl.lit(50.0).alias("composite_risk_score"))

    # Normalise each feature to [0, 1] before weighting
    norm_exprs = []
    for feat, weight in active_weights.items():
        col_min = df[feat].min()
        col_max = df[feat].max()
        span = (col_max - col_min) if col_max != col_min else 1.0
        norm_exprs.append(
            ((pl.col(feat) - col_min) / span * weight).alias(f"_w_{feat}")
        )

    df = df.with_columns(norm_exprs)

    # Sum weighted normalised features
    weighted_cols = [f"_w_{feat}" for feat in active_weights]
    raw_score_expr = sum(pl.col(c) for c in weighted_cols)
    df = df.with_columns(raw_score_expr.alias("_raw_score"))

    # Scale to [0, 100]
    score_min = df["_raw_score"].min()
    score_max = df["_raw_score"].max()
    score_span = (score_max - score_min) if score_max != score_min else 1.0
    df = df.with_columns(
        ((pl.col("_raw_score") - score_min) / score_span * 100.0).alias(
            "composite_risk_score"
        )
    )

    # Drop intermediate columns
    drop_cols = weighted_cols + ["_raw_score"]
    return df.drop(drop_cols)


def _validate_input(df: pl.DataFrame) -> None:
    required = {"driver_id", "distance_km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"trip_features missing required columns: {sorted(missing)}"
        )
