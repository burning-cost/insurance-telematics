"""
feature_extractor.py — Trip-level scalar feature extraction.

Converts a cleaned 1Hz trip DataFrame into one scalar summary row per trip.
These features are the inputs to the HMM state classifier and, after
driver-level aggregation, become the GLM covariates.

Feature definitions
-------------------
All event-rate features are normalised by trip distance (events/km) so that
long motorway trips do not automatically look safer than short urban hops.

Harsh braking threshold   : deceleration < -3.5 m/s²  (standard industry cut-off)
Harsh acceleration threshold : acceleration > +3.5 m/s²
Harsh cornering threshold : lateral acceleration > 3.0 m/s²  — approximated here
                            from heading-change rate × speed because raw lateral
                            accelerometer data is not always available

Speed thresholds by road type:
    urban    < 50 km/h → speeding if speed > 35 km/h  (30 mph limit proxy)
    rural    50-100    → speeding if speed > 96 km/h   (60 mph limit proxy)
    motorway > 100     → speeding if speed > 113 km/h  (70 mph limit proxy)

Night driving: 23:00 – 05:00 (local time not applied; UTC used as proxy)

Academic basis: Henckaerts & Antonio (2022) Table 1 feature set.
"""

from __future__ import annotations

import polars as pl

# Thresholds (m/s²)
_HARSH_BRAKE_THRESHOLD: float = -3.5
_HARSH_ACCEL_THRESHOLD: float = 3.5
_HARSH_CORNER_DEG_PER_S_KMPH: float = 0.15  # heading change rate threshold (deg/s per km/h)

# Speeding thresholds by road type (km/h)
_SPEED_LIMIT_URBAN: float = 35.0    # ~22 mph, conservative urban proxy
_SPEED_LIMIT_RURAL: float = 96.0    # 60 mph
_SPEED_LIMIT_MOTORWAY: float = 113.0  # 70 mph

# Night hours (UTC, approximate)
_NIGHT_HOURS: set[int] = {23, 0, 1, 2, 3, 4}


def extract_trip_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute one summary row per trip from cleaned 1Hz observations.

    Parameters
    ----------
    df:
        Cleaned trip DataFrame as returned by
        :func:`~insurance_telematics.clean_trips`. Must contain:
        ``trip_id``, ``timestamp``, ``speed_kmh``, ``acceleration_ms2``,
        ``road_type``.

    Returns
    -------
    pl.DataFrame
        One row per ``trip_id`` with columns:

        ``trip_id``, ``distance_km``, ``duration_min``, ``mean_speed_kmh``,
        ``p95_speed_kmh``, ``speed_variation_coeff``, ``harsh_braking_rate``,
        ``harsh_accel_rate``, ``harsh_cornering_rate``, ``speeding_fraction``,
        ``night_driving_fraction``, ``urban_fraction``, ``driver_id``
        (if present in input).

    Notes
    -----
    Distance is estimated as the integral of speed over time at 1Hz:
    sum(speed_kmh) / 3600. This is a reasonable proxy when GPS coordinates
    are noisy or unavailable for Haversine distance computation.
    """
    _check_required(df)

    has_driver_id = "driver_id" in df.columns
    has_heading = "heading_deg" in df.columns and df["heading_deg"].null_count() < len(df)

    # Compute per-row flags in a single with_columns call for efficiency
    flags = [
        # Harsh braking flag
        (pl.col("acceleration_ms2") < _HARSH_BRAKE_THRESHOLD)
        .cast(pl.Int32)
        .alias("_harsh_brake"),
        # Harsh acceleration flag
        (pl.col("acceleration_ms2") > _HARSH_ACCEL_THRESHOLD)
        .cast(pl.Int32)
        .alias("_harsh_accel"),
        # Night flag
        pl.col("timestamp")
        .dt.hour()
        .is_in(list(_NIGHT_HOURS))
        .cast(pl.Int32)
        .alias("_night"),
        # Urban flag
        (pl.col("road_type") == "urban")
        .cast(pl.Int32)
        .alias("_urban"),
        # Speeding flag — road-type-specific threshold
        pl.when(pl.col("road_type") == "urban")
        .then(pl.col("speed_kmh") > _SPEED_LIMIT_URBAN)
        .when(pl.col("road_type") == "rural")
        .then(pl.col("speed_kmh") > _SPEED_LIMIT_RURAL)
        .otherwise(pl.col("speed_kmh") > _SPEED_LIMIT_MOTORWAY)
        .cast(pl.Int32)
        .alias("_speeding"),
        # Distance increment: speed_kmh / 3600 = km per second at 1Hz
        (pl.col("speed_kmh") / 3600.0).alias("_dist_km_increment"),
    ]

    if has_heading:
        # Cornering proxy: heading change rate (deg/s) × speed (km/h)
        # Proportional to lateral acceleration; calibrated empirically.
        flags.append(
            (
                pl.col("heading_deg")
                .diff()
                .over("trip_id")
                .abs()
                .fill_null(0.0)
                * pl.col("speed_kmh")
                * _HARSH_CORNER_DEG_PER_S_KMPH
                > 1.0
            )
            .cast(pl.Int32)
            .alias("_harsh_corner"),
        )
    else:
        flags.append(pl.lit(0).cast(pl.Int32).alias("_harsh_corner"))

    df = df.with_columns(flags)

    # Aggregate per trip
    group_cols = ["trip_id"]
    if has_driver_id:
        # Carry driver_id through — take first value (it's constant per trip)
        agg_extra = [pl.col("driver_id").first().alias("driver_id")]
    else:
        agg_extra = []

    agg = df.group_by("trip_id").agg(
        [
            # Basic trip stats
            pl.len().alias("n_obs"),
            (pl.col("_dist_km_increment").sum()).alias("distance_km"),
            (pl.len() / 60.0).alias("duration_min"),
            pl.col("speed_kmh").mean().alias("mean_speed_kmh"),
            pl.col("speed_kmh").quantile(0.95).alias("p95_speed_kmh"),
            pl.col("speed_kmh").std().alias("_speed_std"),
            # Event counts (will divide by distance below)
            pl.col("_harsh_brake").sum().alias("_n_harsh_brake"),
            pl.col("_harsh_accel").sum().alias("_n_harsh_accel"),
            pl.col("_harsh_corner").sum().alias("_n_harsh_corner"),
            # Fraction-based features
            (pl.col("_speeding").mean()).alias("speeding_fraction"),
            (pl.col("_night").mean()).alias("night_driving_fraction"),
            (pl.col("_urban").mean()).alias("urban_fraction"),
        ]
        + agg_extra
    )

    # Derive final rate features
    agg = agg.with_columns(
        [
            # Speed coefficient of variation — std / mean, clamped to avoid div/0
            (
                pl.col("_speed_std")
                / (pl.col("mean_speed_kmh") + 1e-6)
            ).alias("speed_variation_coeff"),
            # Event rates per km — clamp distance to 0.01 km minimum
            (
                pl.col("_n_harsh_brake")
                / (pl.col("distance_km").clip(0.01))
            ).alias("harsh_braking_rate"),
            (
                pl.col("_n_harsh_accel")
                / (pl.col("distance_km").clip(0.01))
            ).alias("harsh_accel_rate"),
            (
                pl.col("_n_harsh_corner")
                / (pl.col("distance_km").clip(0.01))
            ).alias("harsh_cornering_rate"),
        ]
    )

    # Select and order final output columns
    output_cols = [
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
    ]
    if has_driver_id:
        output_cols.append("driver_id")

    return agg.select(output_cols).sort("trip_id")


def _check_required(df: pl.DataFrame) -> None:
    """Raise ValueError if a required column is absent."""
    required = {"trip_id", "timestamp", "speed_kmh", "acceleration_ms2", "road_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for feature extraction: {sorted(missing)}. "
            "Run clean_trips() before extract_trip_features()."
        )
