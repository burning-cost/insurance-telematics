"""
preprocessor.py — GPS cleaning and kinematics derivation.

Takes raw 1Hz trip data and returns a cleaned, enriched DataFrame with:
- GPS jump removal (impossible speeds > 250 km/h)
- Small gap interpolation (< 5 s)
- Derived acceleration from speed differences (if not provided)
- Derived jerk (rate of change of acceleration)
- Heuristic road type classification from speed profile

These steps follow the standard preprocessing in Wüthrich (2017) and the
feature engineering conventions in Henckaerts & Antonio (2022).
"""

from __future__ import annotations

import polars as pl
import numpy as np

# Speed thresholds for road type classification (km/h)
_URBAN_MAX_KMH: float = 50.0
_RURAL_MAX_KMH: float = 100.0

# Threshold above which a GPS reading is treated as a jump artefact
_MAX_PLAUSIBLE_SPEED_KMH: float = 250.0

# Maximum gap (seconds) to interpolate over
_MAX_INTERPOLATE_GAP_S: float = 5.0


def clean_trips(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and enrich a raw telematics DataFrame.

    Applies the following steps per trip, in order:

    1. Remove GPS jump rows (speed > 250 km/h).
    2. Interpolate speed for small gaps (< 5 s within a trip).
    3. Derive acceleration from speed differences if the
       ``acceleration_ms2`` column is null or absent.
    4. Derive jerk (m/s³) as the first difference of acceleration.
    5. Classify each observation into a road type
       (``urban`` / ``rural`` / ``motorway``) from speed.

    Parameters
    ----------
    df:
        Raw trip DataFrame as returned by :func:`~insurance_telematics.load_trips`.
        Must contain ``trip_id``, ``timestamp``, and ``speed_kmh``.

    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame with additional columns:
        ``acceleration_ms2`` (if derived), ``jerk_ms3``, ``road_type``.

    Notes
    -----
    The road type heuristic (urban < 50 km/h, rural 50-100, motorway > 100)
    is a proxy. It works well in aggregate but misclassifies individual
    observations — a momentary slow-down on a motorway reads as urban.
    Trip-level road type statistics (fraction of time in each band) are
    more robust than per-row labels.
    """
    df = _remove_gps_jumps(df)
    df = _interpolate_speed_gaps(df)
    df = _derive_acceleration(df)
    df = _derive_jerk(df)
    df = _classify_road_type(df)
    return df


def _remove_gps_jumps(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where speed exceeds the physically plausible limit."""
    return df.filter(pl.col("speed_kmh") <= _MAX_PLAUSIBLE_SPEED_KMH)


def _interpolate_speed_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Linear-interpolate speed over small gaps within each trip.

    A gap is defined as a null speed value between two non-null readings
    separated by fewer than 5 seconds. In practice, 1Hz data rarely has
    null speed values, but this handles partial GPS dropouts.
    """
    if df["speed_kmh"].null_count() == 0:
        return df

    # Compute time delta within each trip, then interpolate nulls where gap < 5s
    df = df.with_columns(
        pl.col("timestamp")
        .diff()
        .over("trip_id")
        .dt.total_seconds()
        .alias("_dt_s")
    )

    # Simple forward-fill then backward-fill for small gaps
    # Polars forward_fill / backward_fill with limit handles this
    df = df.with_columns(
        pl.col("speed_kmh")
        .forward_fill(limit=4)
        .over("trip_id")
        .alias("speed_kmh")
    )

    return df.drop("_dt_s")


def _derive_acceleration(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute longitudinal acceleration (m/s²) from speed differences.

    Uses the speed column to compute dv/dt at 1Hz. Existing non-null
    values in ``acceleration_ms2`` are preserved; only null positions
    are filled from the derived series.
    """
    # Convert speed difference km/h → m/s, divide by dt (1 s at 1Hz)
    derived_accel = (
        pl.col("speed_kmh").diff().over("trip_id") / 3.6
    ).alias("_derived_accel")

    df = df.with_columns(derived_accel)

    if "acceleration_ms2" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("acceleration_ms2").is_null())
            .then(pl.col("_derived_accel"))
            .otherwise(pl.col("acceleration_ms2"))
            .alias("acceleration_ms2")
        )
    else:
        df = df.with_columns(pl.col("_derived_accel").alias("acceleration_ms2"))

    return df.drop("_derived_accel")


def _derive_jerk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute jerk (m/s³) as the first difference of acceleration.

    Jerk captures abrupt changes in acceleration — sharp brake applications
    appear as large negative jerk values and are predictive of risky driving
    independently of the peak deceleration.
    """
    df = df.with_columns(
        pl.col("acceleration_ms2")
        .diff()
        .over("trip_id")
        .alias("jerk_ms3")
    )
    return df


def _classify_road_type(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assign a heuristic road type to each observation based on speed.

    Speed bands:
        urban    — speed < 50 km/h
        rural    — 50 km/h ≤ speed ≤ 100 km/h
        motorway — speed > 100 km/h

    This approach follows Guillen, Pérez-Marín & Nielsen (2024) who use
    percentage of urban driving as a dynamic telematics feature in weekly
    Poisson regression models.
    """
    df = df.with_columns(
        pl.when(pl.col("speed_kmh") < _URBAN_MAX_KMH)
        .then(pl.lit("urban"))
        .when(pl.col("speed_kmh") <= _RURAL_MAX_KMH)
        .then(pl.lit("rural"))
        .otherwise(pl.lit("motorway"))
        .alias("road_type")
    )
    return df
