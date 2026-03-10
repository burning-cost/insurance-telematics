"""
trip_loader.py — Trip data ingestion.

Accepts CSV and Parquet files with 1Hz telematics observations. Validates
that the expected schema columns are present and returns a normalised
Polars DataFrame ready for the preprocessing stage.

Standard schema
---------------
Required columns:
    trip_id       — string identifier, unique per trip
    timestamp     — datetime (ISO 8601 or Unix epoch)
    latitude      — decimal degrees
    longitude     — decimal degrees
    speed_kmh     — GPS speed in km/h

Optional columns (derived by preprocessor if absent):
    acceleration_ms2 — longitudinal acceleration in m/s²
    heading_deg      — bearing 0-360°
    driver_id        — string identifier (added as "unknown" if absent)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import polars as pl


# Columns the library requires to be present or derivable
REQUIRED_COLUMNS: list[str] = [
    "trip_id",
    "timestamp",
    "latitude",
    "longitude",
    "speed_kmh",
]

OPTIONAL_COLUMNS: list[str] = [
    "acceleration_ms2",
    "heading_deg",
    "driver_id",
]

ALL_SCHEMA_COLUMNS: list[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def load_trips(
    path: str | os.PathLike,
    *,
    schema: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """
    Load telematics trip data from CSV or Parquet, validate schema, and
    return a normalised Polars DataFrame.

    Parameters
    ----------
    path:
        Path to a CSV or Parquet file, or a directory containing Parquet
        files (all will be read and concatenated).
    schema:
        Optional column name overrides for non-standard source files.
        Maps source column names to the standard column names. For example:
        ``{"gps_speed": "speed_kmh", "accel_x": "acceleration_ms2"}``.

    Returns
    -------
    pl.DataFrame
        Normalised DataFrame with standard column names. The ``timestamp``
        column is cast to ``Datetime[us, UTC]``. Missing optional columns
        are added with null values. Rows are sorted by ``trip_id`` then
        ``timestamp``.

    Raises
    ------
    ValueError
        If any required column is missing after applying the schema mapping.
    FileNotFoundError
        If the path does not exist.

    Examples
    --------
    >>> df = load_trips("trips.csv")
    >>> df = load_trips("trips.parquet", schema={"gps_speed": "speed_kmh"})
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in directory: {path}")
        df = pl.concat([pl.read_parquet(f) for f in sorted(parquet_files)])
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        df = pl.read_csv(path, try_parse_dates=True)
    else:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Use CSV or Parquet."
        )

    # Apply column renames from schema mapping
    if schema:
        df = df.rename({src: dst for src, dst in schema.items() if src in df.columns})

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Use the 'schema' parameter to map source column names."
        )

    # Ensure timestamp is datetime
    if df["timestamp"].dtype == pl.Utf8 or df["timestamp"].dtype == pl.String:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(
                format=None, strict=False, use_earliest=True
            )
        )

    if df["timestamp"].dtype not in (pl.Datetime("us", "UTC"), pl.Datetime("ns", "UTC")):
        # Cast to microsecond UTC if not already timezone-aware
        if hasattr(df["timestamp"].dtype, "time_zone") and df["timestamp"].dtype.time_zone:
            df = df.with_columns(
                pl.col("timestamp").dt.convert_time_zone("UTC").dt.cast_time_unit("us")
            )
        else:
            df = df.with_columns(
                pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.cast_time_unit("us")
            )

    # Add missing optional columns as nulls
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == "driver_id":
                df = df.with_columns(pl.lit("unknown").alias("driver_id"))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Enforce numeric types for key columns
    numeric_cols = ["latitude", "longitude", "speed_kmh", "acceleration_ms2", "heading_deg"]
    cast_exprs = [
        pl.col(c).cast(pl.Float64) for c in numeric_cols if c in df.columns
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # Sort for deterministic processing
    df = df.sort(["trip_id", "timestamp"])

    return df


def load_trips_from_dataframe(
    df: pl.DataFrame,
    *,
    schema: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    Accept an existing Polars DataFrame and apply the same normalisation
    as :func:`load_trips`.

    Useful when trip data arrives from a database query rather than a file.

    Parameters
    ----------
    df:
        Input DataFrame. Must contain the required columns (after schema
        mapping).
    schema:
        Optional column rename mapping, same semantics as in
        :func:`load_trips`.

    Returns
    -------
    pl.DataFrame
        Normalised DataFrame, sorted by ``trip_id`` then ``timestamp``.
    """
    if schema:
        df = df.rename({src: dst for src, dst in schema.items() if src in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            if col == "driver_id":
                df = df.with_columns(pl.lit("unknown").alias("driver_id"))
            else:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return df.sort(["trip_id", "timestamp"])
