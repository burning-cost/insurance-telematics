"""
trip_simulator.py — Synthetic telematics trip generator.

Produces realistic 1Hz trip data so teams without raw sensor access can
prototype the full pipeline. Each driver has a latent driving regime mixture
(cautious / normal / aggressive). Speed evolves as an Ornstein-Uhlenbeck
process within each regime. Claims are drawn from a Poisson distribution
with rate proportional to aggressive state fraction.

This removes the data access barrier — the single biggest adoption obstacle
for a telematics library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl


@dataclass
class _RegimeParams:
    """Speed and acceleration parameters for one driving regime."""

    name: str
    mean_speed_kmh: float
    speed_reversion: float  # Ornstein-Uhlenbeck mean-reversion rate (1/s)
    speed_vol: float        # OU volatility (km/h per sqrt(s))
    accel_noise_std: float  # Extra acceleration noise (m/s²)
    base_claim_rate: float  # Annual claim rate for pure exposure in this regime


_REGIMES = {
    "cautious": _RegimeParams(
        name="cautious",
        mean_speed_kmh=35.0,
        speed_reversion=0.15,
        speed_vol=3.0,
        accel_noise_std=0.5,
        base_claim_rate=0.05,
    ),
    "normal": _RegimeParams(
        name="normal",
        mean_speed_kmh=60.0,
        speed_reversion=0.10,
        speed_vol=5.0,
        accel_noise_std=1.2,
        base_claim_rate=0.10,
    ),
    "aggressive": _RegimeParams(
        name="aggressive",
        mean_speed_kmh=90.0,
        speed_reversion=0.08,
        speed_vol=9.0,
        accel_noise_std=2.5,
        base_claim_rate=0.30,
    ),
}


@dataclass
class TripSimulator:
    """
    Generate synthetic 1Hz telematics trip data for a fleet of drivers.

    Each driver is assigned a latent regime mixture (fraction of time spent
    in cautious, normal, and aggressive driving states). Within a trip,
    segments of continuous regime are simulated using an Ornstein-Uhlenbeck
    speed process. A synthetic claim count is generated per driver using a
    Poisson rate proportional to the aggressive fraction.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> sim = TripSimulator(seed=42)
    >>> trips_df, claims_df = sim.simulate(n_drivers=20, trips_per_driver=30)
    >>> trips_df.shape[0]  # ~540,000 rows for 20 drivers × 30 trips × ~15 min avg
    """

    seed: int = 42

    def simulate(
        self,
        n_drivers: int = 100,
        trips_per_driver: int = 50,
        *,
        min_trip_duration_s: int = 300,
        max_trip_duration_s: int = 3600,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Simulate 1Hz trip observations for a fleet of drivers.

        Parameters
        ----------
        n_drivers:
            Number of distinct drivers to simulate.
        trips_per_driver:
            Number of trips per driver.
        min_trip_duration_s:
            Minimum trip length in seconds (default 5 minutes).
        max_trip_duration_s:
            Maximum trip length in seconds (default 60 minutes).

        Returns
        -------
        trips_df:
            Polars DataFrame with one row per 1Hz observation. Columns:
            driver_id, trip_id, timestamp, latitude, longitude,
            speed_kmh, acceleration_ms2, heading_deg.
        claims_df:
            Polars DataFrame with one row per driver. Columns:
            driver_id, n_claims, exposure_years, aggressive_fraction,
            annual_km.
        """
        rng = np.random.default_rng(self.seed)

        # Assign latent regime mixtures to drivers using a Dirichlet draw.
        # Concentration steered towards cautious to mimic real portfolio skew.
        concentration = np.array([3.0, 2.0, 0.5])
        driver_mixtures = rng.dirichlet(concentration, size=n_drivers)

        all_trip_rows: list[dict] = []
        driver_summary: list[dict] = []

        base_lat, base_lon = 51.5, -0.1  # London area

        # Trip IDs are globally unique across drivers
        global_trip_id = 0

        # Reference epoch for reproducible timestamps
        epoch = datetime(2024, 1, 1, 7, 0, 0, tzinfo=timezone.utc)

        for driver_idx in range(n_drivers):
            driver_id = f"DRV{driver_idx:04d}"
            mixture = driver_mixtures[driver_idx]
            cautious_frac, normal_frac, aggressive_frac = mixture

            driver_km = 0.0
            driver_total_seconds = 0

            for trip_num in range(trips_per_driver):
                trip_id = f"TRP{global_trip_id:07d}"
                global_trip_id += 1

                # Trip duration: random between min and max
                duration_s = int(
                    rng.integers(min_trip_duration_s, max_trip_duration_s)
                )
                driver_total_seconds += duration_s

                # Stagger trip timestamps to spread across a year
                days_offset = (driver_idx * trips_per_driver + trip_num) * 3.5 / 365
                hour_offset = rng.uniform(6.0, 22.0)
                trip_start = epoch + timedelta(
                    days=float(days_offset), hours=float(hour_offset)
                )

                rows = self._simulate_trip(
                    rng=rng,
                    driver_id=driver_id,
                    trip_id=trip_id,
                    mixture=mixture,
                    duration_s=duration_s,
                    trip_start=trip_start,
                    base_lat=base_lat,
                    base_lon=base_lon,
                )
                all_trip_rows.extend(rows)

                # Accumulate distance
                for row in rows:
                    driver_km += row["speed_kmh"] / 3600.0  # km per second

            # Exposure: actual driving time scaled to years.
            # Assume each driver drives at this rate year-round, prorated by
            # average UK annual driving time (~200 hours/year).
            # We normalise to observation period: total trip seconds / seconds_per_year.
            # This gives a realistic exposure for a Poisson frequency model.
            exposure_years = driver_total_seconds / 365.25 / 86400
            annual_rate = (
                cautious_frac * _REGIMES["cautious"].base_claim_rate
                + normal_frac * _REGIMES["normal"].base_claim_rate
                + aggressive_frac * _REGIMES["aggressive"].base_claim_rate
            )
            lambda_claims = annual_rate * max(exposure_years, 0.01)
            n_claims = int(rng.poisson(lambda_claims))

            driver_summary.append(
                {
                    "driver_id": driver_id,
                    "n_claims": n_claims,
                    "exposure_years": round(exposure_years, 4),
                    "aggressive_fraction": round(float(aggressive_frac), 4),
                    "normal_fraction": round(float(normal_frac), 4),
                    "cautious_fraction": round(float(cautious_frac), 4),
                    "annual_km": round(driver_km, 1),
                }
            )

        trips_df = pl.DataFrame(all_trip_rows).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        )
        claims_df = pl.DataFrame(driver_summary)

        return trips_df, claims_df

    def _simulate_trip(
        self,
        rng: np.random.Generator,
        driver_id: str,
        trip_id: str,
        mixture: np.ndarray,
        duration_s: int,
        trip_start: datetime,
        base_lat: float,
        base_lon: float,
    ) -> list[dict]:
        """Simulate a single trip at 1Hz. Returns list of row dicts."""
        rows = []

        # Choose regime for each second via regime mixture (multinomial draw)
        regime_names = ["cautious", "normal", "aggressive"]
        regimes_seq = rng.choice(regime_names, size=duration_s, p=mixture)

        speed_kmh = float(rng.uniform(0, 30))  # starting speed
        lat = base_lat + rng.uniform(-0.5, 0.5)
        lon = base_lon + rng.uniform(-0.5, 0.5)
        heading = float(rng.uniform(0, 360))

        for t in range(duration_s):
            regime = _REGIMES[regimes_seq[t]]
            dt = 1.0  # 1Hz → 1 second

            # Ornstein-Uhlenbeck speed update
            mean_rev = regime.mean_speed_kmh
            theta = regime.speed_reversion
            sigma = regime.speed_vol
            dW = float(rng.normal(0, math.sqrt(dt)))
            speed_kmh = (
                speed_kmh
                + theta * (mean_rev - speed_kmh) * dt
                + sigma * dW
            )
            speed_kmh = max(0.0, speed_kmh)

            # Acceleration: derivative of speed + process noise
            # Difference from last step divided by dt, plus noise
            if t == 0:
                prev_speed = speed_kmh
            else:
                prev_speed = rows[-1]["speed_kmh"]

            accel_from_speed = (speed_kmh - prev_speed) / dt / 3.6  # convert to m/s²
            accel_noise = float(rng.normal(0, regime.accel_noise_std * 0.3))
            accel_ms2 = accel_from_speed + accel_noise

            # Update position using heading and speed
            speed_ms = speed_kmh / 3.6
            heading_rad = math.radians(heading)
            # Approximate lat/lon displacement
            dlat = speed_ms * math.cos(heading_rad) * dt / 111_320
            dlon = (
                speed_ms
                * math.sin(heading_rad)
                * dt
                / (111_320 * math.cos(math.radians(lat)))
            )
            lat += dlat
            lon += dlon

            # Slow random heading drift
            heading = (heading + float(rng.normal(0, 2.0))) % 360

            timestamp = trip_start + timedelta(seconds=t)

            rows.append(
                {
                    "driver_id": driver_id,
                    "trip_id": trip_id,
                    "timestamp": timestamp,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "speed_kmh": round(speed_kmh, 2),
                    "acceleration_ms2": round(accel_ms2, 4),
                    "heading_deg": round(heading, 1),
                }
            )

        return rows
