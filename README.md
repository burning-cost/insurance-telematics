# insurance-telematics

Raw telematics trip data to GLM-ready risk scores. Built for UK motor insurance pricing teams.

Most telematics scoring tools are either black-box APIs (you get a number, you cannot audit it) or one-off academic scripts that do not run on your data. This library gives you the full pipeline in Python: load 1Hz GPS/accelerometer data, classify driving behaviour using a Hidden Markov Model, aggregate to driver-level risk scores, and produce a feature DataFrame you can drop into your Poisson frequency GLM alongside traditional rating factors.

The academic basis is Jiang & Shi (2024) in NAAJ: HMM latent states capture driving regimes (cautious, normal, aggressive) and the fraction of time in the aggressive state is more predictive of claim frequency than raw speed or harsh event counts alone.

## Five-line usage

```python
from insurance_telematics import TripSimulator, TelematicsScoringPipeline

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=100, trips_per_driver=50)

pipe = TelematicsScoringPipeline(n_hmm_states=3)
pipe.fit(trips_df, claims_df)
predictions = pipe.predict(trips_df)
```

No raw telematics data? Use `TripSimulator` to generate a realistic synthetic fleet — three driving regimes (cautious, normal, aggressive), Ornstein-Uhlenbeck speed processes, synthetic Poisson claims — so you can prototype the full workflow before your data is available.

## What it does

```
Raw 1Hz trip data (CSV or Parquet)
  → clean_trips()         — GPS jump removal, acceleration derivation, road type
  → extract_trip_features()  — harsh braking rate, speeding fraction, night fraction, etc.
  → DrivingStateHMM       — classify each trip into a latent driving state
  → aggregate_to_driver() — Bühlmann-Straub credibility weighting to driver level
  → TelematicsScoringPipeline — Poisson GLM producing predicted claim frequency
```

## Installation

```bash
pip install insurance-telematics
```

Requires Python 3.10+. Dependencies: polars, numpy, scipy, hmmlearn, statsmodels, scikit-learn.

## Running the full pipeline on your data

```python
from insurance_telematics import load_trips, clean_trips, extract_trip_features
from insurance_telematics import DrivingStateHMM, aggregate_to_driver

# Load from CSV (or Parquet)
trips_raw = load_trips("trips.csv")

# Clean: removes GPS jumps, derives acceleration and jerk, classifies road type
trips_clean = clean_trips(trips_raw)

# Extract trip-level features
features = extract_trip_features(trips_clean)

# Fit HMM and get driver-level state features
model = DrivingStateHMM(n_states=3)
model.fit(features)
states = model.predict_states(features)
driver_hmm_features = model.driver_state_features(features, states)

# Aggregate to driver level with credibility weighting
driver_risk = aggregate_to_driver(features, credibility_threshold=30)
```

## Input data format

The library expects one row per second (1Hz) with these columns:

| Column | Type | Notes |
|---|---|---|
| `trip_id` | string | Unique per trip |
| `timestamp` | datetime | ISO 8601 or Unix epoch |
| `latitude` | float | Decimal degrees |
| `longitude` | float | Decimal degrees |
| `speed_kmh` | float | GPS speed |
| `acceleration_ms2` | float | Optional — derived from speed if absent |
| `heading_deg` | float | Optional — used for cornering estimation |
| `driver_id` | string | Optional — "unknown" if absent |

Use the `schema` parameter to rename non-standard columns:

```python
trips = load_trips("raw_data.csv", schema={"gps_speed": "speed_kmh"})
```

## Features extracted per trip

- `harsh_braking_rate` — events/km where deceleration < -3.5 m/s²
- `harsh_accel_rate` — events/km where acceleration > +3.5 m/s²
- `harsh_cornering_rate` — events/km (estimated from heading-change rate)
- `speeding_fraction` — fraction of time exceeding road-type speed limit
- `night_driving_fraction` — fraction of distance driven 23:00-05:00
- `urban_fraction` — fraction of distance at speeds < 50 km/h
- `mean_speed_kmh`, `p95_speed_kmh`, `speed_variation_coeff`

## HMM state classification

```python
from insurance_telematics import DrivingStateHMM, ContinuousTimeHMM

# Discrete-time (uniform 1Hz intervals) — wraps hmmlearn.GaussianHMM
hmm = DrivingStateHMM(n_states=3)
hmm.fit(trip_features_df)
states = hmm.predict_states(trip_features_df)

# Continuous-time — handles variable trip lengths via expm(Q * dt)
cthmm = ContinuousTimeHMM(n_states=3)
cthmm.fit(trip_features_df, time_deltas=time_intervals_minutes)
```

With three states the HMM typically produces:
- State 0: cautious — low speed, low variance, urban driving
- State 1: normal — mixed road types, moderate speed
- State 2: aggressive — high speed variance, high harsh event rate

The fraction of time in state 2 per driver is the key GLM covariate. Following Jiang & Shi (2024), this outperforms raw feature averages as a predictor of claim frequency.

## Composite risk score

`aggregate_to_driver()` produces a `composite_risk_score` (0-100) as a weighted combination of all features, scaled to the portfolio range. This is a summary diagnostic — use the individual features as GLM covariates for pricing, not the composite score directly.

## Key papers

- Jiang, Q. & Shi, Y. (2024). "Auto Insurance Pricing Using Telematics Data: Application of a Hidden Markov Model." *NAAJ* 28(4), pp.822-839.
- Wüthrich, M.V. (2017). "Covariate Selection from Telematics Car Driving Data." *European Actuarial Journal* 7, pp.89-108.
- Gao, G., Wang, H. & Wüthrich, M.V. (2021). "Boosting Poisson Regression Models with Telematics Car Driving Data." *Machine Learning* 111, pp.1787-1827.
- Henckaerts, R. & Antonio, K. (2022). "The Added Value of Dynamically Updating Motor Insurance Prices with Telematics Data." *Insurance: Mathematics and Economics* 103, pp.79-95.
- Guillen, M., Pérez-Marín, A.M. & Nielsen, J.P. (2024). "Pricing Weekly Motor Insurance Drivers with Behavioural and Contextual Telematics Data." *Heliyon* 10(17).

## Databricks notebook demo

See `notebooks/telematics_demo.py` for a complete walkthrough on a synthetic fleet of 100 drivers — including HMM fitting, state feature extraction, and Poisson GLM training — designed to run on Databricks serverless compute.

## Licence

MIT
