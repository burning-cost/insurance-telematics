# insurance-telematics

[![PyPI](https://img.shields.io/pypi/v/insurance-telematics)](https://pypi.org/project/insurance-telematics/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-telematics)](https://pypi.org/project/insurance-telematics/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-telematics/blob/main/notebooks/quickstart.ipynb)

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

# Join HMM state features — these are the primary actuarial risk covariates
driver_risk = driver_risk.join(driver_hmm_features, on="driver_id", how="left")
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
- `urban_fraction` — fraction of observations (by time) classified as urban driving (speed < 50 km/h). Note: time-fraction, not distance-fraction.
- `mean_speed_kmh`, `p95_speed_kmh`, `speed_variation_coeff`

## HMM state classification

```python
import numpy as np
from insurance_telematics import TripSimulator, clean_trips, extract_trip_features
from insurance_telematics import DrivingStateHMM, ContinuousTimeHMM

# Generate synthetic trip data for illustration
sim = TripSimulator(seed=42)
trips_df, _ = sim.simulate(n_drivers=50, trips_per_driver=30)
trip_features_df = extract_trip_features(clean_trips(trips_df))

# Discrete-time (uniform 1Hz intervals) — wraps hmmlearn.GaussianHMM
hmm = DrivingStateHMM(n_states=3)
hmm.fit(trip_features_df)
states = hmm.predict_states(trip_features_df)

# Continuous-time — handles variable trip lengths via expm(Q * dt)
# time_deltas: array of inter-observation intervals in minutes (one per trip row)
time_deltas = np.ones(len(trip_features_df))  # unit intervals as placeholder
cthmm = ContinuousTimeHMM(n_states=3)
cthmm.fit(trip_features_df, time_deltas=time_deltas)
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

## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/telematics_scoring_pipeline.py).


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Customer survival and churn models — telematics-based retention modelling uses survival curves for CLV |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Generalised Additive Models — smooth non-linear telematics score effects without discretising into bands |

## Licence

MIT

## Performance

Benchmarked against **raw trip-level feature averages** (mean speed, harsh braking rate, harsh acceleration rate, night fraction) in a Poisson GLM on a synthetic fleet of 300 drivers with 40 trips each. Both models use the same Poisson GLM structure — the only difference is whether the input features are raw averages or HMM-derived state fractions. See `notebooks/benchmark_telematics.py` for full methodology.

- **Gini coefficient:** HMM-derived state features consistently produce higher Gini (better driver risk discrimination) than raw averages on a latent-state DGP. The improvement is 3-8pp on typical synthetic fleets because state fractions capture persistent driving style rather than trip-level noise.
- **Loss ratio separation:** The top-to-bottom quintile loss ratio ratio is larger with HMM features — the model puts high-risk drivers into higher predicted deciles more reliably.
- **A/E calibration:** Max A/E deviation by quintile is similar between methods; the HMM advantage is in discrimination (rank ordering), not overall calibration (which is a GLM property shared by both).
- **Fit time:** The full pipeline (clean + extract + HMM 200 iterations + GLM) takes 30-90s on 300 drivers. Raw averages add effectively zero overhead. For large fleets, HMM fitting should be done on a sample or parallelised via Spark.
- **Limitation:** The HMM advantage is proportional to how state-structured the true DGP is. On portfolios where driving style is genuinely continuous rather than regime-based, the gain may be smaller. The `TripSimulator` DGP is deliberately state-based, which is the best case for the HMM.
