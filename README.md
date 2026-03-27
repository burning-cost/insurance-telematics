# insurance-telematics

Turn raw GPS and accelerometer trip data into GLM-ready driver risk features using Hidden Markov Models — auditable, credibility-weighted, and explainable to the FCA.

[![PyPI](https://img.shields.io/pypi/v/insurance-telematics)](https://pypi.org/project/insurance-telematics/) [![Python](https://img.shields.io/pypi/pyversions/insurance-telematics)](https://pypi.org/project/insurance-telematics/) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-telematics/blob/main/LICENSE)

## Why this?

Raw telematics features — mean speed, harsh braking counts — treat a single motorway run as equivalent to a persistent driving style. HMM state classification separates trip-level noise from genuine behavioural regimes (cautious, normal, aggressive), and the fraction of time in the aggressive state is more predictive of claim frequency than raw averages alone (Jiang & Shi, 2024, NAAJ). Unlike vendor scores, every feature is auditable: you can show a regulator exactly which behaviours drive the output.

**Blog post:** [HMM-Based Telematics Risk Scoring for Insurance Pricing](https://burning-cost.github.io/2026/03/13/insurance-telematics/)

## Quickstart

```bash
uv add insurance-telematics
```

```python
from insurance_telematics import TripSimulator, TelematicsScoringPipeline

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=100, trips_per_driver=50)

pipe = TelematicsScoringPipeline(n_hmm_states=3)
pipe.fit(trips_df, claims_df)
predictions = pipe.predict(trips_df)
```

No raw data yet? `TripSimulator` generates a realistic synthetic fleet — three driving regimes, Ornstein-Uhlenbeck speed processes, synthetic Poisson claims — so you can prototype before your data arrives.

## Use cases

### 1. Trip scoring for a new-to-telematics portfolio

Score each trip and aggregate to driver level with Bühlmann-Straub credibility weighting. Drivers with fewer than 10 trips fall back to portfolio means automatically.

```python
from insurance_telematics import load_trips, clean_trips, extract_trip_features
from insurance_telematics import aggregate_to_driver

trips = load_trips("trips.parquet")
features = extract_trip_features(clean_trips(trips))
driver_risk = aggregate_to_driver(features, credibility_threshold=30)
# driver_risk: one row per driver_id, GLM-ready
```

### 2. HMM state classification — extracting driving regime features

Classify each trip into latent driving states and derive the regime fractions that feed your Poisson GLM.

```python
from insurance_telematics import DrivingStateHMM

hmm = DrivingStateHMM(n_states=3)
hmm.fit(features)
states = hmm.predict_states(features)
hmm_features = hmm.driver_state_features(features, states)
# hmm_features includes state_0_fraction, state_1_fraction, state_2_fraction per driver
```

With three states the HMM typically recovers: state 0 = cautious (low speed, urban), state 1 = normal (mixed), state 2 = aggressive (high speed variance, high harsh event rate). The `state_2_fraction` is the primary GLM covariate.

### 3. Variable trip length — continuous-time HMM

For portfolios where observation intervals are irregular (trips logged at variable Hz), use `ContinuousTimeHMM` to avoid biasing state estimates toward shorter trips.

```python
from insurance_telematics import ContinuousTimeHMM
import numpy as np

time_deltas = np.array(features["trip_duration_min"])
cthmm = ContinuousTimeHMM(n_states=3)
cthmm.fit(features, time_deltas=time_deltas)
```

## Full pipeline

```
Raw 1Hz trip data (CSV or Parquet)
  → load_trips()            — load and schema-map
  → clean_trips()           — GPS jump removal, acceleration derivation, road type
  → extract_trip_features() — harsh braking rate, speeding fraction, night fraction
  → DrivingStateHMM         — classify each trip into latent driving states
  → aggregate_to_driver()   — Bühlmann-Straub credibility weighting to driver level
  → TelematicsScoringPipeline — Poisson GLM producing predicted claim frequency
```

## Input data format

One row per second (1Hz):

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

Non-standard column names? Use `schema`:

```python
trips = load_trips("raw_data.csv", schema={"gps_speed": "speed_kmh"})
```

## Features extracted per trip

- `harsh_braking_rate` — events/km where deceleration < −3.5 m/s²
- `harsh_accel_rate` — events/km where acceleration > +3.5 m/s²
- `harsh_cornering_rate` — events/km (estimated from heading-change rate)
- `speeding_fraction` — fraction of time exceeding road-type speed limit
- `night_driving_fraction` — fraction of distance driven 23:00–05:00
- `urban_fraction` — fraction of time at speed < 50 km/h
- `mean_speed_kmh`, `p95_speed_kmh`, `speed_variation_coeff`

## Compared to alternatives

| | Vendor black-box | Raw feature averages | Manual threshold scoring | **insurance-telematics** |
|---|---|---|---|---|
| Auditable methodology | No | Yes | Yes | Yes |
| Captures driving regimes | Possibly | No | Partial | Yes (HMM) |
| Handles sparse new drivers | Varies | No | No | Yes (credibility weighting) |
| GLM-ready output | Varies | Manual | Manual | Yes (Polars DataFrame) |
| FCA-explainable | No | Yes | Yes | Yes |
| Synthetic data for prototyping | No | No | No | Yes (`TripSimulator`) |

## Validated performance

On a synthetic fleet of 5,000 drivers × 30 trips with a known 3-state DGP:

| Approach | Gini improvement | Feature computation |
|---|---|---|
| Raw summary features (mean speed, harsh events) | baseline | < 1s |
| Threshold-based scoring | +1–3pp | < 1s |
| HMM state fractions (this library) | **+5–10pp** | 30–90s |

`state_2_fraction` achieves Spearman rho ≥ 0.70 with the true aggressive fraction from the DGP. Correct identification of top-quartile high-risk drivers: > 50% (vs 25% at random). The HMM advantage is proportional to how regime-structured the true DGP is — on portfolios with continuously varying style, expect closer to 3pp.

Fit time: 30–90 seconds for 5,000 drivers × 30 trips on Databricks serverless. For fleets above 50,000 drivers, batch by cohort or use Spark UDFs.

Full validation notebook: `notebooks/databricks_validation.py`.

## Limitations

- Below 10 trips per driver, state estimation variance is high. Use credibility-weighted summary features below this threshold.
- HMM state labels are not portable across separately fitted models. Do not compare raw state fractions between models fitted on different fleets or time periods.
- `urban_fraction` is a time-fraction, not a distance-fraction. Document this before using it in ceded pricing where some reinsurers define urban exposure on a distance basis.

## Part of the Burning Cost stack

Takes raw trip sensor data (GPS, accelerometer). Feeds HMM-scored, credibility-weighted driver-level features into [insurance-gam](https://github.com/burning-cost/insurance-gam) and [insurance-causal](https://github.com/burning-cost/insurance-causal).

| Library | Role |
|---|---|
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Smooth non-linear telematics score effects without discretising into bands |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML — separates causal driving style effects from correlated demographics |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | FCA proxy discrimination auditing — telematics scores can proxy for protected characteristics |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Drift detection — monitors whether telematics-derived GLM factors remain calibrated |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for telematics models in production |

## References

- Jiang, Q. & Shi, Y. (2024). "Auto Insurance Pricing Using Telematics Data: Application of a Hidden Markov Model." *NAAJ* 28(4), pp.822–839.
- Wüthrich, M.V. (2017). "Covariate Selection from Telematics Car Driving Data." *European Actuarial Journal* 7, pp.89–108.
- Gao, G., Wang, H. & Wüthrich, M.V. (2021). "Boosting Poisson Regression Models with Telematics Car Driving Data." *Machine Learning* 111, pp.1787–1827.
- Henckaerts, R. & Antonio, K. (2022). "The Added Value of Dynamically Updating Motor Insurance Prices with Telematics Data." *Insurance: Mathematics and Economics* 103, pp.79–95.

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-telematics/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-telematics/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 7 covers telematics. £97 one-time.

## Licence

MIT
