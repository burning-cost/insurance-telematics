# insurance-telematics

[![PyPI](https://img.shields.io/pypi/v/insurance-telematics)](https://pypi.org/project/insurance-telematics/) [![Python](https://img.shields.io/pypi/pyversions/insurance-telematics)](https://pypi.org/project/insurance-telematics/) [![Tests](https://github.com/burning-cost/insurance-telematics/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-telematics/actions/workflows/tests.yml) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/burning-cost/insurance-telematics/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-telematics/blob/main/notebooks/quickstart.ipynb) [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-telematics/blob/main/notebooks/quickstart.ipynb)

---

## The problem

Raw telematics data — accelerometer events, GPS speed, harsh braking counts — does not map directly to GLM rating factors. Adding raw trip averages to a Poisson GLM treats a single motorway run as equivalent to a persistent driving style, and most telematics scoring black boxes cannot be audited, retrained, or challenged when the FCA asks how the score was derived.

The FCA's Consumer Duty and its expectations around pricing data require that telematics-based pricing is explainable. A score produced by a vendor's API is not explainable.

**Blog post:** [HMM-Based Telematics Risk Scoring for Insurance Pricing](https://burning-cost.github.io/2026/03/13/insurance-telematics/)

---

## Why this library?

Most telematics scoring tools are either black-box vendor APIs you cannot interrogate, or academic scripts that do not run on production data. This library gives you the full auditable pipeline in Python: GPS cleaning, HMM state classification, credibility-weighted driver scoring, and a Poisson GLM-ready feature DataFrame you can hand to a pricing committee and explain factor by factor.

The academic basis is Jiang & Shi (2024, NAAJ): Hidden Markov Model latent states capture driving regimes — cautious, normal, aggressive — and the fraction of time in the aggressive state is more predictive of claim frequency than raw speed or harsh event counts alone.

---

## Compared to alternatives

| | Vendor black-box | Raw feature averages | Manual threshold scoring | **insurance-telematics** |
|---|---|---|---|---|
| Auditable methodology | No | Yes | Yes | Yes |
| Captures driving regimes | Possibly | No | Partial | Yes (HMM) |
| Handles sparse new drivers | Varies | No | No | Yes (credibility weighting) |
| GLM-ready output | Varies | Manual | Manual | Yes (Polars DataFrame) |
| FCA-explainable | No | Yes | Yes | Yes |
| Synthetic data for prototyping | No | No | No | Yes (`TripSimulator`) |

---

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

No raw data yet? `TripSimulator` generates a realistic synthetic fleet — three driving regimes, Ornstein-Uhlenbeck speed processes, synthetic Poisson claims — so you can prototype the full workflow before your data arrives.

---

## The full pipeline

```
Raw 1Hz trip data (CSV or Parquet)
  → load_trips()            — load and schema-map
  → clean_trips()           — GPS jump removal, acceleration derivation, road type
  → extract_trip_features() — harsh braking rate, speeding fraction, night fraction
  → DrivingStateHMM         — classify each trip into latent driving states
  → aggregate_to_driver()   — Bühlmann-Straub credibility weighting to driver level
  → TelematicsScoringPipeline — Poisson GLM producing predicted claim frequency
```

```python
from insurance_telematics import load_trips, clean_trips, extract_trip_features
from insurance_telematics import DrivingStateHMM, aggregate_to_driver

trips_raw = load_trips("trips.csv")
trips_clean = clean_trips(trips_raw)
features = extract_trip_features(trips_clean)

model = DrivingStateHMM(n_states=3)
model.fit(features)
states = model.predict_states(features)
driver_hmm_features = model.driver_state_features(features, states)

driver_risk = aggregate_to_driver(features, credibility_threshold=30)
driver_risk = driver_risk.join(driver_hmm_features, on="driver_id", how="left")
```

---

## Input data format

One row per second (1Hz) with these columns:

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

Non-standard column names? Use the `schema` parameter:

```python
trips = load_trips("raw_data.csv", schema={"gps_speed": "speed_kmh"})
```

---

## Features extracted per trip

- `harsh_braking_rate` — events/km where deceleration < −3.5 m/s²
- `harsh_accel_rate` — events/km where acceleration > +3.5 m/s²
- `harsh_cornering_rate` — events/km (estimated from heading-change rate)
- `speeding_fraction` — fraction of time exceeding road-type speed limit
- `night_driving_fraction` — fraction of distance driven 23:00–05:00
- `urban_fraction` — fraction of time at speed < 50 km/h
- `mean_speed_kmh`, `p95_speed_kmh`, `speed_variation_coeff`

---

## HMM state classification

With three states the HMM typically produces:
- State 0: cautious — low speed, low variance, urban driving
- State 1: normal — mixed road types, moderate speed
- State 2: aggressive — high speed variance, high harsh event rate

The fraction of time in state 2 per driver is the primary GLM covariate. Following Jiang & Shi (2024), this outperforms raw feature averages as a predictor of claim frequency.

```python
from insurance_telematics import DrivingStateHMM, ContinuousTimeHMM
import numpy as np

hmm = DrivingStateHMM(n_states=3)
hmm.fit(trip_features_df)
states = hmm.predict_states(trip_features_df)

# Continuous-time variant for variable trip lengths
time_deltas = np.ones(len(trip_features_df))  # inter-observation intervals in minutes
cthmm = ContinuousTimeHMM(n_states=3)
cthmm.fit(trip_features_df, time_deltas=time_deltas)
```

---

## Validated performance

On a synthetic fleet of 5,000 drivers with 30 trips each and a known 3-state DGP (cautious/moderate/aggressive, Ornstein-Uhlenbeck speed processes):

| Approach | Gini improvement | Feature computation |
|---|---|---|
| Raw summary features (mean speed, harsh events) | baseline | < 1s |
| Threshold-based scoring | +1–3pp | < 1s |
| HMM state fractions (this library) | **+5–10pp** | 30–90s |

The HMM advantage comes from separating persistent driving style from trip-level noise. The `state_2_fraction` (aggressive driving) achieves Spearman rho ≥ 0.70 with the true aggressive fraction from the DGP. Correct identification of top-quartile high-risk drivers: > 50% (vs 25% at random).

Fit time scales with portfolio size: 30–90 seconds for 5,000 drivers × 30 trips on Databricks serverless. For fleets above 50,000 drivers, batch by cohort or use Spark UDFs.

Full validation notebook: `notebooks/databricks_validation.py`.

---

## FCA context

The FCA's Consumer Duty (2023) and its ongoing scrutiny of data-driven pricing require that pricing models are explainable and do not create unjustified cross-subsidies. A vendor telematics score that you cannot decompose into features is difficult to defend under this framework. The HMM state fractions and the trip-level features computed by this library are fully auditable: you can show a regulator exactly which behaviours drive the score.

---

## Limitations

- The HMM advantage is proportional to how state-structured the true DGP is. On portfolios where driving style varies continuously rather than in discrete regimes, the Gini improvement may be closer to 3pp than 10pp.
- Below 10 trips per driver, state estimation variance is high. Use credibility-weighted summary features below this threshold.
- HMM state labels are not portable across separately fitted models. State 0 being "cautious" depends on the training data distribution. Do not compare raw state fractions between models fitted on different fleets or time periods.
- `urban_fraction` is a time-fraction, not a distance-fraction. Document this before using it in ceded pricing, where some reinsurers define urban exposure on a distance basis.

---

## Part of the Burning Cost stack

Takes raw trip sensor data (GPS, accelerometer). Feeds HMM-scored, credibility-weighted driver-level features into [insurance-gam](https://github.com/burning-cost/insurance-gam) and [insurance-causal](https://github.com/burning-cost/insurance-causal). [See the full stack](https://burning-cost.github.io/stack/)

| Library | Description |
|---|---|
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | GAMs — smooth non-linear telematics score effects without discretising into bands |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — separates causal driving style effects from correlated demographics |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | FCA proxy discrimination auditing — telematics scores can proxy for protected characteristics |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors whether telematics-derived GLM factors remain calibrated |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — sign-off pack for telematics models in production |

---

## References

- Jiang, Q. & Shi, Y. (2024). "Auto Insurance Pricing Using Telematics Data: Application of a Hidden Markov Model." *NAAJ* 28(4), pp.822–839.
- Wüthrich, M.V. (2017). "Covariate Selection from Telematics Car Driving Data." *European Actuarial Journal* 7, pp.89–108.
- Gao, G., Wang, H. & Wüthrich, M.V. (2021). "Boosting Poisson Regression Models with Telematics Car Driving Data." *Machine Learning* 111, pp.1787–1827.
- Henckaerts, R. & Antonio, K. (2022). "The Added Value of Dynamically Updating Motor Insurance Prices with Telematics Data." *Insurance: Mathematics and Economics* 103, pp.79–95.

---

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-telematics/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-telematics/issues)
- **Blog and tutorials:** [burning-cost.github.io](https://burning-cost.github.io)
- **Training course:** [Insurance Pricing in Python](https://burning-cost.github.io/course) — Module 7 covers telematics. £97 one-time.

## Licence

MIT
