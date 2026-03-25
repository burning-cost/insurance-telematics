# insurance-telematics

[![PyPI](https://img.shields.io/pypi/v/insurance-telematics)](https://pypi.org/project/insurance-telematics/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-telematics)](https://pypi.org/project/insurance-telematics/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/insurance-telematics/blob/main/notebooks/quickstart.ipynb)

Your telematics provider scores drivers using a black-box algorithm you cannot audit, retrain, or challenge — and raw harsh-braking counts added directly to a GLM treat a single trip's noise as signal. insurance-telematics gives you the full pipeline in auditable Python: classify trip-level behaviour using a Hidden Markov Model into latent driving regimes, aggregate to driver-level state fractions with Bühlmann-Straub credibility weighting, and produce GLM-ready features you understand and can defend to the FCA.

**Blog post:** [HMM-Based Telematics Risk Scoring for Insurance Pricing](https://burning-cost.github.io/2026/03/13/insurance-telematics/)

The academic basis is Jiang & Shi (2024) in NAAJ: HMM latent states capture driving regimes (cautious, normal, aggressive) and the fraction of time in the aggressive state is more predictive of claim frequency than raw speed or harsh event counts alone.

## Part of the Burning Cost stack

Takes raw trip sensor data (GPS, accelerometer). Feeds HMM-scored, credibility-weighted driver-level features into [insurance-gam](https://github.com/burning-cost/insurance-gam) (as interpretable tariff inputs) and [insurance-causal](https://github.com/burning-cost/insurance-causal) (to separate causal driving style effects from correlated demographics). → [See the full stack](https://burning-cost.github.io/stack/)

## Why use this?

- Most telematics scoring tools are either black-box APIs you cannot audit, or academic scripts that do not run on production data. This library gives you the full auditable pipeline in Python: GPS cleaning, HMM state classification, credibility-weighted driver scoring, and a Poisson GLM-ready feature DataFrame.
- HMM latent states (cautious, normal, aggressive) outperform raw feature averages as claim frequency predictors: the fraction of time in the aggressive state captures persistent driving style rather than trip-level noise — 3–8pp Gini improvement on synthetic fleets following Jiang & Shi (2024, NAAJ).
- Credibility weighting at driver level (Bühlmann-Straub) handles new drivers with few trips without inflating or suppressing their estimated risk profile — the same sound methodology as group credibility pricing, applied per driver.
- No raw telematics data available yet? TripSimulator generates a realistic synthetic fleet (three driving regimes, Ornstein-Uhlenbeck speed processes, Poisson claims) so you can prototype the full workflow and validate your infrastructure before your data is available.
- Output is a Polars DataFrame of named features (harsh_braking_rate, speeding_fraction, HMM state fractions) you drop straight into your existing Poisson frequency GLM alongside traditional rating factors — no glue code required.

## Expected Performance

Validated on a synthetic fleet of 5,000 UK motor drivers with 30 trips each. Known DGP: 3 hidden driving states (calm, moderate, aggressive) with Ornstein-Uhlenbeck speed processes and known accident probabilities per state (4%/yr cautious, 10%/yr moderate, 28%/yr aggressive). Full validation notebook: `notebooks/databricks_validation.py`.

| Approach | Gini (test) | Top/Bottom decile ratio | Feature computation |
|----------|-------------|------------------------|---------------------|
| Simple summary features (mean speed, harsh events) | baseline | baseline | < 1s |
| Threshold-based scoring (% time above speed/braking thresholds) | +1–3pp | +0.1–0.3x | < 1s |
| HMM state fractions (this library) | **+5–10pp** | **+0.5–1.5x** | 30–90s |

**Gini improvement over simple summary features: 5–10 percentage points.** The HMM advantage comes from separating persistent driving style from trip-level noise. A driver who averaged 65 km/h because of one fast motorway run looks identical to a driver who consistently drives at 65 km/h in the summary feature world. The HMM separates them through the temporal state sequence — the consistently moderate driver has a different state_2_fraction profile.

**State classification accuracy:** On the 3-state DGP with 30 trips per driver, HMM state_2_fraction (aggressive) achieves Spearman rho >= 0.70 with the true aggressive fraction from the DGP. Top-quartile aggressive driver overlap is >= 50% (versus 25% at random). This means the HMM correctly identifies more than half of the genuinely high-risk drivers even with a relatively short trip history.

**Where the HMM advantage is largest:** Portfolios where driving style is genuinely regime-based (distinct cautious vs aggressive segments) and drivers have >= 20 trips. The improvement is proportional to how well-separated the latent states are in your data. On portfolios where driving behaviour is genuinely continuous and unimodal, the gain may be closer to 3pp than 10pp.

**Practical limits:**
- Below 10 trips per driver, state estimation variance is high; use credibility-weighted summary features below this threshold
- HMM fit time scales with n_trips: 30-90s for 5,000 drivers x 30 trips on Databricks serverless
- For fleets > 50,000 drivers, batch the HMM fitting by driver cohort or use Spark UDFs

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
uv add insurance-telematics
```

Requires Python 3.10+. Dependencies: polars, numpy, scipy, hmmlearn, statsmodels, scikit-learn.

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-telematics/discussions). Found it useful? A ⭐ helps others find it.

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

## References

- Jiang, Q. & Shi, Y. (2024). "Auto Insurance Pricing Using Telematics Data: Application of a Hidden Markov Model." *NAAJ* 28(4), pp.822-839.
- Wüthrich, M.V. (2017). "Covariate Selection from Telematics Car Driving Data." *European Actuarial Journal* 7, pp.89-108.
- Gao, G., Wang, H. & Wüthrich, M.V. (2021). "Boosting Poisson Regression Models with Telematics Car Driving Data." *Machine Learning* 111, pp.1787-1827.
- Henckaerts, R. & Antonio, K. (2022). "The Added Value of Dynamically Updating Motor Insurance Prices with Telematics Data." *Insurance: Mathematics and Economics* 103, pp.79-95.
- Guillen, M., Pérez-Marín, A.M. & Nielsen, J.P. (2024). "Pricing Weekly Motor Insurance Drivers with Behavioural and Contextual Telematics Data." *Heliyon* 10(17).

## Databricks Notebook

A ready-to-run validation notebook benchmarking this library against simple summary features and threshold-based scoring on a 5,000-driver synthetic fleet is at [`notebooks/databricks_validation.py`](notebooks/databricks_validation.py). It covers DGP construction, HMM state recovery accuracy, GLM discrimination comparison, and the full GLM-ready feature export workflow.


## Limitations

- The HMM advantage is proportional to how state-structured the true DGP is. On portfolios where driving style varies continuously rather than in discrete regimes, the improvement in Gini may be small. The `TripSimulator` DGP is deliberately state-based, which is the best case for the HMM.
- `urban_fraction` is computed as a time-fraction, not a distance-fraction. A driver spending 20 minutes in slow urban traffic and 5 minutes on a motorway will have high `urban_fraction` despite low urban distance. This is consistent with risk (time on road matters for frequency) but differs from how some reinsurers define urban exposure. Document this before using it in ceded pricing.
- HMM state labels are not portable across separately fitted models. State 0 being "cautious" on one portfolio depends on the training data distribution. Do not compare raw state fractions between models fitted on different fleets or time periods.
- The HMM is stateless across trips. Each trip is scored independently using the fitted emission and transition parameters. Within-trip regime transitions are captured, but across-trip learning requires the continuous-time HMM with per-driver state initialisation.
- For large fleets (n > 10,000 drivers, 50+ trips each), HMM fitting should be done on a sample or distributed via Spark. The default implementation fits in Python on a single machine and will be slow above this scale.


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Customer survival and churn models — telematics-based retention modelling uses survival curves for CLV |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Generalised Additive Models — smooth non-linear telematics score effects without discretising into bands |

## Training Course

Want structured learning? [Insurance Pricing in Python](https://burning-cost.github.io/course) is a 12-module course covering the full pricing workflow. Module 7 covers telematics — HMM-derived driving state features, trip aggregation, and integrating behavioural scores into a Poisson GLM. £97 one-time.

## Community

- **Questions?** Start a [Discussion](https://github.com/burning-cost/insurance-telematics/discussions)
- **Found a bug?** Open an [Issue](https://github.com/burning-cost/insurance-telematics/issues)
- **Blog & tutorials:** [burning-cost.github.io](https://burning-cost.github.io)

If this library saves you time, a star on GitHub helps others find it.

## Licence

MIT

## Performance

Benchmarked against **raw trip-level feature averages** (mean speed, harsh braking rate, harsh acceleration rate, night fraction) in a Poisson GLM on a synthetic fleet of 300 drivers with 40 trips each. Both models use the same Poisson GLM structure — the only difference is whether the input features are raw averages or HMM-derived state fractions. See `notebooks/benchmark_telematics.py` for full methodology.

- **Gini coefficient:** HMM-derived state features consistently produce higher Gini (better driver risk discrimination) than raw averages on a latent-state DGP. The improvement is 3-8pp on typical synthetic fleets because state fractions capture persistent driving style rather than trip-level noise.
- **Loss ratio separation:** The top-to-bottom quintile loss ratio ratio is larger with HMM features — the model puts high-risk drivers into higher predicted deciles more reliably.
- **A/E calibration:** Max A/E deviation by quintile is similar between methods; the HMM advantage is in discrimination (rank ordering), not overall calibration (which is a GLM property shared by both).
- **Fit time:** The full pipeline (clean + extract + HMM 200 iterations + GLM) takes 30-90s on 300 drivers. Raw averages add effectively zero overhead. For large fleets, HMM fitting should be done on a sample or parallelised via Spark.
- **Limitation:** The HMM advantage is proportional to how state-structured the true DGP is. On portfolios where driving style is genuinely continuous rather than regime-based, the gain may be smaller. The `TripSimulator` DGP is deliberately state-based, which is the best case for the HMM.

---

## Part of the Burning Cost Toolkit

Open-source Python libraries for UK personal lines insurance pricing. [Browse all libraries](https://burning-cost.github.io/tools/)

| Library | Description |
|---------|-------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | DML causal inference — establishes whether HMM state fractions causally drive claims or proxy for other risk factors |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | FCA proxy discrimination auditing — telematics scores can act as proxies for protected characteristics |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — monitors whether telematics-derived GLM factors remain well-calibrated over time |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals — quantifies uncertainty around the Poisson frequency predictions |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model validation and MRM governance — governance pack for telematics models entering production |
