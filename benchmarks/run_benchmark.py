"""
Benchmark: insurance-telematics
================================

Scenario: A UK motor insurer adds telematics to its product. They have 300
drivers with 40 trips each. We compare two approaches to building the Poisson
frequency model:

  Baseline: Raw trip feature averages in a Poisson GLM
            (mean speed, harsh braking rate, harsh accel rate, night fraction)

  Library:  HMM-derived state fraction features in a Poisson GLM
            (fraction of time in aggressive/cautious/normal driving state)

The DGP is deliberately state-structured (TripSimulator generates three latent
driving regimes). This is the best case for HMM features. Results on real data
where driving style is more continuous will show smaller gains.

Metrics:
  - Gini coefficient (rank discrimination)
  - Poisson deviance on held-out 30% test set
  - Top-quintile vs bottom-quintile A/E ratio (loss ratio separation)
  - A/E by quintile (calibration check)

Seed: 42.
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from insurance_telematics import (
    TripSimulator,
    clean_trips,
    extract_trip_features,
    DrivingStateHMM,
    aggregate_to_driver,
    TelematicsScoringPipeline,
)

print("=" * 60)
print("Benchmark: insurance-telematics")
print("=" * 60)

# ---------------------------------------------------------------------------
# Generate synthetic fleet
# ---------------------------------------------------------------------------
t0 = time.time()
sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=300, trips_per_driver=40)
sim_time = time.time() - t0
print(f"\nSimulated 300 drivers x 40 trips in {sim_time:.2f}s")
print(f"  Total trips: {len(trips_df['trip_id'].unique())}")
print(f"  Claims data: {len(claims_df)} driver records")

# ---------------------------------------------------------------------------
# Pipeline: clean + extract trip features (shared for both approaches)
# ---------------------------------------------------------------------------
t0 = time.time()
trips_clean = clean_trips(trips_df)
trip_features = extract_trip_features(trips_clean)
feature_time = time.time() - t0
print(f"  Clean + extract time: {feature_time:.2f}s")
print(f"  Trip feature columns: {list(trip_features.columns)[:8]}")

# ---------------------------------------------------------------------------
# Train/test split by driver
# ---------------------------------------------------------------------------
driver_ids = claims_df["driver_id"].to_list()
rng = np.random.default_rng(42)
rng.shuffle(driver_ids)
n_test = int(0.3 * len(driver_ids))
test_drivers  = set(driver_ids[:n_test])
train_drivers = set(driver_ids[n_test:])

claims_train = claims_df.filter(
    claims_df["driver_id"].is_in(list(train_drivers))
)
claims_test  = claims_df.filter(
    claims_df["driver_id"].is_in(list(test_drivers))
)

trips_train = trips_clean.filter(
    trips_clean["driver_id"].is_in(list(train_drivers))
)
trips_test  = trips_clean.filter(
    trips_clean["driver_id"].is_in(list(test_drivers))
)

feat_train = trip_features.filter(
    trip_features["driver_id"].is_in(list(train_drivers))
)
feat_test  = trip_features.filter(
    trip_features["driver_id"].is_in(list(test_drivers))
)

print(f"\nTrain: {len(train_drivers)} drivers | Test: {len(test_drivers)} drivers")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def gini_coefficient(y_actual, y_predicted):
    """Normalised Gini from ranked predictions."""
    n = len(y_actual)
    idx = np.argsort(y_predicted)
    y_sorted = np.array(y_actual)[idx]
    lorenz = np.cumsum(y_sorted) / y_sorted.sum()
    gini = 1 - 2 * lorenz.mean()
    return float(gini)


def poisson_deviance(y_true, y_pred):
    y_pred_c = np.maximum(y_pred, 1e-10)
    mask = y_true > 0
    d = np.zeros_like(y_true, dtype=float)
    d[mask] = y_true[mask] * np.log(y_true[mask] / y_pred_c[mask]) - (y_true[mask] - y_pred_c[mask])
    d[~mask] = y_pred_c[~mask]
    return 2.0 * d.mean()


def ae_by_quintile(y_true, y_pred, n_quintiles=5):
    """Return A/E ratios by prediction quintile."""
    quintile_bounds = np.percentile(y_pred, np.linspace(0, 100, n_quintiles + 1))
    ae_vals = []
    for i in range(n_quintiles):
        lo, hi = quintile_bounds[i], quintile_bounds[i + 1]
        mask = (y_pred >= lo) & (y_pred <= hi) if i == n_quintiles - 1 else (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            ae_vals.append(y_true[mask].sum() / y_pred[mask].sum())
        else:
            ae_vals.append(float('nan'))
    return ae_vals


# ---------------------------------------------------------------------------
# Approach A: Baseline — raw trip feature averages in Poisson GLM
# ---------------------------------------------------------------------------
print("\n--- Approach A: Raw feature averages in Poisson GLM ---")

t0 = time.time()
# Aggregate raw features to driver level
driver_raw_train = aggregate_to_driver(feat_train)
driver_raw_test  = aggregate_to_driver(feat_test)

# Join to claims
import polars as pl
train_data_raw = claims_train.join(driver_raw_train, on="driver_id", how="inner")
test_data_raw  = claims_test.join(driver_raw_test, on="driver_id", how="inner")

raw_feat_cols = [
    "mean_speed_kmh",
    "harsh_braking_rate",
    "harsh_accel_rate",
    "night_driving_fraction",
]

# Filter to available columns
available_raw = [c for c in raw_feat_cols if c in train_data_raw.columns]

X_train_raw = train_data_raw.select(available_raw).to_numpy().astype(float)
y_train_raw = train_data_raw["claim_count"].to_numpy().astype(float)
exp_train_raw = train_data_raw["exposure"].to_numpy().astype(float)

X_test_raw = test_data_raw.select(available_raw).to_numpy().astype(float)
y_test_raw = test_data_raw["claim_count"].to_numpy().astype(float)
exp_test_raw = test_data_raw["exposure"].to_numpy().astype(float)

import statsmodels.api as sm

# Replace NaN with column means
for col_idx in range(X_train_raw.shape[1]):
    col_mean = np.nanmean(X_train_raw[:, col_idx])
    X_train_raw[np.isnan(X_train_raw[:, col_idx]), col_idx] = col_mean
    X_test_raw[np.isnan(X_test_raw[:, col_idx]), col_idx] = col_mean

X_sm_raw_train = sm.add_constant(X_train_raw)
X_sm_raw_test  = sm.add_constant(X_test_raw)

glm_raw = sm.GLM(
    y_train_raw, X_sm_raw_train,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_train_raw, 1e-10)),
).fit(disp=False)
t_raw = time.time() - t0

y_pred_raw = glm_raw.predict(X_sm_raw_test, offset=np.log(np.maximum(exp_test_raw, 1e-10)))
dev_raw = poisson_deviance(y_test_raw, y_pred_raw)
gini_raw = gini_coefficient(y_test_raw, y_pred_raw)
ae_quintiles_raw = ae_by_quintile(y_test_raw, y_pred_raw)

print(f"  Fit time: {t_raw:.2f}s")
print(f"  Test deviance: {dev_raw:.4f}")
print(f"  Gini coefficient: {gini_raw:.4f}")
print(f"  A/E by quintile: {' | '.join(f'{v:.3f}' for v in ae_quintiles_raw)}")
if len(ae_quintiles_raw) >= 5:
    top_bottom_raw = ae_quintiles_raw[-1] / ae_quintiles_raw[0] if ae_quintiles_raw[0] > 0 else float('nan')
    print(f"  Top/bottom quintile A/E ratio: {top_bottom_raw:.3f}")

# ---------------------------------------------------------------------------
# Approach B: HMM-derived state features in Poisson GLM
# ---------------------------------------------------------------------------
print("\n--- Approach B: HMM state fraction features in Poisson GLM ---")

t0 = time.time()
hmm = DrivingStateHMM(n_states=3, random_state=42)
hmm.fit(feat_train)
states_train = hmm.predict_states(feat_train)
states_test  = hmm.predict_states(feat_test)

driver_hmm_train = hmm.driver_state_features(feat_train, states_train)
driver_hmm_test  = hmm.driver_state_features(feat_test,  states_test)

train_data_hmm = claims_train.join(driver_hmm_train, on="driver_id", how="inner")
test_data_hmm  = claims_test.join(driver_hmm_test, on="driver_id", how="inner")

# State fraction columns
state_cols = [c for c in driver_hmm_train.columns if c.startswith("state_") and c != "state_assignments"]
available_states = [c for c in state_cols if c in train_data_hmm.columns]

# Use first 2 state fractions (drop one to avoid collinearity)
hmm_feat_cols = available_states[:2] if len(available_states) >= 2 else available_states

X_train_hmm = train_data_hmm.select(hmm_feat_cols).to_numpy().astype(float)
y_train_hmm = train_data_hmm["claim_count"].to_numpy().astype(float)
exp_train_hmm = train_data_hmm["exposure"].to_numpy().astype(float)

X_test_hmm = test_data_hmm.select(hmm_feat_cols).to_numpy().astype(float)
y_test_hmm = test_data_hmm["claim_count"].to_numpy().astype(float)
exp_test_hmm = test_data_hmm["exposure"].to_numpy().astype(float)

# Handle NaN
for col_idx in range(X_train_hmm.shape[1]):
    col_mean = np.nanmean(X_train_hmm[:, col_idx])
    X_train_hmm[np.isnan(X_train_hmm[:, col_idx]), col_idx] = col_mean
    X_test_hmm[np.isnan(X_test_hmm[:, col_idx]), col_idx] = col_mean

X_sm_hmm_train = sm.add_constant(X_train_hmm)
X_sm_hmm_test  = sm.add_constant(X_test_hmm)

glm_hmm = sm.GLM(
    y_train_hmm, X_sm_hmm_train,
    family=sm.families.Poisson(),
    offset=np.log(np.maximum(exp_train_hmm, 1e-10)),
).fit(disp=False)
t_hmm = time.time() - t0

y_pred_hmm = glm_hmm.predict(X_sm_hmm_test, offset=np.log(np.maximum(exp_test_hmm, 1e-10)))
dev_hmm = poisson_deviance(y_test_hmm, y_pred_hmm)
gini_hmm = gini_coefficient(y_test_hmm, y_pred_hmm)
ae_quintiles_hmm = ae_by_quintile(y_test_hmm, y_pred_hmm)

print(f"  HMM fit + state prediction time: {t_hmm:.2f}s")
print(f"  HMM features used: {hmm_feat_cols}")
print(f"  Test deviance: {dev_hmm:.4f}")
print(f"  Gini coefficient: {gini_hmm:.4f}")
print(f"  A/E by quintile: {' | '.join(f'{v:.3f}' for v in ae_quintiles_hmm)}")
if len(ae_quintiles_hmm) >= 5:
    top_bottom_hmm = ae_quintiles_hmm[-1] / ae_quintiles_hmm[0] if ae_quintiles_hmm[0] > 0 else float('nan')
    print(f"  Top/bottom quintile A/E ratio: {top_bottom_hmm:.3f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Method':<30} {'Deviance':>10} {'Gini':>8} {'Top/Bot A/E':>12}")
print("-" * 62)

top_bottom_raw = ae_quintiles_raw[-1] / ae_quintiles_raw[0] if ae_quintiles_raw[0] > 0 else float('nan')
top_bottom_hmm = ae_quintiles_hmm[-1] / ae_quintiles_hmm[0] if ae_quintiles_hmm[0] > 0 else float('nan')

print(f"{'Raw averages (baseline)':<30} {dev_raw:>10.4f} {gini_raw:>8.4f} {top_bottom_raw:>12.3f}")
print(f"{'HMM state fractions (library)':<30} {dev_hmm:>10.4f} {gini_hmm:>8.4f} {top_bottom_hmm:>12.3f}")

gini_lift = (gini_hmm - gini_raw)
print(f"\nGini improvement (HMM - raw): {gini_lift:+.4f} ({gini_lift / abs(gini_raw) * 100:+.1f}%)")
print(f"Deviance change (HMM - raw): {dev_hmm - dev_raw:+.4f}")
print(f"\nTimings: simulation={sim_time:.1f}s  features={feature_time:.1f}s  "
      f"raw GLM={t_raw:.1f}s  HMM+GLM={t_hmm:.1f}s")
