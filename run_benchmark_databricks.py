"""
Run the benchmark notebook on Databricks and capture the Gini numbers.
Execute this locally: python run_benchmark_databricks.py
"""
import os
import sys
import time
import json
import base64

DATABRICKS_HOST = "https://dbc-150a27f5-e1e7.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

# Inline benchmark code as a single notebook
BENCHMARK_CODE = r'''
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Install deps ──────────────────────────────────────────────────────────────
import subprocess
subprocess.run(
    ["pip", "install", "-q",
     "git+https://github.com/burning-cost/insurance-telematics.git",
     "hmmlearn", "statsmodels", "polars", "scikit-learn"],
    check=False
)

from insurance_telematics import (
    TripSimulator,
    TelematicsScoringPipeline,
    DrivingStateHMM,
    clean_trips,
    extract_trip_features,
    aggregate_to_driver,
)

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")

# ── Simulate ──────────────────────────────────────────────────────────────────
N_DRIVERS = 300
TRIPS_PER_DRIVER = 40
RANDOM_STATE = 42

t0_sim = time.perf_counter()
sim = TripSimulator(seed=RANDOM_STATE)
trips_df, claims_df = sim.simulate(n_drivers=N_DRIVERS, trips_per_driver=TRIPS_PER_DRIVER)
sim_time = time.perf_counter() - t0_sim
print(f"Simulation time: {sim_time:.1f}s")

# ── Split ─────────────────────────────────────────────────────────────────────
driver_ids = claims_df["driver_id"].unique().sort().to_list()
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(len(driver_ids))
n_train = int(len(driver_ids) * 0.70)
train_ids = set([driver_ids[i] for i in perm[:n_train]])
test_ids  = set([driver_ids[i] for i in perm[n_train:]])

train_trips = trips_df.filter(pl.col("driver_id").is_in(list(train_ids)))
test_trips  = trips_df.filter(pl.col("driver_id").is_in(list(test_ids)))
train_claims = claims_df.filter(pl.col("driver_id").is_in(list(train_ids)))
test_claims  = claims_df.filter(pl.col("driver_id").is_in(list(test_ids)))

print(f"Train: {len(train_ids)} drivers | Test: {len(test_ids)} drivers")

# ── Baseline ──────────────────────────────────────────────────────────────────
t0_baseline = time.perf_counter()
train_clean = clean_trips(train_trips)
train_feat  = extract_trip_features(train_clean)
test_clean  = clean_trips(test_trips)
test_feat   = extract_trip_features(test_clean)

RAW_FEATURES = ["mean_speed_kmh", "harsh_braking_rate", "harsh_accel_rate", "night_fraction", "distance_km"]

def driver_raw_averages(feat_df, feature_cols):
    present = [c for c in feature_cols if c in feat_df.columns]
    return feat_df.group_by("driver_id").agg([pl.col(c).mean().alias(c) for c in present])

train_raw = driver_raw_averages(train_feat, RAW_FEATURES)
test_raw  = driver_raw_averages(test_feat,  RAW_FEATURES)

train_glm_df = train_raw.join(train_claims, on="driver_id", how="inner").to_pandas()
test_glm_df  = test_raw.join(test_claims,  on="driver_id", how="inner").to_pandas()

feature_cols_baseline = [c for c in RAW_FEATURES if c in train_glm_df.columns]
X_train_b = sm.add_constant(train_glm_df[feature_cols_baseline].fillna(0), has_constant="add")
X_test_b  = sm.add_constant(test_glm_df[feature_cols_baseline].fillna(0),  has_constant="add")
y_train   = train_glm_df["n_claims"].values.astype(float)
y_test    = test_glm_df["n_claims"].values.astype(float)
exp_train = train_glm_df["exposure_years"].values
exp_test  = test_glm_df["exposure_years"].values

glm_baseline = sm.GLM(
    y_train, X_train_b,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(np.clip(exp_train, 1e-6, None)),
).fit(disp=False)
baseline_fit_time = time.perf_counter() - t0_baseline
pred_baseline = glm_baseline.predict(X_test_b)
print(f"Baseline fit time: {baseline_fit_time:.2f}s")

# ── Library (HMM) ─────────────────────────────────────────────────────────────
t0_library = time.perf_counter()
pipe = TelematicsScoringPipeline(n_hmm_states=3, credibility_threshold=20, random_state=RANDOM_STATE)
pipe.fit(train_trips, train_claims)
library_fit_time = time.perf_counter() - t0_library
print(f"Library fit time: {library_fit_time:.2f}s")

pred_library_df = pipe.predict(test_trips)
test_pd = test_claims.to_pandas()
pred_merged = test_pd.merge(pred_library_df.to_pandas(), on="driver_id", how="inner")

pred_library = pred_merged["predicted_claim_frequency"].values * pred_merged["exposure_years"].values
y_test_lib   = pred_merged["n_claims"].values.astype(float)
exp_test_lib = pred_merged["exposure_years"].values

common_ids = pred_merged["driver_id"].values
baseline_merged = test_glm_df.set_index("driver_id").reindex(pred_merged["driver_id"].values)
pred_baseline_aligned = glm_baseline.predict(
    sm.add_constant(baseline_merged[feature_cols_baseline].fillna(0), has_constant="add")
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def poisson_deviance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-10)
    d = 2 * (y_true * np.log(np.where(y_true > 0, y_true / y_pred, 1.0)) - (y_true - y_pred))
    return float(d.mean())

def gini_coefficient(y_true, y_pred, weight=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weight is None:
        weight = np.ones_like(y_true)
    w = np.asarray(weight, dtype=float)
    order = np.argsort(y_pred)
    ys = y_true[order]; ws = w[order]
    cum_w = np.cumsum(ws) / ws.sum()
    cum_y = np.cumsum(ys * ws) / (ys * ws).sum()
    return 2 * float(np.trapz(cum_y, cum_w)) - 1

def ae_by_quintile(y_true, y_pred, weight=None, n=5):
    if weight is None: weight = np.ones_like(y_true)
    cuts = pd.qcut(y_pred, n, labels=False, duplicates="drop")
    rows = []
    for q in range(n):
        mask = cuts == q
        if mask.sum() == 0: continue
        actual = y_true[mask].sum(); expected = y_pred[mask].sum()
        rows.append({"quintile": q+1, "n_drivers": int(mask.sum()),
                     "mean_pred": float(y_pred[mask].mean()),
                     "actual": float(actual), "expected": float(expected),
                     "ae_ratio": float(actual/expected) if expected > 0 else float("nan")})
    return pd.DataFrame(rows)

def loss_ratio_sep(y_true, y_pred, weight, n=5):
    cuts = pd.qcut(y_pred, n, labels=False, duplicates="drop")
    lrs = [y_true[cuts==q].sum() / max(weight[cuts==q].sum(), 1e-6) for q in range(n) if (cuts==q).sum() > 0]
    lrs = np.array(lrs)
    return float(lrs[-1] / lrs[0]) if lrs[0] > 0 else float("nan")

dev_baseline = poisson_deviance(y_test_lib, pred_baseline_aligned)
dev_library  = poisson_deviance(y_test_lib, pred_library)
gini_baseline = gini_coefficient(y_test_lib, pred_baseline_aligned, weight=exp_test_lib)
gini_library  = gini_coefficient(y_test_lib, pred_library,          weight=exp_test_lib)
ae_baseline = ae_by_quintile(y_test_lib, pred_baseline_aligned, weight=exp_test_lib)
ae_library  = ae_by_quintile(y_test_lib, pred_library,          weight=exp_test_lib)
lr_sep_b = loss_ratio_sep(y_test_lib, pred_baseline_aligned, exp_test_lib)
lr_sep_l = loss_ratio_sep(y_test_lib, pred_library,          exp_test_lib)
max_ae_dev_b = (ae_baseline["ae_ratio"] - 1.0).abs().max()
max_ae_dev_l = (ae_library["ae_ratio"]  - 1.0).abs().max()

print("\n" + "="*70)
print("BENCHMARK RESULTS — insurance-telematics HMM vs Raw Trip Feature GLM")
print("="*70)
print(f"  Gini coefficient — Baseline: {gini_baseline:.4f}  |  HMM Library: {gini_library:.4f}  |  Delta: {gini_library - gini_baseline:+.4f}")
print(f"  Poisson deviance — Baseline: {dev_baseline:.4f}  |  HMM Library: {dev_library:.4f}  |  Delta: {dev_library - dev_baseline:+.4f}")
print(f"  Loss ratio sep.  — Baseline: {lr_sep_b:.3f}x  |  HMM Library: {lr_sep_l:.3f}x")
print(f"  Max A/E deviation — Baseline: {max_ae_dev_b:.4f}  |  HMM Library: {max_ae_dev_l:.4f}")
print(f"  Fit time          — Baseline: {baseline_fit_time:.1f}s  |  HMM Library: {library_fit_time:.1f}s")
print()
print("A/E by quintile — BASELINE:")
print(ae_baseline[["quintile", "n_drivers", "mean_pred", "ae_ratio"]].to_string(index=False))
print()
print("A/E by quintile — HMM LIBRARY:")
print(ae_library[["quintile", "n_drivers", "mean_pred", "ae_ratio"]].to_string(index=False))
print()
print(f"BENCHMARK_GINI_BASELINE={gini_baseline:.4f}")
print(f"BENCHMARK_GINI_LIBRARY={gini_library:.4f}")
print(f"BENCHMARK_GINI_DELTA={gini_library - gini_baseline:.4f}")
print(f"BENCHMARK_DEV_BASELINE={dev_baseline:.4f}")
print(f"BENCHMARK_DEV_LIBRARY={dev_library:.4f}")
print(f"BENCHMARK_LRSEP_BASELINE={lr_sep_b:.3f}")
print(f"BENCHMARK_LRSEP_LIBRARY={lr_sep_l:.3f}")
print(f"BENCHMARK_AE_BASELINE={max_ae_dev_b:.4f}")
print(f"BENCHMARK_AE_LIBRARY={max_ae_dev_l:.4f}")
print(f"BENCHMARK_FIT_BASELINE={baseline_fit_time:.1f}")
print(f"BENCHMARK_FIT_LIBRARY={library_fit_time:.1f}")
'''

if __name__ == "__main__":
    print(BENCHMARK_CODE[:200])
    print("Script ready.")
