# Databricks notebook source

# MAGIC %md
# MAGIC # insurance-telematics: Raw Trip Data to GLM-Ready Risk Scores
# MAGIC
# MAGIC This notebook runs the full pipeline end-to-end: simulate a synthetic fleet, classify driving behaviour with a Hidden Markov Model, aggregate to driver-level risk scores, and produce a feature DataFrame you can drop into your Poisson frequency GLM.
# MAGIC
# MAGIC No raw telematics data required — `TripSimulator` generates a realistic synthetic fleet of cautious, normal, and aggressive drivers with Poisson claims.

# COMMAND ----------

# MAGIC %pip install insurance-telematics polars numpy --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from __future__ import annotations

import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Simulate a Synthetic Fleet
# MAGIC
# MAGIC `TripSimulator` generates realistic 1Hz trip data for a mixed fleet. Three driving types (cautious / normal / aggressive) drive Ornstein-Uhlenbeck speed processes with different drift parameters. Claims are Poisson with frequency proportional to time in the aggressive state.

# COMMAND ----------

from insurance_telematics import TripSimulator

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(
    n_drivers=200,
    trips_per_driver=30,
)

print(f"Trips:   {len(trips_df):,} rows")
print(f"Drivers: {trips_df['driver_id'].n_unique()} unique")
print(f"Claims:  {len(claims_df):,} claim events")
print(f"\nTrip columns: {trips_df.columns}")
trips_df.head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Run the Full Scoring Pipeline
# MAGIC
# MAGIC `TelematicsScoringPipeline` chains four stages:
# MAGIC 1. Clean trips (GPS jump removal, acceleration derivation)
# MAGIC 2. Extract trip features (harsh braking rate, speeding fraction, night fraction)
# MAGIC 3. Fit a 3-state HMM on the feature vectors (cautious / normal / aggressive latent states)
# MAGIC 4. Aggregate to driver level using Bühlmann-Straub credibility weighting
# MAGIC
# MAGIC The output is a driver-level DataFrame ready for a Poisson GLM.

# COMMAND ----------

from insurance_telematics import TelematicsScoringPipeline

pipe = TelematicsScoringPipeline(n_hmm_states=3)
pipe.fit(trips_df, claims_df)

driver_scores = pipe.predict(trips_df)

print(f"Driver scores shape: {driver_scores.shape}")
print(f"Score columns: {driver_scores.columns}")
driver_scores.head(8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Validate — Aggressive Drivers Should Have Higher Risk Scores
# MAGIC
# MAGIC The simulator tags each driver with their true driving type. If the pipeline is working, aggressive drivers (driving_type=2) should have materially higher `telematics_risk_score` than cautious drivers (driving_type=0).

# COMMAND ----------

driver_types = sim.driver_metadata()

validation = (
    driver_scores
    .join(driver_types, on="driver_id", how="left")
    .group_by("driving_type")
    .agg([
        pl.col("telematics_risk_score").mean().alias("mean_risk_score"),
        pl.col("telematics_risk_score").std().alias("std_risk_score"),
        pl.len().alias("n_drivers"),
    ])
    .sort("driving_type")
)

type_labels = {0: "cautious", 1: "normal", 2: "aggressive"}
validation = validation.with_columns(
    pl.col("driving_type").replace(type_labels).alias("driver_type")
)
print(validation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Step-by-Step Pipeline
# MAGIC
# MAGIC `TelematicsScoringPipeline` is a convenience wrapper. You can also run each stage independently for custom workflows.

# COMMAND ----------

from insurance_telematics import (
    clean_trips,
    extract_trip_features,
    DrivingStateHMM,
    aggregate_to_driver,
)

# Stage 1: clean
trips_clean = clean_trips(trips_df)

# Stage 2: trip-level features
trip_features = extract_trip_features(trips_clean)
print(f"Trip features: {trip_features.columns}")

# Stage 3: HMM state classification
hmm = DrivingStateHMM(n_states=3)
hmm.fit(trip_features)
trip_features = hmm.predict_states(trip_features)
print(f"State distribution:\n{trip_features['hmm_state'].value_counts().sort('hmm_state')}")

# Stage 4: aggregate to driver level
driver_features = aggregate_to_driver(trip_features)
print(f"\nDriver-level features shape: {driver_features.shape}")
driver_features.head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What You Should See
# MAGIC
# MAGIC - Aggressive drivers should have a risk score 1.5–3x higher than cautious drivers.
# MAGIC - The pipeline produces interpretable features (harsh_braking_rate, speeding_frac, pct_time_aggressive_state) that pricing actuaries can challenge.
# MAGIC - The step-by-step version shows the same result — the pipeline wrapper just chains them.
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC - **`ContinuousTimeHMM`** — continuous-time Markov chain for variable-length trip intervals
# MAGIC - **`load_trips()`** — loader for CSV or Parquet files from real telematics devices
# MAGIC - Integration with `insurance-credibility` for Bühlmann-Straub driver-level weighting
# MAGIC
# MAGIC **GitHub:** https://github.com/burning-cost/insurance-telematics
# MAGIC **PyPI:** https://pypi.org/project/insurance-telematics/
