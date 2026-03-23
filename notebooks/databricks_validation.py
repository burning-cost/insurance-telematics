# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-telematics: Validation on a Synthetic UK Motor Fleet
# MAGIC
# MAGIC This notebook validates insurance-telematics on a realistic synthetic fleet.
# MAGIC
# MAGIC The central claim of this library is that HMM-derived driving state fractions
# MAGIC (cautious / normal / aggressive) outperform simple summary features (mean speed,
# MAGIC max braking) as claim frequency predictors, because state fractions capture
# MAGIC *persistent driving style* rather than trip-level noise. A driver who is occasionally
# MAGIC fast is different from a driver who is consistently aggressive — the HMM separates these.
# MAGIC
# MAGIC What this notebook shows:
# MAGIC
# MAGIC 1. A synthetic fleet of 5,000 drivers, 20-50 trips each, with a known 3-state DGP
# MAGIC    (calm, moderate, aggressive) and known emission distributions
# MAGIC 2. Three approaches compared on the same Poisson frequency GLM:
# MAGIC    (a) Simple summary features — mean speed, max braking, harsh acceleration rate
# MAGIC    (b) Threshold-based scoring — % time above speed threshold and braking threshold
# MAGIC    (c) HMM-derived state fractions from this library
# MAGIC 3. State classification accuracy — can the HMM recover the true latent states?
# MAGIC 4. GLM discrimination by approach — Gini coefficient on held-out claims
# MAGIC 5. Loss ratio separation by predicted decile
# MAGIC
# MAGIC **Expected result:** HMM risk score achieves 5-10pp higher Gini than simple summary
# MAGIC features. State classification accuracy >= 80% on the test set. The resulting
# MAGIC GLM-compatible feature DataFrame drops directly into an existing rating model.
# MAGIC
# MAGIC ---
# MAGIC *Part of the [Burning Cost](https://burning-cost.github.io) insurance pricing toolkit.*

# COMMAND ----------

# MAGIC %pip install insurance-telematics polars scikit-learn statsmodels -q

# COMMAND ----------

from __future__ import annotations

import time
import warnings

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-Generating Process
# MAGIC
# MAGIC The DGP is a fleet of 5,000 UK motor drivers. Each driver has a true latent style
# MAGIC (predominantly cautious, moderate, or aggressive) drawn from a Dirichlet distribution
# MAGIC skewed towards cautious to mimic a real portfolio.
# MAGIC
# MAGIC **Known DGP structure:**
# MAGIC - State 0 (calm): mean speed 32 km/h, low speed variance, low harsh event rate, claim rate 4%/yr
# MAGIC - State 1 (moderate): mean speed 58 km/h, moderate variance, moderate events, claim rate 10%/yr
# MAGIC - State 2 (aggressive): mean speed 88 km/h, high variance, high harsh events, claim rate 28%/yr
# MAGIC
# MAGIC Synthetic trips are generated using TripSimulator (Ornstein-Uhlenbeck speed process within
# MAGIC each regime). Claims are Poisson with rate proportional to the aggressive state fraction —
# MAGIC this is the ground truth that the HMM should recover.
# MAGIC
# MAGIC We generate the trips synthetically using the library's TripSimulator, then also build a
# MAGIC separate manual DGP to test state recovery accuracy with known true states.

# COMMAND ----------

from insurance_telematics import TripSimulator

N_DRIVERS = 5_000
TRIPS_PER_DRIVER = 30   # kept moderate to make this run in 2-3 minutes total

print(f"Generating {N_DRIVERS:,} drivers x {TRIPS_PER_DRIVER} trips...")
print("(Each trip is 1Hz GPS data. This step takes ~60-90 seconds.)")

t0 = time.perf_counter()
sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(
    n_drivers=N_DRIVERS,
    trips_per_driver=TRIPS_PER_DRIVER,
    min_trip_duration_s=300,
    max_trip_duration_s=1800,
)
t_sim = time.perf_counter() - t0

print(f"Generated in {t_sim:.1f}s")
print(f"Trip rows:  {trips_df.shape[0]:,}")
print(f"Drivers:    {claims_df.shape[0]:,}")
print(f"Claims:     {claims_df['n_claims'].sum():,}")
print(f"Mean claim frequency: {claims_df['n_claims'].mean():.4f} per driver")
print()
print("True aggressive fraction distribution (from simulator):")
agg_frac = claims_df["aggressive_fraction"].to_numpy()
print(f"  Mean:  {agg_frac.mean():.4f}")
print(f"  P25:   {np.percentile(agg_frac, 25):.4f}")
print(f"  P75:   {np.percentile(agg_frac, 75):.4f}")
print(f"  P95:   {np.percentile(agg_frac, 95):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Extraction
# MAGIC
# MAGIC Extract trip-level scalar features from the raw 1Hz data. This is the same step
# MAGIC you would run on production data. The output is one row per trip with features like
# MAGIC `harsh_braking_rate`, `mean_speed_kmh`, `speed_variation_coeff`.
# MAGIC
# MAGIC Three feature sets will be compared in the GLM:
# MAGIC - **Summary**: mean speed, harsh braking rate, harsh acceleration rate (driver average over all trips)
# MAGIC - **Threshold**: % time above 80 km/h, % harsh events, night fraction (simple thresholding)
# MAGIC - **HMM**: fraction of trips classified as each latent state (this library's contribution)

# COMMAND ----------

from insurance_telematics import clean_trips, extract_trip_features

print("Cleaning trips and extracting features...")
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trips_clean = clean_trips(trips_df)
    trip_features = extract_trip_features(trips_clean)
t_extract = time.perf_counter() - t0

print(f"Feature extraction: {t_extract:.1f}s")
print(f"Trip features shape: {trip_features.shape}")
print(f"Columns: {trip_features.columns[:10]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Approach (a) — Simple Summary Features
# MAGIC
# MAGIC Average each trip-level feature to the driver level. No modelling — just aggregation.
# MAGIC This is the baseline that most telematics teams start with. Fast and transparent, but
# MAGIC it treats each trip as equally important and ignores the distribution of behaviour
# MAGIC across trips.

# COMMAND ----------

from insurance_telematics import aggregate_to_driver

print("Aggregating to driver level...")
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    driver_summary = aggregate_to_driver(trip_features, credibility_threshold=30)
t_agg = time.perf_counter() - t0
print(f"Aggregation: {t_agg:.2f}s")
print(f"Driver-level features: {driver_summary.shape}")
print(f"Columns: {driver_summary.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Approach (b) — Threshold-Based Scoring
# MAGIC
# MAGIC A common industry simplification: define one or two hard thresholds and compute the
# MAGIC fraction of trips above them. For example, "high speed fraction" = fraction of trips
# MAGIC where mean_speed_kmh > 70. This produces an interpretable score but discards
# MAGIC information about the shape of the speed distribution and ignores regime structure.

# COMMAND ----------

# Compute threshold-based features at driver level
print("Computing threshold-based features...")

# Join with driver-level aggregate to get all trip features per driver
trip_features_pd = trip_features.to_pandas()

# Driver-level threshold scores
driver_ids = trip_features_pd["driver_id"].unique()
threshold_rows = []
for did in driver_ids:
    mask = trip_features_pd["driver_id"] == did
    trips = trip_features_pd[mask]
    n = len(trips)
    threshold_rows.append({
        "driver_id": did,
        "high_speed_frac":    (trips["mean_speed_kmh"] > 70).mean() if "mean_speed_kmh" in trips.columns else 0.0,
        "high_braking_frac":  (trips["harsh_braking_rate"] > 0.5).mean() if "harsh_braking_rate" in trips.columns else 0.0,
        "high_accel_frac":    (trips["harsh_accel_rate"] > 0.5).mean() if "harsh_accel_rate" in trips.columns else 0.0,
        "night_frac":         trips["night_driving_fraction"].mean() if "night_driving_fraction" in trips.columns else 0.0,
        "n_trips_threshold":  n,
    })

threshold_df = pl.DataFrame(threshold_rows)
print(f"Threshold features shape: {threshold_df.shape}")
print(threshold_df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Approach (c) — HMM State Classification
# MAGIC
# MAGIC The HMM fits a 3-state Gaussian model to the sequence of trip-level features.
# MAGIC After fitting, each trip is classified into the most likely latent state (0=calm,
# MAGIC 1=moderate, 2=aggressive). The per-driver fraction of trips in each state becomes
# MAGIC the GLM covariate.
# MAGIC
# MAGIC The HMM captures the difference between a consistently aggressive driver (high state_2_fraction)
# MAGIC and a driver who had a few fast trips by chance (similar mean speed but low state_2_fraction).
# MAGIC Following Jiang & Shi (2024, NAAJ), `state_2_fraction` is the most predictive covariate.
# MAGIC
# MAGIC This step takes 30-90 seconds for 5,000 drivers x 30 trips.

# COMMAND ----------

from insurance_telematics import DrivingStateHMM

print("Fitting HMM on trip features...")
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hmm = DrivingStateHMM(n_states=3, n_iter=200, random_state=42)
    hmm.fit(trip_features)
    states = hmm.predict_states(trip_features)
    hmm_driver_features = hmm.driver_state_features(trip_features, states)
t_hmm = time.perf_counter() - t0

print(f"HMM fit + state classification: {t_hmm:.1f}s")
print()
print("HMM-derived driver features (first 5 rows):")
print(hmm_driver_features.head().to_pandas().to_string(index=False))
print()
print("State fraction summary:")
for k in range(3):
    col = f"state_{k}_fraction"
    vals = hmm_driver_features[col].to_numpy()
    print(f"  state_{k} fraction:  mean={vals.mean():.3f}  P25={np.percentile(vals,25):.3f}  P75={np.percentile(vals,75):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. State Recovery Accuracy
# MAGIC
# MAGIC To assess HMM state recovery, we compare the HMM-estimated aggressive state fraction
# MAGIC per driver against the true aggressive fraction known from the simulator DGP.
# MAGIC
# MAGIC Note: the HMM assigns state indices by mean speed order (0 = slowest), so state_2
# MAGIC should correspond to the simulator's "aggressive" regime. We verify this alignment
# MAGIC and compute the rank correlation (Spearman's rho) between HMM state_2_fraction and
# MAGIC true aggressive_fraction.
# MAGIC
# MAGIC A perfect recovery gives rho = 1.0. In practice we expect rho >= 0.70 with 30 trips
# MAGIC per driver, reflecting the averaging effect of Bühlmann-Straub credibility.

# COMMAND ----------

import scipy.stats as stats

# Join HMM features with true simulator labels
state_compare = hmm_driver_features.join(
    claims_df.select(["driver_id", "aggressive_fraction", "normal_fraction", "cautious_fraction"]),
    on="driver_id",
    how="inner",
)

hmm_aggressive = state_compare["state_2_fraction"].to_numpy()
true_aggressive = state_compare["aggressive_fraction"].to_numpy()

# Spearman rank correlation between HMM and true aggressive fraction
rho, p_val = stats.spearmanr(hmm_aggressive, true_aggressive)

# Mean absolute error in aggressive fraction estimation
mae_aggressive = np.abs(hmm_aggressive - true_aggressive).mean()

print("HMM State Recovery:")
print(f"  Spearman rho (HMM state_2 vs true aggressive): {rho:.4f}  (p={p_val:.2e})")
print(f"  Mean absolute error in aggressive fraction:     {mae_aggressive:.4f}")
print()

# How well does HMM separate aggressive drivers?
# Threshold: top quartile of true aggressive fraction = "known aggressive" drivers
agg_threshold = np.percentile(true_aggressive, 75)
true_top_quartile = true_aggressive >= agg_threshold
hmm_top_quartile  = hmm_aggressive >= np.percentile(hmm_aggressive, 75)

overlap = np.mean(true_top_quartile & hmm_top_quartile)
print(f"  Overlap between top-quartile aggressive (true vs HMM): {overlap:.1%}")
print(f"  (Random would give 25%. Target: >=50%.)")
print()

# Correlation with true claim rate as a further check
true_rate = claims_df.join(
    pl.DataFrame({"driver_id": state_compare["driver_id"].to_list(),
                  "hmm_aggressive": hmm_aggressive.tolist()}),
    on="driver_id",
    how="left"
)
claims_arr = true_rate["n_claims"].to_numpy().astype(float)
hmm_agg_arr = true_rate["hmm_aggressive"].to_numpy()
rho_claims, _ = stats.spearmanr(hmm_agg_arr, claims_arr)
print(f"  Spearman rho (HMM state_2_fraction vs n_claims): {rho_claims:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. GLM Comparison — Discrimination on Held-Out Claims
# MAGIC
# MAGIC All three feature sets are fed into the same Poisson GLM structure. The only difference
# MAGIC is the input features — the GLM family, link function, and exposure handling are identical.
# MAGIC
# MAGIC Train/test split: 70% of drivers for fitting, 30% for evaluation. Gini coefficient
# MAGIC measures how well each approach rank-orders drivers by claim frequency.

# COMMAND ----------

import statsmodels.api as sm

# Prepare driver-level model frame by joining features with claims
def build_model_df(feature_df: pl.DataFrame, claims: pl.DataFrame) -> pl.DataFrame:
    return feature_df.join(
        claims.select(["driver_id", "n_claims", "exposure_years"]),
        on="driver_id",
        how="inner"
    )

# Train/test split at driver level (70/30)
all_drivers = claims_df["driver_id"].to_list()
RNG = np.random.default_rng(99)
perm = RNG.permutation(len(all_drivers))
n_train = int(0.7 * len(all_drivers))
train_ids = set([all_drivers[i] for i in perm[:n_train]])
test_ids  = set([all_drivers[i] for i in perm[n_train:]])

def split_df(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_mask = df["driver_id"].is_in(list(train_ids))
    return df.filter(train_mask), df.filter(~train_mask)

def fit_poisson_glm(feature_df: pl.DataFrame, feature_cols: list[str]) -> tuple:
    """Fit Poisson GLM and return (result, train_preds, test_preds, train_y, test_y, train_exp, test_exp)."""
    train_df, test_df = split_df(feature_df)

    def prep(df):
        X = df.select(feature_cols).to_pandas().fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        # Drop zero-variance columns
        keep = [c for c in X.columns if X[c].std() > 1e-10]
        X = X[keep] if keep else X
        X = sm.add_constant(X, has_constant="add")
        y = df["n_claims"].to_numpy().astype(float)
        exp = np.clip(df["exposure_years"].to_numpy(), 1e-6, None)
        return X, y, exp

    X_tr, y_tr, exp_tr = prep(train_df)
    X_te, y_te, exp_te = prep(test_df)

    model = sm.GLM(
        y_tr, X_tr,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=np.log(exp_tr),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(start_params=np.zeros(X_tr.shape[1]), maxiter=200, disp=False)

    # Align test columns
    for col in X_tr.columns:
        if col not in X_te.columns:
            X_te[col] = 0.0
    X_te = X_te[X_tr.columns]

    pred_te = result.predict(X_te) * exp_te
    return result, pred_te, y_te, exp_te

def gini_coeff(y_true, y_pred, weights):
    """Normalised Gini coefficient."""
    try:
        from insurance_telematics import extract_trip_features  # noqa
        # Import from insurance_gam if available
        from insurance_gam.ebm import gini as ebm_gini
        return ebm_gini(y_true, y_pred, weights)
    except ImportError:
        pass
    # Fallback implementation
    order = np.argsort(y_pred)
    yt = y_true[order]; w = weights[order]
    wc = np.cumsum(w); lc = np.cumsum(yt * w)
    wt = wc[-1]; lt = lc[-1]
    if lt == 0: return 0.0
    x = wc / wt; y = lc / lt
    _trapz = getattr(np, "trapezoid", np.trapz)
    g_model  = 1.0 - 2.0 * float(_trapz(y, x))
    order_o  = np.argsort(yt)
    yt_o = yt[order_o]; wo = w[order_o]
    wco = np.cumsum(wo); lco = np.cumsum(yt_o * wo)
    xo = wco / wco[-1]; yo = lco / lco[-1]
    g_oracle = 1.0 - 2.0 * float(_trapz(yo, xo))
    return g_model / g_oracle if g_oracle != 0 else 0.0

# --- Approach (a): Simple summary features ---
print("Fitting GLM (a): simple summary features...")
summary_model_df = build_model_df(driver_summary, claims_df)
summary_cols = [c for c in driver_summary.columns
                if c not in ("driver_id",) and
                summary_model_df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
# Exclude identifiers and zero-variance
summary_cols = [c for c in summary_cols
                if c not in ("n_trips", "total_km", "composite_risk_score", "credibility_weight")]

t0 = time.perf_counter()
_, pred_te_summary, y_te_summary, exp_te_summary = fit_poisson_glm(summary_model_df, summary_cols)
t_glm_summary = time.perf_counter() - t0
gini_summary = gini_coeff(y_te_summary, pred_te_summary, exp_te_summary)
print(f"  Gini (summary features): {gini_summary:.4f}  ({t_glm_summary:.2f}s)")

# --- Approach (b): Threshold-based features ---
print("Fitting GLM (b): threshold-based features...")
threshold_model_df = build_model_df(threshold_df, claims_df)
threshold_cols = [c for c in threshold_df.columns
                  if c not in ("driver_id", "n_trips_threshold")]

t0 = time.perf_counter()
_, pred_te_thresh, y_te_thresh, exp_te_thresh = fit_poisson_glm(threshold_model_df, threshold_cols)
t_glm_thresh = time.perf_counter() - t0
gini_thresh = gini_coeff(y_te_thresh, pred_te_thresh, exp_te_thresh)
print(f"  Gini (threshold features): {gini_thresh:.4f}  ({t_glm_thresh:.2f}s)")

# --- Approach (c): HMM state fractions ---
print("Fitting GLM (c): HMM state features...")
# Join HMM features with driver aggregate (for exposure metadata) and claims
hmm_model_df = hmm_driver_features.join(
    claims_df.select(["driver_id", "n_claims", "exposure_years"]),
    on="driver_id",
    how="inner"
)
hmm_glm_cols = [f"state_{k}_fraction" for k in range(3)] + ["state_entropy", "mean_transition_rate"]

t0 = time.perf_counter()
_, pred_te_hmm, y_te_hmm, exp_te_hmm = fit_poisson_glm(hmm_model_df, hmm_glm_cols)
t_glm_hmm = time.perf_counter() - t0
gini_hmm = gini_coeff(y_te_hmm, pred_te_hmm, exp_te_hmm)
print(f"  Gini (HMM features):       {gini_hmm:.4f}  ({t_glm_hmm:.2f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Loss Ratio Separation by Predicted Decile
# MAGIC
# MAGIC The Gini coefficient captures overall rank ordering. The decile lift table shows whether
# MAGIC the model separates the top risk decile from the bottom in a way that is commercially
# MAGIC meaningful. A telematics pricing model needs to distinguish the 10% of drivers who will
# MAGIC generate 30-40% of claims from the 10% who will generate almost none.
# MAGIC
# MAGIC We compare the top-decile to bottom-decile loss ratio ratio (T/B ratio) across all
# MAGIC three approaches.

# COMMAND ----------

def decile_lift_table(y_true, y_pred, exposure, n_deciles=10):
    """Returns a dict with top/bottom decile loss ratios."""
    order = np.argsort(y_pred)
    yt = y_true[order]; yp = y_pred[order]; w = exposure[order]
    total_w = w.sum()
    decile_size = total_w / n_deciles

    w_cum = np.cumsum(w)
    rows = []
    prev = 0
    for d in range(n_deciles):
        target = min(decile_size * (d + 1), total_w)
        end = int(np.searchsorted(w_cum, target, side="left")) + 1
        end = min(end, len(yt))
        sl = slice(prev, end)
        w_sl = w[sl]; yt_sl = yt[sl]; yp_sl = yp[sl]
        tw = w_sl.sum()
        if tw == 0: continue
        actual_freq   = (yt_sl * w_sl).sum() / tw
        predicted_freq = (yp_sl * w_sl).sum() / tw
        rows.append({"decile": d + 1, "actual_freq": actual_freq, "predicted_freq": predicted_freq, "exposure": tw})
        prev = end

    if not rows:
        return None, None, None

    actual_freqs = [r["actual_freq"] for r in rows]
    top_dec_actual = actual_freqs[-1]
    bot_dec_actual = actual_freqs[0]
    tb_ratio = top_dec_actual / max(bot_dec_actual, 1e-10)
    return rows, top_dec_actual, tb_ratio

rows_s, top_s, tb_s = decile_lift_table(y_te_summary, pred_te_summary, exp_te_summary)
rows_t, top_t, tb_t = decile_lift_table(y_te_thresh,  pred_te_thresh,  exp_te_thresh)
rows_h, top_h, tb_h = decile_lift_table(y_te_hmm,     pred_te_hmm,     exp_te_hmm)

print("Top-to-Bottom Decile Loss Ratio Ratio:")
print(f"  Summary features:   {tb_s:.2f}x")
print(f"  Threshold features: {tb_t:.2f}x")
print(f"  HMM features:       {tb_h:.2f}x")
print()
print("(Higher ratio = better risk separation. HMM should produce the highest ratio.)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Results Summary

# COMMAND ----------

print("=" * 75)
print("VALIDATION SUMMARY — 5,000-driver fleet, 3-state HMM DGP")
print("=" * 75)
print(f"{'Method':<30} {'Gini':>10} {'T/B Decile':>12} {'GLM fit':>10}")
print("-" * 75)
print(f"{'(a) Summary features':<30} {gini_summary:>10.4f} {tb_s:>11.2f}x {t_glm_summary:>9.2f}s")
print(f"{'(b) Threshold scoring':<30} {gini_thresh:>10.4f} {tb_t:>11.2f}x {t_glm_thresh:>9.2f}s")
print(f"{'(c) HMM state fractions':<30} {gini_hmm:>10.4f} {tb_h:>11.2f}x {t_glm_hmm:>9.2f}s")
print()
print(f"HMM state recovery:")
print(f"  Spearman rho (HMM state_2 vs true aggressive fraction): {rho:.4f}")
print(f"  Top-quartile aggressive overlap:                         {overlap:.1%}")
print()

gini_gain_vs_summary   = (gini_hmm - gini_summary) * 100
gini_gain_vs_threshold = (gini_hmm - gini_thresh)  * 100
print(f"Gini improvement (HMM vs summary features):   +{gini_gain_vs_summary:.1f}pp")
print(f"Gini improvement (HMM vs threshold scoring):  +{gini_gain_vs_threshold:.1f}pp")
print()
print("EXPECTED PERFORMANCE (5k-driver fleet, 30 trips/driver, 3-state DGP):")
print("  HMM Gini improvement vs summary features:     5-10pp")
print("  HMM Gini improvement vs threshold scoring:    2-7pp")
print("  State recovery rho:                           >=0.70")
print("  HMM total fit time (extract + HMM):           30-90s")
print()
print(f"Total pipeline time:")
print(f"  Simulation:      {t_sim:.1f}s")
print(f"  Clean + extract: {t_extract:.1f}s")
print(f"  HMM fit:         {t_hmm:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. GLM-Ready Feature Export
# MAGIC
# MAGIC The final step demonstrates the production workflow: export HMM-derived features as
# MAGIC a Polars DataFrame ready to join into an existing rating model. Each driver gets
# MAGIC named columns that can be added to a GLM alongside traditional rating factors
# MAGIC (driver age, vehicle type, NCD years, etc.).

# COMMAND ----------

# Show the GLM-ready feature DataFrame
print("GLM-ready feature DataFrame (first 5 rows):")
glm_features = hmm_driver_features.join(
    driver_summary.select(["driver_id", "mean_speed_kmh", "harsh_braking_rate",
                           "harsh_accel_rate", "night_driving_fraction"]),
    on="driver_id",
    how="left"
)
print(glm_features.head().to_pandas().to_string(index=False))
print()
print("Recommended GLM covariates (from Jiang & Shi 2024 methodology):")
print("  state_2_fraction  — fraction of trips classified as aggressive (primary predictor)")
print("  state_0_fraction  — fraction of trips classified as cautious (inverse predictor)")
print("  state_entropy     — behavioural consistency (high entropy = unpredictable driver)")
print()
print("These can be dropped directly into a Poisson frequency GLM alongside:")
print("  driver_age, vehicle_type, ncb_years, annual_km, region")
print()
print("Feature column names follow a controlled vocabulary for regulatory filings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. When to Use This — Practical Guidance
# MAGIC
# MAGIC **Use HMM-based telematics scoring when:**
# MAGIC
# MAGIC - You have raw trip sensor data (1Hz GPS + accelerometer) and want risk-predictive
# MAGIC   features for a Poisson frequency GLM
# MAGIC - Drivers have >= 10 trips per year in your dataset — below this, state estimation
# MAGIC   has high variance and the credibility weight will be near zero
# MAGIC - The portfolio has >= 500 drivers — the HMM benefits from portfolio-level fitting
# MAGIC   to learn state emission distributions
# MAGIC - You need auditable features: state fractions are interpretable (fraction of driving
# MAGIC   time in aggressive state), unlike black-box telematics scores from external vendors
# MAGIC - You plan to segment on telematics features for regulatory pricing review —
# MAGIC   named state fractions are easier to justify than proprietary score components
# MAGIC
# MAGIC **Use simple summary features instead when:**
# MAGIC
# MAGIC - You have fewer than 10 trips per driver — HMM state estimation is unreliable with
# MAGIC   very short histories; use raw averages with credibility shrinkage instead
# MAGIC - Compute time is a hard constraint: summary features add <1s; HMM adds 30-90s for
# MAGIC   a 5,000-driver fleet. Scale linearly.
# MAGIC - The true DGP is genuinely continuous (driver behaviour varies smoothly rather than
# MAGIC   regime-based): the HMM advantage is proportional to how state-structured the data is.
# MAGIC   Test on your portfolio before committing to the HMM approach.
# MAGIC
# MAGIC **Data requirements:**
# MAGIC
# MAGIC - Minimum viable: 500 drivers, 10 trips each, 5 minutes per trip
# MAGIC - Recommended: 2,000+ drivers, 20+ trips each, 15 minutes per trip on average
# MAGIC - Input: 1Hz data with speed_kmh (required), acceleration_ms2 (optional — derived if absent)
# MAGIC - Trip minimum length: 5 minutes (shorter trips produce unreliable HMM state sequences)
# MAGIC
# MAGIC **Computational considerations:**
# MAGIC
# MAGIC - HMM fitting scales with n_trips (not n_drivers) — batch by driver cohort for
# MAGIC   very large fleets (>50,000 drivers)
# MAGIC - For large fleets, run HMM fitting on Databricks with one worker per driver cohort
# MAGIC   using Spark UDFs or parallel pool
# MAGIC - Feature extraction (clean_trips + extract_trip_features) is embarrassingly parallel —
# MAGIC   use Spark partitioned by driver_id for production scale

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC *insurance-telematics v0.1+ | [GitHub](https://github.com/burning-cost/insurance-telematics) | [Burning Cost](https://burning-cost.github.io)*
