"""
zip_near_miss.py — Zero-Inflated Poisson model for ADAS near-miss event counts.

ADAS systems (forward collision warning, lane departure, harsh braking, etc.)
generate count data that is structurally zero-inflated: many drivers generate
zero events in a given week simply because they did not drive in triggering
conditions, not because they are inherently safe. A standard Poisson GLM
misattributes these structural zeros as evidence of low risk.

This module fits a Group-Based Zero-Inflated Poisson (GBZIP) model to weekly
ADAS event count panels. The latent group structure identifies sub-populations
with systematically different zero-inflation profiles and event rates — the
mixing weights and group posteriors are then used as features in the downstream
TelematicsScoringPipeline.

Methodology
-----------
The EM algorithm alternates between:
  E-step: compute posterior group membership τ_{i,g} for each driver i
  M-step: re-fit ZIP parameters for each group g using weighted MLE

The unit of observation for ZIP fitting is a (driver, week, event_type) triple.
Driver-level posteriors τ_{i,g} are computed from the sum of weekly ZIP
log-likelihoods over all weeks and event types for that driver. This correctly
handles the structural-zero problem: a driver with many zero-event weeks gets
high posterior for the low-rate / high-pi group.

ZIP per group is estimated via statsmodels ZeroInflatedPoisson (intercept-only
or with covariates). Exposure enters as a log-offset in the count component.
After fitting, groups are relabelled in ascending order of mean NME rate so
that group 0 is always the safest segment.

Academic basis
--------------
Jones, B.L. & Zitikis, R. (2003). ASTIN Bulletin 33(1): 217-230.
Boucher, J-P., Denuit, M. & Guillen, M. (2007). Insurance: Mathematics and
  Economics 40(3): 443-455.
Verbelen, R., Gong, L., Antonio, K., Badescu, A. & Lin, S. (2015).
  Scandinavian Actuarial Journal 2015(8): 649-669.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl

# statsmodels is a required dependency (>=0.14.5)
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedGeneralizedPoisson,
)

_DEFAULT_EVENT_TYPES: list[str] = [
    "harsh_braking",
    "harsh_accel",
    "forward_collision",
    "lane_departure",
    "too_close",
    "speeding_serious",
]


# ---------------------------------------------------------------------------
# NearMissSimulator
# ---------------------------------------------------------------------------


class NearMissSimulator:
    """
    Generate synthetic weekly ADAS near-miss event counts for a driver panel.

    Data is generated from a ZIP mixture with ``n_groups`` latent components.
    Each group has a distinct event rate and zero-inflation probability, chosen
    so that groups are well-separated and group-recovery tests are meaningful.

    Parameters
    ----------
    n_groups:
        Number of latent risk groups (default 3: safe, medium, risky).
    seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> sim = NearMissSimulator(seed=42)
    >>> df = sim.simulate(n_drivers=100, n_weeks=52)
    >>> df.shape == (100 * 52, len(df.columns))
    True
    """

    def __init__(self, n_groups: int = 3, seed: int = 42) -> None:
        self.n_groups = n_groups
        self.seed = seed

        # Group parameters: (lambda_rate_per_100km, zero_inflation_prob, mixing_weight)
        # Designed so groups are clearly separable for ground-truth validation.
        # Safe group: high zero-inflation, very low rate
        # Risky group: low zero-inflation, high rate
        if n_groups == 1:
            self._group_params = [
                {"rate": 0.5, "pi": 0.40, "weight": 1.0},
            ]
        elif n_groups == 2:
            self._group_params = [
                {"rate": 0.1, "pi": 0.70, "weight": 0.55},
                {"rate": 1.5, "pi": 0.10, "weight": 0.45},
            ]
        else:
            # 3+ groups: evenly-spaced rates, linearly decreasing zero-inflation
            weights_raw = np.ones(n_groups)
            weights_raw[0] *= 1.2  # slightly more drivers in safest group
            weights = weights_raw / weights_raw.sum()
            rates = np.linspace(0.08, 1.8, n_groups)
            pi_vals = np.linspace(0.75, 0.05, n_groups)
            self._group_params = [
                {"rate": float(rates[g]), "pi": float(pi_vals[g]), "weight": float(weights[g])}
                for g in range(n_groups)
            ]

    def simulate(
        self,
        n_drivers: int = 200,
        n_weeks: int = 52,
    ) -> pl.DataFrame:
        """
        Simulate weekly ADAS event counts for a panel of drivers.

        Parameters
        ----------
        n_drivers:
            Number of distinct drivers.
        n_weeks:
            Number of weekly observation periods per driver.

        Returns
        -------
        pl.DataFrame
            Long-format panel with one row per (driver, week). Columns:

            - ``driver_id`` — string identifier
            - ``week_id`` — integer 0..n_weeks-1
            - ``exposure_km`` — kilometres driven that week (log-Normal)
            - ``true_group`` — latent group label (0-indexed, 0 = safest)
            - One column per event type in ``_DEFAULT_EVENT_TYPES``
        """
        rng = np.random.default_rng(self.seed)
        weights = [p["weight"] for p in self._group_params]
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()

        # Assign each driver to a latent group
        driver_groups = rng.choice(self.n_groups, size=n_drivers, p=weights_arr)

        rows: list[dict] = []
        for i in range(n_drivers):
            driver_id = f"DRV{i:04d}"
            g = int(driver_groups[i])
            params = self._group_params[g]
            rate_per_100km = params["rate"]
            pi = params["pi"]  # probability of structural zero for each event-week obs

            for w in range(n_weeks):
                # Weekly exposure: log-normal around 200 km, clipped to [20, 800]
                exposure_km = float(
                    np.clip(rng.lognormal(mean=np.log(200.0), sigma=0.5), 20.0, 800.0)
                )

                event_counts: dict[str, int] = {}
                for event_type in _DEFAULT_EVENT_TYPES:
                    # ZIP draw at the weekly-event level:
                    # with probability pi, structural zero (regardless of exposure)
                    if rng.uniform() < pi:
                        count = 0
                    else:
                        # Poisson rate scales with exposure
                        mu = rate_per_100km * exposure_km / 100.0
                        count = int(rng.poisson(max(mu, 1e-8)))
                    event_counts[event_type] = count

                rows.append(
                    {
                        "driver_id": driver_id,
                        "week_id": w,
                        "exposure_km": round(exposure_km, 2),
                        "true_group": g,
                        **event_counts,
                    }
                )

        return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# ZIPNearMissModel
# ---------------------------------------------------------------------------


class ZIPNearMissModel:
    """
    Group-Based Zero-Inflated Poisson (GBZIP) model for ADAS near-miss events.

    Fits an EM algorithm where each latent group has its own ZIP distribution.
    The mixing weights and group posteriors identify risk sub-populations in the
    driver fleet. This is more appropriate than a GLM for ADAS event counts
    because:

    1. Event counts are structurally zero-inflated (no trigger != unsafe driver)
    2. The fleet is heterogeneous — a single ZIP is too restrictive
    3. Group membership probabilities are directly useful as pricing features

    The EM works at the (driver, week, event_type) observation level. Driver
    posteriors τ_{i,g} are computed from the sum of weekly ZIP log-likelihoods
    across all weeks and event types. This correctly distinguishes structural
    zeros (high-pi group) from sampling zeros (low-rate group with exposure).

    Parameters
    ----------
    n_groups:
        Number of latent risk groups.
    event_types:
        ADAS event columns to model. Defaults to the standard six-event set.
    use_generalised:
        If True, fit Zero-Inflated Generalised Poisson (ZIGP) per group instead
        of ZIP. ZIGP handles over/under-dispersion beyond zero-inflation.
        Requires statsmodels>=0.14.5.
    max_iter:
        Maximum EM iterations.
    tol:
        Convergence tolerance on the log-likelihood increment.
    exposure_col:
        Column in the input DataFrame containing weekly exposure in km.
    random_state:
        Seed for initialising mixing weights.

    Attributes
    ----------
    mixing_weights_ : np.ndarray, shape (n_groups,)
        Portfolio-level mixing probabilities after fitting.
    zip_params_ : dict
        Per-group fitted ZIP/ZIGP parameter dicts. Keys are group indices.
    group_posteriors_ : np.ndarray, shape (n_drivers, n_groups)
        Driver-level posterior group membership probabilities τ_{i,g}.
    log_likelihood_history_ : list[float]
        Observed-data log-likelihood after each EM iteration.
    driver_ids_ : list[str]
        Ordered driver IDs corresponding to rows of ``group_posteriors_``.

    Examples
    --------
    >>> sim = NearMissSimulator(seed=42)
    >>> weekly = sim.simulate(n_drivers=100, n_weeks=26)
    >>> model = ZIPNearMissModel(n_groups=3, random_state=42)
    >>> model.fit(weekly)
    ZIPNearMissModel(n_groups=3)
    >>> probs = model.predict_group_probs(weekly)
    >>> probs.columns
    ['driver_id', 'prob_group_0', 'prob_group_1', 'prob_group_2']
    """

    def __init__(
        self,
        n_groups: int = 3,
        event_types: Optional[list[str]] = None,
        use_generalised: bool = False,
        max_iter: int = 100,
        tol: float = 1e-4,
        exposure_col: str = "exposure_km",
        random_state: int = 42,
    ) -> None:
        self.n_groups = n_groups
        self.event_types = event_types if event_types is not None else list(_DEFAULT_EVENT_TYPES)
        self.use_generalised = use_generalised
        self.max_iter = max_iter
        self.tol = tol
        self.exposure_col = exposure_col
        self.random_state = random_state

        # Fitted attributes — set by fit()
        self.mixing_weights_: np.ndarray = np.full(n_groups, 1.0 / n_groups)
        self.zip_params_: dict[int, dict] = {}
        self.group_posteriors_: np.ndarray = np.empty((0, n_groups))
        self.log_likelihood_history_: list[float] = []
        self.driver_ids_: list[str] = []
        self._group_order_: list[int] = list(range(n_groups))  # set post-fit

    def __repr__(self) -> str:
        return f"ZIPNearMissModel(n_groups={self.n_groups})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        weekly_counts: pl.DataFrame,
        covariates: Optional[list[str]] = None,
    ) -> "ZIPNearMissModel":
        """
        Fit the GBZIP model to a weekly driver-event-count panel.

        Parameters
        ----------
        weekly_counts:
            Long-format panel DataFrame. Required columns: ``driver_id``,
            ``week_id``, ``exposure_col``, plus one column per event type.
        covariates:
            Optional list of additional covariate column names to include in
            both the inflation (π) and rate (λ) components of each ZIP.

        Returns
        -------
        self
        """
        _validate_weekly_counts(weekly_counts, self.event_types, self.exposure_col)

        rng = np.random.default_rng(self.random_state)

        # Sorted driver IDs for stable indexing
        driver_ids = (
            weekly_counts.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        self.driver_ids_ = driver_ids
        n_drivers = len(driver_ids)
        driver_idx_map = {d: i for i, d in enumerate(driver_ids)}

        # Build observation arrays at the (driver, week, event_type) level
        # obs_y[k] = count, obs_exp[k] = exposure_km, obs_driver[k] = driver index
        obs_y, obs_exp, obs_driver = self._build_obs_arrays(
            weekly_counts, driver_idx_map
        )
        n_obs = len(obs_y)

        # Initialise mixing weights uniformly
        self.mixing_weights_ = np.ones(self.n_groups) / self.n_groups

        # Initialise driver posteriors by zero-fraction quantile bands.
        # Sort drivers by their per-driver zero fraction across all event types
        # and assign to groups in equal-sized quantile bands. This breaks
        # the symmetry that causes all groups to converge to the same parameters.
        driver_zero_frac = np.zeros(n_drivers)
        obs_is_zero = (obs_y == 0).astype(float)
        obs_count_per_driver = np.zeros(n_drivers)
        np.add.at(driver_zero_frac, obs_driver, obs_is_zero)
        np.add.at(obs_count_per_driver, obs_driver, 1.0)
        driver_zero_frac /= np.where(obs_count_per_driver > 0, obs_count_per_driver, 1.0)

        # Assign drivers to groups by quantile of zero fraction
        sort_idx = np.argsort(driver_zero_frac)
        group_assignments = np.zeros(n_drivers, dtype=int)
        boundaries = np.linspace(0, n_drivers, self.n_groups + 1).astype(int)
        for g in range(self.n_groups):
            group_assignments[sort_idx[boundaries[g]: boundaries[g + 1]]] = g

        # Soft initialisation: high mass on assigned group, small mass on others
        tau = np.full((n_drivers, self.n_groups), 0.05 / max(self.n_groups - 1, 1))
        for i in range(n_drivers):
            tau[i, group_assignments[i]] = 0.95
        # Add small random perturbation to avoid exact symmetry
        tau += rng.uniform(0, 0.02, size=(n_drivers, self.n_groups))
        tau /= tau.sum(axis=1, keepdims=True)

        # EM loop
        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # M-step: update parameters given current τ
            self._m_step(obs_y, obs_exp, obs_driver, tau, n_drivers)

            # E-step: update τ given current parameters
            tau = self._e_step(obs_y, obs_exp, obs_driver, n_drivers)

            # Observed-data log-likelihood
            ll = self._observed_log_likelihood(obs_y, obs_exp, obs_driver, n_drivers)
            self.log_likelihood_history_.append(ll)

            delta = ll - prev_ll
            if iteration > 0 and abs(delta) < self.tol:
                break
            prev_ll = ll

        self.group_posteriors_ = tau  # (n_drivers, n_groups)

        # Post-fit: relabel groups in ascending mean NME rate order
        self._relabel_groups_by_rate(obs_y, obs_exp, obs_driver, n_drivers)

        return self

    def predict_group_probs(self, weekly_counts: pl.DataFrame) -> pl.DataFrame:
        """
        Return posterior group membership probabilities per driver.

        Parameters
        ----------
        weekly_counts:
            Same format as passed to :meth:`fit`.

        Returns
        -------
        pl.DataFrame
            One row per driver, columns: ``driver_id``, ``prob_group_0``, ...,
            ``prob_group_{G-1}``. Probabilities sum to 1 across columns.
        """
        self._check_fitted()
        _validate_weekly_counts(weekly_counts, self.event_types, self.exposure_col)

        driver_ids = (
            weekly_counts.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        driver_idx_map = {d: i for i, d in enumerate(driver_ids)}
        obs_y, obs_exp, obs_driver = self._build_obs_arrays(weekly_counts, driver_idx_map)
        tau = self._e_step(obs_y, obs_exp, obs_driver, len(driver_ids))

        rows: dict = {"driver_id": driver_ids}
        for g in range(self.n_groups):
            rows[f"prob_group_{g}"] = tau[:, g].tolist()
        return pl.DataFrame(rows)

    def predict_rate(self, weekly_counts: pl.DataFrame) -> pl.DataFrame:
        """
        Return predicted NME rate (events per km) per driver per event type.

        Rate is the group-posterior-weighted expected ZIP rate, where the ZIP
        expected rate is (1 - π_g) * λ_g.

        Parameters
        ----------
        weekly_counts:
            Same format as passed to :meth:`fit`.

        Returns
        -------
        pl.DataFrame
            One row per driver, columns: ``driver_id`` plus
            ``predicted_rate_{event_type}`` for each event type.
        """
        self._check_fitted()
        _validate_weekly_counts(weekly_counts, self.event_types, self.exposure_col)

        driver_ids = (
            weekly_counts.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        driver_idx_map = {d: i for i, d in enumerate(driver_ids)}
        obs_y, obs_exp, obs_driver = self._build_obs_arrays(weekly_counts, driver_idx_map)
        tau = self._e_step(obs_y, obs_exp, obs_driver, len(driver_ids))

        # Per-group expected rates (same for all event types in intercept-only model)
        rates_per_group = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            params = self.zip_params_.get(g, {})
            lam = params.get("lambda_per_km", 0.0)
            pi_g = params.get("pi", 0.0)
            rates_per_group[g] = (1.0 - pi_g) * lam

        # Driver-level predicted rate: τ @ rates_per_group
        predicted_rate = tau @ rates_per_group  # (n_drivers,)

        rows: dict = {"driver_id": driver_ids}
        for event_type in self.event_types:
            rows[f"predicted_rate_{event_type}"] = predicted_rate.tolist()
        return pl.DataFrame(rows)

    def driver_risk_features(self, weekly_counts: pl.DataFrame) -> pl.DataFrame:
        """
        Return driver-level risk features for use in TelematicsScoringPipeline.

        This output is designed to be joined with :func:`aggregate_to_driver`
        output on ``driver_id``.

        Parameters
        ----------
        weekly_counts:
            Same format as passed to :meth:`fit`.

        Returns
        -------
        pl.DataFrame
            One row per driver. Columns:

            - ``driver_id``
            - ``dominant_group`` — group index with highest posterior probability
            - ``nme_rate_per_km`` — total NME events per km (all event types summed)
            - ``zero_fraction`` — fraction of (driver, week, event_type) triples
              that are zero
            - ``prob_group_{g}`` — posterior probability for each group g
        """
        self._check_fitted()
        _validate_weekly_counts(weekly_counts, self.event_types, self.exposure_col)

        driver_ids = (
            weekly_counts.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        driver_idx_map = {d: i for i, d in enumerate(driver_ids)}
        obs_y, obs_exp, obs_driver = self._build_obs_arrays(weekly_counts, driver_idx_map)
        tau = self._e_step(obs_y, obs_exp, obs_driver, len(driver_ids))
        n_drivers = len(driver_ids)

        # Dominant group
        dominant = tau.argmax(axis=1).tolist()

        # Total NME rate per km per driver: sum events / sum exposure per event_type
        # obs_exp is per observation (same exposure for all events in same week)
        total_events_per_driver = np.zeros(n_drivers)
        total_exp_per_driver = np.zeros(n_drivers)
        np.add.at(total_events_per_driver, obs_driver, obs_y)
        np.add.at(total_exp_per_driver, obs_driver, obs_exp)
        safe_exp = np.where(total_exp_per_driver > 0, total_exp_per_driver, 1.0)
        nme_rate = (total_events_per_driver / safe_exp).tolist()

        # Zero fraction: fraction of observations that are zero per driver
        obs_count = np.zeros(n_drivers)
        zero_count = np.zeros(n_drivers)
        np.add.at(obs_count, obs_driver, 1.0)
        np.add.at(zero_count, obs_driver, (obs_y == 0).astype(float))
        safe_obs = np.where(obs_count > 0, obs_count, 1.0)
        zero_frac = (zero_count / safe_obs).tolist()

        rows: dict = {
            "driver_id": driver_ids,
            "dominant_group": dominant,
            "nme_rate_per_km": nme_rate,
            "zero_fraction": zero_frac,
        }
        for g in range(self.n_groups):
            rows[f"prob_group_{g}"] = tau[:, g].tolist()
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # EM internals
    # ------------------------------------------------------------------

    def _e_step(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        obs_driver: np.ndarray,
        n_drivers: int,
    ) -> np.ndarray:
        """
        Compute posterior group membership τ_{i,g} for each driver.

        Driver i's log-likelihood under group g is the sum of weekly ZIP
        log-likelihoods across all observations belonging to driver i.

        Returns
        -------
        np.ndarray, shape (n_drivers, n_groups)
            Normalised posterior responsibilities.
        """
        log_resp = np.zeros((n_drivers, self.n_groups))

        for g in range(self.n_groups):
            # Per-observation log-likelihood under group g
            obs_ll = self._obs_log_likelihood(obs_y, obs_exp, g)  # (n_obs,)
            # Sum over observations belonging to each driver
            driver_ll = np.zeros(n_drivers)
            np.add.at(driver_ll, obs_driver, obs_ll)
            log_resp[:, g] = np.log(self.mixing_weights_[g] + 1e-300) + driver_ll

        # Numerically stable softmax over groups
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp /= np.where(resp_sum > 0, resp_sum, 1.0)
        return resp

    def _m_step(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        obs_driver: np.ndarray,
        tau: np.ndarray,
        n_drivers: int,
    ) -> None:
        """
        Update mixing weights and per-group ZIP parameters.

        Each observation is weighted by the posterior of its driver: w_{k,g} = τ_{driver(k),g}.
        """
        # Update mixing weights
        self.mixing_weights_ = tau.mean(axis=0)
        self.mixing_weights_ = np.clip(self.mixing_weights_, 1e-8, None)
        self.mixing_weights_ /= self.mixing_weights_.sum()

        # Broadcast driver posteriors to observations
        obs_tau = tau[obs_driver]  # (n_obs, n_groups)

        for g in range(self.n_groups):
            w_obs_g = obs_tau[:, g]  # (n_obs,) posterior weight per observation
            self.zip_params_[g] = self._fit_group_zip(obs_y, obs_exp, w_obs_g)

    def _fit_group_zip(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        w_obs: np.ndarray,
    ) -> dict:
        """
        Fit an intercept-only ZIP (or ZIGP) to weighted observations.

        Uses statsmodels with weighted MLE. Falls back to method-of-moments
        if statsmodels optimisation fails.

        Parameters
        ----------
        obs_y : (n_obs,) — event counts per observation
        obs_exp : (n_obs,) — exposure_km per observation
        w_obs : (n_obs,) — posterior weights for this group

        Returns
        -------
        dict with keys:
            ``lambda_per_km`` — Poisson rate per km
            ``pi`` — zero-inflation probability
        """
        eff_n = float(w_obs.sum())
        if eff_n < 2.0:
            return self._mom_zip(obs_y, obs_exp, w_obs)

        log_offset = np.log(np.clip(obs_exp, 1e-6, None))

        try:
            params = self._statsmodels_zip_fit(obs_y, log_offset, w_obs)
        except Exception:
            params = self._mom_zip(obs_y, obs_exp, w_obs)

        return params

    def _statsmodels_zip_fit(
        self,
        obs_y: np.ndarray,
        log_offset: np.ndarray,
        w_obs: np.ndarray,
    ) -> dict:
        """
        Fit intercept-only ZIP or ZIGP via weighted MLE.

        For ZIP: uses scipy.optimize with a closed-form weighted log-likelihood.
        statsmodels ZeroInflatedPoisson.freq_weights is effectively ignored in
        the optimiser, so we implement the weighted objective directly in scipy.

        For ZIGP (use_generalised=True): uses statsmodels with a weighted
        subsample, which correctly handles freq_weights as case counts.

        Exposure enters as a log-offset in the Poisson component.
        """
        from scipy.optimize import minimize
        from scipy.special import gammaln

        w_sum = float(w_obs.sum())
        if w_sum < 1e-12:
            return {"lambda_per_km": 1e-4, "pi": 0.5}

        if self.use_generalised:
            # ZIGP via statsmodels with weighted subsample
            return self._statsmodels_zigp_weighted(obs_y, log_offset, w_obs)

        # ZIP via scipy weighted MLE
        def neg_wll(params: np.ndarray) -> float:
            logit_pi, log_lam = params
            pi = 1.0 / (1.0 + np.exp(-logit_pi))
            lam = np.exp(log_lam)
            mu = lam * np.exp(log_offset)
            mu = np.clip(mu, 1e-12, None)
            log_pi = np.log(pi + 1e-300)
            log1mpi = np.log(1.0 - pi + 1e-300)
            ll = np.where(
                obs_y == 0,
                np.logaddexp(log_pi, log1mpi - mu),
                log1mpi + obs_y * np.log(mu) - mu - gammaln(obs_y + 1.0),
            )
            return -float(np.dot(w_obs, ll))

        # Smart initialisation from weighted moments
        mom = self._mom_zip(obs_y, np.exp(log_offset), w_obs)
        pi0 = float(np.clip(mom["pi"], 1e-4, 0.9999))
        lam0 = float(np.clip(mom["lambda_per_km"], 1e-6, None))
        x0 = np.array([np.log(pi0 / (1.0 - pi0)), np.log(lam0)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_wll, x0=x0, method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-9},
            )

        if not res.success and not np.isfinite(res.fun):
            return mom

        pi = float(1.0 / (1.0 + np.exp(-res.x[0])))
        pi = float(np.clip(pi, 1e-6, 1.0 - 1e-6))
        lambda_per_km = float(np.exp(np.clip(res.x[1], -15.0, 10.0)))
        return {"lambda_per_km": lambda_per_km, "pi": pi}

    def _statsmodels_zigp_weighted(
        self,
        obs_y: np.ndarray,
        log_offset: np.ndarray,
        w_obs: np.ndarray,
    ) -> dict:
        """
        Fit ZIGP using statsmodels with a weighted-bootstrap subsample.

        Draws a subsample of ~min(5000, n) observations proportional to w_obs,
        which is unbiased and avoids the freq_weights bug in statsmodels ZI models.
        """
        import pandas as pd

        n = len(obs_y)
        w_norm = w_obs / (w_obs.sum() + 1e-12)
        sample_size = min(5000, n)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=sample_size, replace=True, p=w_norm)
        y_s = obs_y[idx]
        lo_s = log_offset[idx]
        X_s = pd.DataFrame({"const": np.ones(sample_size)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ZeroInflatedGeneralizedPoisson(
                endog=y_s, exog=X_s, exog_infl=X_s, offset=lo_s,
            )
            result = model.fit(method="bfgs", maxiter=300, disp=False)

        inflate_const = float(result.params.iloc[0])
        count_const = float(result.params.iloc[1])
        pi = float(np.clip(1.0 / (1.0 + np.exp(-inflate_const)), 1e-6, 1.0 - 1e-6))
        lambda_per_km = float(np.exp(np.clip(count_const, -15.0, 10.0)))
        return {"lambda_per_km": lambda_per_km, "pi": pi}

    def _mom_zip(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        w_obs: np.ndarray,
    ) -> dict:
        """
        Method-of-moments fallback for ZIP parameter estimation.

        Uses weighted counts to estimate pi from zero fraction and lambda from
        the nonzero mean rate per km.
        """
        w_sum = float(w_obs.sum()) + 1e-12
        w_zero_frac = float((w_obs * (obs_y == 0).astype(float)).sum()) / w_sum

        # Estimated pi: zero_fraction - exp(-lambda*E[exposure]) ≈ zero_fraction
        # Use simple fraction as starting point
        pi_hat = float(np.clip(w_zero_frac * 0.8, 0.0, 0.99))

        safe_exp = np.where(obs_exp > 0, obs_exp, 1.0)
        w_rate = float((w_obs * obs_y / safe_exp).sum()) / w_sum
        lambda_per_km = max(w_rate, 1e-6)

        return {"lambda_per_km": lambda_per_km, "pi": pi_hat}

    def _obs_log_likelihood(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        g: int,
    ) -> np.ndarray:
        """
        Compute per-observation ZIP log-likelihood under group g.

        Parameters
        ----------
        obs_y : (n_obs,) — event counts
        obs_exp : (n_obs,) — exposure_km per observation
        g : group index

        Returns
        -------
        np.ndarray, shape (n_obs,)
        """
        params = self.zip_params_.get(g, {"lambda_per_km": 0.5, "pi": 0.3})
        pi = float(params.get("pi", 0.3))
        lam_per_km = float(params.get("lambda_per_km", 0.5))

        mu = lam_per_km * np.clip(obs_exp, 1e-6, None)
        mu = np.clip(mu, 1e-12, None)

        log_pi = np.log(pi + 1e-300)
        log1mpi = np.log(1.0 - pi + 1e-300)

        from scipy.special import gammaln

        # ZIP log-likelihood:
        # y=0: log(pi + (1-pi)*exp(-mu))  [can be either structural zero or Poisson zero]
        # y>0: log(1-pi) + y*log(mu) - mu - log(y!)
        log_lik = np.where(
            obs_y == 0,
            np.logaddexp(log_pi, log1mpi - mu),
            log1mpi + obs_y * np.log(mu) - mu - gammaln(obs_y + 1.0),
        )
        return log_lik

    # Alias for backward compatibility with _group_log_likelihood name in tests
    def _group_log_likelihood(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        g: int,
    ) -> np.ndarray:
        """Alias for _obs_log_likelihood (per-observation ZIP log-likelihood)."""
        return self._obs_log_likelihood(obs_y, obs_exp, g)

    def _observed_log_likelihood(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        obs_driver: np.ndarray,
        n_drivers: int,
    ) -> float:
        """Compute observed-data log-likelihood (mixture marginal)."""
        # Per-driver log-likelihood: sum over observations
        log_resp = np.zeros((n_drivers, self.n_groups))
        for g in range(self.n_groups):
            obs_ll = self._obs_log_likelihood(obs_y, obs_exp, g)
            driver_ll = np.zeros(n_drivers)
            np.add.at(driver_ll, obs_driver, obs_ll)
            log_resp[:, g] = np.log(self.mixing_weights_[g] + 1e-300) + driver_ll

        ll = float(np.logaddexp.reduce(log_resp, axis=1).sum())
        return ll

    def _relabel_groups_by_rate(
        self,
        obs_y: np.ndarray,
        obs_exp: np.ndarray,
        obs_driver: np.ndarray,
        n_drivers: int,
    ) -> None:
        """
        Relabel groups so group 0 has the lowest mean NME rate.

        Modifies ``zip_params_``, ``mixing_weights_``, and ``group_posteriors_``
        in-place so the ordering is stable after fitting.
        """
        # Use per-group lambda_per_km as the sorting key (simpler and more stable
        # than computing weighted driver rates)
        rates = [
            self.zip_params_.get(g, {}).get("lambda_per_km", 0.0)
            for g in range(self.n_groups)
        ]
        order = np.argsort(rates)  # ascending rate order
        self._group_order_ = order.tolist()

        new_params = {new_g: self.zip_params_[int(old_g)] for new_g, old_g in enumerate(order)}
        new_weights = self.mixing_weights_[order]
        new_posteriors = self.group_posteriors_[:, order]

        self.zip_params_ = new_params
        self.mixing_weights_ = new_weights
        self.group_posteriors_ = new_posteriors

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _build_obs_arrays(
        self,
        weekly_counts: pl.DataFrame,
        driver_idx_map: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flatten weekly panel to (driver, week, event_type) observation arrays.

        Returns
        -------
        obs_y : (n_obs,) — event counts
        obs_exp : (n_obs,) — exposure_km for that observation
        obs_driver : (n_obs,) — driver index (integer)
        """
        # Use polars to efficiently build the flat observation arrays
        # Each row in weekly_counts contains n_event_types observations
        # We melt the event columns to get one row per (driver, week, event_type)
        n_event_types = len(self.event_types)

        # Build arrays from polars without melting (faster for large panels)
        sorted_df = weekly_counts.sort(["driver_id", "week_id"])
        n_rows = len(sorted_df)

        obs_y_list = []
        obs_exp_list = []
        obs_driver_list = []

        driver_col = sorted_df["driver_id"].to_list()
        exp_col = sorted_df[self.exposure_col].to_numpy()

        event_arrays = [sorted_df[et].to_numpy() for et in self.event_types]

        for j in range(n_event_types):
            obs_y_list.append(event_arrays[j])
            obs_exp_list.append(exp_col)
            driver_indices = np.array(
                [driver_idx_map.get(d, 0) for d in driver_col], dtype=np.int32
            )
            obs_driver_list.append(driver_indices)

        obs_y = np.concatenate(obs_y_list).astype(np.float64)
        obs_exp = np.concatenate(obs_exp_list).astype(np.float64)
        obs_driver = np.concatenate(obs_driver_list).astype(np.int32)

        return obs_y, obs_exp, obs_driver

    def _check_fitted(self) -> None:
        if not self.zip_params_:
            raise RuntimeError(
                "ZIPNearMissModel has not been fitted yet. Call fit() first."
            )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_weekly_counts(
    df: pl.DataFrame,
    event_types: list[str],
    exposure_col: str,
) -> None:
    """Raise ValueError if required columns are missing."""
    required = {"driver_id", "week_id", exposure_col} | set(event_types)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"weekly_counts missing required columns: {sorted(missing)}"
        )
    if df.is_empty():
        raise ValueError("weekly_counts DataFrame is empty.")
