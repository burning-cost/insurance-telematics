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
import statsmodels.api as sm
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
        # Medium group: moderate zero-inflation, moderate rate
        # Risky group: low zero-inflation, high rate
        if n_groups == 1:
            self._group_params = [
                {"rate": 0.5, "pi": 0.40, "weight": 1.0},
            ]
        elif n_groups == 2:
            self._group_params = [
                {"rate": 0.3, "pi": 0.60, "weight": 0.55},
                {"rate": 1.5, "pi": 0.15, "weight": 0.45},
            ]
        else:
            # 3+ groups: evenly-spaced rates, linearly decreasing zero-inflation
            weights_raw = np.ones(n_groups)
            weights_raw[0] *= 1.2  # slightly more drivers in safest group
            weights = weights_raw / weights_raw.sum()
            rates = np.linspace(0.2, 2.5, n_groups)
            pi_vals = np.linspace(0.65, 0.10, n_groups)
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
            pi = params["pi"]  # probability of structural zero

            for w in range(n_weeks):
                # Weekly exposure: log-normal around 200 km, clipped to [20, 800]
                exposure_km = float(
                    np.clip(rng.lognormal(mean=np.log(200.0), sigma=0.5), 20.0, 800.0)
                )

                event_counts: dict[str, int] = {}
                for event_type in _DEFAULT_EVENT_TYPES:
                    # ZIP draw: with probability pi, structural zero
                    if rng.uniform() < pi:
                        count = 0
                    else:
                        # Poisson rate scales with exposure
                        mu = rate_per_100km * exposure_km / 100.0
                        # Add small event-type-specific multiplier for realism
                        type_multiplier = 1.0 + 0.3 * _DEFAULT_EVENT_TYPES.index(event_type)
                        count = int(rng.poisson(mu * type_multiplier * 0.5 + 0.01))
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

    1. Event counts are structurally zero-inflated (no exposure ≠ unsafe)
    2. The fleet is heterogeneous — a single ZIP is too restrictive
    3. Group membership probabilities are directly useful as pricing features

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
        Driver-level posterior group membership probabilities τ_{i,g},
        marginalised over weeks.
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

        # Aggregate to driver-level for EM computation
        driver_ids = (
            weekly_counts.select("driver_id")
            .unique()
            .sort("driver_id")["driver_id"]
            .to_list()
        )
        self.driver_ids_ = driver_ids
        n_drivers = len(driver_ids)

        # Build per-driver aggregated arrays: shape (n_drivers, n_event_types)
        # X_counts[i, j] = total events for driver i, event type j
        # X_exposure[i] = total exposure_km for driver i
        X_counts, X_exposure, X_zeros = self._aggregate_driver_arrays(
            weekly_counts, driver_ids, covariates
        )

        # Initialise mixing weights uniformly + small noise
        self.mixing_weights_ = np.ones(self.n_groups) / self.n_groups
        noise = rng.dirichlet(np.ones(self.n_groups) * 10)
        self.mixing_weights_ = 0.9 * self.mixing_weights_ + 0.1 * noise

        # Initialise τ (posteriors) with random assignment
        tau = rng.dirichlet(np.ones(self.n_groups) * 2, size=n_drivers)

        # EM loop
        self.log_likelihood_history_ = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # M-step: update parameters given current τ
            self._m_step(X_counts, X_exposure, tau, covariates)

            # E-step: update τ given current parameters
            tau = self._e_step(X_counts, X_exposure, tau)

            # Observed-data log-likelihood
            ll = self._observed_log_likelihood(X_counts, X_exposure)
            self.log_likelihood_history_.append(ll)

            delta = ll - prev_ll
            if iteration > 0 and abs(delta) < self.tol:
                break
            prev_ll = ll

        self.group_posteriors_ = tau  # (n_drivers, n_groups)

        # Post-fit: relabel groups in ascending mean NME rate order
        self._relabel_groups_by_rate(X_counts, X_exposure)

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
        X_counts, X_exposure, _ = self._aggregate_driver_arrays(
            weekly_counts, driver_ids, covariates=None
        )
        tau = self._e_step(X_counts, X_exposure, tau=None)

        rows = {"driver_id": driver_ids}
        for g in range(self.n_groups):
            rows[f"prob_group_{g}"] = tau[:, g].tolist()

        return pl.DataFrame(rows)

    def predict_rate(self, weekly_counts: pl.DataFrame) -> pl.DataFrame:
        """
        Return predicted NME rate (events per km) per driver per event type.

        Rate is the group-posterior-weighted mean of the per-group Poisson rates,
        marginalised over the driver's exposure distribution.

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
        X_counts, X_exposure, _ = self._aggregate_driver_arrays(
            weekly_counts, driver_ids, covariates=None
        )
        tau = self._e_step(X_counts, X_exposure, tau=None)

        rows: dict[str, list] = {"driver_id": driver_ids}
        for j, event_type in enumerate(self.event_types):
            rates_per_group = np.zeros(self.n_groups)
            for g in range(self.n_groups):
                params = self.zip_params_.get(g, {})
                # lambda per unit exposure; intercept-only model stores mean rate
                lam = params.get("lambda_per_km", 0.0)
                pi_g = params.get("pi", 0.0)
                # Expected rate from ZIP: (1 - pi) * lambda
                rates_per_group[g] = (1.0 - pi_g) * lam

            # Weighted rate: sum_g tau_{i,g} * rate_g
            predicted = tau @ rates_per_group  # (n_drivers,)
            rows[f"predicted_rate_{event_type}"] = predicted.tolist()

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
        X_counts, X_exposure, X_zeros = self._aggregate_driver_arrays(
            weekly_counts, driver_ids, covariates=None
        )
        tau = self._e_step(X_counts, X_exposure, tau=None)

        # Dominant group
        dominant = tau.argmax(axis=1).tolist()

        # Total NME rate per km
        total_counts = X_counts.sum(axis=1)  # (n_drivers,)
        # Avoid division by zero for drivers with no exposure
        safe_exposure = np.where(X_exposure > 0, X_exposure, 1.0)
        nme_rate = (total_counts / safe_exposure).tolist()

        # Zero fraction per driver
        n_event_types = len(self.event_types)
        n_obs_per_driver = X_zeros  # pre-computed in _aggregate_driver_arrays
        zero_frac = (n_obs_per_driver / max(n_event_types, 1)).tolist()

        rows: dict[str, list] = {
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
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
        tau: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute posterior group membership τ_{i,g} for each driver.

        Parameters
        ----------
        X_counts : (n_drivers, n_event_types)
        X_exposure : (n_drivers,)
        tau : current posteriors (unused — recomputed from scratch each step)

        Returns
        -------
        np.ndarray, shape (n_drivers, n_groups)
            Normalised posterior responsibilities.
        """
        n_drivers = X_counts.shape[0]
        log_resp = np.zeros((n_drivers, self.n_groups))

        for g in range(self.n_groups):
            log_resp[:, g] = (
                np.log(self.mixing_weights_[g] + 1e-300)
                + self._group_log_likelihood(X_counts, X_exposure, g)
            )

        # Numerically stable softmax over groups
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(
        self,
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
        tau: np.ndarray,
        covariates: Optional[list[str]],
    ) -> None:
        """
        Update mixing weights and per-group ZIP parameters.

        Parameters
        ----------
        X_counts : (n_drivers, n_event_types)
        X_exposure : (n_drivers,)
        tau : (n_drivers, n_groups) — current posteriors
        covariates : unused in intercept-only mode (reserved for future use)
        """
        # Update mixing weights: ω_g = mean_i τ_{i,g}
        self.mixing_weights_ = tau.mean(axis=0)
        self.mixing_weights_ = np.clip(self.mixing_weights_, 1e-8, None)
        self.mixing_weights_ /= self.mixing_weights_.sum()

        # Update per-group ZIP parameters
        for g in range(self.n_groups):
            w_g = tau[:, g]  # (n_drivers,)
            self.zip_params_[g] = self._fit_group_zip(X_counts, X_exposure, w_g)

    def _fit_group_zip(
        self,
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
        w_g: np.ndarray,
    ) -> dict:
        """
        Fit an intercept-only ZIP (or ZIGP) to event count data for one group.

        Uses weighted MLE via statsmodels. Exposure enters as log-offset.
        Falls back to method-of-moments if statsmodels optimisation fails.

        Parameters
        ----------
        X_counts : (n_drivers, n_event_types) — total counts per driver
        X_exposure : (n_drivers,) — total exposure per driver in km
        w_g : (n_drivers,) — posterior weights for this group

        Returns
        -------
        dict with keys:
            ``lambda_per_km`` — Poisson rate per km (across all event types)
            ``pi`` — zero-inflation probability
        """
        n_drivers = X_counts.shape[0]

        # Sum across event types for a single aggregate count series
        y = X_counts.sum(axis=1)  # (n_drivers,)
        log_offset = np.log(np.clip(X_exposure, 1e-6, None))

        # Only attempt statsmodels fit when there is enough effective weight
        eff_n = float(w_g.sum())
        if eff_n < 2.0 or y.max() == 0:
            return self._mom_zip(y, X_exposure, w_g)

        try:
            params = self._statsmodels_zip_fit(y, log_offset, w_g)
        except Exception:
            params = self._mom_zip(y, X_exposure, w_g)

        return params

    def _statsmodels_zip_fit(
        self,
        y: np.ndarray,
        log_offset: np.ndarray,
        w_g: np.ndarray,
    ) -> dict:
        """
        Fit intercept-only ZIP via statsmodels with exposure offset.

        statsmodels requires pandas DataFrames. We bridge polars -> numpy ->
        pandas here to keep the rest of the code polars-native.
        """
        import pandas as pd

        n = len(y)
        # Intercept-only design matrix
        X_design = pd.DataFrame({"const": np.ones(n)})

        # Normalise weights to sum to sample size (statsmodels convention)
        w_norm = w_g * n / (w_g.sum() + 1e-12)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ModelClass = (
                ZeroInflatedGeneralizedPoisson if self.use_generalised
                else ZeroInflatedPoisson
            )
            model = ModelClass(
                endog=y,
                exog=X_design,
                exog_infl=pd.DataFrame({"const": np.ones(n)}),
                offset=log_offset,
                freq_weights=w_norm,
            )
            result = model.fit(
                method="bfgs",
                maxiter=200,
                disp=False,
            )

        # Extract parameters from fitted result
        # Params layout: [inflate_const, count_const] for intercept-only
        params = result.params
        inflate_const = float(params.iloc[0]) if hasattr(params, "iloc") else float(params[0])
        count_const = float(params.iloc[1]) if hasattr(params, "iloc") else float(params[1])

        # pi = logistic(inflate_const), lambda_per_km = exp(count_const)
        pi = float(1.0 / (1.0 + np.exp(-inflate_const)))
        pi = float(np.clip(pi, 1e-6, 1.0 - 1e-6))
        # count_const is log(lambda * exposure_ref); exposure already in offset
        # so lambda_per_km = exp(count_const) / reference_exposure = exp(count_const)
        lambda_per_km = float(np.exp(np.clip(count_const, -10.0, 10.0)))

        return {"lambda_per_km": lambda_per_km, "pi": pi}

    def _mom_zip(
        self,
        y: np.ndarray,
        X_exposure: np.ndarray,
        w_g: np.ndarray,
    ) -> dict:
        """
        Method-of-moments fallback for ZIP parameter estimation.

        Closed-form: estimate lambda from nonzero mean, pi from zero fraction.
        """
        w_sum = float(w_g.sum()) + 1e-12
        w_y = float((w_g * y).sum()) / w_sum
        w_zero = float((w_g * (y == 0).astype(float)).sum()) / w_sum

        # pi estimate from zero fraction; clamp for numerical stability
        pi_hat = float(np.clip(w_zero - np.exp(-max(w_y, 1e-8)), 0.0, 0.99))

        # Lambda estimate from mean count per km
        safe_exposure = np.where(X_exposure > 0, X_exposure, 1.0)
        rate_per_km = float((w_g * y / safe_exposure).sum() / w_sum)
        lambda_per_km = max(rate_per_km, 1e-6)

        return {"lambda_per_km": lambda_per_km, "pi": pi_hat}

    def _group_log_likelihood(
        self,
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
        g: int,
    ) -> np.ndarray:
        """
        Compute per-driver log-likelihood under group g's ZIP parameters.

        Parameters
        ----------
        X_counts : (n_drivers, n_event_types)
        X_exposure : (n_drivers,)
        g : group index

        Returns
        -------
        np.ndarray, shape (n_drivers,)
            Log-likelihood contribution for each driver under group g.
        """
        params = self.zip_params_.get(g, {"lambda_per_km": 0.5, "pi": 0.3})
        pi = float(params.get("pi", 0.3))
        lam_per_km = float(params.get("lambda_per_km", 0.5))

        # Aggregate counts and compute ZIP log-likelihood per driver
        y = X_counts.sum(axis=1)  # (n_drivers,)
        mu = lam_per_km * np.clip(X_exposure, 1e-6, None)  # expected count
        mu = np.clip(mu, 1e-12, None)

        log_pi = np.log(pi + 1e-300)
        log1mpi = np.log(1.0 - pi + 1e-300)

        # ZIP log-likelihood:
        # y=0: log(pi + (1-pi) * exp(-mu))
        # y>0: log(1-pi) + y*log(mu) - mu - log_factorial(y)
        from scipy.special import gammaln

        log_lik = np.where(
            y == 0,
            np.logaddexp(log_pi, log1mpi - mu),
            log1mpi + y * np.log(mu) - mu - gammaln(y + 1.0),
        )
        return log_lik

    def _observed_log_likelihood(
        self,
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
    ) -> float:
        """Compute observed-data log-likelihood (mixture marginal)."""
        n_drivers = X_counts.shape[0]
        log_resp = np.zeros((n_drivers, self.n_groups))
        for g in range(self.n_groups):
            log_resp[:, g] = (
                np.log(self.mixing_weights_[g] + 1e-300)
                + self._group_log_likelihood(X_counts, X_exposure, g)
            )
        # log-sum-exp over groups
        ll = float(np.logaddexp.reduce(log_resp, axis=1).sum())
        return ll

    def _relabel_groups_by_rate(
        self,
        X_counts: np.ndarray,
        X_exposure: np.ndarray,
    ) -> None:
        """
        Relabel groups so group 0 has the lowest mean NME rate.

        Modifies ``zip_params_``, ``mixing_weights_``, and ``group_posteriors_``
        in-place so the ordering is stable after fitting.
        """
        # Mean NME rate per group: sum_i tau_{i,g} * total_count_i / sum_i tau_{i,g}
        mean_rates = []
        for g in range(self.n_groups):
            w = self.group_posteriors_[:, g]
            w_sum = float(w.sum()) + 1e-12
            total_counts = X_counts.sum(axis=1)
            safe_exp = np.where(X_exposure > 0, X_exposure, 1.0)
            mean_rate = float((w * total_counts / safe_exp).sum() / w_sum)
            mean_rates.append(mean_rate)

        order = np.argsort(mean_rates)  # ascending rate order
        self._group_order_ = order.tolist()

        # Re-index params, weights, posteriors
        new_params = {new_g: self.zip_params_[int(old_g)] for new_g, old_g in enumerate(order)}
        new_weights = self.mixing_weights_[order]
        new_posteriors = self.group_posteriors_[:, order]

        self.zip_params_ = new_params
        self.mixing_weights_ = new_weights
        self.group_posteriors_ = new_posteriors

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _aggregate_driver_arrays(
        self,
        weekly_counts: pl.DataFrame,
        driver_ids: list[str],
        covariates: Optional[list[str]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate weekly panel to driver-level arrays.

        Returns
        -------
        X_counts : (n_drivers, n_event_types) — total event counts per driver
        X_exposure : (n_drivers,) — total exposure_km per driver
        X_zero_frac : (n_drivers,) — fraction of (week, event_type) zeros per driver
        """
        n_drivers = len(driver_ids)
        n_events = len(self.event_types)

        # Build a lookup for driver index
        driver_idx = {d: i for i, d in enumerate(driver_ids)}

        X_counts = np.zeros((n_drivers, n_events), dtype=np.float64)
        X_exposure = np.zeros(n_drivers, dtype=np.float64)
        X_obs_count = np.zeros(n_drivers, dtype=np.float64)  # total (week, event) obs
        X_zero_count = np.zeros(n_drivers, dtype=np.float64)  # total zeros

        # Vectorised aggregation using polars
        agg_cols = [
            pl.col(self.exposure_col).sum().alias("total_exposure"),
            pl.len().alias("n_weeks"),
        ]
        for et in self.event_types:
            agg_cols.append(pl.col(et).sum().alias(f"sum_{et}"))
            agg_cols.append((pl.col(et) == 0).sum().alias(f"zeros_{et}"))

        agg = (
            weekly_counts
            .group_by("driver_id")
            .agg(agg_cols)
            .sort("driver_id")
        )

        for row in agg.iter_rows(named=True):
            i = driver_idx.get(row["driver_id"])
            if i is None:
                continue
            X_exposure[i] = float(row["total_exposure"])
            n_weeks_i = int(row["n_weeks"])
            for j, et in enumerate(self.event_types):
                X_counts[i, j] = float(row[f"sum_{et}"])
                X_zero_count[i] += float(row[f"zeros_{et}"])
            X_obs_count[i] = float(n_weeks_i * n_events)

        # Zero fraction
        safe_obs = np.where(X_obs_count > 0, X_obs_count, 1.0)
        X_zero_frac = X_zero_count / safe_obs

        return X_counts, X_exposure, X_zero_frac

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
