"""
hmm_model.py — HMM-based driving state classification.

Two classes:

DrivingStateHMM
    Discrete-time Gaussian HMM wrapping hmmlearn. Assumes observations at
    uniform time steps (within a trip). State ordering is enforced so that
    state 0 is always the lowest-mean-speed (most cautious) state, making
    the state indices interpretable across fits.

ContinuousTimeHMM
    Continuous-time HMM (CTHMM) for irregularly-sampled data. Uses a
    generator matrix Q with P(Δt) = expm(Q × Δt), so variable trip lengths
    and inter-observation gaps are handled without resampling. Implemented
    from scratch using scipy.linalg.expm and a custom EM algorithm.

Academic basis
--------------
HMM framework: Jiang & Shi (2024), NAAJ 28(4), pp.822-839.
CTHMM mathematics: standard continuous-time Markov chain theory.
State feature extraction: Henckaerts & Antonio (2022).
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import polars as pl
from scipy.stats import entropy as scipy_entropy

try:
    from hmmlearn import hmm as hmmlearn_hmm
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False

try:
    from scipy.linalg import expm as matrix_expm
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# Default feature columns to feed into the HMM
_DEFAULT_HMM_FEATURES: list[str] = [
    "mean_speed_kmh",
    "speed_variation_coeff",
    "harsh_braking_rate",
    "harsh_accel_rate",
]


class DrivingStateHMM:
    """
    Discrete-time Gaussian HMM for classifying driving trip sequences into
    latent driving states (e.g. cautious / normal / aggressive).

    Wraps ``hmmlearn.GaussianHMM`` with state-ordering normalisation and
    a driver-level feature extraction step.

    Parameters
    ----------
    n_states:
        Number of latent states. Default 3. With three states the literature
        consistently produces a cautious / normal / aggressive partition.
    features:
        Trip-level feature columns to use as HMM observations. Defaults to
        mean speed, speed variation, harsh braking rate, harsh acceleration.
    n_iter:
        Maximum EM iterations for hmmlearn (default 200).
    covariance_type:
        Covariance type passed to GaussianHMM (default "diag").
    random_state:
        Random seed for reproducibility.

    Examples
    --------
    >>> model = DrivingStateHMM(n_states=3)
    >>> model.fit(trip_features_df)
    >>> states = model.predict_states(trip_features_df)
    >>> driver_features = model.driver_state_features(trip_features_df, states)
    """

    def __init__(
        self,
        n_states: int = 3,
        features: list[str] | None = None,
        n_iter: int = 200,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        if not _HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for DrivingStateHMM. "
                "Install it with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.features = features or _DEFAULT_HMM_FEATURES
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self._model: Optional[hmmlearn_hmm.GaussianHMM] = None
        self._state_order: Optional[np.ndarray] = None  # indices sorted by mean speed
        self.is_fitted: bool = False

    def fit(self, trip_features: pl.DataFrame) -> "DrivingStateHMM":
        """
        Fit the HMM to a DataFrame of trip-level scalar features.

        Each row is treated as one observation. For a per-driver sequence,
        pass only that driver's trips (ordered by timestamp). For a
        portfolio-level fit, pass all trips — hmmlearn handles the full
        sequence as one long chain, which is appropriate for population-level
        state estimation.

        Parameters
        ----------
        trip_features:
            DataFrame from :func:`~insurance_telematics.extract_trip_features`.
            Must contain the columns listed in ``self.features``.

        Returns
        -------
        self
        """
        X = self._to_matrix(trip_features)
        X = self._standardise(X, fit=True)

        model = hmmlearn_hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        self._model = model

        # Determine state ordering by the mean speed dimension (feature index 0)
        # so that state 0 is always the lowest-speed (most cautious) state
        speed_means = model.means_[:, 0]
        self._state_order = np.argsort(speed_means)
        self._state_rank = np.empty_like(self._state_order)
        self._state_rank[self._state_order] = np.arange(self.n_states)

        self.is_fitted = True
        return self

    def predict_states(self, trip_features: pl.DataFrame) -> np.ndarray:
        """
        Decode the most likely state sequence for a set of trips.

        Parameters
        ----------
        trip_features:
            Same format as used for ``fit()``.

        Returns
        -------
        np.ndarray of shape (n_trips,)
            Integer state labels 0..n_states-1, where 0 = most cautious
            and n_states-1 = most aggressive.
        """
        self._check_fitted()
        X = self._to_matrix(trip_features)
        X = self._standardise(X, fit=False)
        raw_states = self._model.predict(X)
        # Remap to ordered states
        return self._state_rank[raw_states]

    def predict_state_probs(self, trip_features: pl.DataFrame) -> np.ndarray:
        """
        Compute posterior state probabilities for each trip observation.

        Parameters
        ----------
        trip_features:
            Same format as used for ``fit()``.

        Returns
        -------
        np.ndarray of shape (n_trips, n_states)
            Posterior probabilities, summing to 1.0 across states per row.
            Column ordering matches the ordered states (0 = most cautious).
        """
        self._check_fitted()
        X = self._to_matrix(trip_features)
        X = self._standardise(X, fit=False)
        _, posteriors = self._model.score_samples(X)
        # Reorder columns to match state ordering
        return posteriors[:, self._state_order]

    def driver_state_features(
        self,
        trip_features: pl.DataFrame,
        states: np.ndarray,
    ) -> pl.DataFrame:
        """
        Compute per-driver HMM-derived features for use as GLM covariates.

        Parameters
        ----------
        trip_features:
            Trip-level features DataFrame. Must contain ``driver_id``.
        states:
            State sequence as returned by :meth:`predict_states`.

        Returns
        -------
        pl.DataFrame
            One row per driver with columns:

            - ``state_{k}_fraction`` for k in 0..n_states-1
            - ``mean_transition_rate`` — average state transitions per km
            - ``state_entropy`` — Shannon entropy of state distribution
            - ``driver_id``

        Notes
        -----
        State fractions are the primary actuarial risk features. Following
        Jiang & Shi (2024), the fraction of time spent in the highest-risk
        state (state ``n_states - 1``) is the most predictive of claim
        frequency.
        """
        if "driver_id" not in trip_features.columns:
            raise ValueError(
                "trip_features must contain 'driver_id' for driver-level aggregation."
            )

        df = trip_features.with_columns(pl.Series("_state", states.tolist()))

        # Count state occurrences per driver
        driver_rows = []
        driver_ids = df["driver_id"].unique().sort().to_list()

        for driver_id in driver_ids:
            mask = df["driver_id"] == driver_id
            driver_df = df.filter(mask)
            driver_states = driver_df["_state"].to_numpy()
            n_trips = len(driver_states)

            # State fractions
            state_counts = np.bincount(driver_states, minlength=self.n_states)
            state_fractions = state_counts / max(n_trips, 1)

            # Transition rate: number of state changes per km
            n_transitions = np.sum(np.diff(driver_states) != 0)
            total_km = driver_df["distance_km"].sum() if "distance_km" in driver_df.columns else n_trips
            transition_rate = n_transitions / max(total_km, 0.01)

            # State entropy (Shannon)
            state_entropy = float(scipy_entropy(state_fractions))

            row = {"driver_id": driver_id}
            for k in range(self.n_states):
                row[f"state_{k}_fraction"] = float(state_fractions[k])
            row["mean_transition_rate"] = float(transition_rate)
            row["state_entropy"] = float(state_entropy)

            driver_rows.append(row)

        return pl.DataFrame(driver_rows)

    def _to_matrix(self, trip_features: pl.DataFrame) -> np.ndarray:
        """Extract feature columns as float64 numpy array."""
        missing = [c for c in self.features if c not in trip_features.columns]
        if missing:
            raise ValueError(f"Feature columns missing from DataFrame: {missing}")
        return trip_features.select(self.features).to_numpy().astype(np.float64)

    def _standardise(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        """Z-score standardise features. Fit mean/std on first call."""
        if fit:
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0) + 1e-8
        return (X - self._mean) / self._std

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_states().")


class ContinuousTimeHMM:
    """
    Continuous-time HMM for irregularly-sampled telematics observations.

    Uses a generator matrix Q where the transition probability over an
    interval Δt is P(Δt) = expm(Q × Δt). This handles variable trip
    lengths and inter-observation time gaps without resampling.

    Implementation uses an EM algorithm:
    - E-step: forward-backward with time-varying transition matrices P(Δt_i)
    - M-step: update Q via expected transition counts and holding times

    Parameters
    ----------
    n_states:
        Number of latent states (default 3).
    features:
        Feature columns to use as Gaussian emissions per state.
    n_iter:
        Number of EM iterations (default 100).
    tol:
        Convergence tolerance on log-likelihood (default 1e-4).
    random_state:
        Seed for parameter initialisation.

    Notes
    -----
    The M-step update for Q follows the standard uniformisation approach:
    Q_ij (i≠j) ∝ E[N_ij] / E[T_i], where N_ij is the expected number of
    i→j transitions and T_i is the expected total holding time in state i.

    Numerical stability: scipy.linalg.expm is used for matrix exponential.
    For typical generator matrices from driving data (rates on the order of
    0.01-1.0 transitions/minute), expm is stable and accurate.

    Examples
    --------
    >>> model = ContinuousTimeHMM(n_states=3)
    >>> model.fit(trip_features_df, time_deltas=delta_array)
    >>> states = model.predict_states(trip_features_df, time_deltas=delta_array)
    """

    def __init__(
        self,
        n_states: int = 3,
        features: list[str] | None = None,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if not _SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for ContinuousTimeHMM. "
                "Install it with: pip install scipy"
            )

        self.n_states = n_states
        self.features = features or _DEFAULT_HMM_FEATURES
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        self.Q_: Optional[np.ndarray] = None  # Generator matrix (n_states × n_states)
        self.means_: Optional[np.ndarray] = None    # Emission means (n_states × n_features)
        self.covars_: Optional[np.ndarray] = None   # Emission variances (diagonal)
        self.pi_: Optional[np.ndarray] = None       # Initial state distribution
        self.is_fitted: bool = False
        self._state_order: Optional[np.ndarray] = None

    def fit(
        self,
        trip_features: pl.DataFrame,
        time_deltas: np.ndarray | None = None,
    ) -> "ContinuousTimeHMM":
        """
        Fit the CTHMM using the EM algorithm.

        Parameters
        ----------
        trip_features:
            Trip-level features DataFrame (one row per trip, not per second).
        time_deltas:
            Array of shape (n_trips,) giving the time interval in minutes
            between consecutive trips for the same driver. Use a large value
            (e.g. 1440.0 = 24 hours) between trips from different sessions.
            If None, all intervals are set to 1.0 (equivalent to discrete HMM
            with unit time steps).

        Returns
        -------
        self
        """
        X = self._to_matrix(trip_features)
        X = self._standardise(X, fit=True)
        n_obs = X.shape[0]

        if time_deltas is None:
            time_deltas = np.ones(n_obs, dtype=np.float64)
        else:
            time_deltas = np.asarray(time_deltas, dtype=np.float64)
            if len(time_deltas) != n_obs:
                raise ValueError(
                    f"time_deltas length {len(time_deltas)} != n_obs {n_obs}"
                )
        # Clamp to small positive to avoid expm(Q * 0)
        time_deltas = np.clip(time_deltas, 1e-6, None)

        rng = np.random.default_rng(self.random_state)

        # Initialise parameters
        self.pi_ = np.ones(self.n_states) / self.n_states
        self.Q_ = self._init_generator(rng)
        self.means_, self.covars_ = self._init_emissions(X, rng)

        log_likelihood = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            gammas, xis, log_lik_new = self._e_step(X, time_deltas)

            # M-step
            self._m_step(X, gammas, xis, time_deltas)

            # Convergence check
            if abs(log_lik_new - log_likelihood) < self.tol:
                break
            log_likelihood = log_lik_new

        # Order states by mean of the first feature (speed proxy)
        self._state_order = np.argsort(self.means_[:, 0])
        self._state_rank = np.empty_like(self._state_order)
        self._state_rank[self._state_order] = np.arange(self.n_states)

        self.is_fitted = True
        return self

    def predict_states(
        self,
        trip_features: pl.DataFrame,
        time_deltas: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Viterbi decoding of the most likely state sequence.

        Parameters
        ----------
        trip_features:
            Same format as used in ``fit()``.
        time_deltas:
            Inter-observation time intervals, same semantics as in ``fit()``.

        Returns
        -------
        np.ndarray of shape (n_trips,)
            State labels 0..n_states-1 with state 0 = most cautious.
        """
        self._check_fitted()
        X = self._to_matrix(trip_features)
        X = self._standardise(X, fit=False)
        n_obs = X.shape[0]

        if time_deltas is None:
            time_deltas = np.ones(n_obs)
        time_deltas = np.clip(np.asarray(time_deltas, dtype=np.float64), 1e-6, None)

        raw_states = self._viterbi(X, time_deltas)
        return self._state_rank[raw_states]

    def driver_state_features(
        self,
        trip_features: pl.DataFrame,
        states: np.ndarray,
    ) -> pl.DataFrame:
        """
        Compute driver-level HMM features. Same interface as
        :meth:`DrivingStateHMM.driver_state_features`.
        """
        # Delegate to the same logic — identical to discrete case
        if "driver_id" not in trip_features.columns:
            raise ValueError("trip_features must contain 'driver_id'.")

        df = trip_features.with_columns(pl.Series("_state", states.tolist()))
        driver_rows = []
        driver_ids = df["driver_id"].unique().sort().to_list()

        for driver_id in driver_ids:
            driver_df = df.filter(pl.col("driver_id") == driver_id)
            driver_states = driver_df["_state"].to_numpy()
            n_trips = len(driver_states)

            state_counts = np.bincount(driver_states, minlength=self.n_states)
            state_fractions = state_counts / max(n_trips, 1)

            n_transitions = np.sum(np.diff(driver_states) != 0)
            total_km = (
                driver_df["distance_km"].sum()
                if "distance_km" in driver_df.columns
                else float(n_trips)
            )
            transition_rate = n_transitions / max(total_km, 0.01)
            state_entropy = float(scipy_entropy(state_fractions))

            row = {"driver_id": driver_id}
            for k in range(self.n_states):
                row[f"state_{k}_fraction"] = float(state_fractions[k])
            row["mean_transition_rate"] = float(transition_rate)
            row["state_entropy"] = float(state_entropy)
            driver_rows.append(row)

        return pl.DataFrame(driver_rows)

    # ------------------------------------------------------------------
    # Private EM implementation
    # ------------------------------------------------------------------

    def _init_generator(self, rng: np.random.Generator) -> np.ndarray:
        """Initialise a valid generator matrix Q with small random rates."""
        n = self.n_states
        Q = rng.uniform(0.01, 0.1, size=(n, n))
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    def _init_emissions(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialise Gaussian emission parameters by k-means-like split."""
        n, d = X.shape
        # Use equally-spaced quantile points as initial means
        quantiles = np.linspace(0.1, 0.9, self.n_states)
        means = np.array(
            [np.quantile(X, q, axis=0) for q in quantiles], dtype=np.float64
        )
        # Perturb slightly to break symmetry
        means += rng.normal(0, 0.05, size=means.shape)
        covars = np.ones((self.n_states, d), dtype=np.float64)
        return means, covars

    def _transition_matrix(self, dt: float) -> np.ndarray:
        """P(dt) = expm(Q * dt), clamped to be a valid stochastic matrix."""
        P = matrix_expm(self.Q_ * dt)
        # Numerical correction: clip to [0,1] and renormalise rows
        P = np.clip(P.real, 0.0, 1.0)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        return P / row_sums

    def _emission_log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Diagonal Gaussian log-emission probability.

        Returns array of shape (n_obs, n_states).
        """
        n_obs = X.shape[0]
        log_probs = np.zeros((n_obs, self.n_states))
        for k in range(self.n_states):
            diff = X - self.means_[k]
            var = self.covars_[k]
            log_probs[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * var) + diff ** 2 / var, axis=1
            )
        return log_probs

    def _e_step(
        self, X: np.ndarray, dts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-backward algorithm with time-varying transition matrices.

        Returns
        -------
        gammas : (n_obs, n_states)
        xis    : (n_obs-1, n_states, n_states)
        log_likelihood : float
        """
        n_obs, n_states = X.shape[0], self.n_states
        log_emit = self._emission_log_prob(X)  # (n_obs, n_states)

        # Forward pass (log-scale for numerical stability)
        log_alpha = np.full((n_obs, n_states), -np.inf)
        log_alpha[0] = np.log(self.pi_ + 1e-300) + log_emit[0]

        for t in range(1, n_obs):
            P = self._transition_matrix(dts[t - 1])
            log_P = np.log(P + 1e-300)
            # log_alpha[t, j] = log_emit[t, j] + logsumexp_k(log_alpha[t-1, k] + log_P[k, j])
            for j in range(n_states):
                log_alpha[t, j] = log_emit[t, j] + _logsumexp(
                    log_alpha[t - 1] + log_P[:, j]
                )

        log_likelihood = _logsumexp(log_alpha[-1])

        # Backward pass
        log_beta = np.zeros((n_obs, n_states))  # log(1) = 0 at final step
        for t in range(n_obs - 2, -1, -1):
            P = self._transition_matrix(dts[t])
            log_P = np.log(P + 1e-300)
            for i in range(n_states):
                log_beta[t, i] = _logsumexp(
                    log_P[i, :] + log_emit[t + 1] + log_beta[t + 1]
                )

        # Posterior state probabilities gamma
        # Normalise each row by log P(observations) to get P(z_t | x).
        log_gamma = log_alpha + log_beta
        for t in range(n_obs):
            log_gamma[t] -= _logsumexp(log_gamma[t])
        gammas = np.exp(log_gamma)
        gammas = np.clip(gammas, 0.0, None)
        gammas /= gammas.sum(axis=1, keepdims=True)  # numerical guard

        # Pairwise posteriors xi
        # Normalise by P(observations) = exp(log_likelihood), not per-timestep.
        # Per-timestep normalisation is incorrect: it makes each t contribute
        # equally regardless of probability mass, causing EM to converge to a
        # spurious fixed point.
        xis = np.zeros((n_obs - 1, n_states, n_states))
        for t in range(n_obs - 1):
            P = self._transition_matrix(dts[t])
            for i in range(n_states):
                for j in range(n_states):
                    xis[t, i, j] = np.exp(
                        log_alpha[t, i]
                        + np.log(P[i, j] + 1e-300)
                        + log_emit[t + 1, j]
                        + log_beta[t + 1, j]
                        - log_likelihood
                    )

        return gammas, xis, log_likelihood

    def _m_step(
        self,
        X: np.ndarray,
        gammas: np.ndarray,
        xis: np.ndarray,
        dts: np.ndarray,
    ) -> None:
        """Update parameters given posteriors."""
        n_obs, n_states = gammas.shape
        n_features = X.shape[1]

        # Update initial distribution
        self.pi_ = gammas[0] / gammas[0].sum()

        # Update emission parameters
        gamma_sum = gammas.sum(axis=0)  # (n_states,)
        for k in range(n_states):
            w = gammas[:, k]  # (n_obs,)
            w_sum = w.sum() + 1e-300
            self.means_[k] = (w[:, None] * X).sum(axis=0) / w_sum
            diff = X - self.means_[k]
            self.covars_[k] = (w[:, None] * diff ** 2).sum(axis=0) / w_sum + 1e-6

        # Update generator matrix Q
        # Expected transition counts: E[N_ij] = sum_t xi[t, i, j]
        expected_transitions = xis.sum(axis=0)  # (n_states, n_states)

        # Expected holding times: E[T_i] = sum_t gamma[t, i] * dt[t]
        holding_times = np.zeros(n_states)
        for t in range(n_obs - 1):
            holding_times += gammas[t] * dts[t]
        holding_times = np.clip(holding_times, 1e-6, None)

        # New Q: Q_ij = E[N_ij] / E[T_i] for i≠j
        Q_new = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    Q_new[i, j] = expected_transitions[i, j] / holding_times[i]
        np.fill_diagonal(Q_new, -Q_new.sum(axis=1))
        self.Q_ = Q_new

    def _viterbi(self, X: np.ndarray, dts: np.ndarray) -> np.ndarray:
        """Viterbi decoding returning most likely state sequence."""
        n_obs, n_states = X.shape[0], self.n_states
        log_emit = self._emission_log_prob(X)

        log_delta = np.full((n_obs, n_states), -np.inf)
        psi = np.zeros((n_obs, n_states), dtype=int)
        log_delta[0] = np.log(self.pi_ + 1e-300) + log_emit[0]

        for t in range(1, n_obs):
            P = self._transition_matrix(dts[t - 1])
            log_P = np.log(P + 1e-300)
            for j in range(n_states):
                scores = log_delta[t - 1] + log_P[:, j]
                best = int(np.argmax(scores))
                log_delta[t, j] = scores[best] + log_emit[t, j]
                psi[t, j] = best

        # Backtrack
        states = np.zeros(n_obs, dtype=int)
        states[-1] = int(np.argmax(log_delta[-1]))
        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def _to_matrix(self, trip_features: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in trip_features.columns]
        if missing:
            raise ValueError(f"Feature columns missing: {missing}")
        return trip_features.select(self.features).to_numpy().astype(np.float64)

    def _standardise(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0) + 1e-8
        return (X - self._mean) / self._std

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_states().")


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp of a 1D array."""
    a_max = a.max()
    if a_max == -np.inf:
        return float(-np.inf)
    return float(a_max + np.log(np.exp(a - a_max).sum()))
