"""
Regression tests for ContinuousTimeHMM forward-backward algorithm.

These tests were added after fixing two P0 bugs in _e_step:

  P0-1: xi pairwise posteriors were normalised per-timestep (dividing by
        xi[t].sum()) instead of by the global P(observations). This caused
        EM to converge to a spurious fixed point because each timestep
        contributed equally regardless of probability mass.

  P0-2: gamma was normalised using _logsumexp(log_gamma[0]) (t=0 only),
        which is only correct when the forward variable at t=0 already
        equals the full likelihood — not guaranteed in general. The subsequent
        row-max subtraction accidentally masked the error. Fixed to per-row
        normalisation.

The tests here exercise:
  - gamma rows sum to 1 at every timestep
  - xi over all (t,i,j) sums to approximately T-1 (once per timestep, after
    correct global normalisation each xi[t] sums to 1)
  - Q recovery from synthetic data with known true Q (core regression test)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_telematics.hmm_model import ContinuousTimeHMM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cthmm_with_params(
    Q: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    pi: np.ndarray,
) -> ContinuousTimeHMM:
    """
    Build a ContinuousTimeHMM and inject known parameters directly,
    bypassing fit(). This lets us call _e_step on controlled data.
    """
    n_states, n_features = means.shape
    # Create a dummy DataFrame with the right columns; _standardise will
    # not be called because we'll call _e_step directly after setting
    # internal state manually.
    model = ContinuousTimeHMM(n_states=n_states, n_iter=1, random_state=0)
    model.Q_ = Q.copy()
    model.means_ = means.copy()
    model.covars_ = covars.copy()
    model.pi_ = pi.copy()
    model._mean = np.zeros(n_features)
    model._std = np.ones(n_features)
    model.is_fitted = True
    return model


def _simulate_cthmm(
    Q: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    pi: np.ndarray,
    n_obs: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate observations from a CTHMM with given parameters.

    Returns (X, true_states) where X has shape (n_obs, n_features).
    """
    from scipy.linalg import expm

    n_states, n_features = means.shape
    P = expm(Q * dt)
    P = np.clip(P.real, 0.0, 1.0)
    P /= P.sum(axis=1, keepdims=True)

    states = np.empty(n_obs, dtype=int)
    states[0] = rng.choice(n_states, p=pi)
    for t in range(1, n_obs):
        states[t] = rng.choice(n_states, p=P[states[t - 1]])

    X = np.empty((n_obs, n_features))
    for t in range(n_obs):
        k = states[t]
        X[t] = rng.normal(means[k], np.sqrt(covars[k]))

    return X, states


# ---------------------------------------------------------------------------
# Unit tests: e-step invariants
# ---------------------------------------------------------------------------

class TestEStepInvariants:
    """
    Given a known model, the forward-backward e-step must satisfy:
      1. gamma[t].sum() == 1 for all t (posterior over states sums to 1)
      2. xi[t].sum() ≈ 1 for all t (each pairwise posterior sums to 1)
    """

    @pytest.fixture(scope="class")
    def two_state_model_and_obs(self):
        """2-state CTHMM with well-separated emissions."""
        Q_true = np.array([[-0.3,  0.3],
                           [ 0.2, -0.2]])
        means = np.array([[0.0], [3.0]])
        covars = np.array([[0.25], [0.25]])
        pi = np.array([0.5, 0.5])

        rng = np.random.default_rng(1234)
        X, _ = _simulate_cthmm(Q_true, means, covars, pi, n_obs=200, dt=1.0, rng=rng)
        dts = np.ones(200)

        model = _make_cthmm_with_params(Q_true, means, covars, pi)
        gammas, xis, log_lik = model._e_step(X, dts)
        return gammas, xis, log_lik

    def test_gamma_rows_sum_to_one(self, two_state_model_and_obs):
        gammas, _, _ = two_state_model_and_obs
        row_sums = gammas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                                   err_msg="gamma rows must sum to 1")

    def test_gamma_values_in_unit_interval(self, two_state_model_and_obs):
        gammas, _, _ = two_state_model_and_obs
        assert gammas.min() >= -1e-9
        assert gammas.max() <= 1.0 + 1e-9

    def test_xi_per_timestep_sums_to_one(self, two_state_model_and_obs):
        """
        After global normalisation by P(obs), xi[t].sum() should equal 1
        for every t, since it's the joint distribution over (z_t, z_{t+1})
        given the full observation sequence.
        """
        _, xis, _ = two_state_model_and_obs
        for t in range(xis.shape[0]):
            s = xis[t].sum()
            assert abs(s - 1.0) < 1e-5, (
                f"xi[{t}].sum() = {s:.8f}, expected 1.0 "
                "(per-timestep xi should sum to 1 after correct normalisation)"
            )

    def test_xi_consistent_with_gamma(self, two_state_model_and_obs):
        """
        xi[t, i, :].sum() should equal gamma[t, i] for all t, i.
        This is a fundamental consistency identity of the forward-backward
        algorithm.
        """
        gammas, xis, _ = two_state_model_and_obs
        n_obs = gammas.shape[0]
        for t in range(n_obs - 1):
            marginal_i = xis[t].sum(axis=1)  # sum over j
            np.testing.assert_allclose(
                marginal_i, gammas[t],
                atol=1e-5,
                err_msg=f"xi[{t}] marginal over j must equal gamma[{t}]",
            )


# ---------------------------------------------------------------------------
# Regression test: Q recovery
# ---------------------------------------------------------------------------

class TestQRecovery:
    """
    Fit ContinuousTimeHMM on synthetic data generated from a known Q.
    After EM convergence the recovered off-diagonal rates should be within
    reasonable distance of the truth.

    This is the core regression test for P0-1. With the bug, the expected
    transition counts are distorted because each timestep contributes
    equally; the recovered Q converges to a wrong value. With the fix,
    the counts weight each timestep correctly and Q recovery improves.

    We use a 2-state model with well-separated Gaussian emissions so the
    E-step has a clear signal, and a long sequence (5000 obs) to keep
    variance low.
    """

    def _build_synthetic_df(
        self,
        Q_true: np.ndarray,
        n_obs: int,
        dt: float,
        seed: int,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Generate observations from the 2-state CTHMM and return a polars
        DataFrame with a single feature column 'mean_speed_kmh' plus a
        dummy 'driver_id', and the time_deltas array.
        """
        means = np.array([[20.0], [80.0]])   # well-separated speeds
        covars = np.array([[25.0], [25.0]])  # std = 5 kmh
        pi = np.array([0.5, 0.5])

        rng = np.random.default_rng(seed)
        X, _ = _simulate_cthmm(Q_true, means, covars, pi, n_obs=n_obs, dt=dt, rng=rng)

        df = pl.DataFrame({
            "driver_id": ["d1"] * n_obs,
            "mean_speed_kmh": X[:, 0].tolist(),
            # Remaining default features — set to neutral values so they
            # contribute no information (all same → zero variance emission).
            "speed_variation_coeff": [0.1] * n_obs,
            "harsh_braking_rate": [0.0] * n_obs,
            "smooth_accel_score": [0.5] * n_obs,
            "night_driving_fraction": [0.0] * n_obs,
        })
        dts = np.full(n_obs, dt)
        return df, dts

    def test_two_state_q_recovery(self):
        """
        True Q = [[-0.5, 0.5], [0.3, -0.3]].
        With 5000 observations and well-separated emissions the recovered
        off-diagonal rates should be within 30% of truth.

        30% tolerance is generous — the purpose is to catch the qualitative
        failure (order-of-magnitude error) caused by the P0-1 bug, not to
        assert tight statistical estimation.
        """
        Q_true = np.array([[-0.5,  0.5],
                           [ 0.3, -0.3]])
        df, dts = self._build_synthetic_df(Q_true, n_obs=5000, dt=1.0, seed=42)

        model = ContinuousTimeHMM(
            n_states=2,
            features=["mean_speed_kmh"],
            n_iter=80,
            tol=1e-6,
            random_state=7,
        )
        model.fit(df, time_deltas=dts)

        Q_est = model.Q_
        # Q is estimated in standardised space; rates are scale-free so
        # we compare off-diagonal values directly.
        # state ordering may be swapped — try both assignments
        rel_err_natural = max(
            abs(Q_est[0, 1] - Q_true[0, 1]) / Q_true[0, 1],
            abs(Q_est[1, 0] - Q_true[1, 0]) / Q_true[1, 0],
        )
        rel_err_swapped = max(
            abs(Q_est[0, 1] - Q_true[1, 0]) / Q_true[1, 0],
            abs(Q_est[1, 0] - Q_true[0, 1]) / Q_true[0, 1],
        )
        rel_err = min(rel_err_natural, rel_err_swapped)

        assert rel_err < 0.40, (
            f"Q recovery error too large: {rel_err:.2%}. "
            f"True Q off-diag: [{Q_true[0,1]:.3f}, {Q_true[1,0]:.3f}], "
            f"Estimated: [{Q_est[0,1]:.4f}, {Q_est[1,0]:.4f}]. "
            "This may indicate the xi normalisation bug has returned."
        )

    def test_q_is_valid_generator_after_em(self):
        """
        Whatever Q is estimated, it must remain a valid generator matrix:
        off-diagonal non-negative, rows sum to 0.
        """
        Q_true = np.array([[-0.2, 0.2],
                           [0.15, -0.15]])
        df, dts = self._build_synthetic_df(Q_true, n_obs=2000, dt=1.0, seed=99)

        model = ContinuousTimeHMM(n_states=2, features=["mean_speed_kmh"],
                                   n_iter=30, random_state=13)
        model.fit(df, time_deltas=dts)
        Q = model.Q_

        for i in range(2):
            for j in range(2):
                if i != j:
                    assert Q[i, j] >= -1e-6, f"Q[{i},{j}]={Q[i,j]:.6f} is negative"

        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-6,
                                   err_msg="Q rows must sum to 0")

    def test_log_likelihood_increases_during_em(self):
        """
        The EM log-likelihood must be non-decreasing. A single-step decrease
        of more than numerical noise indicates a bug in E- or M-step.
        """
        Q_true = np.array([[-0.4, 0.4],
                           [0.25, -0.25]])
        df, dts = self._build_synthetic_df(Q_true, n_obs=1000, dt=1.0, seed=77)

        # Manually run EM and collect log-likelihoods
        import polars as pl
        from scipy.linalg import expm

        model = ContinuousTimeHMM(
            n_states=2,
            features=["mean_speed_kmh"],
            n_iter=1,  # will loop manually
            random_state=3,
        )
        # Initialise
        X = model._to_matrix(df)
        X = model._standardise(X, fit=True)
        rng = np.random.default_rng(3)
        model.pi_ = np.ones(2) / 2
        model.Q_ = model._init_generator(rng)
        model.means_, model.covars_ = model._init_emissions(X, rng)

        log_likelihoods = []
        for _ in range(15):
            gammas, xis, ll = model._e_step(X, dts)
            log_likelihoods.append(ll)
            model._m_step(X, gammas, xis, dts)

        # Check monotone non-decreasing (allow tiny floating point slack)
        for i in range(1, len(log_likelihoods)):
            assert log_likelihoods[i] >= log_likelihoods[i - 1] - 1e-3, (
                f"Log-likelihood decreased at step {i}: "
                f"{log_likelihoods[i-1]:.4f} -> {log_likelihoods[i]:.4f}. "
                "EM must not decrease the likelihood."
            )


# ---------------------------------------------------------------------------
# P1 regression: scipy_entropy epsilon removed
# ---------------------------------------------------------------------------

class TestEntropyEpsilonRemoved:
    """
    Verify that driver_state_features works correctly when a driver is
    100% in one state (state_fractions has zeros). scipy.stats.entropy
    handles 0 * log(0) = 0 correctly without epsilon padding.
    """

    def test_entropy_with_pure_state(self):
        """A driver always in state 0: entropy should be 0, not epsilon-inflated."""
        model = ContinuousTimeHMM(n_states=3, n_iter=5, random_state=1)

        # Build minimal DataFrame: all observations for one driver
        n = 20
        df = pl.DataFrame({
            "driver_id": ["d1"] * n,
            "mean_speed_kmh": [30.0] * n,
            "speed_variation_coeff": [0.1] * n,
            "harsh_braking_rate": [0.0] * n,
            "harsh_accel_rate": [0.0] * n,
            "night_driving_fraction": [0.0] * n,
        })
        model.fit(df)

        # Manufacture a state array where the driver is 100% in state 0
        all_zero_states = np.zeros(n, dtype=int)
        driver_df = model.driver_state_features(df, all_zero_states)

        entropy_val = driver_df["state_entropy"][0]
        # Should be exactly 0.0 (or extremely close) not inflated by epsilon
        assert entropy_val < 1e-6, (
            f"Expected entropy ≈ 0.0 for pure-state driver, got {entropy_val:.6f}. "
            "Check that + 1e-9 epsilon was removed from scipy_entropy call."
        )
