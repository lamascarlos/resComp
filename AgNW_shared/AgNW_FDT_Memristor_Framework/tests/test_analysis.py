import numpy as np
import pytest
from agnw_sim.analysis import normalize_conductance, autocorrelation, hysteresis_loop_area, entropy_production


def test_normalize_conductance_maps_off_and_on_to_zero_and_one():
    G = np.array([1e-9, 1e-3])
    g = normalize_conductance(G, G_off=1e-9, G_on=1e-3)
    np.testing.assert_allclose(g, [0.0, 1.0])


def test_normalize_conductance_midpoint():
    G_off, G_on = 1e-9, 1e-3
    midpoint = (G_off + G_on) / 2
    g = normalize_conductance(np.array([midpoint]), G_off, G_on)
    np.testing.assert_allclose(g, [0.5])


def test_normalize_conductance_preserves_shape():
    G = np.full((10, 5), 1e-9)
    g = normalize_conductance(G, G_off=1e-9, G_on=1e-3)
    assert g.shape == (10, 5)


def test_autocorrelation_white_noise_decorrelates_immediately():
    rng = np.random.default_rng(1)
    traces = rng.normal(0, 1.0, size=(5000, 20))
    C = autocorrelation(traces, max_lag=10)
    np.testing.assert_allclose(C[0], 1.0, rtol=0.1)
    assert np.all(np.abs(C[1:]) < 0.1)


def test_autocorrelation_constant_series_is_zero():
    traces = np.ones((100, 3)) * 5.0
    C = autocorrelation(traces, max_lag=5)
    np.testing.assert_allclose(C, np.zeros(6), atol=1e-12)


def test_autocorrelation_empty_series_returns_nan():
    traces = np.empty((100, 0))
    C = autocorrelation(traces, max_lag=5)
    assert C.shape == (6,)
    assert np.all(np.isnan(C))


def test_autocorrelation_max_lag_too_large_raises():
    traces = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError):
        autocorrelation(traces, max_lag=10)


def test_autocorrelation_matches_known_ou_decay():
    # Exact discrete Ornstein-Uhlenbeck (AR(1)) process: stationary variance
    # sigma2, correlation time tau_c, sampled at dt=1.0. Its autocorrelation
    # is EXACTLY C(tau) = sigma2 * exp(-tau*dt/tau_c) -- no discretization
    # bias, since this recursion is the exact OU transition kernel, not an
    # Euler-Maruyama approximation.
    rng = np.random.default_rng(123)
    n_steps = 20000
    n_series = 50
    dt = 1.0
    tau_c = 20.0
    sigma2 = 2.0
    phi = np.exp(-dt / tau_c)
    innov_std = np.sqrt(sigma2 * (1 - phi ** 2))

    traces = np.empty((n_steps, n_series))
    traces[0] = rng.normal(0, np.sqrt(sigma2), size=n_series)
    for i in range(1, n_steps):
        traces[i] = traces[i - 1] * phi + rng.normal(0, innov_std, size=n_series)

    max_lag = 100
    C = autocorrelation(traces, max_lag)
    lags = np.arange(max_lag + 1)
    expected = sigma2 * np.exp(-lags * dt / tau_c)

    # Generous statistical tolerance: 50 independent realizations x 20000
    # samples each is enough for the ensemble-averaged estimate to track
    # the analytic curve closely, but this is still a random estimate, not
    # an exact computation, over the first 20 lags (where expected values
    # stay order-1 and the comparison is meaningful).
    np.testing.assert_allclose(C[0], sigma2, rtol=0.1)
    np.testing.assert_allclose(C[:20], expected[:20], atol=0.15, rtol=0.25)


def test_autocorrelation_matches_brute_force_unbiased_normalization():
    # Deterministic (no randomness): pins the spec's unbiased 1/(n-tau)
    # per-lag normalization against an explicit brute-force reference.
    # A regression that divided by the constant n_steps instead of
    # (n_steps - lag) would NOT be caught by any other test in this file
    # (the OU test's tolerance is too generous at its scale to notice),
    # but is caught here: at n_steps=50, max_lag=20, the two divisors
    # (30 vs 50 at the largest lag) differ by ~40%, far outside any
    # floating-point tolerance.
    rng = np.random.default_rng(7)
    n_steps = 50
    n_series = 4
    max_lag = 20
    traces = rng.normal(size=(n_steps, n_series))

    C = autocorrelation(traces, max_lag)

    demeaned = traces - traces.mean(axis=0, keepdims=True)
    C_brute = np.empty((max_lag + 1, n_series))
    for lag in range(max_lag + 1):
        for j in range(n_series):
            x = demeaned[:, j]
            overlap = np.sum(x[:n_steps - lag] * x[lag:])
            C_brute[lag, j] = overlap / (n_steps - lag)
    C_brute = C_brute.mean(axis=1)

    np.testing.assert_allclose(C, C_brute, rtol=1e-12, atol=1e-12)


def test_hysteresis_loop_area_unit_circle():
    theta = np.linspace(0, 2 * np.pi, 100000)
    I = np.sin(theta)
    V = np.cos(theta)
    area = hysteresis_loop_area(I, V)
    np.testing.assert_allclose(abs(area), np.pi, rtol=1e-3)


def test_hysteresis_loop_area_triangle():
    # Vertices (0,0)->(1,0)->(0,1)->(0,0) in (V,I): shoelace area = 0.5.
    V = np.array([0.0, 1.0, 0.0, 0.0])
    I = np.array([0.0, 0.0, 1.0, 0.0])
    area = hysteresis_loop_area(I, V)
    np.testing.assert_allclose(abs(area), 0.5, rtol=1e-9)


def test_hysteresis_loop_area_open_path_is_honest_line_integral():
    # This is an OPEN (non-closed) path -- V does not return to its start.
    # hysteresis_loop_area does not detect or special-case this (per its
    # docstring, closure is the caller's precondition, not something the
    # function enforces -- matching how entropy_production also trusts its
    # inputs without defensive validation). np.trapezoid(I, V) for I=V
    # over [0,1] is the honest line integral of V dV, i.e. V^2/2 evaluated
    # 0->1 = 0.5, NOT zero -- there is nothing "closed" about this path for
    # the formula to detect, so a bare pass-through trapz is exactly right.
    V = np.array([0.0, 1.0])
    I = np.array([0.0, 1.0])
    area = hysteresis_loop_area(I, V)
    np.testing.assert_allclose(area, 0.5, rtol=1e-9)


def test_entropy_production_constant_iv():
    n = 100
    dt = 1e-3
    I = np.full(n, 2.0)
    V = np.full(n, 3.0)
    T_eff = 300.0
    sigma_dot, sigma_total = entropy_production(I, V, dt, T_eff)
    expected_rate = 2.0 * 3.0 / 300.0
    np.testing.assert_allclose(sigma_dot, expected_rate)
    duration = (n - 1) * dt
    expected_total = expected_rate * duration
    np.testing.assert_allclose(sigma_total, expected_total, rtol=1e-9)


def test_entropy_production_default_temperature():
    sigma_dot, _ = entropy_production(np.array([1.0]), np.array([1.0]), dt=1.0)
    np.testing.assert_allclose(sigma_dot, [1.0 / 300.0])


def test_entropy_production_zero_current_gives_zero_rate():
    sigma_dot, sigma_total = entropy_production(np.zeros(10), np.ones(10) * 5.0, dt=1e-3)
    np.testing.assert_allclose(sigma_dot, np.zeros(10))
    assert sigma_total == 0.0


def test_entropy_production_shapes():
    n = 50
    sigma_dot, sigma_total = entropy_production(np.ones(n), np.ones(n), dt=1e-3)
    assert sigma_dot.shape == (n,)
    assert isinstance(sigma_total, float)
