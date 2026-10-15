"""Series memory statistics and the exact-chain FDTest quantities (agnw_sim.markov_tests)."""
import numpy as np
import pytest

from agnw_sim import markov_tests as mt
from agnw_sim.benchmarks import two_state_fdtest, renewal_fdtest
import agnw_sim.fdtest as ft

S0, R0, S1, R1 = 0.0725, 0.0625, 0.09, 0.08


def _series(kind, n_runs=8, n=20000, seed=1):
    """Standardized synthetic runs. ar1, tanh: Markov; ar2, quad: second order."""
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n_runs):
        x, e = np.zeros(n), r.normal(size=n)
        for t in range(2, n):
            if kind == "ar1":
                x[t] = 0.8 * x[t - 1] + 0.5 * e[t]
            elif kind == "ar2":
                x[t] = 0.5 * x[t - 1] + 0.3 * x[t - 2] + 0.5 * e[t]
            elif kind == "tanh":
                x[t] = np.tanh(2 * x[t - 1]) + 0.5 * e[t]
            elif kind == "quad":
                x[t] = 0.5 * x[t - 1] + 0.9 * np.tanh(x[t - 2] ** 2 - 1) + 0.5 * e[t]
        out.append((x - x.mean()) / x.std())
    return out


# ------------------------------------------------------------------------------------- regression gain
def test_gain_zero_for_markov_series():
    for kind in ("ar1", "tanh"):
        g = mt.regression_memory_gain(_series(kind), h=1, rng=np.random.default_rng(0))
        assert abs(g["gain"]) < 1e-3 and g["ci95"][0] < 1e-3, (kind, g)


def test_gain_positive_for_second_order_series():
    for kind, lo in (("ar2", 0.05), ("quad", 0.2)):
        g = mt.regression_memory_gain(_series(kind), h=1, rng=np.random.default_rng(0))
        assert g["ci95"][0] > lo, (kind, g)


def test_polynomial_model_a_prevents_false_positive_on_nonlinear_markov():
    runs = _series("tanh")
    lin = mt.regression_memory_gain(runs, h=1, degree=1, rng=np.random.default_rng(0))
    poly = mt.regression_memory_gain(runs, h=1, rng=np.random.default_rng(0))
    assert lin["ci95"][0] > 0.02 > poly["gain"]


def test_gain_needs_two_runs():
    with pytest.raises(ValueError):
        mt.regression_memory_gain(_series("ar1", n_runs=1, n=500), h=1)


# ------------------------------------------------------------------------------------- CMI
def _cmi(kind, seed=3):
    rng = np.random.default_rng(seed)
    x, y, z = mt.triples(_series(kind, n_runs=4, n=8000), 1, 3, 6000, rng)
    return mt.cmi_test(x, y, z, rng, n_surr=15)


def test_cmi_within_surrogate_band_for_markov_series():
    for kind in ("ar1", "tanh"):
        assert not _cmi(kind)["detected"], kind


def test_cmi_detects_second_order_series():
    for kind in ("ar2", "quad"):
        res = _cmi(kind)
        assert res["detected"] and res["excess"] > 0.02, (kind, res)


def test_surrogate_keeps_marginal_and_conditional_relation():
    rng = np.random.default_rng(0)
    z = rng.normal(size=5000)
    x = z + 0.1 * rng.normal(size=5000)
    xs = mt.markov_surrogate(x, z, rng)
    assert abs(xs.std() / x.std() - 1) < 0.05 and np.corrcoef(xs, z)[0, 1] > 0.99


def test_triples_never_cross_run_boundaries_and_respect_n_max():
    runs = [np.arange(100.0), 1000 + np.arange(100.0)]
    x, y, z = mt.triples(runs, h=5, stride=7, n_max=10_000, rng=np.random.default_rng(0), jitter=0.0)
    assert np.all(x - z == 5) and np.all(z - y == 5)
    x, _, _ = mt.triples(runs, h=5, stride=1, n_max=50, rng=np.random.default_rng(0))
    assert len(x) == 50


def test_correlation_time_of_ar1():
    tau = mt.correlation_time(_series("ar1", n_runs=4), max_lag=200)
    assert 3 <= tau <= 6          # 0.8**tau < 1/e at tau = 5


# ------------------------------------------------------------------------------------- exact chain
def _two_state(s, r):
    return mt.transition_matrix(np.array([[s], [s]]), np.array([[r], [r]]))


def _curves(T0, Te, labels, n_lags):
    p0, pe = mt.stationary_distribution(T0), mt.stationary_distribution(Te)
    z, ex0, exe = mt.conjugate_state_map(labels, pe, p0)
    zm, c = mt.exact_curves(T0, pe, p0, z, n_lags)
    return zm[:, 0], c[:, 0], ex0, exe


def test_two_state_chain_matches_benchmark():
    zm, c, ex0, exe = _curves(_two_state(S0, R0), _two_state(S1, R1), np.array([0, 1]), 200)
    b = two_state_fdtest(S0, R0, S1, R1, 200)
    assert np.abs(zm - b["z_mean"]).max() < 1e-12 and np.abs(c - b["c_zz"]).max() < 1e-12
    assert ex0 == 0 and exe == 0 and mt.exact_delta2(zm, c)[1] < 1e-12


def _renewal_T(s, r, L):
    T = np.zeros((L + 1, L + 1))
    T[0, 0], T[0, 1] = 1 - s, s
    for j in range(L - 1):
        T[j + 1, 0], T[j + 1, j + 2] = r, 1 - r
    T[L, 0] = 1.0
    return T


def test_renewal_chain_with_hidden_age_matches_benchmark():
    L = 8
    labels = np.array([0] + [1] * L)                      # observed through ON/OFF only, the age is hidden
    zm, c, _, _ = _curves(_renewal_T(S0, R0, L), _renewal_T(S0, R1, L), labels, 200)
    b = renewal_fdtest(S0, R0, S0, R1, L, 200)
    assert np.abs(zm - b["z_mean"]).max() < 1e-12 and np.abs(c - b["c_zz"]).max() < 1e-12
    assert ft.delta_squared(c, zm, 200) == pytest.approx(b["delta2"], rel=1e-9) and b["delta2"] > 0.1


def _coupled(coupling, shift):
    """Two junctions: y evolves alone, x switches on faster while y is ON if `coupling`; `shift` perturbs x's ON probability."""
    bits = mt.state_bits(2)
    p_on, p_off = np.zeros((4, 2)), np.zeros((4, 2))
    for s in range(4):
        p_on[s, 0], p_off[s, 0] = (0.02 + coupling * 0.2 * bits[s, 1]) * shift, 0.03
        p_on[s, 1], p_off[s, 1] = 0.01, 0.01
    return mt.transition_matrix(p_on, p_off)


def test_lumping_creates_violation_only_when_the_hidden_variable_matters():
    bits = mt.state_bits(2)
    for coupling, expect_lumped in ((0.0, False), (1.0, True)):
        T0, Te = _coupled(coupling, 1.0), _coupled(coupling, 1.5)
        full = mt.exact_delta2(*_curves(T0, Te, np.arange(4), 600)[:2])[1]
        lumped = mt.exact_delta2(*_curves(T0, Te, bits[:, 0].astype(int), 600)[:2])[1]
        assert full < 1e-10
        assert (lumped > 0.05) if expect_lumped else (lumped < 1e-10)


def test_transition_matrix_rows_are_probabilities_and_stationarity_holds():
    rng = np.random.default_rng(0)
    T = mt.transition_matrix(rng.uniform(0, 0.3, (8, 3)), rng.uniform(0, 0.3, (8, 3)))
    assert np.all(T >= 0) and np.allclose(T.sum(axis=1), 1.0)
    pi = mt.stationary_distribution(T)
    assert np.allclose(pi @ T, pi, atol=1e-12) and pi.sum() == pytest.approx(1.0)


def test_conjugate_state_map_excludes_rare_and_outside_labels():
    pi0 = np.array([0.5, 0.5 - 1e-9, 1e-9, 0.0])
    pie = np.array([0.4, 0.4, 0.1, 0.1])
    z, ex0, exe = mt.conjugate_state_map(np.array([0, 1, 2, -1]), pie, pi0, p_min=1e-6)
    assert z[2] == 0 and z[3] == 0 and z[0] == pytest.approx(-0.2)
    assert ex0 == pytest.approx(1e-9, abs=1e-12) and exe == pytest.approx(0.2)


def test_exact_delta2_reports_window_variants_and_end():
    n = np.arange(400)
    z = 0.9 ** n
    c = z + 0.2 * 0.5 ** n                  # the difference dies out faster than <Z>: delta2 falls as the window grows
    tau, d, dh, d2, hit = mt.exact_delta2(z, c)
    assert not hit and dh > d > d2 > 0 and tau < 60
    assert d == pytest.approx(ft.delta_squared(c, z, tau))
    assert np.isnan(mt.exact_delta2(z[:40], c[:40])[3])
    assert mt.exact_delta2(np.ones(20), np.ones(20))[4]


def test_mutation_swapping_stage_distributions_changes_the_answer():
    """Guard against a sign / roles mistake: swapping pi_0 and pi_eps must change delta2 of the hidden-age chain."""
    L = 8
    T0, Te = _renewal_T(S0, R0, L), _renewal_T(S0, R1, L)
    labels = np.array([0] + [1] * L)
    a = mt.exact_delta2(*_curves(T0, Te, labels, 200)[:2])[1]
    b = mt.exact_delta2(*_curves(Te, T0, labels, 200)[:2])[1]
    assert abs(a - b) > 1e-3
