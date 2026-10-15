"""Unit tests: tasks, readout, metrics, baselines (no network simulation)."""
import numpy as np
import pytest

from agnw_rc import baselines as B
from agnw_rc import tasks as T
from agnw_rc.evaluate import noise_floor, score_iid, score_sinesquare
from agnw_rc.metrics import accuracy, capacity, consistency, nmse, thresholded_sum
from agnw_rc.readout import Ridge, fit_predict


# ------------------------------------------------------------------ tasks
def test_narma10_matches_direct_recurrence():
    s = 0.5 * np.random.default_rng(0).uniform(size=200)
    y = T.narma10(s)
    ref = np.zeros(201)
    for t in range(9, 200):
        ref[t + 1] = (0.3 * ref[t] + 0.05 * ref[t] * sum(ref[t - i] for i in range(10))
                      + 1.5 * s[t - 9] * s[t] + 0.1)
    np.testing.assert_allclose(y, ref)
    tgt = T.narma10_target(2 * s)
    np.testing.assert_allclose(tgt, ref[1:])        # row t = y[t+1]


def test_delayed_alignment():
    u = np.arange(10.0)
    d = T.delayed(u, 3)
    assert np.all(np.isnan(d[:3]))
    np.testing.assert_array_equal(d[3:], u[:7])
    np.testing.assert_array_equal(T.delayed(u, 0), u)


def test_legendre_targets_are_orthonormal():
    u = np.random.default_rng(1).uniform(size=400_000)
    Y, names = T.ipc2_targets(u)
    Y = Y[T.IPC_P2_LAGS:]
    gram = Y.T @ Y / len(Y)
    np.testing.assert_allclose(gram, np.eye(len(names)), atol=0.01)


def test_ema_targets():
    u = T.iid_stream(500, 2)
    z = T.ema_targets(u, (0.9,))[:, 0]
    ref = np.empty(500)
    prev = 0.5
    for t in range(500):
        prev = 0.9 * prev + 0.1 * u[t]
        ref[t] = prev
    np.testing.assert_allclose(z, ref)


def test_leaky_integrator_reconstructs_its_ema(u):
    z = T.ema_targets(u, (0.95,))
    r = score_iid(z, u)                     # the feature IS the lam = 0.95 integrator
    assert r["ema_capacity"][T.EMA_LAMBDAS.index(0.95)] == pytest.approx(1.0, abs=1e-6)
    assert r["MC"] < 1.0                    # yet its per-lag memory is small (spread over many lags)


def test_sinesquare_stream():
    u, lab = T.sinesquare_stream(800, 3)
    assert u.min() >= 0 and u.max() <= 1
    for p in range(100):
        seg, l = u[8 * p:8 * p + 8], lab[8 * p:8 * p + 8]
        assert len(set(l)) == 1
        x = 2 * seg - 1
        if l[0] == 1:
            np.testing.assert_allclose(x, [1, 1, 1, 1, -1, -1, -1, -1])
        else:
            np.testing.assert_allclose(x, np.sin(2 * np.pi * np.arange(8) / 8), atol=1e-12)


def test_split_is_ordered_disjoint_and_skips_lags():
    tr, va, te = T.split_indices(1000)
    assert tr[0] == T.MAX_LAG and tr[-1] < va[0] and va[-1] < te[0] and te[-1] == 999
    assert len(tr) + len(va) + len(te) == 1000 - T.MAX_LAG


# ------------------------------------------------------------------ readout
def test_ridge_svd_matches_closed_form():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 5)) * [1, 10, 0.1, 3, 1] + 7
    Y = rng.normal(size=(200, 2))
    m = Ridge().fit_svd(X, Y)
    Z = (X - X.mean(0)) / X.std(0)
    a = 3.0
    W = np.linalg.solve(Z.T @ Z + a * np.eye(5), Z.T @ (Y - Y.mean(0)))
    np.testing.assert_allclose(m.weights(a), W, atol=1e-10)
    np.testing.assert_allclose(m.predict(X, a), Z @ W + Y.mean(0), atol=1e-10)
    np.testing.assert_allclose(m.predict(X, np.array([a, a])), m.predict(X, a))


def test_ridge_scale_invariance_and_unpenalized_intercept():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = X[:, 0] + 100.0
    p1 = Ridge().fit_svd(X, y).predict(X, 10.0)
    p2 = Ridge().fit_svd(X * [1e-6, 1e4], y).predict(X * [1e-6, 1e4], 10.0)
    np.testing.assert_allclose(p1, p2)
    assert np.mean(p1) == pytest.approx(np.mean(y))        # heavy ridge still keeps the intercept


def test_ridge_drops_constant_and_roundoff_features():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 1))
    X_tr = np.hstack([x, np.full((100, 1), 5.0), 1e-15 * rng.normal(size=(100, 1))])
    X_te = np.hstack([x, np.full((100, 1), 5.0), 1e-3 * rng.normal(size=(100, 1))])
    m = Ridge(min_std=1e-9).fit_svd(X_tr, x[:, 0])
    np.testing.assert_allclose(m.predict(X_te, 1e-8).ravel(), x[:, 0], atol=1e-5)


def test_fit_predict_selects_small_alpha_for_clean_signal():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(1000, 3))
    y = X @ [1.0, -2.0, 0.5]
    tr, va, te = np.arange(600), np.arange(600, 800), np.arange(800, 1000)
    pred, alphas, _ = fit_predict(X, y, tr, va, te)
    assert alphas[0] <= 1e-2
    assert nmse(pred, y[te]) < 1e-6


# ------------------------------------------------------------------ metrics
def test_capacity_and_nmse_known_values():
    t = np.random.default_rng(6).normal(size=1000)
    assert capacity(t, t)[0] == pytest.approx(1.0)
    assert capacity(np.full_like(t, t.mean()), t)[0] == pytest.approx(0.0)
    assert capacity(-t, t)[0] == 0.0                        # clipped
    assert nmse(np.zeros_like(t) + t.mean(), t) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        nmse(t, np.ones_like(t))


def test_thresholded_sum_and_accuracy():
    assert thresholded_sum([0.9, 0.05, 0.5], 0.1) == pytest.approx(1.4)
    assert accuracy([1, -1, 1, 1], [1, -1, -1, 1]) == 0.75


def test_consistency_of_increments_ignores_a_shared_trend():
    """Two replicates sharing a slow trend but with independent fast responses: whole-series consistency is high,
    increment consistency is ~0 (the case the review found in the binary and N4 runs)."""
    rng = np.random.default_rng(8)
    t = np.linspace(0, 1, 3000)[:, None]
    a = 5 * t + rng.normal(size=(3000, 3))
    b = 5 * t + rng.normal(size=(3000, 3))
    assert consistency(a, b) > 0.6
    assert abs(consistency(np.diff(a, axis=0), np.diff(b, axis=0))) < 0.05


def test_consistency():
    rng = np.random.default_rng(7)
    s = rng.normal(size=(2000, 4))
    assert consistency(s, s) == pytest.approx(1.0)
    assert consistency(s, s + rng.normal(size=s.shape)) == pytest.approx(0.5 ** 0.5, abs=0.03)
    assert abs(consistency(s, rng.normal(size=s.shape))) < 0.05


# ------------------------------------------------------------------ capacities end to end
@pytest.fixture(scope="module")
def u():
    return T.iid_stream(2000, 11)


def test_memory_capacity_of_delay_line_equals_depth(u):
    r = score_iid(B.delay_line(u, 5), u, floor=0.02)
    np.testing.assert_allclose(r["memory_function"][:5], 1.0, atol=1e-6)
    assert r["MC"] == pytest.approx(4.0, abs=0.05)         # lags 1..4 (lag 0 reported separately)
    assert r["lag0"] == pytest.approx(1.0)
    assert r["IPC2"] < 0.1                                 # a linear delay line has no degree-2 capacity


def test_memoryless_features_have_no_memory(u):
    r = score_iid(B.input_only(u), u, floor=0.02)
    assert r["lag0"] == pytest.approx(1.0)
    assert r["MC"] == 0.0


def test_quadratic_features_have_degree2_capacity(u):
    r = score_iid(B.quadratic_delay_line(u, 3), u, floor=0.02)
    # P2 at lags 0..2 and cross terms (0,1), (0,2), (1,2): 6 targets captured
    assert r["IPC2"] == pytest.approx(6.0, abs=0.1)


def test_noise_floor_is_small(u):
    assert noise_floor(u, seeds=(0,), dims=(32,)) < 0.05


def test_sinesquare_needs_memory():
    us, lab = T.sinesquare_stream(1200, 1)
    assert score_sinesquare(B.input_only(us), lab)["accuracy"] < 0.75
    assert score_sinesquare(B.quadratic_delay_line(us, 8), lab)["accuracy"] > 0.95


# ------------------------------------------------------------------ ESN
def test_esn_spectral_radius_and_echo_state_property():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 50))
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
    assert np.max(np.abs(np.linalg.eigvals(W))) == pytest.approx(0.9)
    u = T.iid_stream(300, 0)
    a = B.esn_states(u, 50, 0.9, 0.5, seed=4)
    # same weights, different initial state: run with a random prefix, compare the tails
    prefix = T.iid_stream(50, 1)
    b = B.esn_states(np.concatenate([prefix, u]), 50, 0.9, 0.5, seed=4)[50:]
    assert np.abs(a[-50:] - b[-50:]).max() < 1e-6
