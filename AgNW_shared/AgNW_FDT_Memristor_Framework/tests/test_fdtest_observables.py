"""Alternative observables of the FDTest (docs/alt_observables_design.md): ON fraction m_K and the cross-fitted optimal
linear observable. Pure-function tests, exact null on iid Markov junctions, and a smoke test of the driver."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agnw_sim import fdtest as ft
from agnw_sim import fdtest_observables as fo
from scripts import fdtest_observables as drv
from conftest import _mc_cycles, _ar_cycles, PRESET, V0, I0, N_LAGS          # shared Monte-Carlo helpers (tests/conftest.py)

ROOT = Path(__file__).resolve().parents[1]


def _cov(K, seed, big=25.0):
    """Anisotropic noise covariance: identity plus one dominant direction v (variance `big`), returns (Sigma, v)."""
    r = np.random.default_rng(seed)
    v = r.normal(size=K); v /= np.linalg.norm(v)
    return np.eye(K) + (big - 1.0) * np.outer(v, v), v


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class _Perm:
    """Stub rng with a prescribed permutation (the code under test may only call .permutation(n))."""
    def __init__(self, perm):
        self.perm = np.asarray(perm)

    def permutation(self, n):
        assert n == len(self.perm)
        return self.perm.copy()


# ------------------------------------------------------------------------------------ on_fraction
def test_on_fraction_is_the_mean_over_the_junction_axis():
    g = np.array([[[1, 0, 1, 1], [0, 0, 0, 1]], [[1, 1, 1, 1], [0, 0, 0, 0]]], dtype=np.uint8)      # (2 cycles, 2 t, 4 junctions)
    out = fo.on_fraction(g)
    assert out.shape == (2, 2) and out.dtype == np.float64
    np.testing.assert_allclose(out, [[0.75, 0.25], [1.0, 0.0]])
    np.testing.assert_allclose(fo.on_fraction(np.array([1.0, 0.0, 0.0, 0.0])), 0.25)                # a single state vector


# ------------------------------------------------------------------------------------ LDA direction
def test_fit_lda_direction_recovers_the_optimal_direction_with_correlated_noise():
    # Break caught: a direction along the mean difference (ignores the noise covariance) or a broken normalisation.
    K, n_c, T = 15, 60, 120
    Sigma, v = _cov(K, 1)
    delta = np.random.default_rng(2).normal(size=K); delta = 0.8 * delta / np.linalg.norm(delta) + 0.5 * v      # has a component along the noise axis
    truth = np.linalg.solve(Sigma, delta)
    L = np.linalg.cholesky(Sigma)
    r = np.random.default_rng(3)
    g4 = r.normal(size=(n_c, T, K)) @ L.T
    g2 = r.normal(size=(n_c, T, K)) @ L.T + delta
    w, mu0, sd0 = fo.fit_lda_direction(g2, g4, ridge_frac=0.1, stride=1)
    assert w.shape == (K,) and _cos(w, truth) > 0.9
    assert _cos(w, truth) > _cos(delta, truth) + 0.05                                    # better than the naive mean-difference direction
    s4 = g4 @ w
    assert mu0 == pytest.approx(s4.mean()) and sd0 == pytest.approx(s4.std())          # standardisation by stage IV
    assert ((g2 @ w).mean() - mu0) / sd0 > 0                                            # sign convention: positive mean response
    assert (g2.mean(axis=(0, 1)) - g4.mean(axis=(0, 1))) @ w > 0


def test_fit_lda_direction_sign_convention_and_constant_columns():
    # Break caught: a sign that depends on the data order, and a singular covariance from always-OFF / always-ON junctions.
    K = 8
    r = np.random.default_rng(4)
    g4 = (r.random((20, 50, K)) < 0.4).astype(np.uint8)
    g2 = (r.random((20, 50, K)) < 0.4).astype(np.uint8)
    g2[..., 1] = (r.random((20, 50)) < 0.9)                                             # junction 1 is driven ON
    g4[..., 6] = 0; g2[..., 6] = 0                                                       # constant junction (always OFF)
    g4[..., 7] = 1; g2[..., 7] = 1                                                       # constant junction (always ON)
    w, mu0, sd0 = fo.fit_lda_direction(g2, g4, stride=1)
    assert np.isfinite(w).all() and np.isfinite(mu0) and np.isfinite(sd0) and sd0 > 0
    assert w[1] > 0 and w[1] == np.abs(w).max()
    assert w[6] == 0 and w[7] == 0                                                       # no evidence on constant junctions
    w_swapped, _, _ = fo.fit_lda_direction(g4, g2, stride=1)                             # reversed roles: the response is still positive
    assert w_swapped[1] < 0


def test_fit_lda_direction_uses_the_strided_samples_only():
    # Break caught: the stride being ignored (the covariance of 1000 junctions must come from a subsample).
    r = np.random.default_rng(5)
    g4 = r.normal(size=(6, 60, 5)); g2 = r.normal(size=(6, 60, 5)) + 0.3
    a = fo.fit_lda_direction(g2, g4, stride=6)[0]
    b = fo.fit_lda_direction(g2[:, ::6], g4[:, ::6], stride=1)[0]
    np.testing.assert_allclose(a, b)
    c = fo.fit_lda_direction(g2, g4, stride=1)[0]
    assert not np.allclose(a, c)


def test_fit_lda_direction_degenerate_input_gives_a_zero_weight_and_no_warning():
    g = np.ones((4, 10, 3), np.uint8)
    w, mu0, sd0 = fo.fit_lda_direction(g, g, stride=1)
    assert (w == 0).all() and sd0 == 0                                                   # no information: the scores are then NaN


def _explicit_lda(g2, g4, ridge_frac, stride):
    """Independent explicit computation of the documented discriminant (numpy only, on the strided samples)."""
    X2, X4 = (g[:, ::stride, :].reshape(-1, g.shape[2]).astype(float) for g in (g2, g4))
    m2, m4 = X2.mean(axis=0), X4.mean(axis=0)
    sigma = ((X2 - m2).T @ (X2 - m2) + (X4 - m4).T @ (X4 - m4)) / (len(X2) + len(X4) - 2)            # pooled within-stage covariance
    lam = ridge_frac * np.trace(sigma) / g2.shape[2]
    w = np.linalg.solve(sigma + lam * np.eye(g2.shape[2]), m2 - m4)
    return w / np.linalg.norm(w), sigma, lam


@pytest.mark.parametrize("ridge_frac, stride", [(0.1, 1), (0.5, 1), (0.1, 2), (0.0, 1)])
def test_fit_lda_direction_matches_the_explicit_formula_on_a_tiny_problem(ridge_frac, stride):
    # Break caught: a wrong ridge (x10, missing 1/K, ~0) or a covariance that is not the pooled within-stage one
    # (e.g. stage II left out). K = 3 correlated junctions, 6 + 8 samples per stage, so every ingredient matters.
    r = np.random.default_rng(11)
    A = np.array([[2.0, 0.0, 0.0], [1.5, 0.4, 0.0], [0.3, -0.2, 0.05]])                    # anisotropic mixing
    g4 = (r.normal(size=(2, 8, 3)) @ A.T)
    g2 = (r.normal(size=(2, 6, 3)) @ A.T) + np.array([0.7, -0.4, 0.5])
    w, mu0, sd0 = fo.fit_lda_direction(g2, g4, ridge_frac=ridge_frac, stride=stride)
    expected, sigma, lam = _explicit_lda(g2, g4, ridge_frac, stride)
    np.testing.assert_allclose(w, expected, rtol=1e-9, atol=1e-12)
    # the ridge is consequential here: the result differs from the unregularised and the x10 solutions
    d = g2[:, ::stride].reshape(-1, 3).mean(axis=0) - g4[:, ::stride].reshape(-1, 3).mean(axis=0)
    for lam2 in (10 * lam, 0.1 * lam) if lam > 0 else (1.0,):
        other = np.linalg.solve(sigma + lam2 * np.eye(3), d); other /= np.linalg.norm(other)
        assert np.linalg.norm(other - expected) > 1e-3
    s4 = g4 @ w
    assert mu0 == pytest.approx(s4.mean()) and sd0 == pytest.approx(s4.std())


# ------------------------------------------------------------------------------------ leakage control
def _null_pair(seed, n_c=20, T=10, K=60):
    """No shift between the stages: identical distributions, few samples for many junctions (the overfitting regime)."""
    Sigma, _ = _cov(K, 7, big=9.0)
    L = np.linalg.cholesky(Sigma)
    r = np.random.default_rng(seed)
    mk = lambda T_: r.normal(size=(n_c, T_, K)) @ L.T
    return mk(T), mk(T + 4), mk(T)


def test_leakage_in_sample_response_is_positive_and_cross_fitted_response_is_null():
    # Break caught: weights fitted on the cycles that are then scored manufacture a response out of noise (the reason for
    # cross-fitting). Averaged over datasets: the in-sample response is clearly positive, the cross-fitted one is close to 0.
    in_s, cross = [], []
    for seed in range(6):
        g2, g3, g4 = _null_pair(seed)
        w, mu0, sd0 = fo.fit_lda_direction(g2, g4, stride=1)
        in_s.append(((g2 @ w).mean() - mu0) / sd0)
        info = fo.crossfit_scores(g2, g3, g4, np.random.default_rng(seed), stride=1)[3]
        cross.append(info["response_crossfit"])
        assert np.all(np.asarray(info["response_in_sample"]) > 0.3)                       # in-sample per fold, reported in info
    assert np.mean(in_s) > 0.5
    assert abs(np.mean(cross)) < 0.1 and abs(np.mean(cross)) < np.mean(in_s) / 5


# ------------------------------------------------------------------------------------ cross-fitting mechanics
def _mech_data(n_c, K=5, seed=0):
    r = np.random.default_rng(seed)
    g4 = r.normal(size=(n_c, 12, K)) + 0.1 * np.arange(n_c)[:, None, None]                # cycle-specific offsets: order is visible
    g2 = r.normal(size=(n_c, 10, K)) + np.linspace(0.2, 0.9, K) + 0.1 * np.arange(n_c)[:, None, None]
    g3 = r.normal(size=(n_c, 7, K)) + 0.1 * np.arange(n_c)[:, None, None]
    return g2, g3, g4


@pytest.mark.parametrize("n_c, perm", [(6, [0, 1, 2, 3, 4, 5]), (5, [0, 1, 2, 3, 4]), (5, [4, 2, 0, 3, 1]), (7, [6, 5, 4, 3, 2, 1, 0])])
def test_crossfit_scores_each_fold_uses_the_other_folds_weights_and_keeps_the_cycle_order(n_c, perm):
    # Break caught: scoring a fold with its own weights, a scrambled cycle order, a wrong odd-fold split, or the fit
    # fold's standardisation instead of the evaluation fold's own.
    g2, g3, g4 = _mech_data(n_c)
    x2, x3, x4, info = fo.crossfit_scores(g2, g3, g4, _Perm(perm), stride=2)
    assert x2.shape == (n_c, 10) and x3.shape == (n_c, 7) and x4.shape == (n_c, 12)
    perm = np.asarray(perm)
    A, B = perm[: n_c // 2], perm[n_c // 2:]
    np.testing.assert_array_equal(info["fold"][A], 0)
    np.testing.assert_array_equal(info["fold"][B], 1)
    fits = [fo.fit_lda_direction(g2[A], g4[A], stride=2), fo.fit_lda_direction(g2[B], g4[B], stride=2)]
    np.testing.assert_allclose(info["w"], [fits[0][0], fits[1][0]])
    for own, other in ((A, 1), (B, 0)):                                                   # cycles of fold `own` use the weights of the OTHER fold
        w = fits[other][0]
        s4 = g4[own] @ w                                                                  # the evaluation fold's OWN stage-IV statistics
        for full, x in ((g2, x2), (g3, x3), (g4, x4)):
            np.testing.assert_allclose(x[own], (full[own] @ w - s4.mean()) / s4.std())
    assert not np.allclose(fits[0][0], fits[1][0])
    assert info["w"].shape == (2, 5) and len(info["response_in_sample"]) == 2 and np.isfinite(info["response_crossfit"])


def test_crossfit_standardises_each_evaluation_fold_by_its_own_stage_iv_statistics_hand_derived():
    # Hand-derived (K = 1, identity permutation: fold 0 = cycles 0, 1; fold 1 = cycles 2, 3). Fold 0 is perturbed UP
    # (stage II = stage IV + 5) so w0 = +1; fold 1 is perturbed DOWN (stage II = stage IV - 3) so w1 = -1 (positive-response
    # convention). Fold 1's cycles are scored with w0 = +1: stage IV values 10, 12, 11, 13 -> mean 11.5, std sqrt(1.25);
    # fold 0's cycles with w1 = -1: -(0, 2, 1, 3) -> mean -1.5, std sqrt(1.25). Each fold is standardised by its OWN mean/std.
    g4 = np.array([[0, 2], [1, 3], [10, 12], [11, 13]], float)[:, :, None]
    g2 = g4 + np.array([5, 5, -3, -3], float)[:, None, None]
    g3 = g4.copy()
    x2, x3, x4, info = fo.crossfit_scores(g2, g3, g4, _Perm([0, 1, 2, 3]), stride=1)
    sd = np.sqrt(1.25)
    np.testing.assert_allclose(info["w"], [[1.0], [-1.0]])
    np.testing.assert_allclose(x4, [[1.5 / sd, -0.5 / sd], [0.5 / sd, -1.5 / sd], [-1.5 / sd, 0.5 / sd], [-0.5 / sd, 1.5 / sd]])
    np.testing.assert_allclose(x2[0], [-3.5 / sd, -5.5 / sd])                         # -(5, 7) + 1.5
    np.testing.assert_allclose(x2[2], [-4.5 / sd, -2.5 / sd])                         # (7, 9) - 11.5
    np.testing.assert_allclose(info["mu_eval"], [-1.5, 11.5]); np.testing.assert_allclose(info["sd_eval"], [sd, sd])
    np.testing.assert_allclose(info["mu0"], [1.5, -11.5]); np.testing.assert_allclose(info["sd0"], [sd, sd])   # in-fit statistics, reported too


def test_crossfit_scores_have_zero_mean_and_unit_std_in_stage_iv_per_fold_on_null_data():
    # Break caught: standardising with the FIT fold's stage-IV statistics. The weights are fitted on those very samples, so on
    # null data they are inflated in-sample and the held-out stage-IV scores come out with mean 0.5-0.9 and std 1.1-1.6, with a
    # different affine map for each fold (pooling then mixes maps). The evaluation fold's own statistics give exactly 0 / 1.
    biased = []
    for seed in range(6):
        g2, g3, g4 = _null_pair(seed)
        x2, x3, x4, info = fo.crossfit_scores(g2, g3, g4, np.random.default_rng(seed), stride=1)
        for f in (0, 1):
            ev = info["fold"] == f
            assert abs(x4[ev].mean()) < 1e-9 and abs(x4[ev].std() - 1) < 1e-9
            assert abs(info["mu_eval"][f] - (g4[ev] @ info["w"][1 - f]).mean()) < 1e-9
            old = ((g4[ev] @ info["w"][1 - f]).mean() - info["mu0"][1 - f]) / info["sd0"][1 - f]      # the discarded standardisation
            biased.append(abs(old))
        assert abs(x4.mean()) < 1e-9 and abs(x4.std() - 1) < 1e-9
    assert max(biased) > 0.3                                                             # the bias the fix removes is real on this data


def test_crossfit_scores_uses_the_rng_permutation_and_needs_two_cycles():
    g2, g3, g4 = _mech_data(8)
    a = fo.crossfit_scores(g2, g3, g4, np.random.default_rng(3), stride=1)
    b = fo.crossfit_scores(g2, g3, g4, np.random.default_rng(3), stride=1)
    c = fo.crossfit_scores(g2, g3, g4, np.random.default_rng(4), stride=1)
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[3]["fold"], b[3]["fold"])
    assert not np.array_equal(a[3]["fold"], c[3]["fold"])
    assert np.bincount(a[3]["fold"]).tolist() == [4, 4]
    with pytest.raises(ValueError):
        fo.crossfit_scores(g2[:1], g3[:1], g4[:1], np.random.default_rng(0))


def test_crossfit_keeps_a_real_response():
    # Break caught: a cross-fitting that also destroys a genuine signal. With a true shift (Mahalanobis distance 0.66) the
    # cross-fitted response stays close to the in-sample one and to the distance, unlike the null case above.
    K, n_c, T = 15, 60, 40
    Sigma, v = _cov(K, 1)
    delta = np.random.default_rng(2).normal(size=K); delta = 0.8 * delta / np.linalg.norm(delta) + 0.5 * v
    L = np.linalg.cholesky(Sigma)
    r = np.random.default_rng(3)
    g4 = r.normal(size=(n_c, T, K)) @ L.T
    g2 = r.normal(size=(n_c, T, K)) @ L.T + delta
    info = fo.crossfit_scores(g2, g4.copy(), g4, np.random.default_rng(0), stride=1)[3]
    assert info["response_crossfit"] > 0.8 * np.sqrt(delta @ np.linalg.solve(Sigma, delta)) * 0.9
    assert info["response_crossfit"] > 0.85 * np.min(info["response_in_sample"])


# ------------------------------------------------------------------------------------ exact null
def test_on_fraction_of_iid_markov_junctions_satisfies_the_relation_for_a_perturbation_of_both_rates():
    # Exact null: the ON count of K iid identical Markov junctions is a lumpable Markov chain, so delta2(m_K) is 0 up to the
    # finite-M noise. Perturbation of BOTH the SET and the RESET rate (stages I-II at another V and I).
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, 0.36, 3.3e-3, 800, 12, 60, 200, N_LAGS + 1, 200, 31)
    x2, x3, x4 = (fo.on_fraction(g) for g in (g2, g3, g4))
    assert abs(x2.mean() - x4.mean()) > 0.3 * x4.std()                                   # the perturbation is really felt
    res = ft.continuous_fdtest(x2, x3, x4)
    floor = ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(1), n_splits=10)
    boot = ft.bootstrap_continuous_delta2(x2, x3, x4, 40, np.random.default_rng(2))
    vs = ft.violation_summary(res["delta2"], boot, floor)
    assert res["delta2"] < 3 * floor and not vs["excludes_floor"]                      # at the noise floor (both are ~0.03 here)
    assert res["delta2"] < 0.06


def test_on_fraction_of_fixed_lifetime_junctions_violates_the_relation_positive_control():
    # Positive control of the exact-null test: K iid junctions with a FIXED ON lifetime carry a hidden age, so m_K is not a
    # lumpable Markov observable and a RESET-side perturbation (stage I-II at another I) must give delta2(m_K) far above the floor.
    params = dataclasses.replace(PRESET, lifetime_steps=8)
    g2, g3, g4 = _mc_cycles(params, V0, I0, V0, 6e-3, 1000, 8, 60, 200, N_LAGS + 1, 200, 31, lifetime=True)
    x2, x3, x4 = (fo.on_fraction(g) for g in (g2, g3, g4))
    assert abs(x2.mean() - x4.mean()) > 0.5 * x4.std()
    res = ft.continuous_fdtest(x2, x3, x4)
    floor = ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(1), n_splits=10)
    boot = ft.bootstrap_continuous_delta2(x2, x3, x4, 30, np.random.default_rng(2))
    vs = ft.violation_summary(res["delta2"], boot, floor)
    assert res["delta2"] > 0.3 and res["delta2"] > 10 * floor and vs["excludes_floor"]


# ------------------------------------------------------------------------------------ weight map and per-observable analysis
def _info(w0, w1):
    return {"w": np.array([w0, w1], float), "response_in_sample": np.array([1.0, 1.0]), "response_crossfit": 0.5}


def test_weight_map_averages_the_unit_fold_weights_and_reports_hand_derived_associations():
    w0, w1 = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 3.0, 4.0, 0.0])
    u0, u1 = w0 / np.linalg.norm(w0), w1 / np.linalg.norm(w1)
    m = drv.weight_map(_info(w0, w1), strength=np.array([1.0, 2.0, 3.0, 4.0]), share=np.array([0.4, 0.3, 0.2, 0.1]),
                       backbone=np.array([0.0, 0.2, 0.9, 1.0]))
    expected = (u0 + u1) / np.linalg.norm(u0 + u1)
    np.testing.assert_allclose(m["w"], expected)
    assert m["fold_cosine"] == pytest.approx(float(u0 @ u1)) and m["fold_cosine"] > 0.3
    assert expected[0] < expected[1] < expected[2] and expected[2] > expected[3]         # w is not monotone in strength ...
    assert m["spearman_strength"] == pytest.approx(0.4)                                  # ... ranks (1, 3, 4, 2) vs (1, 2, 3, 4): 1 - 6 * 6 / 60
    assert m["spearman_current_share"] == pytest.approx(-0.4)
    assert m["backbone_fraction_weighted"] == pytest.approx(np.abs(expected)[2:].sum() / np.abs(expected).sum())
    assert m["backbone_fraction_recorded"] == 0.5 and m["associations_blanked"] is False


def test_weight_map_blanks_the_associations_when_the_folds_disagree():
    # Two orthogonal fold weight vectors (cosine 0 < 0.3): the average is noise, so the associations are null and flagged.
    m = drv.weight_map(_info([1.0, 0, 0, 0], [0, 1.0, 0, 0]), strength=np.arange(4.0), share=np.arange(4.0), backbone=np.arange(4.0) / 3)
    assert m["fold_cosine"] == pytest.approx(0.0)
    for k in ("spearman_strength", "spearman_current_share", "spearman_backbone", "spearman_abs_strength", "spearman_abs_current_share",
              "backbone_fraction_weighted"):
        assert np.isnan(m[k])
    assert m["associations_blanked"] is True and "fold cosine" in m["associations_note"]
    np.testing.assert_allclose(m["w"], np.array([1, 1, 0, 0]) / np.sqrt(2))              # the map itself is still returned
    zero = drv.weight_map(_info([0.0] * 4, [0.0] * 4), np.arange(4.0), np.arange(4.0), np.arange(4.0))    # degenerate: no weights at all
    assert np.isnan(zero["fold_cosine"]) and zero["associations_blanked"] is True


def test_analyze_observable_reports_defined_values_for_markov_and_hidden_variable_data():
    # A Markov AR observable: response defined and positive, no violation. The same system with a hidden slow variable: violation.
    edges = ft.lag_bin_edges(60)
    markov = drv.analyze_observable(*_ar_cycles(600, hidden=False, seed=3, T2=300, T4=300), 10, np.random.default_rng(0), edges)
    assert 0.4 < markov["achieved_response_sigma"] < 0.9 and markov["z0_snr"] > 3 and np.isfinite(markov["noise_corr_length"])
    for variant in ("unbinned", "binned"):
        v = markov[variant]
        assert np.isfinite(v["delta2"]) and v["delta2"] < 0.1 and v["excludes_floor"] is False
        assert v["ci95_reverse"][0] <= v["delta2"] <= v["ci95_reverse"][1] + 1.0 and np.isfinite(v["noise_floor"]) and v["tau_c"] < 60
    assert markov["binned"]["cutoff_at_end"] in (True, False) and markov["z_mean"].shape == (60,) and markov["c_zz"].shape == (60,)
    hidden = drv.analyze_observable(*_ar_cycles(1600, hidden=True, seed=3, T2=300, T4=300), 10, np.random.default_rng(0), edges)
    for variant in ("unbinned", "binned"):
        v = hidden[variant]
        assert v["delta2"] > 0.15 and v["excludes_floor"] is True and v["excess"] > 0.1
    assert hidden["unbinned"]["delta2"] > 5 * markov["unbinned"]["delta2"]


def test_analyze_observable_degenerate_observable_gives_nan_and_fails_closed():
    x = np.zeros((6, 10)), np.zeros((6, 8)), np.zeros((6, 10))
    out = drv.analyze_observable(*x, 5, np.random.default_rng(0), ft.lag_bin_edges(8))
    assert np.isnan(out["unbinned"]["delta2"]) and out["unbinned"]["excludes_floor"] is False and out["binned"]["excludes_floor"] is False


# ------------------------------------------------------------------------------------ smoke test of the driver
def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    """Tiny N = 150 chain (does not percolate: g_eff ~ 0), two seeds of runs, produced as in test_fdtest_smoke."""
    out = str(tmp_path_factory.mktemp("fdtest_obs_smoke"))
    for seed in ("1", "2"):
        r = _run("scripts.fdtest_run", "150", "--v0", "1.5", "--eps", "0.1", "--t-relax", "100", "--t-samp", "100",
                 "--cycles", "4", "--seed", seed, "--k", "30", "--record-every", "10", "--n-lags", "8", "--out", out)
        assert r.returncode == 0, r.stderr
    return Path(out)


def test_driver_runs_on_the_small_chain_and_writes_strict_json_and_figures(runs_dir):
    r = _run("scripts.fdtest_observables", "150", "--out", str(runs_dir), "--n-boot", "8")
    assert r.returncode == 0, r.stderr
    assert "Warning" not in r.stderr and "Traceback" not in r.stderr
    text = (runs_dir / "fdtest_observables_N150.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    res = json.loads(text)
    assert set(res["observables"]) == {"geff", "m_K", "optimal"}
    for name, o in res["observables"].items():
        for variant in ("unbinned", "binned"):
            for key in ("delta2", "ci95", "ci95_reverse", "noise_floor", "excess", "excludes_floor", "tau_c", "cutoff_at_end"):
                assert key in o[variant], (name, variant, key)
            assert o[variant]["excludes_floor"] is False                                  # fail closed on undefined estimates
        for key in ("achieved_response_sigma", "z0_snr", "noise_corr_length"):
            assert key in o
    w = res["weights"]
    assert len(w["w"]) == res["K"] > 0 and {"spearman_strength", "spearman_current_share", "backbone_fraction_weighted", "fold_cosine", "associations_blanked", "associations_note"} <= set(w)
    assert "does not refit" in res["note"]
    assert res["M"] == 8
    for f in ("fdtest_weights_N150.png", "fdtest_observables_N150.png"):
        assert (runs_dir / f).stat().st_size > 1000
    assert "m_K" in r.stdout and "optimal" in r.stdout and "binned" in r.stdout


def test_driver_rejects_inconsistent_inputs_with_status_1(runs_dir, tmp_path):
    import shutil
    for f in runs_dir.glob("run_N150_seed*.npz"):
        shutil.copy(f, tmp_path / f.name)
    target = tmp_path / "run_N150_seed2.npz"
    z = dict(np.load(target))
    z["sampled_junction_ids"] = z["sampled_junction_ids"][::-1].copy()
    np.savez_compressed(target, **z)
    r = _run("scripts.fdtest_observables", "150", "--out", str(tmp_path), "--n-boot", "8")
    assert r.returncode == 1 and "sampled_junction_ids" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "fdtest_observables_N150.json").exists()
