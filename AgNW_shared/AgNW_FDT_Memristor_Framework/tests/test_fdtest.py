import dataclasses

import numpy as np
import pytest

from agnw_sim import fdtest as ft
from agnw_sim.benchmarks import two_state_fdtest, renewal_fdtest
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics, update_dynamics_lifetime
from conftest import _mc_cycles, _ar_cycles, PRESET, V0, I0, N_LAGS   # shared Monte-Carlo helpers (tests/conftest.py)

# ------------------------------------------------------------------------------------ core helpers
def test_select_cutoff_ignores_a_single_low_lag_and_needs_a_run():
    # Break caught: a cutoff triggered by one noisy lag (would truncate the relaxation window).
    z = [1.0, 0.5, 0.04, 0.5, 0.04, 0.03, 0.02, 0.5]
    assert ft.select_cutoff(z, frac=0.05, run=3) == 4                 # lags 4, 5, 6 are the first run of 3
    assert ft.select_cutoff(z, frac=0.05, run=1) == 2                 # run=1 -> the first low lag


def test_select_cutoff_falls_back_to_the_last_lag():
    assert ft.select_cutoff([1.0, 0.9, 0.8, 0.7], 0.05, 3) == 3       # never decays
    assert ft.select_cutoff([0.0, 0.0, 0.0], 0.05, 3) == 2            # no response at all
    assert ft.select_cutoff([np.nan, 1.0], 0.05, 3) == 1


def test_delta_squared_hand_values():
    # sum (c - z)^2 / sum z^2 over lags 0..tau_c
    assert ft.delta_squared([1, 0.5, 0.25], [1, 0.5, 0.25], 2) == 0.0
    assert ft.delta_squared([1, 1, 1], [1, 0, 0], 2) == pytest.approx(2.0)      # (0 + 1 + 1) / 1
    assert ft.delta_squared([1, 1, 1], [1, 0, 0], 0) == 0.0                     # only lag 0 counts
    assert np.isnan(ft.delta_squared([1, 1], [0, 0], 1))


def test_masked_autocorr_hand_values_with_missing_samples():
    # z = [1, 2, 3]: lag 0 = (1 + 4 + 9)/3, lag 1 = (1*2 + 2*3)/2
    np.testing.assert_allclose(ft.masked_autocorr([[1.0, 2.0, 3.0]], 1), [14 / 3, 4.0])
    # z = [1, nan, 3, 4]: lag 0 = (1 + 9 + 16)/3; lag 1 only the pair (3, 4) = 12; lag 2 only the pair (1, 3) = 3
    np.testing.assert_allclose(ft.masked_autocorr([[1.0, np.nan, 3.0, 4.0]], 2), [26 / 3, 12.0, 3.0])


# ------------------------------------------------------------------------------------ binary observable
def test_binary_curves_match_a_hand_derived_example():
    # Junction 0 (all numbers derived by hand): p_eps = 0.375, p_0 = 0.5 -> Z(OFF) = 0.25, Z(ON) = -0.25.
    # Junction 1 is always OFF and must be excluded (p_0 = 0 < p_min).
    g2 = np.zeros((2, 4, 2), np.uint8); g2[0, :, 0] = [1, 1, 0, 0]; g2[1, :, 0] = [1, 0, 0, 0]
    g4 = np.zeros((2, 4, 2), np.uint8); g4[0, :, 0] = [1, 0, 1, 0]; g4[1, :, 0] = [0, 0, 1, 1]
    g3 = np.zeros((2, 3, 2), np.uint8); g3[0, :, 0] = [1, 1, 0]; g3[1, :, 0] = [0, 1, 0]
    stats = ft.binary_stage_stats(g2, g3, g4, max_lag=2)
    out = ft.binary_curves(stats)
    np.testing.assert_array_equal(out["valid"], [True, False])
    assert out["alpha"][0] == pytest.approx(0.25) and out["beta"][0] == pytest.approx(-0.5)
    # <Z(tau)> = alpha + beta * mean_c g3 = 0.25 - 0.5 * [0.5, 1, 0]
    np.testing.assert_allclose(out["z_mean"][:, 0], [0.0, -0.25, 0.25], atol=1e-7)
    # C_ZZ(l) = -0.0625 + 0.25 * pair(l), pair = mean over cycles of the lag products: [1/2, 1/6, 1/4]
    np.testing.assert_allclose(out["c_zz"][:, 0], [0.0625, -1 / 48, 0.0], atol=1e-7)
    assert np.isnan(out["z_mean"][:, 1]).all() and np.isnan(out["c_zz"][:, 1]).all()


def test_binary_stage_stats_rejects_inconsistent_shapes():
    g = np.zeros((2, 6, 3), np.uint8)
    with pytest.raises(ValueError):
        ft.binary_stage_stats(g, np.zeros((2, 4, 3), np.uint8), g, max_lag=2)          # 4 lags, expected 3
    with pytest.raises(ValueError):
        ft.binary_stage_stats(g, np.zeros((3, 3, 3), np.uint8), g, max_lag=2)          # cycle count mismatch
    with pytest.raises(ValueError):
        ft.binary_stage_stats(g, np.zeros((2, 7, 3), np.uint8), g, max_lag=6)          # lag >= stage IV length


def test_class_sums_are_additive_over_chunks_of_junctions():
    # Break caught: a chunked analysis that changes the class average (the driver processes junctions in chunks).
    rng = np.random.default_rng(0)
    z, c = rng.normal(size=(6, 10)), rng.normal(size=(6, 10))
    z[:, 3] = np.nan; c[:, 3] = np.nan                                       # an invalid junction
    masks = np.array([[1] * 5 + [0] * 5, [0] * 5 + [1] * 5], dtype=bool)
    full = ft.class_sums(z, c, masks)
    a = ft.class_sums(z[:, :4], c[:, :4], masks[:, :4])
    b = ft.class_sums(z[:, 4:], c[:, 4:], masks[:, 4:])
    for whole, pa, pb in zip(full, a, b):
        np.testing.assert_allclose(whole, pa + pb)
    assert full[2].tolist() == [4.0, 5.0]                                    # junction 3 not counted


def _rates(V, I):
    return float(p_set(np.array([V]), PRESET)[0]), float(p_reset(np.array([I]), PRESET)[0])


def _class_result(params, V1, I1, lifetime, n_cycles=3000, K=40, seed=21):
    g2, g3, g4 = _mc_cycles(params, V0, I0, V1, I1, n_cycles, K, 60, 300, N_LAGS + 1, 300, seed, lifetime)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    cur = ft.binary_curves(stats)
    masks = cur["valid"][None, :]
    d2 = ft.class_delta2(cur["z_mean"], cur["c_zz"], masks)[0][0]
    boot = ft.bootstrap_class_delta2(stats, masks, 100, np.random.default_rng(1))[:, 0]
    floor = np.mean([ft.split_half_floor(stats, masks, np.random.default_rng(s))[0] for s in range(5)])
    return d2, np.percentile(boot, [2.5, 97.5]), floor


def test_estimator_sees_no_violation_for_a_markov_junction():
    # Break caught: an estimator that reports memory for a memoryless variable (wrong Z, wrong alignment of the
    # stage-III lags with the stage-IV autocorrelation, or a biased normalization).
    d2, ci, floor = _class_result(PRESET, V0, 3.3e-3, lifetime=False)        # RESET-side perturbation
    assert d2 < 0.01
    assert ci[1] < 0.02
    assert floor < 0.01


def test_estimator_recovers_the_exact_violation_of_the_fixed_lifetime_junction():
    # Break caught: an estimator blind to a hidden variable, or a benchmark/estimator mismatch.
    params = dataclasses.replace(PRESET, lifetime_steps=8)
    s0, r0 = _rates(V0, I0)
    s1, r1 = _rates(V0, 3.3e-3)
    exact = renewal_fdtest(s0, r0, s1, r1, 8, N_LAGS)["delta2"]              # about 0.23
    d2, ci, floor = _class_result(params, V0, 3.3e-3, lifetime=True)
    assert ci[0] <= exact <= ci[1]
    assert d2 > 10 * floor and d2 > 0.1


def test_set_only_perturbation_cannot_expose_the_hidden_age():
    # Break caught: a spurious violation for a perturbation the benchmark says is exactly Markov-like.
    params = dataclasses.replace(PRESET, lifetime_steps=8)
    d2, ci, floor = _class_result(params, 0.36, I0, lifetime=True)
    assert d2 < 0.02


def test_delta2_falls_with_the_number_of_cycles_for_a_markov_junction():
    # Break caught: a finite-M bias that does not shrink (the paper's 1/M behaviour).
    vals = []
    for n_cycles in (100, 3000):
        g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, n_cycles, 40, 60, 300, N_LAGS + 1, 300, 5)
        cur = ft.binary_curves(ft.binary_stage_stats(g2, g3, g4, N_LAGS))
        vals.append(ft.class_delta2(cur["z_mean"], cur["c_zz"], cur["valid"][None, :])[0][0])
    assert vals[1] < vals[0] / 5


def test_bootstrap_sums_are_additive_over_chunks_of_junctions_and_match_the_wrapper():
    # Break caught: the chunked analysis (1000 junctions cannot be held at once) differing from the direct one.
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 300, 6, 60, 300, N_LAGS + 1, 300, 7)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.ones((1, 6), dtype=bool)
    w = ft.bootstrap_weights(300, 5, np.random.default_rng(3))
    whole = ft.bootstrap_class_sums(stats, w, masks)
    halves = []
    for sl in (slice(0, 3), slice(3, 6)):
        part = {k: (v[..., sl] if v.ndim == 3 else v[:, sl]) for k, v in stats.items()}
        halves.append(ft.bootstrap_class_sums(part, w, masks[:, sl]))
    for k in range(3):
        np.testing.assert_allclose(whole[k], halves[0][k] + halves[1][k], atol=1e-9)
    direct = np.stack([ft.delta2_from_sums(whole[0][b], whole[1][b], whole[2][b]) for b in range(5)])
    wrapped = ft.bootstrap_class_delta2(stats, masks, 5, np.random.default_rng(3))
    np.testing.assert_allclose(direct, wrapped)


def test_split_half_floor_sums_are_additive_over_chunks():
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 300, 6, 60, 300, N_LAGS + 1, 300, 8)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.ones((1, 6), dtype=bool)
    perm = np.random.default_rng(2).permutation(300)
    a, b = perm[:150], perm[150:]
    whole = ft.split_half_class_sums(stats, masks, a, b)
    # the Z mapping (alpha, beta) is per junction, so chunks of junctions are independent
    parts = [ft.split_half_class_sums({k: (v[..., sl] if v.ndim == 3 else v[:, sl]) for k, v in stats.items()},
                                      masks[:, sl], a, b) for sl in (slice(0, 3), slice(3, 6))]
    for k in range(4):
        np.testing.assert_allclose(whole[k], parts[0][k] + parts[1][k], atol=1e-9)
    assert ft.floor_from_sums(*whole)[0] >= 0


# ------------------------------------------------------------------------------------ continuous observable
def test_z_map_recovers_the_analytic_conjugate_variable_of_a_shifted_gaussian():
    # Break caught: a wrong normalization of P_eps / P_0 (e.g. swapped sample counts) or bin alignment.
    rng = np.random.default_rng(0)
    mu, sigma = 0.5, 1.0
    x_eps, x_0 = rng.normal(mu, sigma, 400_000), rng.normal(0.0, sigma, 250_000)     # unequal sizes: normalisation matters
    x = np.linspace(-1.5, 1.5, 61)
    z_true = np.exp(mu * x / sigma ** 2 - mu ** 2 / (2 * sigma ** 2)) - 1      # ratio of the two Gaussians
    for estimator in ("hist", "kde"):
        grid, z_grid = ft.z_map(x_eps, x_0, estimator=estimator)
        z = ft.z_eval(x, grid, z_grid)
        assert np.isfinite(z).all()
        assert np.abs(z - z_true).max() < 0.06


def test_z_map_marks_undersampled_bins_and_rejects_unknown_estimators():
    rng = np.random.default_rng(1)
    x_eps, x_0 = rng.normal(0, 1, 5000), rng.normal(3, 1, 5000)               # almost no overlap
    grid, z = ft.z_map(x_eps, x_0, n_bins=40, n_min=50)
    assert np.isnan(z).any()
    assert np.isnan(ft.z_eval(np.array([100.0]), grid, z)).all()               # outside the valid range
    with pytest.raises(ValueError):
        ft.z_map(x_eps, x_0, estimator="wavelet")


def test_continuous_fdtest_separates_a_markov_variable_from_one_with_a_hidden_variable():
    # The paper's validation: the fully observed system obeys the relation, omitting a variable violates it.
    markov = ft.continuous_fdtest(*_ar_cycles(3200, hidden=False, seed=3))
    hidden = ft.continuous_fdtest(*_ar_cycles(3200, hidden=True, seed=3))
    assert markov["delta2"] < 0.05
    assert hidden["delta2"] > 0.15
    assert hidden["delta2"] > 10 * markov["delta2"]
    assert markov["excluded_fraction"] < 0.05


def test_continuous_bootstrap_and_floor_shapes_and_reproducibility():
    x2, x3, x4 = _ar_cycles(400, hidden=False, seed=4)
    a = ft.bootstrap_continuous_delta2(x2, x3, x4, 8, np.random.default_rng(0))
    b = ft.bootstrap_continuous_delta2(x2, x3, x4, 8, np.random.default_rng(0))
    assert a.shape == (8,) and np.array_equal(a, b)
    f1 = ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(0))
    assert f1 >= 0 and f1 == ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(0))


def test_bootstrap_return_z0_shapes():
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 200, 4, 60, 300, N_LAGS + 1, 300, 9)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    d2, z0 = ft.bootstrap_class_delta2(stats, np.ones((2, 4), dtype=bool), 6, np.random.default_rng(0), return_z0=True)
    assert d2.shape == (6, 2) and z0.shape == (6, 2) and (z0 > 0).all()      # <Z(0)> = E_0[Z^2] > 0
    x2, x3, x4 = _ar_cycles(300, hidden=False, seed=6)
    d, z = ft.bootstrap_continuous_delta2(x2, x3, x4, 5, np.random.default_rng(0), return_z0=True)
    assert d.shape == (5,) and z.shape == (5,) and (z > 0).all()


class _IdentityPermutation:
    """Stub rng whose permutation is the identity, so that half A is exactly cycles 0..n//2-1."""
    @staticmethod
    def permutation(n):
        return np.arange(n)


def test_continuous_floor_ignores_lags_without_valid_samples():
    # Break caught: a NaN floor when, with few cycles, a half of the sample has no valid Z at some lag.
    x2, x3, x4 = _ar_cycles(12, hidden=False, seed=5)
    x3 = x3.copy()
    x3[:6, 3] = 1e6                     # exactly half A (cycles 0..5) leaves the valid Z range at lag 3
    floor = ft.continuous_split_half_floor(x2, x3, x4, _IdentityPermutation())
    assert np.isfinite(floor) and floor >= 0


def test_z_map_kde_degrades_to_nan_on_degenerate_input():
    # Break caught: gaussian_kde raising (singular covariance / too few samples) mid-analysis.
    for xe, x0 in ((np.full(100, 1.0), np.random.default_rng(0).normal(size=100)),
                   (np.random.default_rng(1).normal(size=100), np.array([0.3])),
                   (np.array([0.3]), np.array([0.3]))):
        grid, z = ft.z_map(xe, x0, estimator="kde")
        assert np.isnan(z).all() and grid.shape == z.shape



def test_floor_from_sums_is_a_quarter_of_the_half_difference_over_the_full_norm():
    # Break caught: a dropped 1/4 (or a wrong normalisation / window) in the finite-M noise floor.
    zf = np.array([1.0, 0.5, 0.01, 0.01, 0.01, 0.5])           # tau_c = 2 (lags 2, 3, 4 are the first run below 5 %)
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])              # half A = zf + d, half B = zf - d  ->  (za - zb)^2 = 4 d^2
    n = 4.0
    floor = ft.floor_from_sums(np.array([n * zf]), np.array([n * (zf + d)]), np.array([n * (zf - d)]), np.array([n]))
    expected = (0.1 ** 2 + 0.2 ** 2 + 0.3 ** 2) / (1.0 + 0.25 + 0.01 ** 2)      # sum_{lag<=2} d^2 / sum_{lag<=2} zf^2
    assert floor[0] == pytest.approx(expected, rel=1e-12)
    # an empty class gives NaN
    empty = ft.floor_from_sums(np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)), np.array([0.0]))
    assert np.isnan(empty[0])


def test_bootstrap_weights_are_normalised_integer_multiples():
    # Break caught: resampling weights that do not sum to one or are not multinomial counts / n.
    n_cycles, n_boot = 20, 4000
    w = ft.bootstrap_weights(n_cycles, n_boot, np.random.default_rng(0))
    assert w.shape == (n_boot, n_cycles)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    counts = w * n_cycles
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    assert (w >= 0).all()
    np.testing.assert_allclose(w.mean(axis=0), 1.0 / n_cycles, atol=5e-3)   # each cycle is drawn 1/n of the time


def test_bootstrap_return_z0_is_the_class_mean_of_z_at_lag_zero_and_nan_for_an_empty_class():
    # Break caught: a wrong <Z(0)> (SNR numerator) or a division warning / non-NaN for a class without junctions.
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 200, 6, 60, 300, N_LAGS + 1, 300, 11)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.array([[True] * 6, [False] * 6])
    d2, z0 = ft.bootstrap_class_delta2(stats, masks, 7, np.random.default_rng(4), return_z0=True)
    assert d2.shape == z0.shape == (7, 2)
    plain = ft.bootstrap_class_delta2(stats, masks, 7, np.random.default_rng(4))
    np.testing.assert_allclose(d2, plain)                                     # same draws, same delta2
    w = ft.bootstrap_weights(200, 7, np.random.default_rng(4))
    for b in range(7):
        cur = ft.binary_curves(stats, w[b])
        v = cur["valid"]
        assert v.any()
        assert z0[b, 0] == pytest.approx(cur["z_mean"][0, v].mean(), rel=1e-12)
    assert np.isnan(z0[:, 1]).all()


# ------------------------------------------------------------------------------------ final-review fix wave
def test_violation_summary_hand_values_and_reverse_percentile_interval():
    # Break caught: a percentile interval used as the decision rule (it double counts the finite-M bias of the
    # estimator); the reverse interval 2 d - q reflects the bootstrap around the estimate instead.
    boot = np.arange(101) / 100.0                                   # percentiles: q2.5 = 0.025, q97.5 = 0.975 exactly
    s = ft.violation_summary(0.6, boot, 0.2)
    assert s["delta2"] == 0.6
    np.testing.assert_allclose(s["ci95_percentile"], [0.025, 0.975])
    np.testing.assert_allclose(s["ci95_reverse"], [2 * 0.6 - 0.975, 2 * 0.6 - 0.025])       # [0.225, 1.175]
    assert s["excess"] == pytest.approx(0.4)
    assert s["excludes_floor"] is True                               # reverse lower bound 0.225 > floor 0.2
    assert ft.violation_summary(0.6, boot, 0.3)["excludes_floor"] is False                 # 0.225 < 0.3
    # the percentile lower bound (0.025) would have excluded a floor of 0.02 while the reverse bound does not exclude 0.3


def test_violation_summary_fails_closed_on_non_finite_input_without_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for d, boot, floor in ((np.nan, np.arange(11) / 10, 0.1), (0.5, np.full(11, np.nan), 0.1),
                               (0.5, np.arange(11) / 10, np.nan), (0.5, np.array([]), 0.1)):
            s = ft.violation_summary(d, boot, floor)
            assert s["excludes_floor"] is False
        s = ft.violation_summary(0.5, np.r_[np.arange(11) / 10, np.nan, np.nan], 0.1)       # NaN bootstrap draws are dropped
        np.testing.assert_allclose(s["ci95_percentile"], [0.025, 0.975])


def test_split_half_floor_n_splits_one_is_the_old_value_and_n_splits_averages_the_splits():
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 300, 6, 60, 300, N_LAGS + 1, 300, 8)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.ones((1, 6), dtype=bool)
    rng = np.random.default_rng(3)
    old = [ft.floor_from_sums(*ft.split_half_class_sums(stats, masks, p[:150], p[150:]))[0]
           for p in (rng.permutation(300) for _ in range(3))]
    np.testing.assert_allclose(ft.split_half_floor(stats, masks, np.random.default_rng(3))[0], old[0], rtol=1e-12)
    np.testing.assert_allclose(ft.split_half_floor(stats, masks, np.random.default_rng(3), n_splits=3)[0],
                               np.mean(old), rtol=1e-12)


def test_split_half_floor_averaging_over_splits_reduces_its_variability():
    # Break caught: a floor taken from ONE random split (it ranged 5x across replicas).
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 300, 6, 60, 300, N_LAGS + 1, 300, 8)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.ones((1, 6), dtype=bool)
    one = [ft.split_half_floor(stats, masks, np.random.default_rng(s))[0] for s in range(12)]
    many = [ft.split_half_floor(stats, masks, np.random.default_rng(s), n_splits=20)[0] for s in range(12)]
    assert np.std(many) < 0.5 * np.std(one)


def test_continuous_floor_n_splits_one_is_the_old_value_and_averaging_reduces_variability():
    x2, x3, x4 = _ar_cycles(40, hidden=False, seed=9)
    rng = np.random.default_rng(3)
    full = ft.continuous_fdtest(x2, x3, x4)
    manual = []
    for _ in range(3):
        perm = rng.permutation(40)
        za, zb = np.nanmean(full["z3"][perm[:20]], axis=0), np.nanmean(full["z3"][perm[20:]], axis=0)
        ok = np.isfinite(za) & np.isfinite(zb) & np.isfinite(full["z_mean"])
        ok[full["tau_c"] + 1:] = False
        manual.append(0.25 * np.sum((za[ok] - zb[ok]) ** 2) / np.sum(full["z_mean"][ok] ** 2))
    assert ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3)) == pytest.approx(manual[0], rel=1e-12)
    assert ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3), n_splits=3) == pytest.approx(np.mean(manual), rel=1e-12)
    one = [ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(s)) for s in range(12)]
    many = [ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(s), n_splits=20) for s in range(12)]
    assert np.std(many) < 0.5 * np.std(one)


def test_z_map_kde_subsample_is_strided_across_all_cycles_not_the_head_of_the_array():
    # Break caught: xe[:50000] keeps only the first cycles (cycle-major ravel), which here follow a different
    # distribution than the rest of the data.
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    xe = np.r_[rng.normal(-2.0, 1.0, 50_000), rng.normal(0.5, 1.0, 350_000)]         # head = first cycles, shifted
    x0 = np.r_[rng.normal(2.0, 1.0, 50_000), rng.normal(0.0, 1.0, 150_000)]
    f_e, f_0 = 50_000 / 400_000, 50_000 / 200_000
    x = np.linspace(-1.5, 1.5, 31)
    p_e = f_e * norm.pdf(x, -2.0) + (1 - f_e) * norm.pdf(x, 0.5)
    p_0 = f_0 * norm.pdf(x, 2.0) + (1 - f_0) * norm.pdf(x, 0.0)
    grid, z_grid = ft.z_map(xe, x0, estimator="kde")
    z = ft.z_eval(x, grid, z_grid)
    assert np.isfinite(z).all()
    assert np.abs(z - (p_e / p_0 - 1)).max() < 0.1


def test_binary_shift_sigma_hand_values():
    # Break caught: a wrong normalisation of the achieved shift (binomial std of P_0) or a finite value at p_0 in {0, 1}.
    stats = {"m2": np.array([[0.6, 0.5, 0.3], [0.4, 0.5, 0.1]]), "m4": np.array([[0.4, 0.5, 0.0], [0.2, 0.5, 0.0]])}
    s = ft.binary_shift_sigma(stats)
    np.testing.assert_allclose(s[:2], [0.2 / np.sqrt(0.3 * 0.7), 0.0])          # p_eps = .5, p_0 = .3 ; p_eps = p_0 = .5
    assert np.isnan(s[2])                                                        # p_0 = 0: undefined


# ------------------------------------------------------------------------------------ lag binning
def test_lag_bin_edges_hand_values_and_invariants():
    # Break caught: a wrong bin scheme (first single-lag bins, geometric growth with a floor of one lag, clipping).
    # n_lags = 20, n_first = 4, ratio = 1.5: edges 0,1,2,3,4 (four single-lag bins); then
    #   max(4 + 1, floor(4 * 1.5 + .5) = 6) = 6;  max(7, floor(9 + .5) = 9) = 9;  max(10, floor(13.5 + .5) = 14) = 14;
    #   max(15, floor(21 + .5) = 21) = 21 -> clipped to 20.
    edges = ft.lag_bin_edges(20, n_first=4, ratio=1.5)
    assert edges.tolist() == [0, 1, 2, 3, 4, 6, 9, 14, 20]
    # ratio so small that the +1 floor rules: single-lag bins all the way
    assert ft.lag_bin_edges(6, n_first=2, ratio=1.01).tolist() == [0, 1, 2, 3, 4, 5, 6]
    # n_first larger than n_lags
    assert ft.lag_bin_edges(3, n_first=8).tolist() == [0, 1, 2, 3]
    for n in (1, 7, 100, 1400):
        e = ft.lag_bin_edges(n)
        assert e[0] == 0 and e[-1] == n and (np.diff(e) >= 1).all()
    assert len(ft.lag_bin_edges(1400)) - 1 < 60                       # about 45 bins for the default scheme


def test_bin_lags_hand_values_nan_aware_and_no_warnings():
    # Break caught: binning the wrong axis, a NaN lag poisoning its bin, or a RuntimeWarning for an all-NaN bin.
    import warnings
    x = np.array([[1.0, 3.0, np.nan, 5.0, np.nan, np.nan], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]])
    edges = np.array([0, 2, 5, 6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = ft.bin_lags(x, edges)
    np.testing.assert_allclose(b[0, :2], [2.0, 5.0])                  # mean(1, 3); mean(nan, 5, nan) = 5
    assert np.isnan(b[0, 2])                                          # all-NaN bin
    np.testing.assert_allclose(b[1], [1.0, 6.0, 10.0])
    assert b.shape == (2, 3)
    np.testing.assert_array_equal(np.diff(edges), [2, 3, 1])          # the bin widths n_b
    with pytest.raises(ValueError):
        ft.bin_lags(x, np.array([0, 2, 4]))                            # edges must end at the number of lags


def test_bin_scheme_arguments_and_edges_are_validated():
    # Break caught: an uncaught OverflowError / silent nonsense for a non-finite ratio, or an IndexError for degenerate edges.
    for kwargs in (dict(n_lags=0), dict(n_lags=10, n_first=0), dict(n_lags=10, ratio=1.0), dict(n_lags=10, ratio=float("inf")),
                   dict(n_lags=10, ratio=float("nan")), dict(n_lags=10, ratio=0.9)):
        with pytest.raises(ValueError):
            ft.lag_bin_edges(**kwargs)
    for bad in (np.array([]), np.array([0]), np.array([[0, 3]])):
        with pytest.raises(ValueError):
            ft.bin_lags(np.zeros(3), bad)


def test_binned_delta2_hand_values_with_a_nan_lag_and_cutoff_in_lag_units():
    # Break caught: unweighted bins, NaN lags poisoning the bin mean (or averaging Z and C over different lags), a cutoff in
    # bin units instead of lags.
    # z = [1, .8, nan, .4, .2, 0], c = [.9, .7, .6, .4, .3, 0], bins {0} {1, 2} {3, 4, 5}. Lag 2 (Z is NaN) is dropped from
    # BOTH curves, so bin 1 holds the single lag 1: n_b = [1, 1, 3], Zb = [1, .8, .2], Cb = [.9, .7, 0.7 / 3]; no cutoff:
    #   d2 = (1 * .1^2 + 1 * .1^2 + 3 * (0.7/3 - .2)^2) / (1 * 1 + 1 * .64 + 3 * .04)
    z = np.array([[1.0, 0.8, np.nan, 0.4, 0.2, 0.0]])
    c = np.array([[0.9, 0.7, 0.6, 0.4, 0.3, 0.0]])
    edges = np.array([0, 1, 3, 6])
    d2, tau = ft.delta2_from_sums(z, c, np.array([1.0]), edges=edges, return_tau=True)
    expected = (1 * 0.1 ** 2 + 1 * 0.1 ** 2 + 3 * (0.7 / 3 - 0.2) ** 2) / (1 + 0.64 + 3 * 0.04)
    assert d2[0] == pytest.approx(expected, rel=1e-12)
    assert tau[0] == 5                                                # last lag of the last bin (never fell below 5 %)
    # cutoff: Zb = [1, .5, .01, .01, .01] from bins {0} {1} {2, 3} {4, 5} {6, 7}: below 5 % from bin 2 (3 bins) -> bc = 2,
    # cutoff lag = last lag of bin 2 = 3; C = Z + .1 in bin 0 only:  d2 = .1^2 / (1 + .25 + 2 * .01^2)
    z = np.array([[1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
    c = z.copy(); c[0, 0] += 0.1
    d2, tau = ft.delta2_from_sums(z, c, np.array([1.0]), edges=np.array([0, 1, 2, 4, 6, 8]), return_tau=True)
    assert d2[0] == pytest.approx(0.1 ** 2 / (1 + 0.25 + 2 * 0.01 ** 2), rel=1e-12)
    assert tau[0] == 3


def test_binned_floor_from_sums_hand_values_with_unequal_bin_widths_cutoff_and_a_nan_lag():
    # Break caught: a binned floor that drops the bin widths n_b, ignores the cutoff bin, or averages the three curves over
    # different lags. Sums are n * curve with n = 4 junctions; half A = zf + d, half B = zf - d, so ZA - ZB = 2 d.
    n = np.array([4.0])
    floor = lambda zf, d, edges: ft.floor_from_sums(n * zf[None], n * (zf + d)[None], n * (zf - d)[None], n, edges=edges)[0]
    # (1) bins {0} {1, 2} {3, 4, 5}, widths 1, 2, 3, no cutoff (Zb = [1, .7, .2] never below 5 %):
    #   d = [.1, .2, 0, .3, .1, .2] -> d_b = [.1, .1, .2], (ZA - ZB)_b = [.2, .2, .4], Zf_b = [1, .7, .2]
    #   floor = 1/4 (1 * .2^2 + 2 * .2^2 + 3 * .4^2) / (1 * 1 + 2 * .7^2 + 3 * .2^2) = 1/4 * .6 / 2.1
    zf = np.array([1.0, 0.8, 0.6, 0.2, 0.2, 0.2])
    d = np.array([0.1, 0.2, 0.0, 0.3, 0.1, 0.2])
    assert floor(zf, d, np.array([0, 1, 3, 6])) == pytest.approx(0.25 * 0.6 / 2.1, rel=1e-12)
    # (2) cutoff: zf = [1, .5, .01 x 6], bins {0} {1} {2, 3} {4, 5} {6, 7}: Zf_b = [1, .5, .01, .01, .01] -> cutoff bin 2;
    #   d = [.1, .2, .3, .1, .5, .5, .9, .9] -> d_b = [.1, .2, .2, .5, .9]; bins 0..2 only, (ZA - ZB)_b = [.2, .4, .4]
    #   floor = 1/4 (1 * .2^2 + 1 * .4^2 + 2 * .4^2) / (1 + .25 + 2 * .01^2) = 1/4 * .52 / 1.2502
    zf = np.array([1.0, 0.5] + [0.01] * 6)
    d = np.array([0.1, 0.2, 0.3, 0.1, 0.5, 0.5, 0.9, 0.9])
    assert floor(zf, d, np.array([0, 1, 2, 4, 6, 8])) == pytest.approx(0.25 * 0.52 / 1.2502, rel=1e-12)
    # (3) a NaN lag in the full curve is dropped from all three curves (its A and B values, 5 and 7, must not enter):
    #   zf = [1, .8, nan, .2, .2, .2], bins {0} {1, 2} {3, 4, 5}: n_b = [1, 1, 3], Zf_b = [1, .8, .2], (ZA - ZB)_b = [.2, .4, .4]
    #   floor = 1/4 (1 * .2^2 + 1 * .4^2 + 3 * .4^2) / (1 + .64 + 3 * .04) = 1/4 * .68 / 1.76
    zf = np.array([1.0, 0.8, np.nan, 0.2, 0.2, 0.2])
    za = np.array([1.1, 1.0, 5.0, 0.5, 0.3, 0.4])
    zb = np.array([0.9, 0.6, 7.0, -0.1, 0.1, 0.0])
    got = ft.floor_from_sums(n * zf[None], n * za[None], n * zb[None], n, edges=np.array([0, 1, 3, 6]))[0]
    assert got == pytest.approx(0.25 * 0.68 / 1.76, rel=1e-12)


def _reference_binned_floor(za, zb, zf, edges, bc):
    """Plain-loop binned floor from per-lag half curves: only lags where all three curves are finite are used, n_b is their
    count in the bin, bins 0..bc."""
    num = den = 0.0
    for b in range(bc + 1):
        lags = [t for t in range(edges[b], edges[b + 1]) if np.isfinite(za[t]) and np.isfinite(zb[t]) and np.isfinite(zf[t])]
        if lags:
            num += len(lags) * (np.mean([za[t] for t in lags]) - np.mean([zb[t] for t in lags])) ** 2
            den += len(lags) * np.mean([zf[t] for t in lags]) ** 2
    return 0.25 * num / den


def test_continuous_binned_floor_matches_a_plain_loop_reference_including_a_nan_lag_in_a_bin():
    # Break caught: a continuous binned floor that drops the bin widths, ignores the cutoff bin or the joint NaN mask.
    x2, x3, x4 = _ar_cycles(20, hidden=False, seed=5)
    x3 = x3.copy()
    x3[:10, 3] = 1e6                          # half A (cycles 0..9 with the identity stub) has no valid Z at lag 3
    edges = ft.lag_bin_edges(x3.shape[1], n_first=2, ratio=1.7)
    assert (np.diff(edges) > 1).any() and edges[1] <= 3 < edges[-2]                        # lag 3 sits inside a wide bin
    floor, za, zb = ft.continuous_split_half_floor(x2, x3, x4, _IdentityPermutation(), edges=edges, return_halves=True)
    full = ft.continuous_fdtest(x2, x3, x4, edges=edges)
    assert np.isnan(za[3]) and np.isfinite(zb[3])
    expected = _reference_binned_floor(za, zb, full["z_mean"], edges, full["cutoff_bin"])
    assert floor == pytest.approx(expected, rel=1e-12)


def test_continuous_fdtest_with_edges_reports_cutoff_bin_cutoff_lag_and_cutoff_at_end():
    # Break caught: a cutoff in bin units reported as lags, or a wrong cutoff-at-end flag (delta2 truncated).
    x2, x3, x4 = _ar_cycles(300, hidden=False, seed=3)                 # relaxes within the 60-lag window (cutoff at lag 31)
    edges = ft.lag_bin_edges(x3.shape[1], n_first=4, ratio=1.5)
    res = ft.continuous_fdtest(x2, x3, x4, edges=edges)
    ref_d2, ref_tau = _bin_reference(res["z_mean"], res["c_zz"], edges.tolist())
    assert res["delta2"] == pytest.approx(ref_d2, rel=1e-12)
    assert res["tau_c"] == ref_tau                                    # last lag of the cutoff bin
    assert res["cutoff_bin"] == np.searchsorted(edges, ref_tau, side="right") - 1
    assert res["tau_c"] == edges[res["cutoff_bin"] + 1] - 1 and edges[res["cutoff_bin"] + 1] - edges[res["cutoff_bin"]] > 1   # n_b > 1
    assert res["cutoff_at_end"] is False and res["tau_c"] < x3.shape[1] - 1
    # a window too short for the relaxation to decay: the cutoff is the last bin and the flag is set
    short = ft.continuous_fdtest(x2, x3[:, :5], x4, edges=ft.lag_bin_edges(5, n_first=1, ratio=2.0))
    assert short["cutoff_at_end"] is True and short["tau_c"] == 4 and short["cutoff_bin"] == len(ft.lag_bin_edges(5, 1, 2.0)) - 2


def test_edges_none_is_the_old_behaviour_and_single_lag_bins_reproduce_it():
    # Break caught: an `edges` argument that changes the default results, or bin weights that break the n_b = 1 limit.
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 300, 6, 60, 300, N_LAGS + 1, 300, 8)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.array([[True] * 6, [True, True, True, False, False, False]])
    cur = ft.binary_curves(stats)
    single = np.arange(N_LAGS + 2)                                    # every lag its own bin
    a = ft.class_delta2(cur["z_mean"], cur["c_zz"], masks)
    same = ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, edges=None)
    np.testing.assert_array_equal(a[0], same[0])
    np.testing.assert_array_equal(a[1], same[1])
    one = ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, edges=single)
    np.testing.assert_allclose(a[0], one[0], rtol=1e-12)
    np.testing.assert_array_equal(a[1], one[1])
    perm = np.random.default_rng(2).permutation(300)
    sums = ft.split_half_class_sums(stats, masks, perm[:150], perm[150:])
    f0 = ft.floor_from_sums(*sums)
    np.testing.assert_array_equal(f0, ft.floor_from_sums(*sums, edges=None))
    np.testing.assert_allclose(f0, ft.floor_from_sums(*sums, edges=single), rtol=1e-12)
    x2, x3, x4 = _ar_cycles(40, hidden=True, seed=4)
    r0 = ft.continuous_fdtest(x2, x3, x4)
    r1 = ft.continuous_fdtest(x2, x3, x4, edges=np.arange(x3.shape[1] + 1))
    assert r1["delta2"] == pytest.approx(r0["delta2"], rel=1e-12) and r1["tau_c"] == r0["tau_c"]
    assert set(r0) <= set(r1)
    assert ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3)) == pytest.approx(
        ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3), edges=np.arange(x3.shape[1] + 1)), rel=1e-12)


def _bin_reference(z, c, edges):
    """Independent (plain loops) binned delta2 and cutoff lag from exact curves; the cutoff rule is the documented one:
    first bin from which |Zb| stays below 5 % of |Zb[0]| for 3 consecutive bins, else the last bin."""
    nb = len(edges) - 1
    zb, cb, w = [], [], []
    for b in range(nb):
        zs, cs = 0.0, 0.0
        for t in range(edges[b], edges[b + 1]):
            zs += z[t]; cs += c[t]
        n = edges[b + 1] - edges[b]
        zb.append(zs / n); cb.append(cs / n); w.append(n)
    bc = nb - 1
    for i in range(nb - 2):
        if all(abs(zb[j]) < 0.05 * abs(zb[0]) for j in (i, i + 1, i + 2)):
            bc = i
            break
    num = sum(w[b] * (cb[b] - zb[b]) ** 2 for b in range(bc + 1))
    den = sum(w[b] * zb[b] ** 2 for b in range(bc + 1))
    return num / den, edges[bc + 1] - 1


def test_binned_delta2_is_zero_for_the_markov_benchmark_and_exact_for_the_fixed_lifetime_junction():
    # Break caught: a binned estimator that reports memory for a Markov junction (unequal bin weights, misaligned C and Z),
    # or that does not reproduce the exact binned violation of the hidden-age junction.
    s0, r0 = _rates(V0, I0)
    s1, r1 = _rates(V0, 3.3e-3)
    n = N_LAGS + 1
    edges = ft.lag_bin_edges(n, n_first=4, ratio=1.5)
    assert edges.tolist() == [0, 1, 2, 3, 4, 6, 9, 14, 21, 31]
    mk = two_state_fdtest(s0, r0, s1, r1, N_LAGS)
    d2 = ft.delta2_from_sums(mk["z_mean"][None, :], mk["c_zz"][None, :], np.array([1.0]), edges=edges)[0]
    assert d2 < 1e-20
    lt = renewal_fdtest(s0, r0, s1, r1, 8, N_LAGS)
    d2, tau = ft.delta2_from_sums(lt["z_mean"][None, :], lt["c_zz"][None, :], np.array([1.0]), edges=edges, return_tau=True)
    ref, ref_tau = _bin_reference(lt["z_mean"], lt["c_zz"], [0, 1, 2, 3, 4, 6, 9, 14, 21, 31])
    assert d2[0] == pytest.approx(ref, rel=1e-12) and tau[0] == ref_tau
    assert ref > 0.05                                                 # a real violation survives the binning


FAST = SimParams.from_step_probs(1.0, 0.1, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3)   # relaxation within a few lags


def test_binned_floor_is_lower_than_the_unbinned_floor_for_white_scatter_and_delta2_bin_does_not_explode():
    # Break caught: binning that does not average down the lag-to-lag scatter of <Z> (floor unchanged), or a binned
    # delta2 that explodes for a memoryless variable. SCOPE: this covers ONLY the white-scatter regime (a fast Markov
    # junction with few cycles: the relaxation is over after a few lags and the 100-lag window is noise, the cutoff falls at
    # the last lag). It does NOT cover the correlated-scatter regime (slow relaxation), where binning gains little
    # (lag_noise_correlation reports it), and it does not show delta2_bin == floor: with noise in both Z and the
    # normalisation delta2_bin sits about 1.5 x above the floor here (0.30 against 0.20; 0.29 / 0.18 and 0.03 / 0.11 with
    # other seeds), so only a loose upper bound is asserted. The exact statements are the benchmark and hand-value tests.
    n_lags = 100
    g2, g3, g4 = _mc_cycles(FAST, V0, I0, V0, 3.3e-3, 40, 40, 60, 300, n_lags + 1, 300, 21)
    stats = ft.binary_stage_stats(g2, g3, g4, n_lags)
    cur = ft.binary_curves(stats)
    masks = cur["valid"][None, :]
    edges = ft.lag_bin_edges(n_lags + 1, n_first=4, ratio=1.5)
    fl = np.mean([ft.split_half_floor(stats, masks, np.random.default_rng(s))[0] for s in range(10)])
    fb = np.mean([ft.split_half_floor(stats, masks, np.random.default_rng(s), edges=edges)[0] for s in range(10)])
    d_b = ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, edges=edges)[0][0]
    assert fb < 0.7 * fl                                              # measured: 0.20 against 0.37
    assert d_b < 3 * fb + 0.05                                        # measured 0.30 against 0.20: a loose "does not explode"
    perm = np.random.default_rng(0).permutation(40)
    sa, sb, n = *ft.split_half_class_sums(stats, masks, perm[:20], perm[20:])[1:3], cur["valid"].sum()
    assert ft.lag_noise_correlation(sa[0] / n, sb[0] / n)[1] < 3      # the scatter across lags is close to white


def test_lag_noise_correlation_hand_values():
    # Break caught: a wrong correlation length (missing 2x, wrong stopping rule, NaN lags mishandled).
    # d = [1, 1, -1, -1, 1, 1, -1, -1] (mean 0): raw0 = 1; raw1 = (1 - 1 + 1 - 1 + 1 - 1 + 1) / 7 = 1/7;
    # raw2 = -1 (all six pairs); rho1 = 1/7 > 0, rho2 = -1 <= 0 stops the sum -> length = 1 + 2/7.
    d = np.array([1.0, 1, -1, -1, 1, 1, -1, -1])
    rho, length = ft.lag_noise_correlation(d, np.zeros(8), max_lag=4)
    assert rho[0] == pytest.approx(1.0) and rho[1] == pytest.approx(1 / 7) and rho[2] == pytest.approx(-1.0)
    assert length == pytest.approx(1 + 2 / 7)
    # a NaN lag is dropped (pairs need both samples): d = [1, 1, nan, -1, -1, 1, 1, -1, -1]
    #   raw0 = 1; raw1 = (1 + 1 - 1 + 1 - 1 + 1) / 6 = 1/3; raw2 = -5 / 5 = -1  -> length = 1 + 2/3
    d = np.array([1.0, 1, np.nan, -1, -1, 1, 1, -1, -1])
    rho, length = ft.lag_noise_correlation(d, np.zeros(9), max_lag=3)
    assert rho[1] == pytest.approx(1 / 3) and rho[2] == pytest.approx(-1.0)
    assert length == pytest.approx(1 + 2 / 3)
    # white noise: first non-positive rho ends the sum immediately or nearly so
    w = np.random.default_rng(0).normal(size=20000)
    assert ft.lag_noise_correlation(w, np.zeros(20000))[1] < 1.15
    # a smooth, strongly correlated sequence has a long correlation length
    t = np.arange(400)
    assert ft.lag_noise_correlation(np.sin(2 * np.pi * t / 400), np.zeros(400))[1] > 10
    # too few finite lags: undefined
    assert np.isnan(ft.lag_noise_correlation(np.array([1.0, np.nan]), np.zeros(2))[1])


# ------------------------------------------------------------------------------------ fixed window (cutoff)
# `cutoff` = inclusive last lag of the analysis window. Without `edges` the window is the lags 0..cutoff (clipped to the last
# lag); with `edges` it is every bin that STARTS at a lag <= cutoff (so the bin containing the cutoff lag is used whole) and
# the reported tau_c is the last lag of the last used bin. cutoff=None keeps the data-dependent select_cutoff rule.
def test_delta2_from_sums_fixed_cutoff_hand_values_and_none_is_the_old_rule():
    # Break caught: a `cutoff` that is ignored (select_cutoff still decides), off by one (exclusive), or not clipped.
    # z = [1, .5, .01, .01, .01, .5]: the data-dependent cutoff is lag 2 (lags 2, 3, 4 are the first run below 5 %).
    # c = z + [0, 0, .1, 0, 0, .3]
    z = np.array([[1.0, 0.5, 0.01, 0.01, 0.01, 0.5]])
    c = z + np.array([[0.0, 0.0, 0.1, 0.0, 0.0, 0.3]])
    one = np.array([1.0])
    d_old, t_old = ft.delta2_from_sums(z, c, one, return_tau=True)
    assert t_old[0] == 2 and d_old[0] == pytest.approx(0.1 ** 2 / (1 + 0.25 + 0.01 ** 2), rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, cutoff=None)
    np.testing.assert_array_equal(d, d_old); np.testing.assert_array_equal(t, t_old)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, cutoff=5)                  # whole window, inclusive last lag
    assert t[0] == 5 and d[0] == pytest.approx((0.1 ** 2 + 0.3 ** 2) / (1 + 0.25 + 3 * 0.01 ** 2 + 0.25), rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, cutoff=1)                  # before the perturbed lag
    assert t[0] == 1 and d[0] == 0.0
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, cutoff=2)                  # inclusive: lag 2 is in
    assert d[0] == pytest.approx(0.1 ** 2 / (1 + 0.25 + 0.01 ** 2), rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, cutoff=99)                 # beyond the window: clipped
    assert t[0] == 5 and d[0] == pytest.approx((0.1 ** 2 + 0.3 ** 2) / (1 + 0.25 + 3 * 0.01 ** 2 + 0.25), rel=1e-12)
    d, t = ft.delta2_from_sums(z * 3, c * 3, np.array([3.0]), return_tau=True, cutoff=5)     # sums / count
    assert d[0] == pytest.approx((0.1 ** 2 + 0.3 ** 2) / (1 + 0.25 + 3 * 0.01 ** 2 + 0.25), rel=1e-12)
    d, t = ft.delta2_from_sums(np.zeros((2, 6)), np.zeros((2, 6)), np.array([0.0, 0.0]), return_tau=True, cutoff=3)
    assert np.isnan(d).all() and (t == -1).all()                                       # empty classes stay empty
    with pytest.raises(ValueError):
        ft.delta2_from_sums(z, c, one, cutoff=-1)


def test_delta2_from_sums_fixed_cutoff_with_edges_uses_the_bins_starting_at_or_before_the_cutoff():
    # Break caught: an `edges` window that ignores the cutoff, cuts a bin in the middle, or reports tau_c in bin units.
    # bins {0} {1} {2, 3} {4, 5} {6, 7}; z = [1, .5, .4, .2, .1, .1, .05, .05]; c = z + [0, 0, .1, .3, 0, 0, .2, .2]
    # Zb = [1, .5, .3, .1, .05], Cb = [1, .5, .5, .1, .25], n_b = [1, 1, 2, 2, 2]. Zb never stays below 5 %: no automatic cutoff.
    z = np.array([[1.0, 0.5, 0.4, 0.2, 0.1, 0.1, 0.05, 0.05]])
    c = z + np.array([[0.0, 0.0, 0.1, 0.3, 0.0, 0.0, 0.2, 0.2]])
    edges, one = np.array([0, 1, 2, 4, 6, 8]), np.array([1.0])
    d_all, t_all = ft.delta2_from_sums(z, c, one, return_tau=True, edges=edges)
    assert t_all[0] == 7 and d_all[0] == pytest.approx(0.16 / 1.455, rel=1e-12)
    for cutoff in (2, 3):                                              # both inside bin 2: bins 0..2, tau_c = last lag of bin 2
        d, t = ft.delta2_from_sums(z, c, one, return_tau=True, edges=edges, cutoff=cutoff)
        assert t[0] == 3 and d[0] == pytest.approx(0.08 / 1.43, rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, edges=edges, cutoff=4)     # first lag of bin 3: bins 0..3
    assert t[0] == 5 and d[0] == pytest.approx(0.08 / 1.45, rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, edges=edges, cutoff=7)
    assert t[0] == 7 and d[0] == pytest.approx(d_all[0], rel=1e-12)
    d, t = ft.delta2_from_sums(z, c, one, return_tau=True, edges=edges, cutoff=0)     # only the single lag 0
    assert t[0] == 0 and d[0] == 0.0


def test_floor_from_sums_fixed_cutoff_hand_values_plain_and_binned():
    # Break caught: a floor computed over a window other than the one of delta2 (the estimate and its floor must share it).
    n = np.array([4.0])
    zf = np.array([1.0, 0.5, 0.01, 0.01, 0.01, 0.5])
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    args = (np.array([n * zf]), np.array([n * (zf + d)]), np.array([n * (zf - d)]), n)
    # window lags 0..4: 1/4 * 4 * (.1^2 + .2^2 + .3^2 + .4^2 + .5^2) / (1 + .25 + 3 * .01^2)
    assert ft.floor_from_sums(*args, cutoff=4)[0] == pytest.approx((0.01 + 0.04 + 0.09 + 0.16 + 0.25) / 1.2503, rel=1e-12)
    assert ft.floor_from_sums(*args, cutoff=1)[0] == pytest.approx((0.01 + 0.04) / 1.25, rel=1e-12)
    assert ft.floor_from_sums(*args, cutoff=None)[0] == ft.floor_from_sums(*args)[0]         # default: data-dependent tau_c = 2
    assert ft.floor_from_sums(*args)[0] == pytest.approx((0.01 + 0.04 + 0.09) / 1.2501, rel=1e-12)
    # binned: bins {0} {1} {2, 3} {4, 5}; cutoff lag 3 -> bins 0..2; (ZA - ZB) = 2 d -> d_b = [.1, .2, .35], Zf_b = [1, .5, .01]
    edges = np.array([0, 1, 2, 4, 6])
    exp = (0.01 + 0.04 + 2 * 0.35 ** 2) / (1 + 0.25 + 2 * 0.01 ** 2)
    assert ft.floor_from_sums(*args, edges=edges, cutoff=3)[0] == pytest.approx(exp, rel=1e-12)
    assert ft.floor_from_sums(*args, edges=edges, cutoff=2)[0] == pytest.approx(exp, rel=1e-12)      # lag 2 starts bin 2


def test_fixed_cutoff_at_the_last_lag_equals_the_old_result_when_the_old_cutoff_was_the_last_lag():
    # Break caught: an off-by-one in the fixed window (last lag dropped or an extra lag) or a changed default path.
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 60, 6, 60, 300, N_LAGS + 1, 300, 8)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.array([[True] * 6, [True, True, True, False, False, False]])
    # a window shorter than the relaxation: the old cutoff is the last lag of the window
    short = 4
    stats_s = {**stats, "pair": stats["pair"][:, :short + 1], "g3": stats["g3"][:, :short + 1]}
    cur = ft.binary_curves(stats_s)
    old, tau = ft.delta2_from_sums(*ft.class_sums(cur["z_mean"], cur["c_zz"], masks), return_tau=True)
    assert (tau == short).all()
    np.testing.assert_array_equal(ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, cutoff=short)[0], old)
    perm = np.random.default_rng(2).permutation(60)
    sums = ft.split_half_class_sums(stats_s, masks, perm[:30], perm[30:])
    np.testing.assert_array_equal(ft.floor_from_sums(*sums, cutoff=short), ft.floor_from_sums(*sums))
    a = ft.bootstrap_class_delta2(stats_s, masks, 6, np.random.default_rng(0))
    np.testing.assert_array_equal(ft.bootstrap_class_delta2(stats_s, masks, 6, np.random.default_rng(0), cutoff=short), a)
    np.testing.assert_array_equal(ft.split_half_floor(stats_s, masks, np.random.default_rng(1), cutoff=short),
                                  ft.split_half_floor(stats_s, masks, np.random.default_rng(1)))
    x2, x3, x4 = _ar_cycles(60, hidden=False, seed=3)
    x3 = x3[:, :5]                                                     # the relaxation does not fall below 5 % in 5 lags
    r0, r1 = ft.continuous_fdtest(x2, x3, x4), ft.continuous_fdtest(x2, x3, x4, cutoff=4)
    assert r0["tau_c"] == 4 and r1["tau_c"] == 4 and r1["delta2"] == r0["delta2"]
    assert ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3), cutoff=4) == \
        ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3))
    np.testing.assert_array_equal(ft.bootstrap_continuous_delta2(x2, x3, x4, 4, np.random.default_rng(0), cutoff=4),
                                  ft.bootstrap_continuous_delta2(x2, x3, x4, 4, np.random.default_rng(0)))


def test_fixed_cutoff_binary_chain_matches_a_manual_window_for_estimate_bootstrap_and_floor():
    # Break caught: any of the class-level functions ignoring `cutoff`, or using a different window than the others.
    g2, g3, g4 = _mc_cycles(PRESET, V0, I0, V0, 3.3e-3, 80, 6, 60, 300, N_LAGS + 1, 300, 12)
    stats = ft.binary_stage_stats(g2, g3, g4, N_LAGS)
    masks = np.ones((1, 6), dtype=bool)
    cut = 7
    cur = ft.binary_curves(stats)
    z, c = np.nanmean(cur["z_mean"], axis=1), np.nanmean(cur["c_zz"], axis=1)      # all junctions valid
    assert cur["valid"].all()
    manual = ft.delta_squared(c, z, cut)
    assert ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, cutoff=cut)[0][0] == pytest.approx(manual, rel=1e-12)
    assert manual != ft.class_delta2(cur["z_mean"], cur["c_zz"], masks)[0][0]      # the window matters here
    w = ft.bootstrap_weights(80, 5, np.random.default_rng(4))
    boot = ft.bootstrap_class_delta2(stats, masks, 5, np.random.default_rng(4), cutoff=cut)
    for b in range(5):
        cb = ft.binary_curves(stats, w[b])
        assert boot[b, 0] == pytest.approx(ft.delta_squared(np.nanmean(cb["c_zz"], axis=1), np.nanmean(cb["z_mean"], axis=1), cut), rel=1e-10)
    perm = np.random.default_rng(6).permutation(80)
    za = ft.binary_curves(stats)["alpha"] + ft.binary_curves(stats)["beta"] * stats["g3"][perm[:40]].mean(axis=0)
    zb = ft.binary_curves(stats)["alpha"] + ft.binary_curves(stats)["beta"] * stats["g3"][perm[40:]].mean(axis=0)
    manual_floor = 0.25 * np.sum((za.mean(axis=1)[:cut + 1] - zb.mean(axis=1)[:cut + 1]) ** 2) / np.sum(z[:cut + 1] ** 2)
    assert ft.split_half_floor(stats, masks, np.random.default_rng(6), cutoff=cut)[0] == pytest.approx(manual_floor, rel=1e-10)


def test_continuous_fixed_cutoff_matches_manual_window_including_bootstrap_floor_and_edges():
    # Break caught: continuous functions ignoring `cutoff` (bootstrap kwargs, floor window, binned bins).
    x2, x3, x4 = _ar_cycles(60, hidden=True, seed=7)
    cut = 12
    res = ft.continuous_fdtest(x2, x3, x4, cutoff=cut)
    full = ft.continuous_fdtest(x2, x3, x4)
    assert res["tau_c"] == cut and res["delta2"] == pytest.approx(ft.delta_squared(full["c_zz"], full["z_mean"], cut), rel=1e-12)
    assert full["tau_c"] != cut
    # bootstrap: same resamples, each one evaluated with the same fixed window
    boot = ft.bootstrap_continuous_delta2(x2, x3, x4, 4, np.random.default_rng(1), cutoff=cut)
    rng = np.random.default_rng(1)
    for b in range(4):
        idx = rng.integers(0, 60, 60)
        assert boot[b] == pytest.approx(ft.continuous_fdtest(x2[idx], x3[idx], x4[idx], cutoff=cut)["delta2"], rel=1e-12)
        assert boot[b] != ft.continuous_fdtest(x2[idx], x3[idx], x4[idx])["delta2"]
    # floor: lags 0..cut of the half difference (same manual recipe as the existing floor test)
    rng = np.random.default_rng(3)
    perm = rng.permutation(60)
    za, zb = np.nanmean(res["z3"][perm[:30]], axis=0), np.nanmean(res["z3"][perm[30:]], axis=0)
    ok = np.isfinite(za) & np.isfinite(zb) & np.isfinite(res["z_mean"])
    ok[cut + 1:] = False
    manual = 0.25 * np.sum((za[ok] - zb[ok]) ** 2) / np.sum(res["z_mean"][ok] ** 2)
    assert ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng(3), cutoff=cut) == pytest.approx(manual, rel=1e-12)
    # with edges: bins starting at a lag <= cutoff, tau_c the last lag of the last of them
    edges = ft.lag_bin_edges(x3.shape[1], n_first=4, ratio=1.5)
    rb = ft.continuous_fdtest(x2, x3, x4, edges=edges, cutoff=cut)
    bc = int(np.searchsorted(edges, cut, side="right") - 1)
    assert rb["cutoff_bin"] == bc and rb["tau_c"] == edges[bc + 1] - 1 and rb["tau_c"] >= cut
    zb_, cb_ = ft.bin_lags(res["z_mean"], edges), ft.bin_lags(res["c_zz"], edges)
    w = np.diff(edges)[:bc + 1]
    assert rb["delta2"] == pytest.approx(np.sum(w * (cb_[:bc + 1] - zb_[:bc + 1]) ** 2) / np.sum(w * zb_[:bc + 1] ** 2), rel=1e-12)


# ------------------------------------------------------------------------------------ interval methods
def test_violation_summary_default_is_unchanged_and_reports_the_method():
    # Break caught: a changed default decision rule (the analyzer's `reverse` rule must not move) or a missing method name.
    boot = np.arange(101) / 100.0
    old = ft.violation_summary(0.6, boot, 0.2)
    assert old["method"] == "reverse" and old["excludes_floor"] is True
    assert ft.violation_summary(0.6, boot, 0.2, method="reverse") == old
    np.testing.assert_allclose(old["ci95_reverse"], [0.225, 1.175])
    np.testing.assert_allclose(old["ci95"], old["ci95_reverse"])                        # the interval used for the decision
    with pytest.raises(ValueError):
        ft.violation_summary(0.6, boot, 0.2, method="bca")


def test_violation_summary_percentile_method_decides_with_the_percentile_lower_end():
    # Break caught: `percentile` deciding with the reverse interval (or vice versa).
    boot = np.arange(101) / 100.0                                    # q2.5 = 0.025, q97.5 = 0.975 exactly
    s = ft.violation_summary(0.6, boot, 0.02, method="percentile")
    assert s["method"] == "percentile" and s["excludes_floor"] is True             # 0.025 > 0.02
    np.testing.assert_allclose(s["ci95"], [0.025, 0.975])
    assert ft.violation_summary(0.6, boot, 0.03, method="percentile")["excludes_floor"] is False
    assert ft.violation_summary(0.6, boot, 0.03, method="reverse")["excludes_floor"] is True      # reverse: 0.225 > 0.03


def test_violation_summary_sqrt_reverse_hand_values_clipping_and_decision():
    # Break caught: the reverse rule applied on the delta2 scale, no back-transform, or a lower end below zero.
    # boot = (k / 100)^2 so that s* = k / 100 (q2.5 = .025, q97.5 = .975 exactly); d = .36 -> s_hat = .6:
    #   s interval = [1.2 - .975, 1.2 - .025] = [.225, 1.175] -> delta2 interval [.050625, 1.380625]
    boot = (np.arange(101) / 100.0) ** 2
    s = ft.violation_summary(0.36, boot, 0.05, method="sqrt_reverse")
    assert s["method"] == "sqrt_reverse"
    np.testing.assert_allclose(s["ci95_sqrt_reverse"], [0.225 ** 2, 1.175 ** 2])
    np.testing.assert_allclose(s["ci95"], s["ci95_sqrt_reverse"])
    assert s["excludes_floor"] is True                                              # .050625 > .05
    assert ft.violation_summary(0.36, boot, 0.06, method="sqrt_reverse")["excludes_floor"] is False
    np.testing.assert_allclose(s["ci95_reverse"], [0.72 - 0.9506, 0.72 - 0.0006], atol=1e-3)   # the other intervals are still reported
    assert s["excess"] == pytest.approx(0.31)
    # clipped at 0 on the s scale: d = .0025 -> s_hat = .05, s interval [.1 - .975, .1 - .025] = [-.875, .075] -> [0, .075^2]
    c = ft.violation_summary(0.0025, boot, 0.0, method="sqrt_reverse")
    np.testing.assert_allclose(c["ci95_sqrt_reverse"], [0.0, 0.075 ** 2])
    assert c["excludes_floor"] is False                                             # 0 > 0 is False
    # both ends below zero -> [0, 0]
    z = ft.violation_summary(0.0, boot, 0.0, method="sqrt_reverse")
    np.testing.assert_allclose(z["ci95_sqrt_reverse"], [0.0, 0.0])


def test_violation_summary_all_methods_fail_closed_on_bad_input_without_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in ("percentile", "reverse", "sqrt_reverse"):
            for d, boot, floor in ((np.nan, np.arange(11) / 10, 0.1), (0.5, np.full(11, np.nan), 0.1), (0.5, np.arange(11) / 10, np.nan),
                                   (0.5, np.array([]), 0.1), (-0.2, np.arange(11) / 10, 0.1), (np.inf, np.arange(11) / 10, 0.1)):
                s = ft.violation_summary(d, boot, floor, method=method)
                assert s["excludes_floor"] is False and s["method"] == method
            # NaN draws are dropped; a negative draw (cannot occur for delta2) does not raise a warning either
            s = ft.violation_summary(0.5, np.r_[np.arange(11) / 10, np.nan, -0.1], 0.1, method=method)
            assert np.isfinite(s["ci95"]).all()


def test_studentized_interval_hand_values_dropped_draws_and_clipping():
    # Break caught: a wrong sign / quantile order in the bootstrap-t interval, standardising by the wrong se, keeping zero or
    # non-finite se draws, or a lower end below zero.
    # d_hat = 1, d* = 1 + [-.4, -.2, 0, .2, .4], se* = .2 everywhere -> t* = [-2, -1, 0, 1, 2]; alpha = .5: q_.25 = -1, q_.75 = 1;
    # se_hat = std(d*, ddof=1) = sqrt(.1); interval = [1 - 1 * se_hat, 1 + 1 * se_hat]
    d_boot = 1 + np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    lo, hi = ft.studentized_interval(1.0, d_boot, np.full(5, 0.2), alpha=0.5)
    assert lo == pytest.approx(1 - np.sqrt(0.1), rel=1e-12) and hi == pytest.approx(1 + np.sqrt(0.1), rel=1e-12)
    # asymmetric t* = [-1, 0, 0, 0, 3] (d* - d_hat = [-.2, 0, 0, 0, .6], se* = .2); alpha = .1: linear-interpolated percentiles
    # q_.05 = -1 + .2 * 1 = -.8, q_.95 = 0 + .8 * 3 = 2.4; the interval is REVERSED: [d_hat - q_.95 se_hat, d_hat - q_.05 se_hat]
    d2 = 1 + np.array([-0.2, 0.0, 0.0, 0.0, 0.6])
    se_hat = np.std(d2, ddof=1)
    assert se_hat == pytest.approx(np.sqrt(0.092), rel=1e-12)                        # deviations from the mean .08: sum of squares .368, / 4
    lo, hi = ft.studentized_interval(1.0, d2, np.full(5, 0.2), alpha=0.1)
    assert lo == pytest.approx(1.0 - 2.4 * se_hat, rel=1e-12) and hi == pytest.approx(1.0 + 0.8 * se_hat, rel=1e-12)
    # zero and non-finite se draws are dropped from t* (their d* still enters se_hat, taken over all finite d*)
    d_more = np.r_[d_boot, 1.0, 1.0, 1.0]
    se_more = np.r_[np.full(5, 0.2), 0.0, np.nan, np.inf]
    lo2, hi2 = ft.studentized_interval(1.0, d_more, se_more, alpha=0.5)
    sh = np.std(d_more, ddof=1)
    assert lo2 == pytest.approx(1 - sh, rel=1e-12) and hi2 == pytest.approx(1 + sh, rel=1e-12)
    # clipped at zero
    # d_hat = .1, d* = [-.3, -.1, .1, .3, .5], se* = .2 -> t* = [-2, -1, 0, 1, 2], interval [.1 - .316, .1 + .316] -> [0, .416]
    lo3, hi3 = ft.studentized_interval(0.1, d_boot - 0.9, np.full(5, 0.2), alpha=0.5)
    assert lo3 == 0.0 and hi3 == pytest.approx(0.1 + np.sqrt(0.1), rel=1e-12)


def test_studentized_interval_fails_closed_without_warnings():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for d, db, se in ((np.nan, np.arange(5.0), np.ones(5)), (1.0, np.full(5, np.nan), np.ones(5)), (1.0, np.arange(5.0), np.zeros(5)),
                          (1.0, np.array([1.0]), np.array([1.0])), (1.0, np.array([]), np.array([])), (1.0, np.ones(5), np.ones(5))):
            lo, hi = ft.studentized_interval(d, db, se)
            assert np.isnan(lo) and np.isnan(hi)                 # NaN interval: the caller's `lo > floor` is False (fails closed)


# ------------------------------------------------------------------------------------ noise-based cutoff
def _z3_with_known_se(mean, s, n=100):
    """(n, len(mean)) per-cycle curves whose per-lag mean is EXACTLY `mean` and whose standard error is EXACTLY s / sqrt(n):
    alternating +-1 deviations scaled to unit sample std (ddof = 1)."""
    d = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * np.sqrt((n - 1) / n)
    return np.asarray(mean)[None, :] + d[:, None] * np.asarray(s)[None, :]


def test_noise_cutoff_is_the_analytic_crossing_of_a_clean_decay():
    # <z> = 5 exp(-l / 10) over 60 lags, SE = 0.1 at every lag: crossing 5 exp(-l/10) = 0.1 at l = 10 ln 50 = 39.1
    l = np.arange(60)
    z3 = _z3_with_known_se(5 * np.exp(-l / 10), np.full(60, 1.0))
    cut, hit_end = ft.noise_cutoff(z3)
    assert 38 <= cut <= 41 and not hit_end


def test_noise_cutoff_uses_the_median_not_the_mean_of_the_standard_error():
    # Break caught: a mean-based noise level. 10 of the 60 lags have a huge scatter (SE 100 against 0.1): the median SE is 0.1
    # (crossing at lag ~39) whereas the mean SE is ~16.7 (which would put the cutoff near lag 12).
    l = np.arange(60)
    s = np.ones(60); s[50:] = 1000.0
    z3 = _z3_with_known_se(5 * np.exp(-l / 10), s)
    assert np.mean(s / 10) > 10 and np.median(s / 10) == pytest.approx(0.1)
    cut, hit_end = ft.noise_cutoff(z3)
    assert 38 <= cut <= 41 and not hit_end


def test_noise_cutoff_pure_noise_and_never_decaying_signal_hit_the_end():
    rng = np.random.default_rng(0)
    cut, hit_end = ft.noise_cutoff(rng.normal(size=(200, 40)))                # <z> ~ SE at every lag: below the noise from lag 0
    assert cut == 39 and hit_end
    z3 = _z3_with_known_se(np.full(40, 5.0), np.ones(40))                     # never decays
    assert ft.noise_cutoff(z3) == (39, True)


def test_noise_cutoff_needs_a_sustained_run_and_survives_a_single_dip():
    # Break caught: a cutoff triggered by one lag below the noise. <z> dips to 0 at lag 10 only (smooth = 1: no averaging).
    z = np.r_[np.full(10, 5.0), 0.0, np.full(19, 5.0), np.zeros(10)]
    z3 = _z3_with_known_se(z, np.ones(len(z)))
    assert ft.noise_cutoff(z3, smooth=1, sustain=3) == (30, False)
    assert ft.noise_cutoff(z3, smooth=1, sustain=1) == (10, False)


def test_noise_cutoff_is_robust_to_nan():
    l = np.arange(60)
    z3 = _z3_with_known_se(5 * np.exp(-l / 10), np.ones(60))
    z3[:5, 3] = np.nan                                                        # a few missing cycles at one lag
    z3[:, 20] = np.nan                                                        # a lag without any valid cycle
    cut, hit_end = ft.noise_cutoff(z3)
    assert 37 <= cut <= 42 and not hit_end
    assert ft.noise_cutoff(np.full((10, 5), np.nan)) == (4, True)             # nothing valid: last lag, flagged


def test_noise_level_and_smoothing_at_the_edges():
    # the edge-safe moving average shrinks its window at the ends instead of biasing towards zero
    z3 = _z3_with_known_se(np.full(10, 2.0), np.full(10, 1.0), n=100)
    sigma, m, sm = ft.noise_level(z3, smooth=7)
    assert sigma == pytest.approx(0.1) and np.allclose(m, 2.0) and np.allclose(sm, 2.0)


def test_noise_level_with_fewer_lags_than_the_smoothing_window():
    # Break caught: np.convolve(mode="same") returns max(len, window) points, so the smoothed curve was longer than the lags.
    z3 = _z3_with_known_se(np.full(5, 2.0), np.full(5, 1.0), n=100)
    sigma, m, sm = ft.noise_level(z3, smooth=7)
    assert sm.shape == (5,) and np.allclose(sm, 2.0) and sigma == pytest.approx(0.1)
    assert ft.noise_cutoff(z3, smooth=7, sustain=3) == (4, True)
