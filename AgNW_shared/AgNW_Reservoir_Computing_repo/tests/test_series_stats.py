"""N11 stage-0 estimators (agnw_rc/series_stats.py, docs/PLAN_N11_CALIBRATION.md section 5)."""
import numpy as np
import pytest

from agnw_rc import series_stats as S


def test_clean_drops_nan_padding_and_repeated_times():
    t = np.array([0.0, 0.1, 0.1, 0.2, np.nan, 0.3, np.nan])
    G = np.array([1.0, 2.0, 9.0, 3.0, np.nan, 4.0, np.nan])
    tc, Gc = S.clean(t, G)
    np.testing.assert_array_equal(tc, [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(Gc, [1.0, 2.0, 3.0, 4.0])


def test_to_grid_takes_nearest_samples_without_reuse():
    t = np.arange(0, 10.0001, 0.1)
    tg, Gg = S.to_grid(t, t * 2, 0.6)
    np.testing.assert_allclose(np.diff(tg), 0.6, atol=1e-9)
    np.testing.assert_allclose(Gg, 2 * tg)
    tg2, _ = S.to_grid(np.arange(0, 10, 0.6), np.ones(17), 0.3)       # coarser data: no sample used twice
    assert len(np.unique(tg2)) == len(tg2)


def test_interpolation_index_tells_interpolated_from_raw_records():
    rng = np.random.default_rng(0)
    raw = 1 + 1e-3 * rng.standard_normal(600)
    lin = np.interp(np.arange(600) / 6, np.arange(100), raw[:100])     # linear interpolation x6
    assert S.interpolation_index(raw) < 0.05 and S.interpolation_index(lin) > 0.7


def test_clipped_sd_ignores_jumps():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, 20_000)
    m = rng.random(20_000) < 0.05
    x[m] += rng.choice([-1, 1], m.sum()) * 8.0
    assert S.clipped_sd(x) == pytest.approx(1.0, rel=0.03)


def test_jump_stats_rejects_records_at_the_floor():
    with pytest.raises(ValueError):
        S.jump_stats(np.arange(5.0), np.array([1e-13, -1e-14, 2e-13, 1e-13, 3e-14]))


def _long_series(seed, rate, size_min, sd=4.8e-4, T=20_000.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, S.GRID_DT)
    return t, S.synthetic_series(rng, t, np.full(len(t), 3e-5), sd, rate, 3.0, size_min)


def test_validation_sd_within_10_percent_at_data_like_jumps():
    """Spec section 5: sd within 10 % (jumps at the measured rate, sizes from the threshold up)."""
    for seed in (1, 2, 3):
        t, G = _long_series(seed, 0.08, 2.6 * 4.8e-4)
        assert S.jump_stats(t, G)["sd_rel"] == pytest.approx(4.8e-4, rel=0.10)


def test_validation_rate_within_15_percent_for_clear_jumps():
    """Spec section 5: lambda within 15 % for jumps clearly above the threshold (sizes >= 5 sd)."""
    for seed in (1, 2, 3):
        t, G = _long_series(seed, 0.08, 5 * 4.8e-4)
        js = S.jump_stats(t, G)
        assert js["rate_per_s"] == pytest.approx(0.08, rel=0.15)
        assert 0.4 < js["up_share"] < 0.6
        lo, hi = js["rate_ci_per_s"]
        assert lo < js["rate_per_s"] < hi


def test_no_jumps_gives_a_rate_near_zero():
    t, G = _long_series(4, 0.0, 1e-3)
    assert abs(S.jump_stats(t, G)["rate_per_s"]) < 0.003


def test_poisson_ci_and_block_bootstrap():
    lo, hi = S.poisson_ci(40)
    assert lo == pytest.approx(28.58, abs=0.05) and hi == pytest.approx(54.47, abs=0.05)
    assert S.poisson_ci(0)[0] == 0.0
    x = np.random.default_rng(2).normal(5.0, 1.0, 2000)
    b = S.block_bootstrap(x, np.mean, 20, n_boot=300, rng=np.random.default_rng(3))
    assert b["lo"] < x.mean() < b["hi"] and 0 < b["log_sd"] < 0.02   # interval around the sample mean
    short = S.block_bootstrap(x[:5], np.mean, 20, n_boot=50)          # block longer than the record
    assert np.isfinite(short["lo"])


def test_onset_interpolates_in_log_and_handles_floor_values():
    V = np.array([2.2, 2.3, 2.4, 2.5, 2.6])
    G = np.array([-1e-14, 2e-13, 1e-10, 1e-7, 1e-5])
    v = S.onset(V, G, G_ref=1e-4)                                       # threshold 1e-7: reached exactly at 2.5 V
    assert v == pytest.approx(2.5)
    v2 = S.onset(V, G, G_ref=1e-5)                                      # threshold 1e-8: between 2.4 and 2.5
    assert v2 == pytest.approx(2.4 + 0.1 * np.log(1e-8 / 1e-10) / np.log(1e-7 / 1e-10))
    assert S.onset(V, G, G_ref=1.0) is None


def test_theta_recovered_within_20_percent():
    """Spec section 5: theta within 20 % on noisy, jumpy transients sampled like Fig. 2a (80 s, 0.1 and 0.6 s)."""
    for theta, dt, seed in ((0.1, 0.6, 1), (1.0, 0.1, 2), (0.3, 0.1, 3)):
        rng = np.random.default_rng(seed)
        t = np.arange(0, 80, dt)
        G = S.synthetic_series(rng, t, S.approach(t, 3e-5, 1e-7, theta), 4.8e-4, 0.08, 3.0, 2.6 * 4.8e-4)
        f = S.fit_theta(t, G, n_boot=20, rng=rng)
        assert f["theta_per_s"] == pytest.approx(theta, rel=0.20) and not f["unresolved"]
        assert f["lo"] <= f["theta_per_s"] <= f["hi"]


def test_instant_transient_is_flagged_not_crashed():
    t = np.arange(0, 80, 0.6)
    G = 3e-5 * (1 + 1e-4 * np.random.default_rng(0).standard_normal(len(t)))
    f = S.fit_theta(t, G, n_boot=5)
    assert f["unresolved"] and np.isfinite(f["theta_per_s"])


def test_relaxation_prefers_two_exponentials_only_when_present():
    t = np.arange(0, 250, 0.25)
    rng = np.random.default_rng(0)
    one = 2.5e-2 + 3e-4 * np.exp(-t / 40) + 2e-7 * rng.standard_normal(len(t))
    two = 2.5e-2 + 2e-4 * np.exp(-t / 3) + 2e-4 * np.exp(-t / 80) + 2e-7 * rng.standard_normal(len(t))
    r1, r2 = S.fit_relaxation(t, one), S.fit_relaxation(t, two)
    assert r1["best"] == "one" and r1["one"]["tau_s"] == pytest.approx(40, rel=0.05)
    assert r2["best"] == "two" and sorted([r2["two"]["tau1_s"], r2["two"]["tau2_s"]]) == pytest.approx([3, 80], rel=0.1)
    flat = S.fit_relaxation(t, np.full(len(t), 2.5e-2))                # no decay at all: no crash
    assert flat["best"] in ("one", "two")


def test_loglog_slope():
    x = np.array([1.0, 2.0, 4.0])
    assert S.loglog_slope(x, 3 * x ** 2.5) == pytest.approx(2.5)


# ---- final-review fixes (2026-10-13)
def test_flat_or_drifting_transient_is_flagged_unresolved():
    """A record with no exponential approach (flat + slow drift) must not return a usable theta ~ 0 with an absurd
    plateau (review: 4.6 V gave theta 8e-7 /s, G_inf 0.25 S, unresolved False)."""
    t = np.arange(0, 80, 0.1)
    rng = np.random.default_rng(1)
    G = 3e-5 * (1 + 0.3 * t / 80) * (1 + 2e-3 * rng.standard_normal(len(t)))       # linear rise, no plateau
    f = S.fit_theta(t, G, n_boot=5)
    assert f["unresolved"]


def test_stationarity_check_against_the_steady_state_point():
    assert S.is_stationary(1.05e-5, 1.0e-5) and not S.is_stationary(0.4e-5, 1.0e-5)
    assert not S.is_stationary(1.0e-5, 0.0)


def test_loglog_fit_gives_a_standard_error():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    slope, se = S.loglog_fit(x, 3 * x ** 2.5)
    assert slope == pytest.approx(2.5) and se == pytest.approx(0.0, abs=1e-9)
    slope, se = S.loglog_fit(x, np.array([1.0, 2.5, 3.0, 9.0]))
    assert se > 0


def test_stage0_json_has_per_voltage_jump_sizes():
    """Part 2 needs the measured event sizes per voltage for the stage-2 KS term."""
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parents[1] / "results" / "n11_data.json").read_text())
    rows = [r for r in d["inset"] if "jumps" in r]
    assert rows and all(len(r["sizes"]) == r["jumps"]["events"] for r in rows)
