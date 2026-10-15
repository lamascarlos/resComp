"""Driver of the g_eff FDTest with the switch-on doubling and the noise-based cutoff (scripts/fdtest_geff.py), on synthetic
run files: chained AR(1) series with a known real shift between the perturbed and unperturbed stages (a Markov variable, so
the OFF, ON and BOTH violations must sit near their noise floors) and the same with a hidden slow variable (violated)."""
import json

import numpy as np
import pytest

from agnw_sim import fdtest as ft
from agnw_sim import fdtest_protocol as fp
from scripts import fdtest_geff as fg

N, V0, EPS = 100, 1.0, 0.1


def _meta(nc, T_relax=60, T_samp=100, n_lags=40, r=1):
    return {"V0": V0, "eps": EPS, "T_relax": T_relax, "T_samp": T_samp, "n_cycles": nc, "record_every": r, "n_lags": n_lags,
            "network_seed": 1, "k": 10, "p_relax": 0.0, "N": N}


def _ar_series(sch, seed, hidden=False, a=0.8, f=0.06, c=0.5, d=0.95, start=0.0):
    """Chained AR(1) g_eff series, one record per step: x_{t+1} = a x_t (+ c y_t) + f [stage I/II] + noise (y slow hidden)."""
    r = np.random.default_rng(seed)
    x, y = start, 0.0
    out = np.empty(sch.n_steps)
    for t in range(sch.n_steps):
        out[t] = x
        force = f if t % sch.T_cycle < sch.T_relax + sch.T_samp else 0.0
        x, y = a * x + (c * y if hidden else 0.0) + force + 0.3 * r.normal(), d * y + 0.3 * r.normal()
    return out


def _write_run(path, sch, geff):
    steps = np.arange(sch.n_steps)
    V_in = np.where(steps % sch.T_cycle < sch.T_relax + sch.T_samp, sch.V0 + sch.eps, sch.V0)
    meta = _meta(sch.n_cycles, sch.T_relax, sch.T_samp, sch.n_lags, sch.record_every)
    np.savez(path, I_total=geff * V_in, schedule_json=json.dumps(meta))


def _sched(nc, **kw):
    m = _meta(nc, **kw)
    return fp.Schedule(V0=V0, eps=EPS, T_relax=m["T_relax"], T_samp=m["T_samp"], n_cycles=nc, record_every=m["record_every"], n_lags=m["n_lags"])


def _make_dir(tmp_path, n_runs=2, nc=5, hidden=False, seed=0, geff_fn=None, f=0.06, **kw):
    sch = _sched(nc, **kw)
    for i in range(n_runs):
        g = _ar_series(sch, seed + i, hidden=hidden, f=f) if geff_fn is None else geff_fn(sch, i)
        _write_run(tmp_path / f"run_N{N}_seed{i + 1}.npz", sch, g + 10.0)          # offset keeps g_eff and I_total positive
    return sch


# ------------------------------------------------------------------------------------ stage arrays
def test_stage_arrays_drop_cycle_zero_of_every_run_for_all_directions(tmp_path):
    # g_eff = record index (times 1): every array entry identifies its record. 2 runs of 4 cycles: cycles 1..3 of each are used.
    sch = _sched(4, T_relax=60, T_samp=100, n_lags=40)
    _make_dir(tmp_path, n_runs=2, nc=4, geff_fn=lambda s, i: np.arange(s.n_steps) - 10.0)
    from scripts.fdtest_analyze import load_runs
    runs, meta, _ = load_runs(str(tmp_path), N)
    x2, x3, x4, xI = fg.stage_arrays(runs, sch)
    assert x2.shape[0] == x3.shape[0] == x4.shape[0] == xI.shape[0] == 6 and xI.shape[1] == 40
    T = sch.T_cycle
    np.testing.assert_allclose(xI[0], 10.0 + np.arange(1 * T, 1 * T + 40) - 10.0 + 0.0, atol=1e-9)   # cycle 1 of run 1, not cycle 0
    np.testing.assert_allclose(xI[3][0], T, atol=1e-9)                                                 # first used cycle of run 2 is cycle 1
    np.testing.assert_allclose(x2[0][0], T + sch.T_relax, atol=1e-9)
    np.testing.assert_allclose(x4[0][0], T + 2 * sch.T_relax + sch.T_samp, atol=1e-9)


# ------------------------------------------------------------------------------------ estimator identities
def test_fixed_window_delta2_equals_the_estimate_with_the_cutoff_argument():
    # The bootstrap loop derives the fixed-window value from the c_zz and z_mean of the free estimate; it must equal
    # continuous_fdtest(cutoff=...) on the same data (same z map, same curves).
    rng = np.random.default_rng(3)
    x2 = rng.normal(0.4, 1, (60, 80)); x3 = rng.normal(0.4, 1, (60, 30)); x4 = rng.normal(0, 1, (60, 80))
    free = ft.continuous_fdtest(x2, x3, x4)
    for cut in (0, 7, 29, 500):
        d = fg.delta2_at(free, cut)
        assert d == pytest.approx(ft.continuous_fdtest(x2, x3, x4, cutoff=cut)["delta2"], rel=1e-12, abs=1e-15)


def test_bootstrap_shares_the_cycle_draws_between_the_directions(tmp_path):
    # BOTH = the mean of the OFF and ON draws of the SAME resample; OFF alone reproduces ft.bootstrap_continuous_delta2 with the
    # same rng (same integers), and ON reproduces the same call with the roles swapped -> the indices are shared.
    sch = _make_dir(tmp_path, nc=9, n_lags=20)
    from scripts.fdtest_analyze import load_runs
    runs, _, _ = load_runs(str(tmp_path), N)
    x2, x3, x4, xI = fg.stage_arrays(runs, sch)
    boot = fg.bootstrap_switch(x2, x3, x4, xI, 6, np.random.default_rng(5), cut_off=8, cut_on=12)
    ref_off = ft.bootstrap_continuous_delta2(x2, x3, x4, 6, np.random.default_rng(5))
    ref_on = ft.bootstrap_continuous_delta2(x4, xI, x2, 6, np.random.default_rng(5))
    np.testing.assert_allclose(boot["c5"]["off"], ref_off, rtol=1e-12)
    np.testing.assert_allclose(boot["c5"]["on"], ref_on, rtol=1e-12)
    np.testing.assert_allclose(boot["c5"]["both"], 0.5 * (ref_off + ref_on), rtol=1e-12)
    ref_off_n = ft.bootstrap_continuous_delta2(x2, x3, x4, 6, np.random.default_rng(5), cutoff=8)
    ref_on_n = ft.bootstrap_continuous_delta2(x4, xI, x2, 6, np.random.default_rng(5), cutoff=12)
    np.testing.assert_allclose(boot["noise"]["off"], ref_off_n, rtol=1e-12)
    np.testing.assert_allclose(boot["noise"]["on"], ref_on_n, rtol=1e-12)
    ref_both = 0.5 * (ft.bootstrap_continuous_delta2(x2, x3, x4, 6, np.random.default_rng(5), cutoff=12)
                      + ft.bootstrap_continuous_delta2(x4, xI, x2, 6, np.random.default_rng(5), cutoff=12))
    np.testing.assert_allclose(boot["noise"]["both"], ref_both, rtol=1e-12)                     # common window = the larger cutoff


# ------------------------------------------------------------------------------------ driver
def test_driver_smoke_writes_strict_json_and_figure(tmp_path, capsys):
    _make_dir(tmp_path, n_runs=2, nc=5, n_lags=20)
    assert fg.main([str(N), "--out", str(tmp_path), "--n-boot", "6", "--n-lags", "20"]) == 0
    text = (tmp_path / f"fdtest_geff_N{N}.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    res = json.loads(text)
    assert res["n_cycles_used"] == 8 and res["cycle0_excluded"] is True and res["schedule"]["n_lags"] == 20
    assert set(res["variants"]) == {"OFF", "ON", "BOTH"}
    for v in res["variants"].values():
        assert set(v) == {"c5", "noise"}
        for block in v.values():
            for key in ("delta2", "ci95_reverse", "noise_floor", "excess", "excludes_floor", "cutoff", "hit_end", "achieved_response_sigma",
                        "z0_snr", "noise_corr_length"):
                assert key in block
    assert (tmp_path / f"fdtest_geff_switch_N{N}.png").stat().st_size > 1000
    assert "OFF" in capsys.readouterr().out


def test_driver_exits_1_on_inconsistent_inputs(tmp_path):
    _make_dir(tmp_path, n_runs=2, nc=5, n_lags=20)
    with pytest.raises(SystemExit) as e:                                        # n_lags larger than the stage III / I window
        fg.main([str(N), "--out", str(tmp_path), "--n-lags", "500"])
    assert e.value.code not in (0, None) and "n-lags" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        fg.main(["77", "--out", str(tmp_path)])                                 # no run files for N = 77
    assert e.value.code not in (0, None)
    # a run with another schedule
    sch = _sched(5, T_samp=120, n_lags=20)
    _write_run(tmp_path / f"run_N{N}_seed9.npz", sch, _ar_series(sch, 9) + 10.0)
    with pytest.raises(SystemExit) as e:
        fg.main([str(N), "--out", str(tmp_path), "--n-lags", "20"])
    assert e.value.code not in (0, None) and "schedule" in str(e.value.code)


def test_markov_ar_series_is_not_a_violation_in_any_direction(tmp_path):
    _make_dir(tmp_path, n_runs=8, nc=41, hidden=False, seed=10)
    assert fg.main([str(N), "--out", str(tmp_path), "--n-boot", "40", "--n-lags", "40"]) == 0
    res = json.loads((tmp_path / f"fdtest_geff_N{N}.json").read_text())
    for name in ("OFF", "ON", "BOTH"):
        for rule in ("c5", "noise"):
            b = res["variants"][name][rule]
            assert b["excess"] < 0.3 and not b["excludes_floor"], (name, rule, b)      # robust over seeds (max delta2 0.24)
    assert res["variants"]["ON"]["noise"]["z0_snr"] > 3                          # the mirror transient carries a real signal


def test_hidden_variable_violates_in_off_on_and_both(tmp_path):
    _make_dir(tmp_path, n_runs=8, nc=41, hidden=True, seed=10, f=0.3)
    assert fg.main([str(N), "--out", str(tmp_path), "--n-boot", "40", "--n-lags", "40"]) == 0
    res = json.loads((tmp_path / f"fdtest_geff_N{N}.json").read_text())
    for name in ("OFF", "ON", "BOTH"):
        for rule in ("c5", "noise"):
            b = res["variants"][name][rule]
            assert b["delta2"] > 1.0 and b["excess"] > 0.5, (name, rule, b)      # an order of magnitude above the Markov values (~0.1)
