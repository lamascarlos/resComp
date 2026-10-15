import numpy as np
import pandas as pd
import pytest

from agnw_sim import experiment as ex


def _sheet(n_on=400, n_off=408, shift=0, g0=7.5e-3, slope=0.0, jump=0.0, step=0.0, noise=0.0, rng=None):
    """One synthetic sheet: ON / OFF / ON / OFF at 1.05 / 1.0 V. G = g0 * (1 + slope * t) * (1 + jump + step) during ON
    (jump = instantaneous readout offset, step = slow response, here also instantaneous for simplicity), g0 * (1 + slope * t)
    during OFF. `shift` moves all three switches earlier, shortening segment A (as in Run47/Run60)."""
    lens = [n_on - shift, n_off, n_on, n_off]
    V = np.concatenate([np.full(n, v) for n, v in zip(lens, (1.05, 1.0, 1.05, 1.0))])
    t = 0.0288 * (np.arange(len(V)) + 0.8)
    rel = np.where(V > 1.02, jump + step, 0.0)
    if noise:
        rel = rel + noise * rng.standard_normal(len(V))
    G = g0 * (1 + slope * t) * (1 + rel)
    return t, G * V, V


def _frame(sheets):
    return ex.to_long([(f"Run{k + 2}", t, I, V) for k, (t, I, V) in enumerate(sheets)])


def test_to_long_labels_cycles_and_stages():
    df = _frame([_sheet(), _sheet(shift=6)])
    assert list(df.columns) == ["run", "sheet", "cycle", "stage", "sample_in_stage", "sample", "t_s", "V", "I", "R"]
    s0 = df[df.sheet == 1]
    assert len(s0) == 1616
    assert (s0.groupby(["cycle", "stage"]).size().to_dict()
            == {(1, "perturbed"): 400, (1, "unperturbed"): 408, (2, "perturbed"): 400, (2, "unperturbed"): 408})
    s1 = df[df.sheet == 2]
    assert len(s1) == 1610 and (s1[(s1.cycle == 1) & (s1.stage == "perturbed")].shape[0] == 394)
    assert np.allclose(df.R, df.V / df.I)
    assert set(df.loc[df.stage == "perturbed", "V"]) == {1.05}


def test_to_long_drops_rows_without_a_measurement():
    t, I, V = _sheet()
    I[-4:] = np.nan
    V[-4:] = np.nan
    df = ex.to_long([("Run2", t, I, V)])
    assert len(df) == 1612 and np.isfinite(df.I).all()


def test_detrend_removes_linear_drift_but_keeps_the_response():
    df = _frame([_sheet(slope=4e-3, step=-4e-4)])
    sh = ex.sheet_segments(df)[0]
    x = ex.detrend(sh, tail=200)
    assert np.allclose(x[1], 0.0, atol=1e-12) and np.allclose(x[3], 0.0, atol=1e-12)   # OFF segments flat at 0
    # ON segments: (1 + step) exactly, the trend divided out
    assert np.allclose(x[2], -4e-4, atol=1e-12) and np.allclose(x[0], -4e-4, atol=1e-12)


def test_readout_jump_recovers_offset_and_correction_removes_it():
    rng = np.random.default_rng(0)
    sheets = [_sheet(jump=-3e-4, noise=1e-5, rng=rng) for _ in range(40)]
    segs = [ex.detrend(s, tail=200) for s in ex.sheet_segments(_frame(sheets))]
    d = ex.readout_jump(segs)
    assert d == pytest.approx(-3e-4, abs=5e-6)


def test_stage_arrays_shapes_and_cycle1_switch_on_is_missing():
    rng = np.random.default_rng(1)
    segs = [ex.detrend(s, tail=200) for s in ex.sheet_segments(_frame([_sheet(noise=1e-4, rng=rng), _sheet(shift=6, noise=1e-4, rng=rng)]))]
    x2, x3, x4, xI = ex.stage_arrays(segs, tail=200, n_lags=150, cycles=(1, 2), jump=0.0)
    assert x2.shape == (4, 200) and x3.shape == (4, 150) and x4.shape == (4, 200) and xI.shape == (4, 150)
    assert np.isnan(xI[0::2]).all() and np.isfinite(xI[1::2]).all()                 # cycle 1 rows (even) have no valid switch-on
    # x3 starts at the first unperturbed sample, x4 is the last `tail` samples of the same segment
    assert np.array_equal(x3[0], segs[0][1][:150]) and np.array_equal(x4[3], segs[1][3][-200:])
    x2c, *_ = ex.stage_arrays(segs, tail=200, n_lags=150, cycles=(2,), jump=1e-3)
    assert x2c.shape == (2, 200) and np.allclose(x2c[0], segs[0][2][-200:] - 1e-3)


def test_stage_arrays_rejects_overlapping_windows():
    segs = [ex.detrend(s, tail=200) for s in ex.sheet_segments(_frame([_sheet()]))]
    with pytest.raises(ValueError):
        ex.stage_arrays(segs, tail=300, n_lags=150)


def test_max_inner_step_flags_a_sudden_jump_but_not_the_switches():
    segs = [ex.detrend(s, tail=200) for s in ex.sheet_segments(_frame([_sheet(jump=-3e-4), _sheet(jump=-3e-4)]))]
    segs[1][1][100:] += 2e-3
    m = ex.max_inner_step(segs)
    assert m[0] < 1e-12 and m[1] == pytest.approx(2e-3)
