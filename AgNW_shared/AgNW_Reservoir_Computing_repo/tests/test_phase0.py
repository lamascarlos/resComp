"""Phase 0 of the decisive plan (review 2026-09-27): washout, stride, dense virtual nodes, criteria, drift."""
from dataclasses import replace

import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc import criteria as K
from agnw_rc.drive import predrive_stream, virtual_offsets
from agnw_rc.filament import step_lambda
from agnw_rc.multidrive import simulate_multi


def test_model_washout_covers_three_forgetting_times():
    assert C.model_washout_steps("stochastic", 1e-4) == 30_000
    assert C.model_washout_steps("stochastic", 1e-5) == 300_000
    d = C.N4_PARAMS
    t_dissolve = 1.5 * d["lambda_crit"] / (d["b"] * d["V_set"] / 2)
    assert C.model_washout_steps("filament") * C.DT == pytest.approx(3 * t_dissolve, rel=1e-3)


def test_e4_specs_washout_is_whole_blocks_and_predrive_matches():
    for s in C.e4_specs():
        assert s.washout % s.hold == 0 and s.washout >= C.model_washout_steps(s.model)
        assert s.hold % s.stride == 0
        assert len(predrive_stream(s)) * s.hold == s.washout
    assert {s.stride for s in C.e4_specs(("filament",))} == {C.E4_STRIDE}


def test_dense_virtual_nodes():
    assert len(virtual_offsets(3000, 50)) == 50
    assert len(virtual_offsets(30, 50)) == 30               # never more nodes than steps
    s = C.RunSpec("iid", 3000, 1.75, 1, n_virtual=50)
    assert s.n_virtual_eff == 50 and C.RunSpec("iid", 30, 1.75, 1).n_virtual_eff == C.N_VIRTUAL


def test_filament_stride_is_one_euler_step_of_stride_ms():
    """filament_params(stride=k) gives dt = k ms: one coarse step equals the explicit Euler step of length k ms."""
    fp1, fp10 = C.filament_params(stride=1), C.filament_params(stride=10)
    assert fp10.dt == pytest.approx(10 * fp1.dt)
    rng = np.random.default_rng(0)
    lam = np.full(5, 0.3 * fp1.lambda_crit)
    V = np.linspace(-0.2, 0.2, 5)
    from agnw_rc.filament import lambda_rate
    want = np.clip(lam + 10 * C.DT * lambda_rate(V, lam, fp1), -fp1.lambda_max, fp1.lambda_max)
    np.testing.assert_allclose(step_lambda(lam, V, fp10, rng), want, rtol=1e-12)


def test_stride_rejected_for_stochastic_and_non_divisible_hold():
    spec = C.RunSpec("iid", 30, 1.75, 1, layout="two_bands", encoding="direct", stride=10, n_samples=5)
    with pytest.raises(ValueError):
        simulate_multi(spec, n_samples=2, washout_steps=30)
    spec = replace(spec, model="filament", hold=35)
    with pytest.raises(ValueError):
        simulate_multi(spec, n_samples=2, washout_steps=35)


def test_drift():
    assert K.drift(np.full(100, 0.3)) == 0.0
    on = np.linspace(0.3, 0.33, 100)
    assert K.drift(on) == pytest.approx((on[-20:].mean() - on[:20].mean()) / on.mean())


def _run(hold, vc, rep, mc, ipc, narma, val_bonus=0.0, stream="iid", acc=None):
    s = {"MC": mc, "IPC2": ipc, "narma10_nmse": narma, "val_MC": mc + val_bonus, "val_IPC2": ipc,
         "memory_function": [1.0, 0.5, 0.3, 0.0]}
    if stream == "sinesquare":
        s = {"accuracy": acc}
    return {"stream": stream, "hold": hold, "v_center": vc, "replicate": rep, "scores": {"current": s}}


def test_operating_point_is_chosen_on_validation_not_test():
    # config A has the better TEST scores, config B the better VALIDATION scores: B must be chosen
    runs = [_run(300, 2.25, r, 3.0, 2.0, 0.5, val_bonus=-2.5) for r in range(4)]
    runs += [_run(1000, 2.25, r, 1.0, 1.0, 0.8, val_bonus=+1.0) for r in range(4)]
    assert K.select_operating_point(runs, "current") == (1000, 2.25)


def test_criteria_levels():
    runs = [_run(300, 2.25, r, 3.0 + 0.1 * r, 2.0, 0.5) for r in range(4)]
    runs += [_run(300, 2.25, r, 0, 0, 0, stream="sinesquare", acc=0.97) for r in range(2)]
    cons = [{"hold": 300, "v_center": 2.25, "consistency": {"current_state_inc": 0.9}}]
    r = K.evaluate(runs, cons, "current", n_channels=50, esn_totals={32: 8.0, 200: 12.0}, floor=0.1)
    assert r["G0"] and r["G1"] and r["esn_units"] == 200 and not r["G2"]    # 5.15 < 0.5 * 12
    r = K.evaluate(runs, cons, "current", n_channels=20, esn_totals={32: 8.0, 200: 12.0}, floor=0.1)
    assert r["G2"]
    r = K.evaluate(runs, cons, "current", n_channels=20, esn_totals={32: 8.0}, floor=0.4)
    assert r["G0"] and not r["G1"]                           # only one lag above the floor
    cons[0]["consistency"]["current_state_inc"] = 0.5
    r = K.evaluate(runs, cons, "current", n_channels=20, esn_totals={32: 8.0}, floor=0.1)
    assert not r["G0"] and not r["G1"] and not r["G2"]
