"""Smoke tests of the two Markov-test drivers on tiny inputs (fixed seeds): the outputs exist, are valid strict JSON and are finite."""
import json

import numpy as np
import pytest

from agnw_sim import exact_chain as ec
from agnw_sim import markov_tests as mt
from scripts import tiny_circuit_exact as tce
from scripts.fdtest_analyze import sanitize


def _bridge():
    params, D = tce.working_model()
    net = ec.bridge_network()
    return net, ec.dynamic_junctions(net, D), params, D


def test_tiny_circuit_driver_end_to_end_on_the_bridge(tmp_path):
    net, ids, params, D = _bridge()
    res = tce.run_tiny(net, ids, params, D, str(tmp_path), v_grid=None, eps_grid=None, v0=0.6, eps=0.05, mc_steps=0,
                       trace_steps=3000, trace_seeds=(1,), n_start=300, cap=600, log=lambda *_: None)
    data = json.loads((tmp_path / "tiny_circuit_exact.json").read_text())
    assert data["circuit"]["n_dynamic"] == 4 and set(data["table"]) == set(tce.OBSERVABLES)
    assert data["table"]["full"]["delta2"] < 1e-10                     # the full state is Markov
    assert data["operating_point"]["geff_drive_dependence"] < 1e-9
    assert (tmp_path / "tiny_circuit_exact.png").exists()
    tr = np.load(tmp_path / "tiny_circuit_mc_trace.npz")
    assert tr["geff_raw"].shape == (1, 3000) and str(tr["control_key"]) in ("geff_20_jit", "on_count_jit")
    assert np.all(np.isfinite(tr["on_count_jit"]))
    assert tce.gate({**res, "validation": {"rows": {"max_deviation_sigma": 1.0}}}) == []


def test_gate_flags_a_nonzero_full_state_delta2():
    bad = tce.gate({"table": {"full": {"delta2": 1e-3}}, "validation": {"rows": {"max_deviation_sigma": 9.0}}})
    assert len(bad) == 2 and "FULL" in bad[0]


def test_exact_table_is_deterministic_and_grows_the_window():
    net, ids, params, D = _bridge()
    T0, pi0, g = tce.chain_at(net, params, D, ids, 0.6)
    Te, pie, _ = tce.chain_at(net, params, D, ids, 0.65)
    labels = tce.build_labels(g, mt.state_bits(4), pi0, pie)
    t1, _ = tce.exact_table(T0, pi0, pie, labels, n_start=100, cap=2 ** 12)
    t2, _ = tce.exact_table(T0, pi0, pie, labels, n_start=100, cap=2 ** 12)
    assert json.dumps(sanitize(t1)) == json.dumps(sanitize(t2)) and t1["on_count"]["n_lags"] > 100


def test_validate_trajectories_runs_on_the_bridge():
    net, ids, params, D = _bridge()
    T0, pi0, g = tce.chain_at(net, params, D, ids, 0.6)
    out = tce.validate_trajectories(net, params, D, ids, 0.6, pi0, seeds=(1,), n_steps=600, burn_in=200, block=100)
    assert out["n_blocks"] == 6 and np.isfinite(out["max_abs_z"])


def test_sample_chain_follows_the_stationary_distribution():
    T = mt.transition_matrix(np.array([[0.2], [0.2]]), np.array([[0.1], [0.1]]))
    pi = mt.stationary_distribution(T)
    st = tce.sample_chain(T, pi, 200_000, np.random.default_rng(0))
    assert abs(st.mean() - pi[1]) < 0.01


# ------------------------------------------------------------------------------------------------ test 4 driver
from scripts import markov_geff_test as mg

META = {"V0": 1.0, "eps": 1e-6, "T_relax": 60, "T_samp": 100, "n_cycles": 40, "record_every": 1, "n_lags": 40,
        "network_seed": 1, "k": 10, "p_relax": 1e-4, "N": 100}


def _write_runs(path, kind, n_runs=5, seed=0):
    """Run files whose I_total is a positive AR series (kind 'ar1': Markov; 'ar2': second order)."""
    n = META["n_cycles"] * 2 * (META["T_relax"] + META["T_samp"])
    rng = np.random.default_rng(seed)
    for i in range(n_runs):
        x, e = np.zeros(n), rng.normal(size=n)
        for t in range(2, n):
            x[t] = (0.8 * x[t - 1] if kind == "ar1" else 0.5 * x[t - 1] + 0.3 * x[t - 2]) + 0.5 * e[t]
        np.savez(path / f"run_N100_seed{i + 1}.npz", I_total=10.0 + x, schedule_json=json.dumps({**META, "dynamics_seed": i + 1}))


def test_geff_series_drops_cycle_zero_and_rebuilds_v_in(tmp_path):
    _write_runs(tmp_path, "ar1", n_runs=1)
    z = np.load(tmp_path / "run_N100_seed1.npz")
    s = mg.geff_series([z], META)[0]
    T = 2 * (META["T_relax"] + META["T_samp"])
    assert len(s) == (META["n_cycles"] - 1) * T
    assert s[0] == pytest.approx(z["I_total"][T] / (META["V0"] + META["eps"]))
    with pytest.raises(ValueError):
        mg.geff_series([{"I_total": np.zeros(10)}], META)


@pytest.mark.parametrize("kind, detected", [("ar1", False), ("ar2", True)])
def test_test4_driver_end_to_end(tmp_path, kind, detected):
    _write_runs(tmp_path, kind)
    rc = mg.main(["--dir", str(tmp_path), "--N", "100", "--lags", "1,3", "--n-boot", "20", "--n-surr", "10", "--n-max", "2000", "--skip-controls"])
    assert rc == 0
    out = json.loads((tmp_path / "markov_geff_test_N100.json").read_text())
    assert (tmp_path / "markov_geff_test_N100.png").exists()
    assert (out["real"]["n_detections"] > 0) == detected
    assert ("memory detected" in out["verdict"]["text"] and "no memory" not in out["verdict"]["text"]) == detected


def test_test4_driver_rejects_bad_input(tmp_path):
    _write_runs(tmp_path, "ar1", n_runs=3)
    with pytest.raises(SystemExit):
        mg.main(["--dir", str(tmp_path), "--N", "100", "--skip-controls"])                # fewer than 4 runs
    _write_runs(tmp_path, "ar1", n_runs=5)
    with pytest.raises(SystemExit):
        mg.main(["--dir", str(tmp_path), "--N", "100", "--lags", "100000", "--skip-controls"])   # lag beyond a tenth of the series


def test_controls_and_verdict_logic():
    ok = {"a": {"required": True, "passed": True, "analysis": {}}, "b": {"required": False, "passed": False, "analysis": {}}}
    bad = {"a": {"required": True, "passed": False, "analysis": {}}}
    real0, real1 = {"n_detections": 0, "lags": []}, {"n_detections": 1, "lags": [{"h": 5, "gain_detected": True, "cmi_detected": False}]}
    assert "no memory" in mg.verdict(real0, ok)["text"]
    assert "h=5" in mg.verdict(real1, ok)["text"]
    assert "no verdict" in mg.verdict(real1, bad)["text"] and mg.verdict(real1, bad)["failed_controls"] == ["a"]


def test_synthetic_controls_pass_on_a_small_size():
    rng = np.random.default_rng(1)
    ctl = mg.run_controls(rng, [1, 3], None, 6000, k=3, n_boot=20, n_surr=10, n_max=1500)
    assert set(ctl) == {"isolated_junction", "nonlinear_markov"} and all(c["passed"] for c in ctl.values())


def test_tiny_circuit_trace_control_is_required_only_when_the_exact_delta2_is_large(tmp_path):
    rng = np.random.default_rng(2)
    x = np.zeros((4, 4000))
    for i in range(4):
        for t in range(2, 4000):
            x[i, t] = 0.5 * x[i, t - 1] + 0.3 * x[i, t - 2] + 0.5 * rng.normal()
    for exact, required in ((0.5, True), (1e-4, False)):
        path = tmp_path / f"trace_{required}.npz"
        np.savez(path, on_count_jit=x, control_key="on_count_jit", control_exact_delta2=exact)
        ctl = mg.run_controls(rng, [1, 3], str(path), 3000, k=3, n_boot=20, n_surr=10, n_max=1500)
        assert ctl["tiny_circuit_trace"]["required"] == required and ctl["tiny_circuit_trace"]["passed"]
