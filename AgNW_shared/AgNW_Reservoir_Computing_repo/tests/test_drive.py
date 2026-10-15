"""Tests against the framework: configuration contract and alignment of the simulated observables."""
import dataclasses

import numpy as np
import pytest

from agnw_sim.presets import lamas2026_fig4

from agnw_rc import config as C
from agnw_rc.drive import (input_stream, load_raw, save_raw, simulate, to_voltage, unpack_flags,
                           virtual_offsets)
from agnw_rc.evaluate import evaluate_consistency, evaluate_iid_run, evaluate_sinesquare_run


def test_config_is_framework_working_model():
    params, wd = C.network_params()
    ref, kw = lamas2026_fig4(1e-3)
    assert params == dataclasses.replace(ref, p_relax=1e-4)
    assert wd == kw["wire_diameter"] == 90e-9
    assert (C.N_WIRES, C.AREA, C.LENGTH, C.DT, C.SOLVER) == (2000, 1e-6, 70e-6, 1e-3, "cached")


def test_network_depends_only_on_seed():
    a, b = C.make_network(5), C.make_network(5)
    np.testing.assert_array_equal(a.wire_endpoints, b.wire_endpoints)
    np.testing.assert_array_equal(a.junctions, b.junctions)
    assert not np.array_equal(a.wire_endpoints, C.make_network(6).wire_endpoints)


def test_virtual_offsets():
    np.testing.assert_array_equal(virtual_offsets(10), np.arange(10))
    off = virtual_offsets(100)
    assert len(off) == 10 and off[0] == 0 and off[-1] == 99
    np.testing.assert_array_equal(virtual_offsets(3), [0, 1, 2])


def test_replicates_share_input_but_not_predrive():
    a, b = C.RunSpec("iid", 10, 1.75, 1, 0), C.RunSpec("iid", 10, 1.75, 1, 1)
    np.testing.assert_array_equal(input_stream(a)[0], input_stream(b)[0])
    from agnw_rc.drive import dynamics_seed, predrive_stream
    assert not np.array_equal(predrive_stream(a), predrive_stream(b))
    assert dynamics_seed(a) != dynamics_seed(b)


@pytest.fixture(scope="module")
def short_runs():
    specs = [C.RunSpec("iid", 5, 1.75, 1, r) for r in (0, 1)] + [C.RunSpec("sinesquare", 5, 1.75, 1, 0)]
    return specs, [simulate(s, n_samples=200, washout_steps=500) for s in specs]


def test_recomputed_current_equals_simulated(short_runs):
    _, (raw, _, _) = short_runs
    np.testing.assert_allclose(raw["i_recomputed"], raw["i_simulated"], rtol=1e-6)
    # the last virtual node is the block's last step, where the snapshot was taken
    np.testing.assert_allclose(raw["current"][:, -1], raw["i_simulated"])


def test_voltage_encoding(short_runs):
    specs, (raw, _, _) = short_runs
    np.testing.assert_allclose(raw["v_in"], to_voltage(raw["u"], specs[0]))
    assert raw["v_in"].min() >= specs[0].v_lo and raw["v_in"].max() <= specs[0].v_hi
    # the scored current trace holds each input for exactly `hold` steps: I/V constant within a block
    # is not guaranteed (junctions switch), but the trace length is
    assert len(raw["i_total_scored"]) == 200 * 5


def test_flags_roundtrip_and_save_load(short_runs, tmp_path):
    specs, (raw, _, _) = short_runs
    flags = unpack_flags(raw)
    assert flags.shape == (200, raw["n_junctions"])
    np.testing.assert_allclose(flags.mean(axis=1), raw["on_fraction"])
    save_raw(specs[0], raw, tmp_path)
    back = load_raw(specs[0], tmp_path)
    np.testing.assert_array_equal(back["flags_packed"], raw["flags_packed"])
    assert float(back["spec_v_center"]) == 1.75


def test_evaluation_smoke(short_runs):
    _, (a, b, ss) = short_runs
    r = evaluate_iid_run(a, floor=0.05)
    for obs in ("current", "probes", "junctions"):
        assert len(r[obs]["memory_function"]) == 31
        assert np.isfinite(r[obs]["narma10_nmse"])
    assert set(evaluate_sinesquare_run(ss)) == {"current", "probes", "junctions"}
    c = evaluate_consistency(a, b)
    assert all(-1 <= v <= 1 for v in c.values())


def test_state_observables_do_not_depend_on_applied_voltage():
    """I/V_in and V_probe/V_in are functions of the junction states only (linear circuit)."""
    from agnw_sim.dualnode import DualNodeCircuit
    params, wd = C.network_params()
    net = C.make_network(1)
    circuit = DualNodeCircuit(net, wd)
    G = np.where(np.random.default_rng(0).random(len(net.junctions)) < 0.25, params.G_on, params.G_off)
    V1, V2 = circuit.solve(G, 1.0), circuit.solve(G, 2.5)
    np.testing.assert_allclose(V2 / 2.5, V1 / 1.0, atol=1e-12)
    assert circuit.total_current(V2) / 2.5 == pytest.approx(circuit.total_current(V1))


def test_experiment_run_names_are_unique():
    """Run names format p_relax / sigma with one significant digit and window centres with two decimals; make sure
    no two specs of the defined experiments collide (a collision would overwrite a raw result)."""
    specs = (C.e1_specs() + C.e2_specs(100, 1.75) + C.e3_specs(1.75) + C.m1_specs() + C.n2_specs()
             + C.n7a_specs() + C.n7b_specs() + C.n5a_specs() + C.n4_1_specs() + C.e4_specs() + C.n10_1_specs() + C.n10m_specs())
    by_name = {}
    for s in specs:
        other = by_name.setdefault(s.name, s)
        assert other == s, f"name collision: {s} vs {other}"
