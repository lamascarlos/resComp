"""N5 multi-electrode drive: encodings, and a short end-to-end run whose recorded readout matches a re-solve."""
import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc.electrodes import make_layout
from agnw_rc.evaluate import evaluate_consistency, evaluate_iid_run, state_observables
from agnw_rc.multicircuit import MultiTerminalCircuit
from agnw_rc.multidrive import electrode_voltages, input_weights, simulate_multi


def test_weights_encoding_stays_in_window():
    spec = C.RunSpec("iid", 10, 1.5, 2, layout="pads", encoding="weights")
    u = np.random.default_rng(0).uniform(size=500)
    V = electrode_voltages(u, spec, 8)
    assert V.shape == (500, 8)
    assert V.min() >= spec.v_lo - 1e-12 and V.max() <= spec.v_hi + 1e-12
    w = input_weights(spec, 8)
    np.testing.assert_allclose(V, 1.5 + 0.5 * (2 * u[:, None] - 1) * w[None, :])
    np.testing.assert_array_equal(w, input_weights(C.RunSpec("iid", 30, 1.0, 2, 1, layout="pads"), 8))


def test_delays_encoding():
    spec = C.RunSpec("iid", 10, 1.5, 2, layout="pads", encoding="delays")
    u_pre = np.array([0.1, 0.2, 0.3])
    u = np.array([0.6, 0.7, 0.8, 0.9])
    V = electrode_voltages(u, spec, 3, u_before=u_pre)
    x = lambda a: 1.5 + 0.5 * (2 * np.asarray(a) - 1)
    np.testing.assert_allclose(V[:, 0], x(u))                        # d = 0
    np.testing.assert_allclose(V[:, 1], x([0.3, 0.6, 0.7, 0.8]))    # d = 1
    np.testing.assert_allclose(V[:, 2], x([0.2, 0.3, 0.6, 0.7]))    # d = 2


@pytest.fixture(scope="module")
def short_runs():
    specs = [C.RunSpec("iid", 5, 1.0, 1, r, layout="split_bands") for r in (0, 1)]
    return specs, [simulate_multi(s, n_samples=80, washout_steps=300, save_states=True) for s in specs]


def test_recorded_currents_match_resolve(short_runs):
    specs, (raw, _) = short_runs
    params, wd = C.network_params()
    net = C.make_network(1)
    els = make_layout("split_bands", net)
    mc = MultiTerminalCircuit(net, wd, els)
    flags = np.unpackbits(raw["flags_packed"], axis=1, count=int(raw["n_junctions"])).astype(bool)
    driven = [k for k, i in enumerate(mc.fixed_electrodes) if els[i].role == "in"]
    for k in (10, 40, 79):
        Vf = np.zeros(len(mc.fixed)); Vf[driven] = raw["v_electrodes"][k]
        V = mc.solve(np.where(flags[k], params.G_on, params.G_off), Vf, method="direct")
        n_e = int(raw["n_electrodes"])
        np.testing.assert_allclose(raw["current"][k][-n_e:], mc.electrode_currents(V), rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(raw["probe_v"][k], V[raw["probe_nodes"]], atol=1e-6)
    assert abs(raw["current"][10][-n_e:].sum()) < 1e-9                 # charge conservation


def test_multi_evaluation_smoke(short_runs):
    _, (a, b) = short_runs
    r = evaluate_iid_run(a, floor=0.05)
    assert set(r) == {"current", "probes", "junctions", "electrodes"}       # electrodes: multi-electrode readout (N10-E)
    S = state_observables(a)
    assert S["current"].shape == a["current"].shape
    c = evaluate_consistency(a, b)
    assert all(np.isnan(v) or -1 <= v <= 1 for v in c.values())
    assert {"current_state_inc", "probes_state_inc", "junctions_state_inc"} <= set(c)
