"""N4 in the network: recorded readout matches a re-solve, determinism, noise reproducibility, direct encoding, and
the framework Fig. 4 metric function."""
import json

import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc.electrodes import make_layout
from agnw_rc.fig4 import AUDIT, FRAMEWORK, loss, metrics_fn, paper_targets
from agnw_rc.filament import conductance
from agnw_rc.multicircuit import MultiTerminalCircuit
from agnw_rc.multidrive import electrode_voltages, simulate_multi


def _spec(rep=0, sigma=0.0):
    return C.RunSpec("iid", 5, 1.75, 1, rep, layout="two_bands", encoding="direct", model="filament", sigma=sigma)


@pytest.fixture(scope="module")
def runs():
    return {k: simulate_multi(s, n_samples=60, washout_steps=300, save_states=True)
            for k, s in {"a": _spec(), "a2": _spec(), "noisy": _spec(sigma=3e-3), "noisy2": _spec(sigma=3e-3)}.items()}


def test_direct_encoding_matches_e1_input():
    spec = _spec()
    u = np.linspace(0, 1, 11)
    np.testing.assert_allclose(electrode_voltages(u, spec, 1)[:, 0], spec.v_lo + (spec.v_hi - spec.v_lo) * u)


def test_recorded_readout_matches_resolve(runs):
    raw = runs["a"]
    fp = C.filament_params(_spec())
    net = C.make_network(1)
    mc = MultiTerminalCircuit(net, C.network_params()[1], make_layout("two_bands", net))
    for k in (20, 59):
        lam = raw["junction_lambda"][k].astype(float) * fp.lambda_crit
        V = mc.solve(conductance(lam, fp), [raw["v_electrodes"][k, 0], 0.0], method="direct")
        np.testing.assert_allclose(raw["current"][k][-2:], mc.electrode_currents(V), rtol=1e-4, atol=1e-10)
        np.testing.assert_allclose(raw["probe_v"][k], V[raw["probe_nodes"]], atol=1e-4)


def test_noise_free_is_deterministic_and_noise_is_seeded(runs):
    np.testing.assert_array_equal(runs["a"]["current"], runs["a2"]["current"])
    np.testing.assert_array_equal(runs["noisy"]["current"], runs["noisy2"]["current"])
    assert not np.array_equal(runs["a"]["junction_lambda"], runs["noisy"]["junction_lambda"])


def test_filament_params_from_config():
    fp = C.filament_params(_spec(sigma=1e-3), V_set=0.2)
    assert fp.V_set == 0.2 and fp.V_reset == pytest.approx(0.1) and fp.sigma == 1e-3
    fw, _ = C.network_params()
    assert (fp.G_on, fp.G_off, fp.dt) == (fw.G_on, fw.G_off, fw.dt)
    assert fp.lambda_max == pytest.approx(1.5 * fp.lambda_crit)
    assert _spec(sigma=3e-3).name.endswith("_fil_sig3e-03")


def test_framework_fig4_metrics_reproduced():
    m = metrics_fn()
    z = np.load(FRAMEWORK / "validation_output/lamas_fig4/N2000_lamas_eps1e-4_seed1.npz")
    got = m(z["V_in"], z["I_total"] * 1e3)
    ref = json.loads((AUDIT / "fig4_metrics.json").read_text())["relaxation eps=1e-4 N=2000"]
    for k in ("Imax", "onset_V", "area_pos", "area_neg", "ratio"):
        assert got[k] == pytest.approx(ref[k], rel=1e-9)
    t = paper_targets()[2000]
    assert loss(t, t) == 0.0
