"""N11 strip protocols (agnw_rc/n11sim.py)."""
import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc.n11sim import SAMPLE_DT, run_constant, strip_circuit
from agnw_rc.volatile import junction_kd0


def _setup(**kw):
    s, mc = strip_circuit(1, 1.5, 0.3)
    vp = C.volatile_params(None, tau_s=0.3, tau_spread=1.0, dt=2e-3, **kw)
    return s, mc, vp, junction_kd0(vp, len(s.network.junctions), 1)


def test_sampling_times_and_reproducibility():
    s, mc, vp, kd0 = _setup()
    a = run_constant(mc, vp, kd0, 3.0, 3.0)
    b = run_constant(mc, vp, kd0, 3.0, 3.0)
    assert len(a["t"]) == 5 and np.allclose(np.diff(a["t"]), SAMPLE_DT)
    np.testing.assert_allclose(a["G"], b["G"], rtol=1e-4)     # pcg warm start depends on solve history
    assert np.all(a["G"] > 0) and a["mean_rupture_rate"] == 0.0


def test_tiny_bias_does_not_switch():
    s, mc, vp, kd0 = _setup()
    a = run_constant(mc, vp, kd0, 0.01, 3.0)
    # x relaxes from 0 toward k_p0 / (k_p0 + k_d) ~ 1e-6: G moves by ~ 1e-6 x ln(G_on / G_off) ~ 1e-5, not more
    assert np.std(a["G"]) / np.mean(a["G"]) < 1e-4 and np.isnan(a["P_formed_median"])


def test_continuing_from_the_state_equals_one_long_run():
    s, mc, vp, kd0 = _setup()
    one = run_constant(mc, vp, kd0, 3.0, 2.4)
    a = run_constant(mc, vp, kd0, 3.0, 1.2)
    b = run_constant(mc, vp, kd0, 3.0, 1.2, state=a["state"], t0=1.2)
    np.testing.assert_allclose(np.r_[a["t"], b["t"]], one["t"])
    # equal to the pcg precision (warm start depends on the solve history; currents of ~1e-9 S networks: ~3e-5 rel.)
    np.testing.assert_allclose(np.r_[a["G"], b["G"]], one["G"], rtol=1e-4)


def test_rupture_rate_is_recorded_per_law():
    s, mc, vp, kd0 = _setup(rupture_rate=0.5)
    a = run_constant(mc, vp, kd0, 3.0, 1.2, rng=np.random.default_rng(0))
    assert a["mean_rupture_rate"] == pytest.approx(0.5)
    _, _, vp2, _ = _setup(rupture_rate=0.5, rupture_law="power", rupture_alpha=1.0, rupture_P0=1e-9)
    b = run_constant(mc, vp2, kd0, 3.0, 1.2, rng=np.random.default_rng(0))
    assert np.isfinite(b["mean_rupture_rate"]) and b["mean_rupture_rate"] > 0
