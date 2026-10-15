"""N11 strip geometry (agnw_rc/strip.py, docs/PLAN_N11_CALIBRATION.md section 3)."""
import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc.electrodes import two_bands
from agnw_rc.multicircuit import MultiTerminalCircuit
from agnw_rc.strip import make_strip


def _same_network(a, b):
    assert a.n_wires == b.n_wires and a.wire_length == b.wire_length
    assert a.area == pytest.approx(b.area, rel=1e-12)            # L_x L_y vs AREA side^2: last bit may differ
    for f in ("wire_endpoints", "junctions", "junction_pos", "electrode_in", "electrode_gnd", "isolated_wires"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_square_strip_is_the_framework_network_bit_for_bit():
    for seed, side in ((1, 1.0), (2, 1.0), (3, 2.0)):
        s = make_strip(seed, side, side)
        ref = C.make_network(seed) if side == 1.0 else C.make_network(seed, side=side)
        _same_network(s.network, ref)
        for e, r in zip(s.electrodes, two_bands(ref)):
            assert (e.name, e.role) == (r.name, r.role)
            np.testing.assert_array_equal(e.wires, r.wires)
            np.testing.assert_array_equal(e.t, r.t)


def test_strip_has_the_working_density_and_bands_at_its_short_ends():
    s = make_strip(1, 3.0, 0.5)
    ends = s.network.wire_endpoints
    assert s.network.n_wires == round(C.N_WIRES * 3.0 * 0.5) and s.length == 3e-3 and s.width == 0.5e-3
    c = ends.mean(axis=1)
    assert c[:, 0].min() >= 0 and c[:, 0].max() <= 3e-3 and c[:, 1].min() >= 0 and c[:, 1].max() <= 0.5e-3
    ein, egnd = s.electrodes[0].wires, s.electrodes[1].wires
    assert np.all(ends[ein, :, 0].min(axis=1) <= 0.05 * 3e-3)
    assert np.all(ends[egnd, :, 0].max(axis=1) >= 3e-3 - 0.05 * 3e-3)
    assert len(ein) > 20 and len(egnd) > 20 and not set(ein) & set(egnd)


def test_strip_conducts_in_the_multi_terminal_circuit():
    s = make_strip(1, 1.5, 0.3)
    _, wd = C.network_params()
    mc = MultiTerminalCircuit(s.network, wd, s.electrodes)
    V = mc.solve(np.full(len(s.network.junctions), 1e-4), np.array([1.0, 0.0]), method="pcg")
    I = mc.electrode_currents(V)
    assert I[mc.fixed_electrodes[0]] > 0
