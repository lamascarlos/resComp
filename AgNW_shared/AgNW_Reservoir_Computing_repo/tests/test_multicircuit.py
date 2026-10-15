"""N5 multi-terminal circuit: the two-band special case reproduces the framework circuit exactly; electrode
geometry; superposition and current conservation."""
import numpy as np
import pytest

from agnw_sim.dualnode import DualNodeCircuit

from agnw_rc import config as C
from agnw_rc.electrodes import Electrode, _clip_segment_to_box, pad_grid, split_bands, two_bands
from agnw_rc.multicircuit import MultiTerminalCircuit


@pytest.fixture(scope="module")
def setup():
    params, wd = C.network_params()
    net = C.make_network(1)
    return params, wd, net


def _random_G(params, n, seed, frac=0.25):
    return np.where(np.random.default_rng(seed).random(n) < frac, params.G_on, params.G_off)


def test_two_bands_reproduces_framework_circuit(setup):
    """Same graph (node numbering, reduced matrix to round-off) as DualNodeCircuit. The solutions agree to the
    conditioning of the system (conductances span 1e-9..1e3 S): both satisfy Kirchhoff's law to ~1e-14 A, and
    differ by < 1e-5 V (the readout resolution)."""
    import scipy.sparse as sp
    params, wd, net = setup
    ref = DualNodeCircuit(net, wd)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    assert mc.n == ref.n                                  # same node count and numbering
    assert list(mc.super) == [ref.vin, ref.gnd]
    n = ref.n
    for seed in range(3):
        G = _random_G(params, len(net.junctions), seed)
        off = ref._fixed + sp.coo_matrix((np.r_[-G, -G], (np.r_[ref.jn_a, ref.jn_b], np.r_[ref.jn_b, ref.jn_a])),
                                         shape=(n, n)).tocsr()
        Y = off + sp.diags(-np.asarray(off.sum(axis=1)).ravel())
        Y_ff = Y[mc.free][:, mc.free]
        assert abs(mc._matrix(G) - Y_ff).max() < 1e-12 * abs(Y_ff).max()
        V_ref = ref.solve(G, 2.0)
        V = mc.solve(G, [2.0, 0.0], method="direct")
        assert np.abs((Y @ V)[mc.free]).max() < 1e-12            # KCL residual (A)
        np.testing.assert_allclose(V, V_ref, atol=1e-5)
        I = mc.electrode_currents(V)
        assert I[0] == pytest.approx(ref.total_current(V_ref), rel=1e-5)


def test_current_conservation_and_superposition(setup):
    params, wd, net = setup
    mc = MultiTerminalCircuit(net, wd, split_bands(net))
    G = _random_G(params, len(net.junctions), 7)
    k = len(mc.fixed)
    rng = np.random.default_rng(0)
    va, vb = rng.uniform(0, 2, k), rng.uniform(0, 2, k)
    Va, Vb, Vab = (mc.solve(G, v, method="direct") for v in (va, vb, va + vb))
    np.testing.assert_allclose(Vab, Va + Vb, atol=1e-9)             # linear in the electrode voltages
    I = mc.electrode_currents(Va)
    assert abs(I.sum()) < 1e-9 * np.abs(I).max()                    # Kirchhoff: total current is zero


def test_reuse_equals_direct(setup):
    params, wd, net = setup
    mc = MultiTerminalCircuit(net, wd, pad_grid(net))
    G1 = _random_G(params, len(net.junctions), 1)
    G2 = G1.copy(); G2[:5] = params.G_on
    v = np.linspace(0, 2, len(mc.fixed))
    for G in (G1, G1, G2, G2):
        np.testing.assert_allclose(mc.solve(G, v), mc.solve(G, v, method="direct"), atol=1e-12)
    mc.solve(G2, v); mc.solve(G2, v)
    assert mc.stats["reuses"] >= 1


def test_pcg_matches_direct(setup):
    params, wd, net = setup
    mc = MultiTerminalCircuit(net, wd, pad_grid(net))
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 0.3, len(net.junctions))
    v = rng.uniform(0, 3, len(mc.fixed))
    worst = 0.0
    for _ in range(10):
        P = np.clip(P + rng.normal(0, 0.003, P.shape), 0, 1)
        G = params.G_off + P * (params.G_on - params.G_off)
        worst = max(worst, np.abs(mc.solve(G, v, method="pcg") - mc.solve(G, v, method="direct")).max())
    assert worst < 1e-5


def test_layout_geometry(setup):
    _, _, net = setup
    sb = split_bands(net)
    tb = two_bands(net)
    assert [e.role for e in sb] == ["in"] * 8 + ["gnd"] * 8
    # the split segments partition the framework bands exactly
    assert sorted(np.concatenate([e.wires for e in sb[:8]])) == sorted(tb[0].wires)
    assert sorted(np.concatenate([e.wires for e in sb[8:]])) == sorted(tb[1].wires)
    pads = pad_grid(net)
    assert len(pads) == 16 and sum(e.role == "in" for e in pads) == 8
    assert all(len(e.wires) > 0 for e in pads)                       # every pad touches wires at this density
    for e in pads:
        assert np.all((e.t >= 0) & (e.t <= net.wire_length))


def test_clip_segment_to_box():
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert _clip_segment_to_box(np.array([-1.0, 0.5]), np.array([2.0, 0.5]), lo, hi) == pytest.approx((1 / 3, 2 / 3))
    assert _clip_segment_to_box(np.array([2.0, 2.0]), np.array([3.0, 3.0]), lo, hi) is None
    assert _clip_segment_to_box(np.array([0.2, 0.2]), np.array([0.4, 0.4]), lo, hi) == pytest.approx((0.0, 1.0))


def test_electrode_role_validation():
    with pytest.raises(ValueError):
        Electrode("x", "bogus", np.array([1]), np.array([0.0]))
