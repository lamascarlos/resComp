import numpy as np
import pytest
from agnw_sim.geometry import Network, build_network
from agnw_sim.circuit import solve_circuit, total_current, G_CONTACT
from agnw_sim.dualnode import DualNodeCircuit, RHO_AG
from agnw_sim.dynamics import SimParams
from agnw_sim.simulate import run_simulation
from agnw_sim.presets import lamas2026_fig4


def _two_wire_chain():
    """Wire 0 (horizontal, touches V_IN at x=0) crosses wire 1 (touches GND at
    its far end) at (30 um, 0). Junction is at t=30 um on wire 0 and at the
    midpoint (t=35 um) of wire 1: a hand-checkable series circuit."""
    L = 70e-6
    w0 = np.array([[0.0, 0.0], [L, 0.0]])
    mid = np.array([30e-6, 0.0])
    d = np.array([0.6, 0.8])
    w1 = np.array([mid - 0.5 * L * d, mid + 0.5 * L * d])
    return Network(
        n_wires=2, wire_endpoints=np.stack([w0, w1]),
        junctions=np.array([[0, 1]]), junction_pos=mid[None, :],
        electrode_in=np.array([0]), electrode_gnd=np.array([1]),
        isolated_wires=np.array([], dtype=int), area=1e-8, wire_length=L,
    )


def test_series_chain_matches_hand_calculation():
    net = _two_wire_chain()
    D = 100e-9
    area_cs = np.pi * (D / 2) ** 2
    G_j = 1e-3
    R_expected = (1 / G_CONTACT + RHO_AG * 30e-6 / area_cs + 1 / G_j
                  + RHO_AG * 35e-6 / area_cs + 1 / G_CONTACT)
    circuit = DualNodeCircuit(net, D)
    V = circuit.solve(np.array([G_j]), 1.0)
    np.testing.assert_allclose(circuit.total_current(V), 1.0 / R_expected, rtol=1e-9)
    # junction voltage drop is I / G_j, signed from wire-0 node to wire-1 node
    drops = circuit.junction_voltage_drops(V)
    np.testing.assert_allclose(drops, (1.0 / R_expected) / G_j, rtol=1e-9)


def test_wire_resistance_lowers_current_vs_ideal_wires():
    net = _two_wire_chain()
    G = np.array([1e-3])
    I_ideal = total_current(net, G, solve_circuit(net, G, 1.0))
    c = DualNodeCircuit(net, 50e-9)
    I_wired = c.total_current(c.solve(G, 1.0))
    assert I_wired < I_ideal


def test_fat_wires_recover_wire_as_node_model():
    rng = np.random.default_rng(3)
    # N=1500 (seed 3) percolates; below ~1150 wires there is no V_IN-GND path and
    # both currents are numerical zero, which would make this test vacuous.
    net = build_network(1500, 1e-6, 70e-6, rng)
    nJ = net.junctions.shape[0]
    G = np.full(nJ, 1e-3)                  # all ON: well-conditioned comparison
    V_ref = solve_circuit(net, G, 1.0)
    I_ref = total_current(net, G, V_ref)
    assert I_ref > 1e-4                    # guard: network actually conducts
    c = DualNodeCircuit(net, 10e-6)        # 10 um "wires": negligible resistance
    V = c.solve(G, 1.0)
    np.testing.assert_allclose(c.total_current(V), I_ref, rtol=2e-3)
    ref_drop = V_ref[net.junctions[:, 0]] - V_ref[net.junctions[:, 1]]
    np.testing.assert_allclose(c.junction_voltage_drops(V), ref_drop, atol=1e-3 * abs(ref_drop).max())


def test_no_junctions_or_electrodes_does_not_crash():
    rng = np.random.default_rng(2)
    net = build_network(2, 1e-6, 1e-9, rng)
    c = DualNodeCircuit(net, 90e-9)
    V = c.solve(np.zeros(net.junctions.shape[0]), 1.0)
    assert np.isfinite(V).all()
    assert c.total_current(V) == 0.0 or np.isfinite(c.total_current(V))


def test_rejects_nonpositive_diameter():
    net = _two_wire_chain()
    with pytest.raises(ValueError):
        DualNodeCircuit(net, 0.0)
    with pytest.raises(ValueError):
        DualNodeCircuit(net, -1e-9)


def test_run_simulation_with_wire_diameter():
    rng = np.random.default_rng(1)
    result = run_simulation(
        n_wires=150, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.5, n_steps=20, dt=1e-3, rng=rng,
        wire_diameter=90e-9,
    )
    assert result.wire_diameter == 90e-9
    assert result.I_total.shape == (20,)
    assert np.isfinite(result.I_total).all()


def test_run_simulation_default_is_ideal_wires():
    result = run_simulation(
        n_wires=50, area=1e-6, length=70e-6, waveform_fn=lambda t: 0.5,
        n_steps=3, dt=1e-3, rng=np.random.default_rng(1),
    )
    assert result.wire_diameter is None


def test_lamas_preset_values():
    params, kwargs = lamas2026_fig4()
    assert isinstance(params, SimParams)
    assert params.dt == 1e-3
    np.testing.assert_allclose(params.dt * params.P0, 0.4)
    assert params.alpha == 1.0
    assert params.V_th == 0.1                 # inferred (Table 1 prints 0.01 V)
    assert kwargs["wire_diameter"] == 90e-9   # inferred (paper gives none)
