import numpy as np
from agnw_sim.geometry import Network
from agnw_sim.circuit import solve_circuit, G_CONTACT, junction_currents, total_current


def _make_network(n_wires, junctions, electrode_in, electrode_gnd, isolated_wires=None):
    junctions = np.array(junctions, dtype=int).reshape(-1, 2) if junctions else np.empty((0, 2), dtype=int)
    return Network(
        n_wires=n_wires,
        wire_endpoints=np.zeros((n_wires, 2, 2)),
        junctions=junctions,
        junction_pos=np.zeros((len(junctions), 2)),
        electrode_in=np.array(electrode_in, dtype=int),
        electrode_gnd=np.array(electrode_gnd, dtype=int),
        isolated_wires=np.array(isolated_wires or [], dtype=int),
        area=1e-6,
        wire_length=70e-6,
    )


def test_solve_circuit_series_two_junctions():
    # VIN -- wire0 --(G1)-- wire1 -- GND
    net = _make_network(n_wires=2, junctions=[[0, 1]], electrode_in=[0], electrode_gnd=[1])
    G1 = 1e-3
    V_in = 1.0
    V = solve_circuit(net, G=np.array([G1]), V_in=V_in)

    Gc = G_CONTACT
    R_total = 1 / Gc + 1 / G1 + 1 / Gc
    I = V_in / R_total
    expected_v0 = V_in - I / Gc
    expected_v1 = I / Gc

    np.testing.assert_allclose(V[0], expected_v0, rtol=1e-9)
    np.testing.assert_allclose(V[1], expected_v1, rtol=1e-9)
    np.testing.assert_allclose(V[2], V_in)
    np.testing.assert_allclose(V[3], 0.0)


def test_solve_circuit_floating_component_gets_zero_voltage():
    # wire0-wire1 connected to each other but neither touches an electrode;
    # wire2-wire3 form the actual VIN-GND path.
    net = _make_network(n_wires=4, junctions=[[0, 1], [2, 3]], electrode_in=[2], electrode_gnd=[3])
    V = solve_circuit(net, G=np.array([1e-3, 1e-3]), V_in=1.0)
    assert V[0] == 0.0
    assert V[1] == 0.0


def test_solve_circuit_isolated_wire_does_not_break_solve():
    net = _make_network(n_wires=3, junctions=[[0, 1]], electrode_in=[0], electrode_gnd=[1], isolated_wires=[2])
    V = solve_circuit(net, G=np.array([1e-3]), V_in=1.0)
    assert V[2] == 0.0
    assert np.isfinite(V).all()


def test_solve_circuit_no_junctions_at_all():
    net = _make_network(n_wires=2, junctions=[], electrode_in=[0], electrode_gnd=[1], isolated_wires=[0, 1])
    V = solve_circuit(net, G=np.empty(0), V_in=1.0)
    assert np.isfinite(V).all()
    assert V[2] == 1.0
    assert V[3] == 0.0


def test_junction_currents_matches_ohms_law():
    net = _make_network(n_wires=2, junctions=[[0, 1]], electrode_in=[0], electrode_gnd=[1])
    G = np.array([1e-3])
    V = solve_circuit(net, G, V_in=1.0)
    I = junction_currents(net, G, V)
    expected = G * (V[0] - V[1])
    np.testing.assert_allclose(I, expected)


def test_junction_currents_empty_network():
    net = _make_network(n_wires=2, junctions=[], electrode_in=[0], electrode_gnd=[1], isolated_wires=[0, 1])
    V = solve_circuit(net, np.empty(0), V_in=1.0)
    I = junction_currents(net, np.empty(0), V)
    assert I.shape == (0,)


def test_total_current_matches_junction_current_in_series_circuit():
    net = _make_network(n_wires=2, junctions=[[0, 1]], electrode_in=[0], electrode_gnd=[1])
    G = np.array([1e-3])
    V = solve_circuit(net, G, V_in=1.0)
    I_junction = junction_currents(net, G, V)
    I_tot = total_current(net, G, V)
    np.testing.assert_allclose(I_tot, I_junction[0], rtol=1e-9)


def test_total_current_zero_when_no_electrode_in_wires():
    net = _make_network(n_wires=2, junctions=[[0, 1]], electrode_in=[], electrode_gnd=[1])
    G = np.array([1e-3])
    V = solve_circuit(net, G, V_in=1.0)
    assert total_current(net, G, V) == 0.0
