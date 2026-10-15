"""Exact one-step data of a small circuit (agnw_sim.exact_chain) against the package dynamics."""
import dataclasses

import numpy as np
import pytest

from agnw_sim import exact_chain as ec
from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.dynamics import update_dynamics
from agnw_sim.markov_tests import state_bits, transition_matrix, stationary_distribution
from agnw_sim.presets import lamas2026_fig4

PARAMS, KW = lamas2026_fig4(1e-3)
PARAMS = dataclasses.replace(PARAMS, p_relax=0.05)          # large, so that dropping the relaxation term is detectable
D = KW["wire_diameter"]


def test_bridge_network_has_four_dynamic_junctions():
    assert list(ec.dynamic_junctions(ec.bridge_network(), D)) == [0, 1, 2, 3]


def test_no_dynamic_junctions_without_a_path_between_the_electrodes():
    net = dataclasses.replace(ec.bridge_network(), electrode_gnd=np.array([], dtype=int))
    assert ec.dynamic_junctions(net, D).size == 0


def test_g_eff_does_not_depend_on_the_drive_voltage():
    net = ec.bridge_network()
    ids = ec.dynamic_junctions(net, D)
    g1 = ec.state_data(net, PARAMS, D, ids, 0.5)[2]
    g2 = ec.state_data(net, PARAMS, D, ids, 0.9)[2]
    assert np.abs(g1 - g2).max() < 1e-9 * g1.max() and g1.max() > 100 * g1.min()


def test_too_many_junctions_refused():
    net = ec.bridge_network()
    with pytest.raises(ValueError):
        ec.state_data(net, PARAMS, D, np.arange(ec.MAX_JUNCTIONS + 1), 0.5)


def test_transition_rows_match_update_dynamics_monte_carlo():
    net = ec.bridge_network()
    ids = ec.dynamic_junctions(net, D)
    V = 0.6
    p_on, p_off, _ = ec.state_data(net, PARAMS, D, ids, V)
    T = transition_matrix(p_on, p_off)
    assert np.allclose(T.sum(axis=1), 1.0) and np.all(T >= 0)
    circuit = DualNodeCircuit(net, D)
    bits = state_bits(4)
    rng = np.random.default_rng(0)
    M = 200_000
    for s in (0, 5, 10, 15):
        G = np.where(bits[s] == 1, PARAMS.G_on, PARAMS.G_off)
        dV = circuit.junction_voltage_drops(circuit.solve(G, V))
        new = update_dynamics(np.tile(G, M), np.tile(np.abs(dV), M), np.tile(G * dV, M), PARAMS, rng).reshape(M, 4)
        idx = ((new > PARAMS.G_off) * (1 << np.arange(4))).sum(axis=1)
        emp = np.bincount(idx, minlength=16) / M
        sigma = np.sqrt(np.maximum(T[s] * (1 - T[s]), 1e-12) / M)
        assert np.all(np.abs(emp - T[s]) < 5 * sigma + 2e-6), (s, np.abs(emp - T[s]).max())


def test_stationary_distribution_of_the_circuit_chain():
    net = ec.bridge_network()
    ids = ec.dynamic_junctions(net, D)
    p_on, p_off, _ = ec.state_data(net, PARAMS, D, ids, 0.6)
    T = transition_matrix(p_on, p_off)
    pi = stationary_distribution(T)
    assert np.allclose(pi @ T, pi, atol=1e-12)
