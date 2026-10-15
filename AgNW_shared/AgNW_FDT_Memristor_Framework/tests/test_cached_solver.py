"""Tests for the cached / rank-update circuit solver.

The reference for every comparison is the existing direct solver
(DualNodeCircuit.solve), which is verified independently in test_dualnode.py,
or a hand-derived formula (linearity in V_in, a series circuit). Accuracy is
judged on the quantities the dynamics actually use: switching probabilities
and the total current.

Achievable agreement (measured, not assumed): the conductances span 1e-9..1e3 S,
so even two CORRECT direct factorizations (different column orderings) differ
by dP_SET ~ 6e-7 and rel. dI_total ~ 7e-6 in an OFF-dominated state (30% ON,
I ~ 1e-9 A), but only by ~2e-8 and ~1e-10 in a conducting state (70-90% ON, I ~ 1e-4 A).
Tests that need tight tolerances therefore use conducting states; the one test
that runs from the all-OFF state uses the measured OFF-regime noise level.
"""
import dataclasses

import numpy as np
import pytest

from agnw_sim.geometry import Network, build_network
from agnw_sim.circuit import G_CONTACT
from agnw_sim.dualnode import DualNodeCircuit, RHO_AG
from agnw_sim.cached_solver import CachedSolver
from agnw_sim.dynamics import SimParams, p_set, p_reset
from agnw_sim.simulate import run_simulation
from agnw_sim.presets import lamas2026_fig4

D_WIRE = 90e-9
PARAMS, _ = lamas2026_fig4()


def _percolating(seed=3, n=1500):
    return build_network(n, 1e-6, 70e-6, np.random.default_rng(seed))


def _probs_and_current(circuit, V, G):
    dV = circuit.junction_voltage_drops(V)
    return (p_set(np.abs(dV), PARAMS), p_reset(np.abs(G * dV), PARAMS),
            circuit.total_current(V))


def _conducting_state(rng, n_junctions, on_fraction=0.9):
    """A junction-state vector with a conducting ON backbone (I ~ 3e-4 A at 2.5 V).
    (70% ON is borderline for percolation of the ON subgraph: some realizations do
    not conduct, hence 90%.)"""
    return np.where(rng.random(n_junctions) < on_fraction, 1e-3, 1e-9)


def _assert_same_physics(circuit, V_cached, V_direct, G, p_tol=1e-6, i_rtol=1e-7):
    ps_c, pr_c, i_c = _probs_and_current(circuit, V_cached, G)
    ps_d, pr_d, i_d = _probs_and_current(circuit, V_direct, G)
    assert np.abs(ps_c - ps_d).max() <= p_tol
    assert np.abs(pr_c - pr_d).max() <= p_tol
    assert abs(i_d) > 1e-6, "state does not conduct: the tight tolerances would be meaningless"
    assert abs(i_c - i_d) <= i_rtol * abs(i_d)


def _two_wire_chain():
    """Hand-checkable series circuit: V_IN wire -- one junction -- GND wire."""
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


def test_unchanged_states_are_rescaled_not_resolved():
    # Break caught: re-factoring on every call, or a wrong V_in scaling. The
    # expectation is hand-derived from linearity: V(V_in) = V_in * V(1).
    net = _percolating()
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit)
    G = _conducting_state(np.random.default_rng(0), net.junctions.shape[0])
    V_unit = solver.solve(G, 1.0)
    n_fact = solver.stats["refactorizations"]
    V = solver.solve(G, 2.5)
    assert solver.stats["refactorizations"] == n_fact
    np.testing.assert_array_equal(V, 2.5 * V_unit)
    _assert_same_physics(circuit, V, circuit.solve(G, 2.5), G)


def test_negative_and_zero_drive():
    # Break caught: sign or zero handling of the linear rescaling.
    net = _percolating()
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit)
    G = _conducting_state(np.random.default_rng(1), net.junctions.shape[0])
    V_unit = solver.solve(G, 1.0)
    np.testing.assert_array_equal(solver.solve(G, -1.7), -1.7 * V_unit)
    assert np.all(solver.solve(G, 0.0) == 0.0)
    _assert_same_physics(circuit, solver.solve(G, -1.7), circuit.solve(G, -1.7), G)


def test_random_flip_sequence_matches_direct_solver():
    # Break caught: wrong rank-update algebra (signs, incidence vectors,
    # stale cached columns) after a few junctions change.
    net = _percolating()
    nJ = net.junctions.shape[0]
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit)
    rng = np.random.default_rng(2)
    G = _conducting_state(rng, nJ)
    calls = 0
    for step in range(40):
        for j in rng.choice(nJ, size=int(rng.integers(0, 4)), replace=False):
            G[j] = 1e-3 if G[j] < 5e-4 else 1e-9
        V_in = float(rng.choice([0.7, 1.9, 2.5, 3.6]))
        V = solver.solve(G, V_in)
        calls += 1
        _assert_same_physics(circuit, V, circuit.solve(G, V_in), G)
    stats = solver.stats
    assert stats["rank_updates"] > 0                       # the fast path was really used
    assert stats["refactorizations"] < calls
    # With correct algebra the residual guard never fires on well-conditioned flips.
    # (A wrong update would be rescued by the guard -- correct answers, silently slow --
    # so the counter, not the physics, is what exposes such a bug.)
    assert stats["guard_rejections"] == 0
    assert stats["refactorizations"] + stats["rank_updates"] + stats["rescales"] == calls


def test_residual_guard_rejection_falls_back_to_a_fresh_solve():
    # Break caught: the guard is missing, or its fallback returns a wrong/None
    # result. An unreachable tolerance forces every rank update to be rejected.
    net = _percolating()
    nJ = net.junctions.shape[0]
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit, residual_tol=1e-30)
    rng = np.random.default_rng(5)
    G = _conducting_state(rng, nJ)
    solver.solve(G, 2.5)
    for j in rng.choice(nJ, size=2, replace=False):
        G[j] = 1e-3 if G[j] < 5e-4 else 1e-9
    V = solver.solve(G, 2.5)
    assert solver.stats["guard_rejections"] >= 1
    assert solver.stats["rank_updates"] == 0
    _assert_same_physics(circuit, V, circuit.solve(G, 2.5), G)


def test_many_simultaneous_flips_fall_back_and_stay_accurate():
    # Break caught: no limit on the update rank (cost explosion) or a wrong
    # fallback path.
    net = _percolating()
    nJ = net.junctions.shape[0]
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit, max_updates=8)
    rng = np.random.default_rng(3)
    G = _conducting_state(rng, nJ)
    solver.solve(G, 2.5)
    before = solver.stats["refactorizations"]
    flip = rng.choice(nJ, size=300, replace=False)
    G[flip] = np.where(G[flip] < 5e-4, 1e-3, 1e-9)
    V = solver.solve(G, 2.5)
    assert solver.stats["refactorizations"] == before + 1
    _assert_same_physics(circuit, V, circuit.solve(G, 2.5), G)


@pytest.mark.parametrize("start_on", [True, False])
def test_bridge_junction_switching_is_accurate(start_on):
    # Break caught: catastrophic cancellation in the rank update when the only
    # junction of a series circuit switches (effective resistance jumps from
    # ~1e3 to ~1e9 ohm); the residual guard must fall back to a fresh solve.
    net = _two_wire_chain()
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit)
    G_a = np.array([1e-3 if start_on else 1e-9])
    G_b = np.array([1e-9 if start_on else 1e-3])
    solver.solve(G_a, 1.0)
    V = solver.solve(G_b, 1.0)
    area = np.pi * (D_WIRE / 2) ** 2
    R_wires = RHO_AG * (30e-6 + 35e-6) / area              # hand-derived: 30 um + 35 um of wire
    R_total = 2.0 / G_CONTACT + R_wires + 1.0 / G_b[0]
    assert circuit.total_current(V) == pytest.approx(1.0 / R_total, rel=1e-6)


def test_floating_clusters_are_zero_like_the_direct_solver():
    # Break caught: singular matrix / NaN from wire clusters that touch no
    # electrode (a sub-percolating network has many).
    net = build_network(300, 1e-6, 70e-6, np.random.default_rng(4))
    nJ = net.junctions.shape[0]
    circuit = DualNodeCircuit(net, D_WIRE)
    solver = CachedSolver(circuit)
    rng = np.random.default_rng(4)
    G = np.where(rng.random(nJ) < 0.5, 1e-3, 1e-9)
    solver.solve(G, 1.0)
    for j in rng.choice(nJ, size=3, replace=False):
        G[j] = 1e-3 if G[j] < 5e-4 else 1e-9
    V = solver.solve(G, 2.0)
    assert np.isfinite(V).all()
    np.testing.assert_allclose(V, circuit.solve(G, 2.0), rtol=1e-6, atol=1e-9)


def test_run_simulation_cached_solver_reproduces_direct_trajectory():
    # Break caught: the option changes the physics of a full run.
    net_seed = 3
    params = dataclasses.replace(SimParams(P0=1.0, alpha=1000.0, V_th=0.001, dt=1e-3),
                                 p_relax=2e-2)
    common = dict(n_wires=1500, area=1e-6, length=70e-6, waveform_fn=lambda t: 1.0,
                  n_steps=80, dt=1e-3, params=params, wire_diameter=D_WIRE,
                  n_sampled_junctions=100)
    direct = run_simulation(rng=np.random.default_rng(net_seed), **common)
    cached = run_simulation(rng=np.random.default_rng(net_seed), solver="cached", **common)
    assert direct.solver_stats is None
    assert cached.solver_stats["refactorizations"] >= 1
    np.testing.assert_array_equal(cached.G_sample, direct.G_sample)
    # the run starts all-OFF (I ~ 1e-10 A, the ill-conditioned regime): two correct
    # direct solves already differ by ~1e-5 relative there (see module docstring)
    np.testing.assert_allclose(cached.I_total, direct.I_total, rtol=1e-4, atol=1e-30)


def test_solver_option_is_validated():
    # Break caught: silently ignoring a cached request without the dual-node
    # model, or an unknown solver name.
    common = dict(n_wires=50, area=1e-6, length=70e-6, waveform_fn=lambda t: 0.5,
                  n_steps=3, dt=1e-3, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        run_simulation(solver="cached", **common)               # no wire_diameter
    with pytest.raises(ValueError):
        run_simulation(solver="bogus", wire_diameter=D_WIRE, **common)
