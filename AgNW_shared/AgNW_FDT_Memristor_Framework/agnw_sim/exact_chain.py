"""Exact one-step transition data of a small dual-node circuit: every one of the 2**J junction states of the J junctions
connected to both electrodes is solved once per drive voltage (agnw_sim.dualnode), and the per-junction flip probabilities
of agnw_sim.dynamics are read off. Feed the result to agnw_sim.markov_tests.transition_matrix."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.dynamics import p_set, p_reset
from agnw_sim.geometry import Network
from agnw_sim.markov_tests import state_bits

MAX_JUNCTIONS = 13          # dense 2**J x 2**J float64 matrices: 0.5 GB at J = 13


def dynamic_junctions(network, wire_diameter):
    """Indices of the junctions whose two nodes lie in the connected component of the electrodes (the other junctions
    see 0 V and never switch); () if the electrodes are not connected."""
    c = DualNodeCircuit(network, wire_diameter)
    J = len(network.junctions)
    if J == 0 or len(network.electrode_in) == 0 or len(network.electrode_gnd) == 0:
        return np.empty(0, dtype=int)
    both = sp.coo_matrix((np.ones(2 * J), (np.r_[c.jn_a, c.jn_b], np.r_[c.jn_b, c.jn_a])), shape=(c.n, c.n)).tocsr()
    _, label = connected_components((c._fixed != 0) + both, directed=False)
    if label[c.vin] != label[c.gnd]:
        return np.empty(0, dtype=int)
    return np.nonzero(label[c.jn_a] == label[c.vin])[0]


def state_data(network, params, wire_diameter, dyn_ids, V_in):
    """For every state s (bit j of s = junction dyn_ids[j] ON) at drive V_in: (p_on, p_off, g_eff). p_on[s, j] = P(OFF -> ON in one step)
    (p_set of the memristor voltage), p_off[s, j] = P(ON -> OFF) = 1 - (1 - p_reset(I_mem)) (1 - p_relax) (as in update_dynamics),
    g_eff[s] = I_total / V_in. Junctions outside dyn_ids stay OFF."""
    J = len(dyn_ids)
    if J > MAX_JUNCTIONS:
        raise ValueError(f"{J} dynamic junctions exceed the limit {MAX_JUNCTIONS} (dense 2^J x 2^J matrices)")
    circuit = DualNodeCircuit(network, wire_diameter)
    bits = state_bits(J)
    S = 2 ** J
    p_on, p_off, g_eff = np.zeros((S, J)), np.zeros((S, J)), np.zeros(S)
    for s in range(S):
        G = np.full(len(network.junctions), params.G_off)
        G[dyn_ids[bits[s] == 1]] = params.G_on
        V = circuit.solve(G, V_in)
        dV = circuit.junction_voltage_drops(V)[dyn_ids]
        p_on[s] = p_set(np.abs(dV), params)
        p_off[s] = 1.0 - (1.0 - p_reset(G[dyn_ids] * dV, params)) * (1.0 - params.p_relax)
        g_eff[s] = circuit.total_current(V) / V_in
    return p_on, p_off, g_eff


def bridge_network():
    """Hand-built 4-junction test network: wire 0 (electrode in) and wire 2 (electrode gnd) are horizontal, wires 1 and 3 are
    vertical and cross both; all wires are 60 um long."""
    um = 1e-6
    ends = np.array([[[0, 0], [60, 0]], [[30, -30], [30, 30]], [[20, 20], [80, 20]], [[45, -10], [45, 50]]], dtype=float) * um
    junctions = np.array([[0, 1], [0, 3], [1, 2], [2, 3]])
    pos = np.array([[30, 0], [45, 0], [30, 20], [45, 20]], dtype=float) * um
    return Network(n_wires=4, wire_endpoints=ends, junctions=junctions, junction_pos=pos,
                   electrode_in=np.array([0]), electrode_gnd=np.array([2]), isolated_wires=np.array([], dtype=int),
                   area=(100 * um) ** 2, wire_length=60 * um)
