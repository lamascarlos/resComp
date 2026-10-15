"""Circuit solve: sparse admittance assembly and Dirichlet reduction."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

# Ideal electrode-contact conductance (S). Much larger than G_ON=1e-3 S so
# bus-bar contact resistance is negligible next to junction resistance.
G_CONTACT = 1.0


def _build_full_admittance(network, G):
    """Build the (n_wires+2) x (n_wires+2) admittance matrix. Node
    n_wires is the V_IN electrode super-node, node n_wires+1 is GND."""
    n = network.n_wires + 2
    vin_node = network.n_wires
    gnd_node = network.n_wires + 1

    rows, cols, vals = [], [], []

    def add_edge(p, q, g):
        rows.extend([p, q])
        cols.extend([q, p])
        vals.extend([-g, -g])

    if network.junctions.size:
        for (k, l), g in zip(network.junctions, G):
            add_edge(int(k), int(l), g)

    for w in network.electrode_in:
        add_edge(int(w), vin_node, G_CONTACT)
    for w in network.electrode_gnd:
        add_edge(int(w), gnd_node, G_CONTACT)

    if not rows:
        return sp.csr_matrix((n, n))

    Y_offdiag = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    diag = -np.asarray(Y_offdiag.sum(axis=1)).ravel()
    Y = Y_offdiag + sp.diags(diag)
    return Y.tocsr()


def solve_circuit(network, G, V_in):
    """Solve for node voltages given junction conductances and the applied
    V_IN electrode voltage (GND fixed at 0V).

    Wires whose connected component (via junctions of any conductance)
    does not include either electrode are set to 0.0V, per spec §9 -- they
    carry no current and their absolute potential is undefined.

    Returns:
        V: (n_wires+2,) voltage array.
    """
    n = network.n_wires + 2
    vin_node = network.n_wires
    gnd_node = network.n_wires + 1

    Y = _build_full_admittance(network, G)

    V = np.zeros(n)
    V[vin_node] = V_in
    V[gnd_node] = 0.0

    if Y.nnz == 0:
        return V

    _, labels = connected_components(Y, directed=False)
    electrode_components = {labels[vin_node], labels[gnd_node]}
    reachable = np.isin(labels, list(electrode_components))
    reachable[network.isolated_wires] = False

    fixed_mask = np.zeros(n, dtype=bool)
    fixed_mask[vin_node] = True
    fixed_mask[gnd_node] = True
    free_mask = reachable & ~fixed_mask

    free_idx = np.nonzero(free_mask)[0]
    if free_idx.size:
        fixed_idx = np.array([vin_node, gnd_node])
        Y_ff = Y[free_idx][:, free_idx].tocsc()
        Y_fd = Y[free_idx][:, fixed_idx]
        V_d = V[fixed_idx]
        rhs = -(Y_fd @ V_d)
        lu = splu(Y_ff)
        V_f = lu.solve(np.asarray(rhs).ravel())
        V[free_idx] = V_f

    return V


def junction_currents(network, G, V):
    """Signed current through each junction: I_kl = G_kl * (V_k - V_l)."""
    if network.junctions.size == 0:
        return np.zeros(0)
    k = network.junctions[:, 0]
    l = network.junctions[:, 1]
    return G * (V[k] - V[l])


def total_current(network, G, V):
    """Total current delivered by the V_IN electrode into the network
    (positive when current flows from V_IN into the network)."""
    vin_node = network.n_wires
    if network.electrode_in.size == 0:
        return 0.0
    contact_currents = G_CONTACT * (V[vin_node] - V[network.electrode_in])
    return float(np.sum(contact_currents))
