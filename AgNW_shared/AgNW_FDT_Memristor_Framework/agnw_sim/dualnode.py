"""Dual-node circuit model (Lamas et al. 2026, Sec. 2.2).

Each junction j between wires a and b becomes two graph nodes, 2j (on wire a)
and 2j+1 (on wire b), joined by the memristor edge G[j]. Consecutive nodes on
the same wire are joined by a resistive segment with conductance
G_seg = A_cs / (rho * d), A_cs = pi (D/2)^2, d = distance along the wire.
Electrode wires get one extra node at the wire end inside the electrode band,
tied to the electrode by G_CONTACT.

The paper prints G_seg = 1/(d*rho), which omits the cross-section; the wire
diameter D is not stated there, so it is a parameter here.

Conditioning: G_seg spans up to ~1e2-1e3 S for very close junctions against
G_OFF = 1e-9 S. Segments are capped at G_SEG_MAX; for (near-)ideal wires use
the wire-as-node model in circuit.py instead (fat wires with D >~ 1 um make
the direct solve lose accuracy in the all-OFF regime).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

from .circuit import G_CONTACT

RHO_AG = 1.59e-8       # ohm m (Lamas 2026)
G_SEG_MAX = 1e3        # S, cap for (near-)coincident junctions on one wire


class DualNodeCircuit:
    """Sparse Kirchhoff solver on the dual-node graph of a Network. Build once
    per network and wire diameter; call solve() every time step."""

    def __init__(self, network, wire_diameter, rho=RHO_AG, g_seg_max=G_SEG_MAX):
        if not wire_diameter > 0:
            raise ValueError(f"wire_diameter must be > 0, got {wire_diameter!r}")
        self.network = network
        n_junctions = network.junctions.shape[0]
        ends = network.wire_endpoints
        L = network.wire_length
        area_cs = np.pi * (wire_diameter / 2) ** 2

        # (distance along wire, node id) for every node, grouped by wire
        per_wire = [[] for _ in range(network.n_wires)]
        for j, (a, b) in enumerate(network.junctions):
            for side, w in enumerate((a, b)):
                p1, p2 = ends[w]
                t = float(np.dot(network.junction_pos[j] - p1, (p2 - p1) / L))
                per_wire[w].append((t, 2 * j + side))
        n_nodes = 2 * n_junctions
        vin_nodes, gnd_nodes = [], []
        for group, out, left in ((network.electrode_in, vin_nodes, True),
                                 (network.electrode_gnd, gnd_nodes, False)):
            for w in group:
                x = ends[w][:, 0]
                k = int(np.argmin(x)) if left else int(np.argmax(x))
                per_wire[w].append((0.0 if k == 0 else L, n_nodes))
                out.append(n_nodes)
                n_nodes += 1

        rows, cols, vals = [], [], []
        for nodes in per_wire:
            nodes.sort()
            for (t1, n1), (t2, n2) in zip(nodes[:-1], nodes[1:]):
                dist = t2 - t1
                g = g_seg_max if dist <= 0 else min(area_cs / (rho * dist), g_seg_max)
                rows += [n1, n2]; cols += [n2, n1]; vals += [-g, -g]

        self.n = n_nodes + 2
        self.vin = n_nodes
        self.gnd = n_nodes + 1
        for nodes_, sup in ((vin_nodes, self.vin), (gnd_nodes, self.gnd)):
            for nd in nodes_:
                rows += [nd, sup]; cols += [sup, nd]; vals += [-G_CONTACT] * 2
        self._fixed = sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()
        self.jn_a = 2 * np.arange(n_junctions)
        self.jn_b = self.jn_a + 1
        self._vin_nodes = np.array(vin_nodes, dtype=int)

    def solve(self, G, V_in):
        """Node voltages for junction conductances G and electrode voltage V_in
        (GND = 0). Nodes not connected to either electrode are set to 0 V."""
        n = self.n
        V = np.zeros(n)
        V[self.vin] = V_in
        if self._fixed.nnz == 0 and len(G) == 0:
            return V
        off = sp.coo_matrix((np.concatenate([-G, -G]),
                             (np.concatenate([self.jn_a, self.jn_b]),
                              np.concatenate([self.jn_b, self.jn_a]))),
                            shape=(n, n)).tocsr()
        Y_off = self._fixed + off
        Y = (Y_off + sp.diags(-np.asarray(Y_off.sum(axis=1)).ravel())).tocsr()
        _, label = connected_components(Y_off != 0, directed=False)
        reachable = np.isin(label, [label[self.vin], label[self.gnd]])
        reachable[[self.vin, self.gnd]] = False        # Dirichlet nodes are not unknowns
        free = np.nonzero(reachable)[0]
        if free.size:
            fixed_idx = np.array([self.vin, self.gnd])
            rhs = -(Y[free][:, fixed_idx] @ V[fixed_idx])
            V[free] = splu(Y[free][:, free].tocsc()).solve(np.asarray(rhs).ravel())
        return V

    def junction_voltage_drops(self, V):
        """Signed V(node on wire a) - V(node on wire b) across each memristor."""
        return V[self.jn_a] - V[self.jn_b]

    def total_current(self, V):
        """Current delivered by the V_IN electrode into the network."""
        return float(np.sum(G_CONTACT * (V[self.vin] - V[self._vin_nodes])))
