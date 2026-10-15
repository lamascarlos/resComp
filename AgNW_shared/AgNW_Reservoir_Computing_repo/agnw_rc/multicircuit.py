"""Multi-terminal Kirchhoff solver on the framework's dual-node graph (NEXT_STEPS N5).

Same graph as agnw_sim.dualnode.DualNodeCircuit: each junction j is two nodes (2j on its first wire, 2j+1 on its
second) joined by the memristor edge G[j]; consecutive nodes on a wire are joined by resistive segments
G_seg = min(pi (D/2)^2 / (rho d), G_SEG_MAX); every wire touching an electrode gets a contact node at its contact
point, tied to the electrode's super-node by G_CONTACT. The difference is the electrode list: any number of electrodes,
each driven ("in"), grounded ("gnd") or floating ("float"). With agnw_rc.electrodes.two_bands the graph, node
numbering and solution coincide with DualNodeCircuit (tested).

Driven and grounded super-nodes are Dirichlet nodes. Junction edges never touch a super-node, so the right-hand side
is B @ V_electrodes with a constant matrix B. The factorization of the reduced matrix is reused while the junction
conductances do not change (most steps of the stochastic dynamics); `solve(..., method="pcg")` instead runs
LU-preconditioned conjugate gradients, for the mean-field dynamics where every junction changes a little each step.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg, splu

from agnw_sim.circuit import G_CONTACT
from agnw_sim.dualnode import G_SEG_MAX, RHO_AG


class MultiTerminalCircuit:
    def __init__(self, network, wire_diameter, electrodes, rho=RHO_AG, g_seg_max=G_SEG_MAX):
        if not wire_diameter > 0:
            raise ValueError(f"wire_diameter must be > 0, got {wire_diameter!r}")
        self.network = network
        self.electrodes = list(electrodes)
        n_j = network.junctions.shape[0]
        ends = network.wire_endpoints
        L = network.wire_length
        area_cs = np.pi * (wire_diameter / 2) ** 2

        per_wire = [[] for _ in range(network.n_wires)]
        for j, (a, b) in enumerate(network.junctions):
            for side, w in enumerate((a, b)):
                p1, p2 = ends[w]
                t = float(np.dot(network.junction_pos[j] - p1, (p2 - p1) / L))
                per_wire[w].append((t, 2 * j + side))
        n_nodes = 2 * n_j
        contacts = []                                   # per electrode: its contact node ids
        for e in self.electrodes:
            ids = []
            for w, t in zip(e.wires, e.t):
                per_wire[int(w)].append((float(t), n_nodes))
                ids.append(n_nodes)
                n_nodes += 1
            contacts.append(np.array(ids, dtype=int))

        rows, cols, vals = [], [], []
        for nodes in per_wire:
            nodes.sort()
            for (t1, n1), (t2, n2) in zip(nodes[:-1], nodes[1:]):
                dist = t2 - t1
                g = g_seg_max if dist <= 0 else min(area_cs / (rho * dist), g_seg_max)
                rows += [n1, n2]; cols += [n2, n1]; vals += [-g, -g]
        self.super = n_nodes + np.arange(len(self.electrodes))
        self.n = n_nodes + len(self.electrodes)
        for ids, sup in zip(contacts, self.super):
            for nd in ids:
                rows += [nd, sup]; cols += [sup, nd]; vals += [-G_CONTACT] * 2
        off = sp.coo_matrix((vals, (rows, cols)), shape=(self.n, self.n)).tocsr()
        L_fixed = (off + sp.diags(-np.asarray(off.sum(axis=1)).ravel())).tocsr()
        self.contacts = contacts
        self.jn_a = 2 * np.arange(n_j)
        self.jn_b = self.jn_a + 1

        # Dirichlet (driven/grounded) super-nodes and the free nodes reachable from them
        self.fixed = np.array([s for s, e in zip(self.super, self.electrodes) if e.role != "float"], dtype=int)
        if n_j:
            jpat = sp.coo_matrix((np.ones(2 * n_j), (np.r_[self.jn_a, self.jn_b], np.r_[self.jn_b, self.jn_a])),
                                 shape=(self.n, self.n))
            pattern = (off != 0) + (jpat != 0)
        else:
            pattern = off != 0
        _, label = connected_components(pattern, directed=False)
        reachable = np.isin(label, label[self.fixed])
        reachable[self.fixed] = False
        self.free = np.nonzero(reachable)[0]
        pos = -np.ones(self.n, dtype=int)
        pos[self.free] = np.arange(len(self.free))
        self._pos = pos

        self._L_ff = L_fixed[self.free][:, self.free].tocsr()
        self._B = -L_fixed[self.free][:, self.fixed].tocsr()              # rhs = B @ V_fixed
        inc = sp.lil_matrix((n_j, self.n))
        inc[np.arange(n_j), self.jn_a] = 1.0
        inc[np.arange(n_j), self.jn_b] = -1.0
        self._J = inc.tocsr()[:, self.free].tocsr()                       # junction incidence on free nodes
        self._lu = None
        self._G_lu = None
        self._x = None
        self.stats = {"factorizations": 0, "reuses": 0, "cg_solves": 0}

    # ------------------------------------------------------------------ helpers
    @property
    def fixed_electrodes(self):
        """Indices (into self.electrodes) of the driven/grounded electrodes, in the order of the V vector."""
        return [i for i, e in enumerate(self.electrodes) if e.role != "float"]

    def _matrix(self, G):
        return (self._L_ff + self._J.T @ sp.diags(G) @ self._J).tocsc()

    def _factor(self, G):
        self._A = self._matrix(G)
        self._lu = splu(self._A)
        self._G_lu = G.copy()
        self._M = LinearOperator(self._A.shape, matvec=self._lu.solve, dtype=float)
        self.stats["factorizations"] += 1

    # ---------------------------------------------------------------------- API
    def solve(self, G, V_fixed, method="reuse", rtol=1e-13, max_iter=30):
        """Node voltages for junction conductances G and the voltages of the driven/grounded electrodes (in the
        order of `fixed_electrodes`). Nodes not connected to any of them are 0 V.

        method: "reuse" (refactorize only when G changed; exact), "direct" (always refactorize), or "pcg"
        (conjugate gradients preconditioned by the last factorization; for slowly, continuously changing G)."""
        G = np.asarray(G, dtype=float)
        V_fixed = np.asarray(V_fixed, dtype=float)
        V = np.zeros(self.n)
        V[self.fixed] = V_fixed
        if len(self.free) == 0:
            return V
        b = self._B @ V_fixed
        if method == "direct" or self._lu is None:
            self._factor(G)
            x = self._lu.solve(b)
        elif method == "reuse":
            if np.array_equal(G, self._G_lu):
                self.stats["reuses"] += 1
            else:
                self._factor(G)
            x = self._lu.solve(b)
        elif method == "pcg":
            A = self._matrix(G)
            it = [0]

            def count(_):
                it[0] += 1
            x, info = cg(A, b, x0=self._x, M=self._M, rtol=rtol, atol=0.0, maxiter=max_iter, callback=count)
            self.stats["cg_solves"] += 1
            if info != 0 or it[0] > max_iter // 2:
                self._factor(G)
                x = self._lu.solve(b)
        else:
            raise ValueError(f"unknown method {method!r}")
        self._x = x
        V[self.free] = x
        return V

    def junction_voltage_drops(self, V):
        return V[self.jn_a] - V[self.jn_b]

    def electrode_currents(self, V):
        """Current flowing from each electrode into the network (A), in electrode order. They sum to zero."""
        return np.array([float(np.sum(G_CONTACT * (V[s] - V[ids]))) for s, ids in zip(self.super, self.contacts)])
