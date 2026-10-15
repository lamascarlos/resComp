"""Opt-in accelerated solver for the dual-node circuit.

Between two time steps only a handful of junctions change state (about 0.5 per
step in steady state), yet DualNodeCircuit.solve() rebuilds and refactorizes
the whole sparse matrix every step; the factorization is ~85% of the cost.
CachedSolver exploits three facts:

1. The topology never changes (G_OFF > 0 keeps every junction connected), so
   the reachable set of nodes is computed once.
2. The node voltages are linear in V_in. If no junction changed since the
   previous call, the previous unit-drive solution is simply rescaled (exact).
3. A junction that differs from the cached factorization is a rank-1 change
   A = A0 + d u u^T with u = e_a - e_b (a, b the two nodes of its memristor).
   A few of them are handled with the Woodbury identity: one extra triangular
   solve per newly changed junction plus a small dense system.

The conductances span 1e-9 to 1e3 S, so the rank update can lose accuracy
through cancellation (e.g. when a bridge junction switches). Every update is
therefore checked by the residual of the CURRENT system, and a failed check,
too many changed junctions, or an exhausted column cache triggers a fresh
factorization. The direct solver remains the default of run_simulation();
this class is opt-in (`solver="cached"`).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

from .circuit import G_CONTACT


class CachedSolver:
    """Cached / rank-update solver bound to one DualNodeCircuit.

    Args:
        circuit: the DualNodeCircuit whose graph is solved (its private
            fixed-edge matrix is read; both classes live in this package).
        max_updates: maximum number of junctions that may differ from the cached
            factorization before it is rebuilt.
        residual_tol: accepted relative residual (||b - A x|| / ||b||) of a
            rank-update solution.

    `stats` counts every call exactly once as a `rescale`, a `rank_update` or a
    `refactorization` (a rejected update is counted as both a
    `guard_rejection` and the refactorization that follows).
    """

    def __init__(self, circuit, max_updates=40, residual_tol=1e-10):
        self.circuit = circuit
        self.max_updates = int(max_updates)
        self.residual_tol = float(residual_tol)
        self._max_cols = 4 * self.max_updates
        self.stats = {"refactorizations": 0, "rank_updates": 0,
                      "rescales": 0, "guard_rejections": 0}

        n, vin, gnd = circuit.n, circuit.vin, circuit.gnd
        ja_all, jb_all = circuit.jn_a, circuit.jn_b
        n_junctions = len(ja_all)
        fixed = circuit._fixed.tocsr()

        # ---- static topology: reachable nodes never change
        if n_junctions:
            junction_pattern = sp.coo_matrix(
                (np.ones(2 * n_junctions),
                 (np.r_[ja_all, jb_all], np.r_[jb_all, ja_all])), shape=(n, n))
            all_edges = fixed + junction_pattern
        else:
            all_edges = fixed
        _, label = connected_components(all_edges != 0, directed=False)
        reachable = np.isin(label, [label[vin], label[gnd]])
        reachable[[vin, gnd]] = False
        self._n = n
        self._vin = vin
        self._free = np.nonzero(reachable)[0]
        n_free = len(self._free)
        position = -np.ones(n, dtype=int)
        position[self._free] = np.arange(n_free)

        # junctions inside the reachable set (both endpoints are, or neither is)
        valid = (position[ja_all] >= 0) & (position[jb_all] >= 0)
        self._jv = np.nonzero(valid)[0]
        self._ja = position[ja_all[self._jv]]
        self._jb = position[jb_all[self._jv]]

        # ---- reduced system  A x = b1  for a unit drive (V_IN = 1, GND = 0)
        self._b1 = np.zeros(n_free)
        for node in circuit._vin_nodes:
            if position[node] >= 0:
                self._b1[position[node]] += G_CONTACT
        if n_free == 0:
            return
        diag_fixed = -np.asarray(fixed.sum(axis=1)).ravel()            # all incident conductances
        off_fixed = fixed[self._free][:, self._free].tocoo()
        rows = np.r_[off_fixed.row, np.arange(n_free), self._ja, self._jb]
        cols = np.r_[off_fixed.col, np.arange(n_free), self._jb, self._ja]
        vals = np.r_[off_fixed.data, diag_fixed[self._free],
                     -np.ones(2 * len(self._jv))]                         # junction slots (overwritten)
        template = sp.coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsc()
        template.sort_indices()
        self._indices, self._indptr = template.indices, template.indptr
        marker = sp.csc_matrix((np.arange(1, template.nnz + 1, dtype=float),
                                self._indices, self._indptr), shape=(n_free, n_free))

        def where(r, c):
            return np.asarray(marker[r, c]).ravel().astype(int) - 1

        self._pos_ab, self._pos_ba = where(self._ja, self._jb), where(self._jb, self._ja)
        self._pos_aa, self._pos_bb = where(self._ja, self._ja), where(self._jb, self._jb)
        data0 = template.data.copy()
        data0[self._pos_ab] = 0.0
        data0[self._pos_ba] = 0.0
        self._data0 = data0                                              # diag holds the fixed part only

        self._lu = None
        self._G_last = None
        self._x_last = None

    # ------------------------------------------------------------------ helpers
    def _assemble(self, Gv):
        data = self._data0.copy()
        data[self._pos_ab] = -Gv
        data[self._pos_ba] = -Gv
        data[self._pos_aa] += Gv
        data[self._pos_bb] += Gv
        n_free = len(self._free)
        return sp.csc_matrix((data, self._indices, self._indptr), shape=(n_free, n_free))

    def _refactor(self, Gv):
        self._A0 = self._assemble(Gv)
        self._lu = splu(self._A0)
        self._x0 = self._lu.solve(self._b1)
        self._G0v = Gv.copy()
        self._W = np.empty((len(self._free), self._max_cols))
        self._col = {}
        self.stats["refactorizations"] += 1
        return self._x0

    def _rank_update(self, Gv):
        """Unit-drive solution for Gv via Woodbury on the cached factorization, or
        None if the update is not possible / not accurate enough."""
        changed = np.nonzero(Gv != self._G0v)[0]
        if len(changed) > self.max_updates:
            return None
        if len(changed) == 0:
            return self._x0
        n_free = len(self._free)
        for k in changed:
            if k not in self._col:
                if len(self._col) >= self._max_cols:
                    return None
                u = np.zeros(n_free)
                u[self._ja[k]] += 1.0
                u[self._jb[k]] -= 1.0
                self._col[k] = len(self._col)
                self._W[:, self._col[k]] = self._lu.solve(u)
        columns = [self._col[k] for k in changed]
        W = self._W[:, columns]
        a, b = self._ja[changed], self._jb[changed]
        d = Gv[changed] - self._G0v[changed]
        S = np.diag(1.0 / d) + (W[a, :] - W[b, :])
        try:
            y = np.linalg.solve(S, self._x0[a] - self._x0[b])
        except np.linalg.LinAlgError:
            return None
        x = self._x0 - W @ y
        # residual of the current system A = A0 + sum_k d_k u_k u_k^T
        ux = x[a] - x[b]
        residual = self._b1 - self._A0 @ x
        residual[a] -= d * ux
        residual[b] += d * ux
        if np.linalg.norm(residual) > self.residual_tol * np.linalg.norm(self._b1):
            self.stats["guard_rejections"] += 1
            return None
        return x

    # ---------------------------------------------------------------------- API
    def solve(self, G, V_in):
        """Node voltages for junction conductances G and electrode voltage V_in
        (GND = 0); same contract as DualNodeCircuit.solve()."""
        V = np.zeros(self._n)
        V[self._vin] = V_in
        if len(self._free) == 0:
            self.stats["rescales"] += 1
            return V
        G = np.asarray(G, dtype=float)
        if self._G_last is not None and np.array_equal(G, self._G_last):
            self.stats["rescales"] += 1
            x = self._x_last
        else:
            Gv = G[self._jv]
            x = self._rank_update(Gv) if self._lu is not None else None
            if x is None:
                x = self._refactor(Gv)
            else:
                self.stats["rank_updates"] += 1
            self._G_last = G.copy()
            self._x_last = x
        V[self._free] = x * V_in
        return V
