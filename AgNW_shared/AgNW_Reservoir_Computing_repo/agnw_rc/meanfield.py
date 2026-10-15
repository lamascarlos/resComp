"""Mean-field ablation (NEXT_STEPS N1): the framework's switching rates without the switching noise.

Each junction carries its ON probability P_j instead of a Bernoulli state. One step of the framework's
update_dynamics turns an OFF junction ON with probability p_set(|dV|) and an ON junction OFF with probability
p_off = 1 - (1 - p_reset(I_on)) (1 - p_relax), I_on = G_on |dV| (the current it would carry if ON). The exact
expectation of that update, given the voltages, is

    P <- P (1 - p_off) + (1 - P) p_set.

The circuit is solved with the mean conductances G_j = G_off + P_j (G_on - G_off). This is a mean-field
approximation (E[V(G)] != V(E[G])), which is exactly the point: same network, circuit, rates and interface,
no noise. p_set / p_reset are the framework's functions (agnw_sim.dynamics); the circuit is the framework's
dual-node graph.

Every junction changes a little at every step, so the framework's cached solver (which relies on few changed
junctions) does not help. MeanFieldSolver instead solves each step by preconditioned conjugate gradients, using
an LU factorization of a recent matrix as the preconditioner and refactorizing when CG needs too many iterations.
"""
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, splu

from agnw_sim.cached_solver import CachedSolver
from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.dynamics import p_reset, p_set

from . import config as C
from .drive import input_stream, predrive_stream, probe_nodes, to_voltage, virtual_offsets


class MeanFieldSolver:
    """Node voltages for continuous junction conductances; same contract as DualNodeCircuit.solve()."""

    def __init__(self, circuit: DualNodeCircuit, rtol: float = 1e-13, max_iter: int = 30):
        self._cs = CachedSolver(circuit)          # reduced-system structure of the framework's graph
        self.rtol, self.max_iter = rtol, max_iter
        self._lu = None
        self._x = None
        self.stats = {"refactorizations": 0, "cg_solves": 0, "cg_iterations": 0}

    def _refactor(self, A):
        self._lu = splu(A)
        self._M = LinearOperator(A.shape, matvec=self._lu.solve, dtype=float)
        self.stats["refactorizations"] += 1

    def solve(self, G, V_in):
        cs = self._cs
        V = np.zeros(cs._n)
        V[cs._vin] = V_in
        if len(cs._free) == 0:
            return V
        A = cs._assemble(np.asarray(G, dtype=float)[cs._jv])
        if self._lu is None:
            self._refactor(A)
            x = self._lu.solve(cs._b1)
        else:
            it = [0]

            def count(_):
                it[0] += 1
            x, info = cg(A, cs._b1, x0=self._x, M=self._M, rtol=self.rtol, atol=0.0, maxiter=self.max_iter,
                         callback=count)
            self.stats["cg_solves"] += 1
            self.stats["cg_iterations"] += it[0]
            if info != 0 or it[0] > self.max_iter // 2:
                self._refactor(A)
                x = self._lu.solve(cs._b1)
        self._x = x
        V[cs._free] = x * V_in
        return V


def meanfield_step(P, dV, params, p_relax=None):
    """Expected ON probability after one framework update step, given junction voltage drops dV. p_relax may be a
    per-junction array (NEXT_STEPS N2); default params.p_relax."""
    v = np.abs(dV)
    ps = p_set(v, params)
    pr = p_reset(params.G_on * v, params)
    p_relax = params.p_relax if p_relax is None else p_relax
    p_off = 1.0 - (1.0 - pr) * (1.0 - p_relax)
    return P * (1.0 - p_off) + (1.0 - P) * ps


def simulate_meanfield(spec: C.RunSpec, n_samples: int | None = None, washout_steps: int | None = None) -> dict:
    """Mean-field counterpart of drive.simulate (same inputs, same observables). The junction observable is the
    continuous P_j (stored as float16 in 'junction_p'); replicates differ only in the pre-drive."""
    t0 = time.time()
    params, wire_diameter = C.network_params(spec.p_relax)
    network = C.make_network(spec.network_seed)
    circuit = DualNodeCircuit(network, wire_diameter)
    solver = MeanFieldSolver(circuit)
    u, labels = input_stream(spec)
    if n_samples is not None:
        u = u[:n_samples]
        labels = None if labels is None else labels[:n_samples]
    u_pre = predrive_stream(spec)
    if washout_steps is not None:
        u_pre = u_pre[:-(-washout_steps // spec.hold)]
    hold = spec.hold
    v_pre, v_in = to_voltage(u_pre, spec), to_voltage(u, spec)
    probes = probe_nodes(network, circuit, C.N_PROBES, spec.network_seed)
    offsets = virtual_offsets(hold)
    n_j = network.junctions.shape[0]
    p_relax = C.junction_p_relax(spec, n_j)

    P = np.zeros(n_j)
    current = np.empty((len(u), len(offsets)))
    probe_v = np.empty((len(u), len(probes)))
    junction_p = np.empty((len(u), n_j), dtype=np.float16)
    i_scored = np.empty(len(u) * hold, dtype=np.float32)

    def step(P, V_applied):
        G = params.G_off + P * (params.G_on - params.G_off)
        V = solver.solve(G, V_applied)
        return meanfield_step(P, circuit.junction_voltage_drops(V), params, p_relax), V

    for v in v_pre:
        for _ in range(hold):
            P, _ = step(P, v)
    for k, v in enumerate(v_in):
        for s in range(hold):
            P_before = P
            P, V = step(P, v)
            i = circuit.total_current(V)
            i_scored[k * hold + s] = i
            if s == hold - 1:                       # state that produced V at the block's last step
                probe_v[k] = V[probes]
                junction_p[k] = P_before
        current[k] = i_scored[k * hold + offsets]

    return {
        "u": u, "labels": labels if labels is not None else np.empty(0, dtype=int),
        "v_in": v_in, "current": current, "virtual_offsets": offsets,
        "probe_v": probe_v, "probe_nodes": probes, "junction_p": junction_p,
        "n_junctions": n_j, "on_fraction": junction_p.astype(float).mean(axis=1),
        "i_total_scored": i_scored,
        "solver_stats": np.array([solver.stats[k] for k in sorted(solver.stats)]),
        "seconds": time.time() - t0, "model": "meanfield",
    }
