#!/usr/bin/env python3
"""Is a GPU worth it for the N12 runs? Times the per-step Kirchhoff solve (98-99 % of a run) on CPU and GPU.

Each simulation step solves A(G) x = b, A = the sparse reduced Laplacian of the network (12k unknowns uniform, 33k
clusters4_dense) with conductances spanning 1e-9 .. 1e-3 S. The project solves it on the CPU with conjugate gradients
preconditioned by an LU factorization of an earlier A, warm-started from the previous step (agnw_rc.multicircuit,
rtol 1e-13). An LU preconditioner needs sparse triangular solves, which GPUs do badly, so the GPU candidate here is
conjugate gradients with a Jacobi (diagonal) preconditioner, in float64 and float32, from the same warm start.

The script drives the network for a few hundred steps on the CPU (realistic, partly formed filaments), then times one
step k -> k+1 both ways and checks the GPU answer against an exact sparse LU solve:

    python scripts/gpu_benchmark.py [--geometry clusters4_dense] [--steps 300]

Needs CuPy for the GPU part (pip install cupy-cuda12x, or cupy-cuda11x for an older CUDA driver; check with
nvidia-smi). Without CuPy it runs the same Jacobi CG on the CPU with NumPy, which tests the code but not the GPU.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402


def jacobi_cg(xp, A, b, x0, rtol, maxiter):
    """Jacobi-preconditioned conjugate gradients with any array module xp (numpy or cupy). Returns x, iterations."""
    dinv = 1.0 / A.diagonal()
    x = x0.copy()
    r = b - A @ x
    z = dinv * r
    p = z.copy()
    rz = xp.dot(r, z)
    bnorm = float(xp.linalg.norm(b))
    for k in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rz / xp.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if float(xp.linalg.norm(r)) <= rtol * bnorm:
            return x, k
        z = dinv * r
        rz_new = xp.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="clusters4_dense")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--maxiter", type=int, default=20000)
    a = ap.parse_args()

    net = C.make_network(1, geometry=a.geometry)
    _, wd = C.network_params()
    mc = MultiTerminalCircuit(net, wd, make_layout("pads", net))
    vp = C.volatile_params(None, tau_s=0.3, tau_spread=1.0, dt=2e-3)
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    vf = np.zeros(len(mc.fixed_electrodes))
    driven = [k for k, i in enumerate(mc.fixed_electrodes) if mc.electrodes[i].role == "in"]
    vf[driven] = 0.45
    print(f"{a.geometry}: {len(net.junctions)} junctions, {mc.n} nodes; driving {a.steps} steps on the CPU ...", flush=True)
    t = time.time()
    for _ in range(a.steps):
        V = mc.solve(conductance(state, vp), vf, method="pcg")
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    cpu_step = (time.time() - t) / a.steps
    x_prev = V[mc.free].copy()
    G = conductance(state, vp)
    A = mc._matrix(G).tocsr()
    b = mc._B @ vf
    exact = splu(A.tocsc()).solve(b)
    print(f"G spans {G.min():.1e} .. {G.max():.1e} S; CPU (project solver, LU-preconditioned CG): "
          f"{cpu_step * 1e3:.1f} ms per step\n", flush=True)

    try:
        import cupy as xp
        import cupyx.scipy.sparse as xsp
        dev = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        where = f"GPU ({dev})"
    except Exception as err:                                        # no CuPy / no GPU: test the code on the CPU
        xp, xsp, where = np, sp, f"CPU with NumPy (no GPU: {type(err).__name__})"

    def sync():
        if xp is not np:
            xp.cuda.Stream.null.synchronize()

    print(f"Jacobi-preconditioned CG on {where}, warm start from the previous step:")
    for dtype, rtol in ((np.float64, 1e-13), (np.float64, 1e-10), (np.float32, 1e-6)):
        Ad = xsp.csr_matrix(A.astype(dtype))
        bd, x0 = xp.asarray(b.astype(dtype)), xp.asarray(x_prev.astype(dtype))
        jacobi_cg(xp, Ad, bd, x0, rtol, 5)                          # warm-up (kernel compilation)
        sync()
        t = time.time()
        x, it = jacobi_cg(xp, Ad, bd, x0, rtol, a.maxiter)
        sync()
        dt = time.time() - t
        x = np.asarray(x.get() if xp is not np else x, dtype=float)
        err = np.max(np.abs(x - exact)) / np.max(np.abs(exact))
        flag = "" if it < a.maxiter else "  (did NOT converge)"
        print(f"  {np.dtype(dtype).name:8s} rtol {rtol:.0e}: {it:6d} iterations, {dt * 1e3:8.1f} ms per step, "
              f"max error vs exact {err:.1e}{flag}")
    print("\nThe GPU pays off only if its ms per step (at an accuracy the results tolerate, ~1e-8 or better) is well "
          "below the CPU figure divided by the number of CPU cores (the CPU runs one simulation per core).")


if __name__ == "__main__":
    main()
