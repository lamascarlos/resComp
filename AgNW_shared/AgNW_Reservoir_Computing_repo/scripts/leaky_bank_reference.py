#!/usr/bin/env python3
"""Idealized upper reference for relaxational junctions: a noise-free bank of K independent leaky integrators
x_i[t] = a_i x_i[t-1] + (1 - a_i) u[t], with time constants tau_i (in input samples) log-uniform between tau_min and
tau_max. This is what a network of first-order, purely relaxational elements driven by one scalar input could at best
provide to a linear readout (no noise, no circuit coupling, perfectly linear). A variant adds a static nonlinearity
(x_i -> x_i + x_i^2) for degree-2 capacity. Compared with the ESN, which has recurrent, rotating modes.

Scored with the same readout and targets as every reservoir run (agnw_rc.evaluate.score_iid), on the i.i.d. stream of
network seed 1.

    python scripts/leaky_bank_reference.py   ->  results/leaky_bank_reference.json
"""
import json
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import input_stream  # noqa: E402
from agnw_rc.evaluate import noise_floor, score_iid  # noqa: E402
from agnw_rc.metrics import capacity  # noqa: E402
from agnw_rc.readout import ALPHAS, fit_predict  # noqa: E402
from agnw_rc.tasks import memory_targets, split_indices  # noqa: E402


def leaky_bank(u, taus, quadratic=False):
    a = np.exp(-1.0 / np.asarray(taus, dtype=float))
    x = np.zeros(len(a))
    X = np.empty((len(u), len(a)))
    for t, ut in enumerate(u):
        x = a * x + (1 - a) * ut
        X[t] = x
    return np.hstack([X, X ** 2]) if quadratic else X


def main():
    u, _ = input_stream(C.RunSpec("iid", 10, 1.0, 1))
    floor = noise_floor(u)
    out = {"noise_floor": floor, "banks": {}}
    for K in (32, 200):
        for tau_max in (10, 100, 1000):
            for quad in (False, True):
                taus = np.logspace(0, np.log10(tau_max), K)          # 1 .. tau_max samples
                r = score_iid(leaky_bank(u, taus, quad), u, floor)
                key = f"K={K}, tau=1..{tau_max}" + (", +x^2" if quad else "")
                out["banks"][key] = {k: r[k] for k in ("MC", "IPC2", "narma10_nmse", "lag0")}
                print(f"{key:28s} MC {r['MC']:.2f}  IPC2 {r['IPC2']:.2f}  NARMA {r['narma10_nmse']:.3f}", flush=True)
    # Review 2026-09-27: with the default ridge grid (alpha >= 1e-6) the noise-free bank is limited by the grid
    # floor, not by its dynamics. Report it also with a wider grid (alpha >= 1e-14) and with additive feature noise
    # (relative SNR per feature), which is what limits a physical, noisy bank.
    u_ = u
    taus = np.logspace(0, 2, 200)
    X = leaky_bank(u_, taus)
    mem = memory_targets(u_)
    tr, va, te = split_indices(len(u_))
    out["noise"] = {}
    for label, snr, alphas in (("noise-free, default grid", 0, ALPHAS),
                               ("noise-free, wide grid (alpha >= 1e-14)", 0, np.logspace(-14, 4, 19)),
                               ("SNR 1e4", 1e4, np.logspace(-14, 4, 19)),
                               ("SNR 1e2", 1e2, np.logspace(-14, 4, 19))):
        Xn = X if not snr else X + np.random.default_rng(1).standard_normal(X.shape) * X.std(0) / np.sqrt(snr)
        pred, best, _ = fit_predict(Xn, mem, tr, va, te, alphas=alphas, min_std=1e-14)
        c = capacity(pred, mem[te])
        out["noise"][label] = {"MC": float(c[1:].sum()), "alpha_at_grid_min": float(np.mean(best == alphas[0]))}
        print(f"K=200, tau=1..100, {label:40s} MC {c[1:].sum():.2f} (alpha at grid min for "
              f"{np.mean(best == alphas[0]):.0%} of lags)", flush=True)
    path = ROOT / "results" / "leaky_bank_reference.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
