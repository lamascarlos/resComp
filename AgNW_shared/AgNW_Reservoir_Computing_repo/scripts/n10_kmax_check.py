#!/usr/bin/env python3
"""N10: why the step-size check failed, and the fix (2026-09-27). Milano's potentiation rate k_p explodes above
threshold; with k dt >> 1 the exact junction update jumps to the equilibrium for the frozen voltage, and the
junction-circuit feedback makes the result depend on the step. Compare an N10-1-like run (hold 100 ms, 60 samples,
tau_s 0.3 s, 2.75 V) at stride 1 vs 2 and 10, for caps k_max on k_p (inf = Milano) and two rupture rates k_P0.

    python scripts/n10_kmax_check.py     -> results/n10_kmax_check.json

Result: k_P0 does not matter here (no rupture in this window); k_max = inf or 1e3: stride 2 off by 5 % (current)
and 21 % (states); k_max = 100: 0.2 % / 1.6 % -> config.VOLATILE_K_MAX = 100, N10_STRIDE = 2.
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
import agnw_rc.multidrive as M  # noqa: E402

OUT = ROOT / "results" / "n10_kmax_check.json"
K_MAX = (np.inf, 1e3, 1e2)
K_P0 = (1e3, 1e2)
STRIDES = (1, 2, 10)


def run(a):
    kmax, kP0, stride = a
    orig = C.volatile_params
    C.volatile_params = lambda spec=None, **o: orig(spec, **{"k_max": kmax, "k_P0": kP0, **o})
    spec = C.RunSpec("iid", 100, 2.75, 1, 0, n_samples=60, layout="two_bands", encoding="direct", model="volatile",
                     tau_s=0.3, stride=stride, washout=3000, n_virtual=10)
    raw = M.simulate_multi(spec, save_states=True)
    n_e = int(raw["n_electrodes"])
    i_end = raw["current"][:, -n_e:][:, raw["electrode_roles"] == "in"].sum(axis=1)
    return a, i_end, raw["junction_x"].astype(float)


def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def main():
    cases = [(k, kp, s) for k in K_MAX for kp in K_P0 for s in STRIDES]
    with ProcessPoolExecutor(min(18, len(cases))) as pool:
        res = {a: (i, x) for a, i, x in pool.map(run, cases)}
    rows = []
    for k in K_MAX:
        for kp in K_P0:
            i1, x1 = res[(k, kp, 1)]
            rows.append({"k_max": str(k), "k_P0": kp, "mean_x": float(x1.mean()), "I_mean_mA": float(i1.mean() * 1e3),
                         "rel_err": {str(s): {"current": rel(res[(k, kp, s)][0], i1), "states": rel(res[(k, kp, s)][1], x1)}
                                     for s in STRIDES[1:]}})
    OUT.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
