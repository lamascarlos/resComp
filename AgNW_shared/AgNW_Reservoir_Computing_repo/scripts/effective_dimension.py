#!/usr/bin/env python3
"""Effective dimension of each readout's features: participation ratio PR = (sum l_i)^2 / sum l_i^2 of the
eigenvalues l_i of the feature covariance (after z-scoring features; features varying less than the measurement
resolution are dropped). PR ~ number of independent directions the state explores. It is compared with the echo-state
networks used as baselines, on the same i.i.d. input stream (network seed 1, replicate 0; test of the "all junctions
driven by one scalar -> correlated, low-dimensional state" hypothesis of docs/NEXT_STEPS.md, N2 result).

    python scripts/effective_dimension.py   ->  results/effective_dimension.json
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

from agnw_rc import baselines as B  # noqa: E402
from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import load_raw  # noqa: E402
from agnw_rc.evaluate import MEAS_RES, OBSERVABLES, observables, state_observables  # noqa: E402
from agnw_rc.tasks import MAX_LAG  # noqa: E402

RAW = ROOT / "results" / "raw"

CASES = {   # label: RunSpec (seed 1, replicate 0, i.i.d. stream)
    "stochastic E1 (hold 30, 1.75 V)": C.RunSpec("iid", 30, 1.75, 1, 0),
    "stochastic E3 (hold 1000, 1.75 V)": C.RunSpec("iid", 1000, 1.75, 1, 0, n_samples=C.E3_N_SAMPLES),
    "mean-field M1 (hold 30, 5.5 V)": C.RunSpec("iid", 30, 5.5, 1, 0, model="meanfield"),
    "mean-field N2 log-unif. 1e-4..1e-1 (hold 30, 5.5 V)": C.RunSpec("iid", 30, 5.5, 1, 0, model="meanfield",
                                                                     p_relax_max=1e-1),
}


def participation_ratio(F: np.ndarray) -> float:
    F = np.asarray(F, dtype=float)
    keep = F.std(axis=0) > MEAS_RES
    if not keep.any():
        return 0.0
    Z = (F[:, keep] - F[:, keep].mean(axis=0)) / F[:, keep].std(axis=0)
    lam = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def main():
    out = {}
    for label, spec in CASES.items():
        raw = load_raw(spec, RAW)
        F, S = observables(raw), state_observables(raw)
        out[label] = {}
        for obs in OBSERVABLES:
            out[label][obs] = {"n_features": int(F[obs].shape[1]),
                               "PR": participation_ratio(F[obs][MAX_LAG:]),
                               "PR_state": participation_ratio(S[obs][MAX_LAG:])}
        print(label, {k: round(v["PR_state"], 2) for k, v in out[label].items()}, flush=True)
    u = load_raw(C.RunSpec("iid", 30, 1.75, 1, 0), RAW)["u"]
    for n in (32, 200):
        X = B.esn_states(u, n, 0.9, 0.5, seed=0)
        out[f"ESN {n} units (rho 0.9, sigma 0.5)"] = {"states": {"n_features": n,
                                                                 "PR": participation_ratio(X[MAX_LAG:])}}
        print(f"ESN {n}", round(out[f"ESN {n} units (rho 0.9, sigma 0.5)"]["states"]["PR"], 2))
    path = ROOT / "results" / "effective_dimension.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
