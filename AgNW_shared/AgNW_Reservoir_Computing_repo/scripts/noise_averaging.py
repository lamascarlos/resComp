#!/usr/bin/env python3
"""N7a noise-averaging curve (docs/NEXT_STEPS.md): how much of the stochastic model's deficit is switching noise
that averaging removes?

For each network seed, 8 noise replicates of the same E1 point (i.i.d. stream, hold 30, 1.25-2.25 V; each replicate
has its own switching noise and random pre-drive) are loaded. The readout features are averaged over R = 1, 2, 4, 8
replicates, the equivalent of R identical devices driven in parallel (or R repeated trials), and scored with the
usual tasks. Consistency (whole series and sample-to-sample increments, see evaluate.evaluate_consistency) is
measured between two disjoint groups of R replicates (R <= 4).

    python scripts/noise_averaging.py   ->  results/noise_averaging.json
"""
import json
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import load_raw  # noqa: E402
from agnw_rc.evaluate import (MEAS_RES, OBSERVABLES, noise_floor, observables, score_iid,  # noqa: E402
                              state_observables)
from agnw_rc.metrics import consistency  # noqa: E402
from agnw_rc.tasks import MAX_LAG  # noqa: E402

RAW = ROOT / "results" / "raw"
RS = (1, 2, 4, 8)
KEEP = ("lag0", "MC", "IPC2", "narma10_nmse", "memory_function", "ema_capacity")


def one_seed(seed):
    specs = [C.RunSpec("iid", C.N7A_HOLD, C.N7A_CENTER, seed, rep) for rep in C.N7A_REPLICATES]
    raws = [load_raw(s, RAW) for s in specs]
    u = raws[0]["u"]
    floor = noise_floor(u)
    feats = [observables(r) for r in raws]
    states = [state_observables(r) for r in raws]
    out = {"floor": floor, "scores": {}, "consistency": {}, "consistency_inc": {}}
    for R in RS:
        out["scores"][R] = {}
        for obs in OBSERVABLES:
            F = np.mean([f[obs] for f in feats[:R]], axis=0)
            s = score_iid(F, u, floor)
            out["scores"][R][obs] = {k: s[k] for k in KEEP}
        if 2 * R <= len(raws):
            out["consistency"][R], out["consistency_inc"][R] = {}, {}
            for obs in OBSERVABLES:
                a = np.mean([st[obs] for st in states[:R]], axis=0)[MAX_LAG:]
                b = np.mean([st[obs] for st in states[R:2 * R]], axis=0)[MAX_LAG:]
                out["consistency"][R][obs] = consistency(a, b, min_std=MEAS_RES)
                out["consistency_inc"][R][obs] = consistency(np.diff(a, axis=0), np.diff(b, axis=0),
                                                             min_std=MEAS_RES)
    print(f"seed {seed}: MC by R (junctions)",
          [round(out["scores"][R]["junctions"]["MC"], 2) for R in RS], flush=True)
    return seed, out


def main():
    with ProcessPoolExecutor(max_workers=len(C.E1_SEEDS)) as pool:
        res = dict(pool.map(one_seed, C.E1_SEEDS))
    path = ROOT / "results" / "noise_averaging.json"
    path.write_text(json.dumps({"hold": C.N7A_HOLD, "v_center": C.N7A_CENTER, "Rs": list(RS),
                                "per_seed": {str(k): v for k, v in res.items()}}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
