#!/usr/bin/env python3
"""N4 check: calibrated continuous-filament network under a ZERO-MEAN (bipolar) input, V = A (2u - 1), A = 3 V,
two bands, hold 30 steps, 400 samples after a 3000-step pre-drive, 2 replicates (network seed 1). Tests whether the
model's bipolarity (negative V drives Lambda back through zero) makes the state follow the input; the N4-1 DC-biased
drive left the network static (docs/RESULTS.md, N4-1 section).

    python scripts/n4_bipolar_check.py   ->  results/n4_bipolar_check.json   (~4 min)
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
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

A = 3.0


def main():
    specs = [C.RunSpec("iid", 30, 0.0, 1, r, v_width=2 * A, layout="two_bands", encoding="direct",
                       model="filament") for r in (0, 1)]
    raws = [simulate_multi(s, n_samples=400, washout_steps=3000) for s in specs]
    floor = E.noise_floor(raws[0]["u"], seeds=(0,), dims=(32,))
    scores = E.evaluate_iid_run(raws[0], floor)
    cons = E.evaluate_consistency(*raws)
    out = {"amplitude_V": A, "hold": 30, "n_samples": 400,
           "closed_fraction_mean": float(raws[0]["on_fraction"].mean()),
           "closed_fraction_max": float(raws[0]["on_fraction"].max()),
           "max_abs_lambda_over_crit": float(np.abs(raws[0]["junction_lambda"]).max()),
           "consistency": {k: (None if np.isnan(v) else v) for k, v in cons.items()},
           "scores": {o: {k: s[k] for k in ("lag0", "MC", "IPC2", "narma10_nmse")} for o, s in scores.items()}}
    (ROOT / "results" / "n4_bipolar_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("closed_fraction_mean", "closed_fraction_max",
                                          "max_abs_lambda_over_crit")}, indent=1))
    print("saved results/n4_bipolar_check.json")


if __name__ == "__main__":
    main()
