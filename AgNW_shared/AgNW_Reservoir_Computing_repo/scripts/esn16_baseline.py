#!/usr/bin/env python3
"""ESN baselines with 16 and 32 units on the 800-sample i.i.d. streams of networks 1-3 (as in evaluate_all's
short-stream baselines), for the G2 comparison of 16-electrode readouts (N10-E). -> results/esn16_baseline.json"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import input_stream  # noqa: E402


def one(seed):
    u = input_stream(C.RunSpec("iid", 10, 1.0, seed, n_samples=C.E4_N_SAMPLES))[0]
    b = E.iid_baselines(u, E.noise_floor(u), esn_sizes=(16, 32))
    return seed, {k: {m: b[k][m] for m in ("MC", "IPC2", "narma10_nmse")} for k in ("esn_16", "esn_32")}


if __name__ == "__main__":
    with ProcessPoolExecutor(3) as pool:
        res = dict(pool.map(one, C.E1_SEEDS))
    mean = {k: {m: sum(res[s][k][m] for s in res) / len(res) for m in ("MC", "IPC2", "narma10_nmse")}
            for k in ("esn_16", "esn_32")}
    (ROOT / "results" / "esn16_baseline.json").write_text(json.dumps({"per_seed": res, "mean": mean}, indent=1))
    print(mean)
