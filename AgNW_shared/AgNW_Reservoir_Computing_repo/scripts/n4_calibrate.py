#!/usr/bin/env python3
"""N4 calibration against Lamas Fig. 4 (docs/N4_DESIGN.md section 5).

    python scripts/n4_calibrate.py grid [--procs 14]          # coarse grid, N = 2000
    python scripts/n4_calibrate.py refine [--procs 14]        # local grid around the current best
    python scripts/n4_calibrate.py grid2 [--procs 14]         # slow dissolution x bipolar/unipolar (A-/A+ test)
    python scripts/n4_calibrate.py validate --V-set .. --b .. --lambda-crit ..   # N = 2000 and 3000, 4 seeds

Each point: one 50 s cycle of a 3.6 V sine (framework Fig. 4 protocol), noise-free (sigma = 0), network seed 1.
Traces go to results/n4_calib/*.npz; metrics and losses to results/n4_calibration.json (entries are merged, so
the stages accumulate). Loss: sum of squared relative errors on I_max, onset, A+ and A-/A+ vs the paper (N = 2000).
"""
import argparse
import itertools
import json
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.fig4 import loss, metrics_fn, paper_targets, sine_protocol  # noqa: E402

OUT = ROOT / "results" / "n4_calib"
JSON = ROOT / "results" / "n4_calibration.json"
GRID = {"V_set": (0.05, 0.1, 0.15, 0.2), "b": (1.0, 10.0, 100.0), "lambda_crit": (0.003, 0.01, 0.03)}


GRID2 = {"V_set": (0.05, 0.06, 0.07), "b": (0.01, 0.1, 1.0), "lambda_crit": (0.01, 0.03), "unipolar": (False, True)}


def key(V_set, b, lambda_crit, n_wires=2000, seed=1, unipolar=False):
    return f"N{n_wires}_s{seed}_Vset{V_set:g}_b{b:g}_lc{lambda_crit:g}" + ("_uni" if unipolar else "")


def run_point(args):
    V_set, b, lambda_crit, n_wires, seed, *rest = args
    unipolar = bool(rest[0]) if rest else False
    k = key(V_set, b, lambda_crit, n_wires, seed, unipolar)
    path = OUT / f"{k}.npz"
    t0 = time.time()
    if not path.exists():
        fp = C.filament_params(V_set=V_set, b=b, lambda_crit=lambda_crit, unipolar=unipolar)
        v, i, on = sine_protocol(fp, n_wires=n_wires, network_seed=seed, record_every=100)
        np.savez_compressed(path, V_in=v, I_total=i, on_frac=on)
    z = np.load(path)
    m = metrics_fn()(z["V_in"], z["I_total"] * 1e3)
    target = paper_targets()[n_wires]
    return k, {"V_set": V_set, "b": b, "lambda_crit": lambda_crit, "unipolar": unipolar, "n_wires": n_wires,
               "network_seed": seed,
               "metrics": m, "loss": loss(m, target), "seconds": time.time() - t0}


def run_all(points, procs):
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads(JSON.read_text()) if JSON.exists() else {}
    with ProcessPoolExecutor(max_workers=procs) as pool:
        futs = [pool.submit(run_point, p) for p in points]
        for i, f in enumerate(as_completed(futs), 1):
            k, r = f.result()
            res[k] = r
            m = r["metrics"]
            print(f"[{i}/{len(points)}] {k}: loss {r['loss']:.3f} | Imax {m['Imax']:.2f} onset {m['onset_V']:.2f} "
                  f"A+ {m['area_pos']:.2f} A-/A+ {m['ratio']:.2f} ({r['seconds']:.0f}s)", flush=True)
            JSON.write_text(json.dumps(res, indent=1))
    return res


def best(res, n_wires=2000):
    rows = [r for r in res.values() if r["n_wires"] == n_wires and r["network_seed"] == 1]
    return min(rows, key=lambda r: r["loss"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "grid2", "refine", "validate"])
    ap.add_argument("--unipolar", action="store_true")
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--V-set", type=float)
    ap.add_argument("--b", type=float)
    ap.add_argument("--lambda-crit", type=float)
    a = ap.parse_args()
    if a.stage == "grid":
        pts = [(v, b, lc, 2000, 1) for v, b, lc in itertools.product(*GRID.values())]
    elif a.stage == "grid2":
        pts = [(v, b, lc, 2000, 1, uni) for v, b, lc, uni in itertools.product(*GRID2.values())]
    elif a.stage == "refine":
        r = best(json.loads(JSON.read_text()))
        pts = [(r["V_set"] * fv, r["b"] * fb, r["lambda_crit"] * fl, 2000, 1, r.get("unipolar", False))
               for fv, fb, fl in itertools.product((0.8, 1.0, 1.25), (0.5, 1.0, 2.0), (0.7, 1.0, 1.4))]
    else:
        if None in (a.V_set, a.b, a.lambda_crit):
            ap.error("validate needs --V-set, --b and --lambda-crit")
        pts = [(a.V_set, a.b, a.lambda_crit, n, s, a.unipolar) for n in (2000, 3000) for s in (1, 2, 3, 4)]
    res = run_all(pts, a.procs)
    r = best(res)
    print(f"best (N=2000, seed 1): V_set {r['V_set']:g} b {r['b']:g} lambda_crit {r['lambda_crit']:g} "
          f"unipolar {r.get('unipolar', False)} loss {r['loss']:.3f}")


if __name__ == "__main__":
    main()
