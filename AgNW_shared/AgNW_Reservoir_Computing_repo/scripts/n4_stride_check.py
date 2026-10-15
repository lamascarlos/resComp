#!/usr/bin/env python3
"""Review P0.3: validate a coarser N4 integration step (stride x 1 ms) before using it in E4.

Two checks at the calibrated N4 parameters (config.N4_PARAMS), noise-free:
  1. Fig. 4 protocol (N = 2000, network seed 1) at stride 1, 2, 5, 10: every calibration metric (I_max, onset,
     A+, A-/A+) within 2 % of stride 1.
  2. Reservoir features: an E4-like run (hold 300 ms, washout shortened to WASHOUT_CHECK, 40 samples) at stride 1
     and config.E4_STRIDE (first run 10, which failed check 1; now 2). End-of-hold junction states (Lambda / Lambda_crit) and end-of-hold driven current compared
     sample by sample (relative RMS difference and correlation).

    python scripts/n4_stride_check.py [--procs 6]

Output: results/n4_stride_check.json.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.fig4 import LOSS_KEYS, metrics_fn, sine_protocol  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

OUT = ROOT / "results" / "n4_stride_check.json"
CACHE = ROOT / "results" / "n4_stride_check"            # traces, so an interrupted check resumes
STRIDES = (1, 2, 5, 10)
TOL = 0.02
FEATURE_SAMPLES = 40
WASHOUT_CHECK = 30_000       # 30 s: enough to compare two integrators started identically (not a stationarity run)


def fig4_point(stride):
    path = CACHE / f"fig4_stride{stride}.npz"
    if not path.exists():
        v, i, _ = sine_protocol(C.filament_params(stride=stride))
        np.savez_compressed(path, V_in=v, I_total=i)
    z = np.load(path)
    return stride, metrics_fn()(z["V_in"], z["I_total"] * 1e3)


def feature_run(stride):
    path = CACHE / f"features_stride{stride}.npz"
    if not path.exists():
        spec = replace(C.e4_specs(("filament",))[0], stride=stride, washout=WASHOUT_CHECK)
        raw = simulate_multi(spec, n_samples=FEATURE_SAMPLES)
        n_e = int(raw["n_electrodes"])
        i_end = raw["current"][:, -n_e:][:, raw["electrode_roles"] == "in"].sum(axis=1)   # last virtual node
        np.savez_compressed(path, lam=raw["junction_lambda"].astype(float), i_end=i_end)
    z = np.load(path)
    return stride, {"lam": z["lam"], "i_end": z["i_end"]}


def compare(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return {"rel_rms": float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-30)),
            "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.procs) as pool:
        f_fig4 = [pool.submit(fig4_point, s) for s in STRIDES]
        f_feat = [pool.submit(feature_run, s) for s in (1, C.E4_STRIDE)]
        fig4 = dict(f.result() for f in f_fig4)
        feat = dict(f.result() for f in f_feat)
    ref = fig4[1]
    out = {"tolerance": TOL, "fig4": {}, "features": {}}
    for s in STRIDES:
        dev = {k: float((fig4[s][k] - ref[k]) / ref[k]) for k in LOSS_KEYS}
        out["fig4"][str(s)] = {"metrics": {k: float(fig4[s][k]) for k in LOSS_KEYS}, "rel_dev": dev,
                               "pass": bool(all(abs(x) <= TOL for x in dev.values()))}
    a, b = feat[1], feat[C.E4_STRIDE]
    out["features"] = {"stride": C.E4_STRIDE, "n_samples": FEATURE_SAMPLES,
                       "junction_lambda": compare(a["lam"], b["lam"]), "current_end": compare(a["i_end"], b["i_end"])}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
