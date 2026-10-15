#!/usr/bin/env python3
"""N10-OR: does online (periodic) retraining of the readout rescue the N10 reservoir from junction ruptures?

No new simulations. N10-R runs (N10-E setup with random filament ruptures, rates 1e-5 to 3 /s per junction) and their
rupture-free twins, 16 electrode currents. The usual readout is trained once (first 60 % train, next 20 % validation)
and scored on the last 20 % (test). Here the test segment is predicted in blocks of R inputs; before each block the
ridge readout is retrained CAUSALLY on the W inputs just before it (last 20 % of the window chooses alpha), so a
readout can follow the network after a rupture rerouted the current. Scored exactly as score_iid on the concatenated
test predictions (MC, degree-2 capacity above the noise floor, NARMA-10 NMSE).
Settings: (W, R) = (600, 40) [long window], (300, 20) [main], (150, 10) [fast], and the static readout for reference (same code path check).

    python scripts/n10r_retrain.py [--procs N]   -> results/n10r_retrain.json, figures/fig27_n10r_retrain.png
"""
import argparse
import dataclasses
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
sys.path.insert(0, str(ROOT / "scripts"))

from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw, raw_path  # noqa: E402
from agnw_rc.metrics import capacity, nmse, thresholded_sum  # noqa: E402
from agnw_rc.readout import fit_predict  # noqa: E402
from agnw_rc.tasks import ipc2_targets, memory_targets, narma10_target, split_indices  # noqa: E402
import n10r_noise as R  # noqa: E402

OUT = ROOT / "results" / "n10r_retrain.json"
FIG = ROOT / "figures" / "fig27_n10r_retrain.png"
OBS = "electrodes"
MODES = {"static": None, "W600_R40": (600, 40), "W300_R20": (300, 20), "W150_R10": (150, 10)}


def score(F, u, floor, mode):
    mem, (ipc, _) = memory_targets(u), ipc2_targets(u)
    Y = np.hstack([mem, ipc, narma10_target(u)[:, None]])
    train, val, test = split_indices(len(u))
    if MODES[mode] is None:
        pred, _, _ = fit_predict(F, Y, train, val, test, min_std=E.MEAS_RES)
    else:
        W, Rb = MODES[mode]
        parts = []
        for k0 in range(0, len(test), Rb):
            blk = test[k0:k0 + Rb]
            win = np.arange(max(train[0], blk[0] - W), blk[0])
            cut = int(round(0.8 * len(win)))
            p, _, _ = fit_predict(F, Y, win[:cut], win[cut:], blk, min_std=E.MEAS_RES)
            parts.append(p)
        pred = np.vstack(parts)
    c = capacity(pred[:, :-1], Y[test, :-1])
    n_mem = mem.shape[1]
    return {"MC": thresholded_sum(c[1:n_mem], floor), "IPC2": thresholded_sum(c[n_mem:], floor),
            "narma10_nmse": nmse(pred[:, -1], Y[test, -1])}


def job(spec):
    out = {}
    for tag, sp in (("rupture", spec), ("no_rupture", dataclasses.replace(spec, rupture=0.0))):
        raw = load_raw(sp, R.RAW)
        F, u = E.observables(raw)[OBS], raw["u"]
        floor = E.noise_floor(u)
        out[tag] = {m: score(F, u, floor, m) for m in MODES}
    return spec, out


def main(procs):
    sp = [s for s in R.specs(("rupture_vvlow", "rupture_vlow", "rupture_low", "rupture")) if s.stream == "iid"
          and raw_path(s, R.RAW).exists() and raw_path(dataclasses.replace(s, rupture=0.0), R.RAW).exists()]
    with ProcessPoolExecutor(procs) as pool:
        res = list(pool.map(job, sp, chunksize=2))
    rows, summ = [], {}
    for s, o in res:
        rows.append({"rupture": s.rupture, "v_center": s.v_center, "network_seed": s.network_seed,
                     "replicate": s.replicate, **o})
        for m in MODES:
            tot = lambda d: d["MC"] + d["IPC2"]  # noqa: E731
            e = summ.setdefault(f"{s.rupture:g}", {}).setdefault(m, {"total": [], "ratio": [], "narma": []})
            e["total"].append(tot(o["rupture"][m]))
            e["ratio"].append(tot(o["rupture"][m]) / max(tot(o["no_rupture"]["static"]), 1e-9))
            e["narma"].append(o["rupture"][m]["narma10_nmse"])
    ref = {m: float(np.mean([r["no_rupture"][m]["MC"] + r["no_rupture"][m]["IPC2"] for r in rows])) for m in MODES}
    summary = {}
    for r, d in sorted(summ.items(), key=lambda kv: float(kv[0])):
        summary[r] = {m: {"total_mean": float(np.mean(v["total"])), "total_median": float(np.median(v["total"])),
                          "ratio_vs_static_twin_median": float(np.median(v["ratio"])),
                          "collapsed": int(np.sum(np.array(v["ratio"]) < 0.5)), "n": len(v["ratio"]),
                          "narma_median": float(np.median(v["narma"]))} for m, v in d.items()}
    OUT.write_text(json.dumps({"modes": {k: v for k, v in MODES.items()}, "no_rupture_total_mean": ref,
                               "summary": summary, "rows": rows}, indent=1))
    print("no-rupture twins, mean MC + IPC2:", {m: round(v, 2) for m, v in ref.items()})
    for r, d in summary.items():
        print(f"rupture {r:>6}/s: " + " | ".join(
            f"{m}: total {v['total_median']:.2f}, ratio {v['ratio_vs_static_twin_median']:.2f}, collapsed "
            f"{v['collapsed']}/{v['n']}, NARMA {v['narma_median']:.2f}" for m, v in d.items()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [float(r) for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in MODES:
        axes[0].plot(rates, [summary[f"{r:g}"][m]["total_median"] for r in rates], "o-", label=m)
        axes[1].plot(rates, [summary[f"{r:g}"][m]["collapsed"] / summary[f"{r:g}"][m]["n"] for r in rates], "o-", label=m)
    for m, ls in zip(MODES, ("-", "--", ":", "-.")):
        axes[0].axhline(ref[m], color="0.5", ls=ls, lw=0.8)
    axes[0].set(xscale="log", xlabel="rupture rate per junction (1/s)", ylabel="MC + degree-2 (median over runs)",
                title="capacity (grey: rupture-free twins)")
    axes[1].set(xscale="log", xlabel="rupture rate per junction (1/s)", ylabel="fraction of runs collapsed",
                title="collapses (< 0.5 x static rupture-free twin)")
    axes[0].legend(fontsize=8)
    fig.suptitle("N10-OR: periodic causal retraining of the readout (window W inputs, every R inputs) vs ruptures")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    main(ap.parse_args().procs)
