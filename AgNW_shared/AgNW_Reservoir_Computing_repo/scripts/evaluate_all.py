#!/usr/bin/env python3
"""Score every raw run in results/raw plus the no-nanowire baselines; write results/metrics.json and
results/summary.csv (mean and std over network seeds and replicates).

    python scripts/evaluate_all.py [--procs 12]
"""
import argparse
import csv
import json
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.criteria import drift  # noqa: E402
from agnw_rc.drive import input_stream, load_raw  # noqa: E402

RAW = ROOT / "results" / "raw"
SPEC_FIELDS = ("stream", "hold", "v_center", "network_seed", "replicate", "v_width", "p_relax", "n_samples",
               "model", "p_relax_max", "layout", "encoding", "sigma", "washout", "n_virtual", "stride",
               "tau_s", "tau_spread", "consolidation", "v_scale", "k_max", "rupture")


def spec_of(path: Path) -> C.RunSpec:
    with np.load(path) as f:
        kw = {k: f[f"spec_{k}"].item() for k in SPEC_FIELDS if f"spec_{k}" in f.files}
    return C.RunSpec(**kw)


def floor_and_baselines(seed: int):
    u, _ = input_stream(C.RunSpec("iid", 10, 1.0, seed))
    floor = E.noise_floor(u)
    us, lab = input_stream(C.RunSpec("sinesquare", 10, 1.0, seed))
    return seed, floor, E.iid_baselines(u, floor), E.sinesquare_baselines(us, lab)


def floor_for_length(args):
    """Noise floor and baselines for a shorter stream (fewer test samples -> larger chance capacities; the
    baselines must be scored on the same length, review M6)."""
    seed, n = args
    u = input_stream(C.RunSpec("iid", 10, 1.0, seed, n_samples=n))[0]
    us, lab = input_stream(C.RunSpec("sinesquare", 10, 1.0, seed, n_samples=n))
    floor = E.noise_floor(u)
    return (seed, n), (floor, E.iid_baselines(u, floor), E.sinesquare_baselines(us, lab))


def driven_current(raw) -> np.ndarray:
    """Total current into the network per sample at the block's last virtual node (A). Multi-electrode runs store
    the currents of all electrodes, which sum to zero, so only the driven ones are added."""
    if "electrode_roles" in raw:
        n_e = int(raw["n_electrodes"])
        last = raw["current"][:, -n_e:]
        return last[:, raw["electrode_roles"] == "in"].sum(axis=1)
    return raw["current"][:, -1]


def score_run(args):
    spec, floor = args
    raw = load_raw(spec, RAW)
    base = {k: getattr(spec, k) for k in SPEC_FIELDS}
    base["seconds"] = float(raw["seconds"])
    base["on_fraction_mean"] = float(raw["on_fraction"].mean())
    base["current_mean_mA"] = float(driven_current(raw).mean() * 1e3)
    base["drift"] = drift(raw["on_fraction"])
    if spec.stream == "iid":
        return {**base, "scores": E.evaluate_iid_run(raw, floor)}
    return {**base, "scores": E.evaluate_sinesquare_run(raw)}


def score_consistency(pair):
    a, b = pair
    base = {k: getattr(a, k) for k in SPEC_FIELDS if k != "replicate"}
    return {**base, "consistency": E.evaluate_consistency(load_raw(a, RAW), load_raw(b, RAW))}


KEY = ("stream", "hold", "v_center", "v_width", "p_relax", "n_samples", "model", "p_relax_max", "layout", "encoding",
       "sigma", "washout", "n_virtual", "stride", "tau_s", "tau_spread", "consolidation", "v_scale", "k_max", "rupture")


def group_key(r):
    return tuple(r[k] for k in KEY)


METRICS = {"iid": ("lag0", "MC", "IPC2", "narma10_nmse"), "sinesquare": ("accuracy",)}


def summarize(runs, consistency):
    """Mean and std per configuration over network seeds and the two standard replicates (0, 1). The extra
    replicates of N7a (2-7) are analysed by scripts/noise_averaging.py and kept out of the summary so that every
    cell has the same n."""
    rows = []
    groups = defaultdict(list)
    for r in runs:
        if r["replicate"] in C.E1_REPLICATES:
            groups[group_key(r)].append(r)
    cons = defaultdict(list)
    for c in consistency:
        cons[("iid",) + tuple(c[k] for k in KEY[1:])].append(c)
    for key, rs in sorted(groups.items()):
        stream = key[0]
        for obs in E.OBSERVABLES:
            row = dict(zip(KEY, key), observable=obs,
                       n=len(rs))
            for m in METRICS[stream]:
                v = [r["scores"][obs][m] for r in rs]
                row[f"{m}_mean"], row[f"{m}_std"] = float(np.mean(v)), float(np.std(v))
            if stream == "iid" and cons[key]:
                for suffix in ("", "_state", "_state_inc"):
                    v = [c["consistency"][obs + suffix] for c in cons[key]]
                    row[f"consistency{suffix}_mean"] = float(np.mean(v))
                    row[f"consistency{suffix}_std"] = float(np.std(v))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    specs = sorted((spec_of(p) for p in RAW.glob("*.npz") if not p.name.endswith(".tmp.npz")),
                   key=lambda s: s.name)
    seeds = sorted({s.network_seed for s in specs})
    print(f"{len(specs)} raw runs, network seeds {seeds}", flush=True)
    with ProcessPoolExecutor(max_workers=a.procs) as pool:
        fb = {s: (f, bi, bs) for s, f, bi, bs in pool.map(floor_and_baselines, seeds)}
        floors = {s: v[0] for s, v in fb.items()}
        short = sorted({(s.network_seed, s.n_samples) for s in specs if s.n_samples})
        short_fb = dict(pool.map(floor_for_length, short))
        short_floors = {k: v[0] for k, v in short_fb.items()}
        runs = list(pool.map(score_run, [(s, short_floors[(s.network_seed, s.n_samples)] if s.n_samples
                                          else floors[s.network_seed]) for s in specs]))
        by = {s.name: s for s in specs}
        pairs = []
        for s in specs:
            if s.stream == "iid" and s.replicate == 0:
                other = C.RunSpec(**{**s.__dict__, "replicate": 1})
                if other.name in by:
                    pairs.append((s, other))
        consistency = list(pool.map(score_consistency, pairs))
    out = {
        "framework_commit": C.framework_commit(),
        "config": {"N_WIRES": C.N_WIRES, "AREA": C.AREA, "LENGTH": C.LENGTH, "DT": C.DT, "P_RELAX": C.P_RELAX,
                   "N_SAMPLES": C.N_SAMPLES, "WASHOUT_STEPS": C.WASHOUT_STEPS, "N_PROBES": C.N_PROBES,
                   "N_VIRTUAL": C.N_VIRTUAL, "SPLIT": C.SPLIT},
        "noise_floor": {str(s): floors[s] for s in seeds},
        "noise_floor_short": {f"{s}_{n}": v for (s, n), v in short_floors.items()},
        "baselines_short": {f"{s}_{n}": {"iid": v[1], "sinesquare": v[2]} for (s, n), v in short_fb.items()},
        "baselines": {str(s): {"iid": fb[s][1], "sinesquare": fb[s][2]} for s in seeds},
        "runs": runs,
        "consistency": consistency,
        "summary": summarize(runs, consistency),
    }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=1))
    rows = out["summary"]
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                    KEY + ("observable", "n"), k))
    with open(ROOT / "results" / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote results/metrics.json and results/summary.csv")


if __name__ == "__main__":
    main()
